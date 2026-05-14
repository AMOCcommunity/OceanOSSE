# ===================================================================
# Copyright 2025 National Oceanography Centre
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#  http://www.apache.org/licenses/LICENSE-2.0.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
# ===================================================================

"""
sampler_nearest_neighbour.py

Description: Sampling module for OceanOSSE package.

Created By: OceanOSSE Development Team (NOC, UK)
"""

# -- Import Dependencies -- #
from __future__ import annotations

import logging

import xarray as xr
import numpy as np

from OceanOSSE.utils import import_class
from OceanOSSE.sampling.sampler import ErrorKernel, ObsSampler

logger = logging.getLogger(__name__)


class NNSampler(ObsSampler):
    """
    Class for sampling gridded ocean model output analogously
    to an ocean observing platform (e.g., Argo floats).
    """
    def __init__(self, error_kernels: list[ErrorKernel] | None = None):
        # -- Validate Inputs -- #
        if error_kernels is not None:
            if not isinstance(error_kernels, list):
                raise TypeError(
                    "`error_kernels` must be a list of ErrorKernel instances."
                )
            for n, kernel in enumerate(error_kernels):
                if not isinstance(kernel, ErrorKernel):
                    raise TypeError(f"`error_kernels[{n}]` must be an ErrorKernel.")

        # -- Class Attributes -- #
        self._error_kernels = error_kernels
        
    def from_config(self, config: dict) -> Self:
        """
        Parameterss
        ----------
        config : dict
            Configuration dictionary containing input parameters from .toml
            configuration file.

        Returns
        -------
        self
            Initialised ObsSampler instance.
        """
        
        return self
    

    def collect_samples(self, ds, profile) -> xr.Dataset:
        """
        Parameters
        ----------
        ds : xarray.Dataset
            Gridded ocean model output dataset.
        profile : xarray.Dataset
            loaded observation data locations.
            
        Returns
        -------
        xarray.Dataset
            Sampled synthetic observations dataset.
        """
        i_nn, j_nn = self.find_nearest_ij(ds, profile)
        t_nn = self.find_nearest_time(ds, profile)
        
        ds_synth = self.extract_locations(ds, i_nn, j_nn, t_nn)
        
        return ds_synth
    
    
    def apply_errors(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Parameters
        ----------
        ds : xarray.Dataset
            Synthetic observations dataset.

        Returns
        -------
        xarray.Dataset
            Synthetic observations dataset with all error kernels
            applied in order.
        """
        # -- Apply each Error Kernel sequentially -- #
        if self._error_kernels is not None:
            for kernel in self._error_kernels:
                logger.debug(f"Applying ErrorKernel --> {repr(kernel)}")
                ds = kernel.apply(ds)
            logging.info(
                "--> Completed: Applied ErrorKernels to synthetic observations."
            )

        return ds
    
    
    def sample(self, ds: xr.Dataset, profile: xr.Dataset) -> xr.Dataset:
        """
        Perform sampling pipeline for chosen ocean observing platform.
        
        Parameters
        ----------
        ds : xarray.Dataset
            Gridded ocean model dataset.
        profile : xarray.Dataset observation profile dataset

        Returns
        -------
        xarray.Dataset
            Synthetic observations dataset with errors applied.
        """
        # -- Sample the gridded ocean model output -- #
        ds_sampled = self.collect_samples(ds, profile)
        logging.info(
            "--> Completed: Collected samples from ocean model dataset using ObsSampler."
        )

        # -- Apply error kernels sequentially to the synthetic observations -- #
        ds_obs = self.apply_errors(ds_sampled)

        return ds_obs
    
    
    def find_nearest_ij(self, ds, profile):
        """
        Turn observation lat and lon into model index

        Parameters
        ----------
        ds : xarray.Dataset
            Gridded ocean model dataset.
        profile : xarray.Dataset 
            observation profile dataset

        Return
        index: indicies of model in i an j
        """

        lon_sub = np.abs(ds.lon - profile.lon)
        lat_sub = np.abs(ds.lat - profile.lat)
        dist = ((lon_sub + lat_sub) / 2)
        # Stack along new gridpoint dimension
        dist = dist.stack(gridpoint=("j", "i"))
        
        # Tiny tie-break penalties to sort dist, j , i
        # Gives consitent results and 0.5 rounds up
        if (dist.min("gridpoint") == 0.5).any():
            score = (
                dist
                - 1e-6 * dist["j"]
                - 1e-9 * dist["i"]
                )
        else:
            score = dist
        
        # Find nearest
        nearest = score.argmin("gridpoint")
        ji = score["gridpoint"].isel(gridpoint=nearest)

        i_nn = ji["i"]
        j_nn = ji["j"]
        i_nn = i_nn.drop_vars("gridpoint")
        j_nn = j_nn.drop_vars("gridpoint")

        return i_nn, j_nn

    
    def find_nearest_time(self, ds, profile, thresh=10):
        """
        Turn observation time into model time index

        Parameters
        ----------
        ds : xarray.Dataset
            Gridded ocean model dataset.
        profile : xarray.Dataset 
            observation profile dataset
        thresh : int 
            threshold in model timesteps for a profile being out of time bounds

        Return
        index: indicies of model in time
        """
        # Time difference in microsec
        time_delta = np.abs(ds.time - profile.time)

        # Find nearest and take first occurance (i.e. round down)
        nearest = time_delta.argmin("t")
        t_near = time_delta.isel(t=nearest)

        t_nn = t_near["t"]
        
        n_profile = len(profile.coords['profile_id'])
        flag = np.zeros((n_profile))
        for p in range(n_profile):
            ps = profile.coords['profile_id'][p].to_numpy()
            if time_delta.isel(profile_id=ps).min() > ((ds.time.isel(t=1) - ds.time.isel(t=0)) * thresh):
                flag[p] = 1
                # I'm not sure what the best behaviour is for out of bounds profiles:
                # raise error, return a flag that will skip the profile 
                # or fill a blank profile
                raise ValueError("Profile time is outside model time bounds.")
        
        return t_nn#, flag
    
    
    def extract_locations(self, ds, i_index, j_index, t_index):
        """
        Extract a model profile at the specified model index.

        Parameters
        ----------
        ds : xarray.Dataset
            Gridded ocean model dataset.
        i_index : observation index on model grid in i direction
        j_index : observation index on model grid in j direction

        Return
        xarray.Dataset
            Synthetic observations dataset
        """

        ds_synth = ds.isel(i=i_index, j=j_index, t=t_index)
        
        return ds_synth
        