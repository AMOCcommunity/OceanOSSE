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
from xarray.indexes import NDPointIndex
from xoak import SklearnGeoBallTreeAdapter

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
    

    def collect_samples(self, ds, profile, ij=True) -> xr.Dataset:
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
        if ij:
            i_nn, j_nn = self.find_nearest_ij(ds, profile)
            ds_synth = self.extract_locations_ij(ds, i_nn, j_nn)
        
        else:
            ds = self.find_nearest_geoball(ds)
            ds_synth = self.extract_locations_geoball(ds, profile)
        
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
    
    
    def sample(self, ds: xr.Dataset, profile: xr.Dataset, ij=True) -> xr.Dataset:
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
        ds_sampled = self.collect_samples(ds, profile, ij)
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
        profile : xarray.Dataset observation profile dataset

        Return
        index: indicies of model in i an j
        """

        lon_sub = np.abs(ds.lon - profile.lon)
        lat_sub = np.abs(ds.lat - profile.lat)
        dist = ((lon_sub + lat_sub) / 2)
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
        
        nearest = score.argmin("gridpoint")
        ji = score["gridpoint"].isel(gridpoint=nearest)        

        i_nn = ji["i"]
        j_nn = ji["j"]
        i_nn = i_nn.drop_vars("gridpoint")
        j_nn = j_nn.drop_vars("gridpoint")

        return i_nn, j_nn

    
    def extract_locations_ij(self, ds, i_index, j_index):
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
            Model profile dataset
        """

        ds_model_profile = ds.isel(i=i_index, j=j_index)
        
        return ds_model_profile
        

    def find_nearest_geoball(self, ds):
        """
        Assign geoball distance indexer with lat and lon.

        Parameters
        ----------
        ds : xarray.Dataset
            Gridded ocean model dataset.

        Return
        index: indicies of model in i an j
        """

        self.lat_name = 'lat'
        self.lon_name = 'lon'
        ds = (ds.assign_coords({
                self.lat_name: ds[self.lat_name], 
                self.lon_name: ds[self.lon_name]}).set_xindex(
                (self.lat_name, self.lon_name), 
                NDPointIndex, 
                tree_adapter_cls=SklearnGeoBallTreeAdapter))
        print(ds)
        return ds

    def extract_locations_geoball(self, ds, profile):
        """
        Extract a model profile at the obs profile lat and lon.

        Parameters
        ----------
        ds : xarray.Dataset
            Gridded ocean model dataset.
        profile : xarray.Dataset observation profile dataset

        Return
        xarray.Dataset
            Model profile dataset
        """
        self.prof_lat_name = 'lat'
        self.prof_lon_name = 'lon'
        ds_model_profile = ds.sel({
            self.lat_name: profile[self.prof_lat_name], 
            self.lon_name: profile[self.prof_lon_name]}, 
            method='nearest')  
      
        return ds_model_profile

