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


class RandomSampler(ObsSampler):
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
    

    def collect_samples(self, ds, prob=None) -> xr.Dataset:
        """
        Parameters
        ----------
        ds : xarray.Dataset
            Gridded ocean model output dataset.
        prob : xarray.Dataset
            Loaded probability distribution.
            
        Returns
        -------
        xarray.Dataset
            Sampled synthetic observations dataset.
        """
        if prob is not None:
            prob = prob.interp_like(ds, method="linear")
        
        # from config
        from_config = False
        if from_config:
            None
            # load n_samples from config
        else:
            # for tests
            n_sample = 100
        
        t_index = np.zeros((ds['t'].size * n_sample), dtype=int)
        i_index = np.zeros((ds['t'].size * n_sample), dtype=int)
        j_index = np.zeros((ds['t'].size * n_sample), dtype=int)
        for t in range(ds['t'].size):
            st_i = t * n_sample
            en_i = (t + 1) * n_sample
            t_index[st_i:en_i] = t
            i_prob, j_prob = self.random_sample(ds, n_sample, prob=prob)
            i_index[st_i:en_i] = i_prob
            j_index[st_i:en_i] = j_prob
        
        prof_id = xr.DataArray(np.arange(n_sample * ds['t'].size), dims="profile_id")
        i_index = xr.DataArray(i_index, dims="profile_id", coords={"profile_id": prof_id})
        j_index = xr.DataArray(j_index, dims="profile_id", coords={"profile_id": prof_id})
        t_index = xr.DataArray(t_index, dims="profile_id", coords={"profile_id": prof_id})

        ds_synth = self.extract_locations_ij(ds, i_index, j_index, t_index)

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
    
    
    def sample(self, ds: xr.Dataset, prob: Optional[xr.DataArray] = None) -> xr.Dataset:
        """
        Perform sampling pipeline for chosen ocean observing platform.
        
        Parameters
        ----------
        ds : xarray.Dataset
            Gridded ocean model dataset.
        prob : xarray.Dataset 
            Loaded probability distribution.

        Returns
        -------
        xarray.Dataset
            Synthetic observations dataset with errors applied.
        """
        # -- Sample the gridded ocean model output -- #
        ds_sampled = self.collect_samples(ds, prob)
        logging.info(
            "--> Completed: Collected samples from ocean model dataset using ObsSampler."
        )

        # -- Apply error kernels sequentially to the synthetic observations -- #
        ds_obs = self.apply_errors(ds_sampled)

        return ds_obs

    
    def random_sample(self, ds, n_sample, prob=None):
        """
        Take a random set of profiles in the model domain. 
        If probability is given take a semi-random set of profiles 
        in the model domain based on a probability distribution.
        
        Parameters
        ----------
        ds : xarray.Dataset
            Gridded ocean model dataset.
        n_sample : int 
            Number of samples desired
        prob : xarray.Dataset
            Probability distribution

        Return
        coords : xarray.Dataset
            model i and j coordinates for profiles
        """
           
        sizes = ds.sizes

        mask = xr.DataArray(ds.votemper.isel({"d": 0, "t": 0}).notnull())
        rng = np.random.default_rng()
        
        i_index = np.array([], dtype=int)
        j_index = np.array([], dtype=int)

        while len(i_index) < n_sample:
            get_sample = n_sample - len(i_index)
            if prob is not None:
                flat_idx = np.random.choice(mask.size, size=get_sample, p=prob.values.ravel())
            else:
                flat_idx = np.random.choice(mask.size, size=get_sample)
                
            j_random, i_random = np.unravel_index(flat_idx, mask.shape)
            sel_bool = mask.values[j_random, i_random]
            i_index = np.append(i_index, i_random[sel_bool])
            j_index = np.append(j_index, j_random[sel_bool])
            
        i_index = i_index[:n_sample]
        j_index = j_index[:n_sample]

        return i_index, j_index

    
    def extract_locations_ij(self, ds, i_index, j_index, t_index):
        """
        Extract a model profile at the specified model index.

        Parameters
        ----------
        ds : xarray.Dataset
            Gridded ocean model dataset.
        i_index : observation index on model grid in i direction
        j_index : observation index on model grid in j direction
        t_index : observation index in time

        Return
        xarray.Dataset
            Model profile dataset
        """

        ds_model_profile = ds.isel(i=i_index, j=j_index, t=t_index)
        
        return ds_model_profile
        



