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
from typing import Self

import numpy as np
import xarray as xr

from OceanOSSE.sampling.sampler import ErrorKernel, ObsSampler
from OceanOSSE.sampling.utils import _extract_locations_ij

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
        #self.n_samples = config["sampling"]["kwargs"]["n_samples"]
        
        return self
    

    def collect_samples(self, ds: xr.Dataset, prob: xr.DataArray | None = None) -> xr.Dataset:
        """
        Parameters
        ----------
        ds : xarray.Dataset
            Gridded ocean model output dataset.
        prob : xarray.DataArray | None
            Loaded probability distribution.
            
        Returns
        -------
        xarray.Dataset
            Sampled synthetic observations dataset.
        """
        
        # from config
        from_cf = False
        if from_cf:
            None
            n_samples = self.n_samples
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

        ds_synth = _extract_locations_ij(ds, i_index, j_index, t_index)

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
    
    
    def sample(self, ds: xr.Dataset, prob: xr.DataArray | None = None) -> xr.Dataset:
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

    
    def random_sample(self, ds: xr.Dataset, n_sample: int, prob: xr.DataArray | None = None):
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
        prob : xarray.DataArray | None
            Probability distribution

        Return
        coords : xarray.Dataset
            model i and j coordinates for profiles
        """

        sizes = ds.sizes

        # surface layer: 0 land, 1 sea
        mask = ds.mask.isel({"lev": 0})
                            
        if prob is not None:
            # change provided probability to zero where land is present
            prob = prob.where(mask).fillna(0)
        else:
            # make a probability map where land is zero
            prob = xr.ones_like(mask).where(mask == 1).fillna(0)
            
        # normalise the probabilities to make them sum to 1.
        prob = prob / prob.sum()

        flat_idx = np.random.choice(mask.size, size=n_sample, p=prob.values.ravel())
        j_random, i_random = np.unravel_index(flat_idx, mask.shape)
        
        return i_random, j_random

