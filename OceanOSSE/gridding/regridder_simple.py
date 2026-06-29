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
from OceanOSSE.gridding.regridder import Regridder

logger = logging.getLogger(__name__)


class SwapRegridder(Regridder):
    """
    Regridding class for synthetic ocean observations onto
    the original model grid using climatology and exchanging profiles.

    Parameters
    ----------
    target_grid : xarray.Dataset or None, optional
        Dataset describing the target grid (coordinates, masks, etc.).
    """

    def __init__(
        self,
        target_grid: xr.Dataset | None = None,
    ) -> None:
        if target_grid is not None and not isinstance(target_grid, xr.Dataset):
            raise TypeError("``target_grid`` must be an xarray.Dataset or None.")
        self.target_grid = target_grid

    def __repr__(self) -> str:
        has_grid = self._target_grid is not None
        return f"{type(self).__name__}(target_grid={'<Dataset>' if has_grid else None})"


    def from_config(cls, config: dict) -> Self:
        """
        Construct a Regridder from the from the `[regridding]` table of
        the .toml configuration file.

        Parameters
        ----------
        config : dict
            Configuration dictionary containing input parameters from .toml
            configuration file.

        Returns
        -------
        Self
            Initialised Regridder instance.
        """
        
        return self

    
    def regrid(self, ds_profile: xr.Dataset) -> xr.Dataset:
        """
        Regrid the synthetic observation dataset into the target grid.

        Parameters
        ----------
        ds : xarray.Dataset
            Synthetic observations dataset.

        Returns
        -------
        xarray.Dataset
            Dataset of synthetic observations placed into target grid.
        """
        # Use indices in synthetic profile set to replace data in the 
        # climatology with model data
        ds_model = self.target_grid # in future this will already be a climatology
        
        # calculate month for profiles to put them on climatology
        ds_profile = ds_profile.assign_coords(
            month=("profile_id", ds_profile.t.dt.strftime("%m").astype(int).data)
        )
        
        n_profile = len(ds_profile.coords['profile_id'])

        # loop over profiles
        for p in range(n_profile):
            i_ind = ds_profile.coords['i'][p].to_numpy()
            j_ind = ds_profile.coords['j'][p].to_numpy()
            t_ind = ds_profile.coords['t'][p].to_numpy()
            ps = ds_profile.coords['profile_id'][p].to_numpy()

            profile = ds_profile['votemper'].isel(profile_id=ps)
            
            ds_model['votemper'].loc[
                dict(
                    #month=ds_profile.month.sel(profile_id=ps),
                    t=ds_profile.t.sel(profile_id=ps),
                    j=ds_profile.j.sel(profile_id=ps),
                    i=ds_profile.i.sel(profile_id=ps))
                    #d=ds_profile.d.sel(profile_id=ps))
                ] = profile.values
            
        return ds_model
    
    
