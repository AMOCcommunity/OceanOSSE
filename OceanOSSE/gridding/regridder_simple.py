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
        self._target_grid = target_grid
        

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

    
    def regrid(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Regrid the synthetic observation dataset onto the target grid.

        Parameters
        ----------
        ds : xarray.Dataset
            Synthetic observations dataset.

        Returns
        -------
        xarray.Dataset
            Dataset of synthetic observations regridded onto target grid.
        """
        # Use indices in synthetic profile set to replace data in the climatology with model data
        ds_model = self._climatology()
        
        return ds_model
    
    def _climatology(self):
        """
        Calculate the climatology of the target grid.
        
        Returns
        -------
        xarray.Dataset
            Dataset of monthly means.
        """
        ds = self._target_grid.assign_coords(
            dayofyear=("t", self._target_grid.t.dt.dayofyear.data)
        )
        # calculate climatology
        ds_clim = ds.groupby('dayofyear').mean()
        
        # tile the climatology data back over full time series
        ds_clim_full = ds_clim.sel(dayofyear=ds.dayofyear)
        
        # Remove not needed time dim from variables
        for v in ["lat", "lon", "depth"]:
            ds_clim_full[v] = ds_clim_full[v].isel(t=0, drop=True)

        print(ds_clim_full)
        
        return ds_clim_full
    