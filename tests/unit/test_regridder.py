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
test_sampler.py

Description:
This module includes unit tests for extracting profiles.

Author:
Benjamin Barton (benbar@noc.ac.uk)
"""

import pytest
import datetime as dt
import numpy as np
import xarray as xr
import pandas as pd

from OceanOSSE.gridding.regridder import SwapRegridder

@pytest.mark.parametrize('construct_ds, construct_profile_ds', [(150, 150)], indirect=True)
def test_regrid_upper(construct_ds, construct_profile_ds):
    """
    Test replacing profiles in climatology with model data in shallow water.
    """
    ds_profile = construct_profile_ds
    ds = construct_ds
    ds_clim = climatology(ds)
    ds_clim.votemper.data[:, 0, :, :] = 0
    
    regrid_data = SwapRegridder(["votemper", "vosaline"], "mask")
    ds_regrid = regrid_data.regrid(ds_profile, ds, ds_clim)
    print(ds_regrid.isel(i=3, j=5, t=1).votemper, ds.isel(i=3, j=5, t=1).votemper, ds_clim.isel(i=3, j=5, t=1).votemper)
    assert (ds_regrid.votemper == ds_clim.votemper).all() == False
    assert (ds_regrid.votemper == ds.votemper).all() == False
    assert (ds_regrid.isel(i=3, j=5, t=1).votemper == ds.isel(i=3, j=5, t=1).votemper).all()
    assert (ds_regrid.isel(i=4, j=4, t=1).votemper == ds_clim.isel(i=4, j=4, t=1).votemper).all()
    

@pytest.mark.parametrize('construct_ds, construct_profile_ds', [(3500, 2000)], indirect=True)
def test_regrid_lower(construct_ds, construct_profile_ds):
    """
    Test replacing profiles in climatology with model data in deep water.
    """
    ds_profile = construct_profile_ds
    ds = construct_ds
    ds_clim = climatology(ds)
    ds_clim.votemper.data[:, 0, :, :] = 0
    
    regrid_data = SwapRegridder(["votemper", "vosaline"], "mask", 2000)
    ds_regrid = regrid_data.regrid(ds_profile, ds, ds_clim)
    expected_slabs = xr.where(cond=ds.depth > 2000, x=ds_clim, y=ds)
    
    assert (ds_regrid.votemper == ds_clim.votemper).all() == False
    assert (ds_regrid.votemper == ds.votemper).all() == False
    assert (ds_regrid.isel(i=3, j=5, t=1).votemper == expected_slabs.isel(i=3, j=5, t=1).votemper).all()
    assert (ds_regrid.isel(i=4, j=4, t=1).votemper == ds_clim.isel(i=4, j=4, t=1).votemper).all()

    
@pytest.fixture(scope="function")
def construct_ds(request) -> xr.Dataset:
    """
    Build a dataset for testing.
    """
    lat = np.arange(0, 8)
    lon = np.arange(0, 10)
    depth = np.arange(0, request.param, 10)

    model_dates = pd.date_range(start="2020-05-01", periods=24, freq="MS")
    model_day = (model_dates - model_dates[0]).days

    # Broadcast to 4D (time, depth, lat, lon)
    t, d, y, x = np.meshgrid(model_day, depth, lat, lon, indexing='ij')

    votemper = 15 - (y * 0.4) + (x * 0.2) - (d * 0.005) + (t * 0.005)
    vosaline = 33 + (y * 0.4) - (x * 0.2) + (d * 0.005) + (t * 0.005)
    mask = np.ones_like(votemper[0, :, :, :], dtype=bool)
    
    # Build dataset
    ds = xr.Dataset(
        {
            "votemper": (("t", "lev", "j", "i"), votemper),
            "vosaline": (("t", "lev", "j", "i"), vosaline),

            "lat": (("j", "i"), y[0, 0, :, :]),
            "lon": (("j", "i"), x[0, 0, :, :]),
            "depth": (("lev", "j", "i"), d[0, :, :, :]),
            "mask": (("lev", "j", "i"), mask),
            "time": (("t"), t[:, 0, 0, 0])
        },
        coords={
            "lev": depth,
            "j": lat,
            "i": lon,
            "t": model_dates.to_numpy()
        },
    )
    
    return ds


@pytest.fixture(scope="function")
def construct_profile_ds(request)  -> xr.Dataset:
    d = np.arange(0, request.param, 10)
    profile_id = np.arange(2)

    j = np.array([5, 6], dtype=int)
    i = np.array([3, 8], dtype=int)

    depth = np.tile(d[:, None], (1, profile_id.size))

    # Time coordinate
    st_date = dt.datetime(2020, 5, 1)
    time = np.array([
        dt.datetime(2020, 6, 1),
        dt.datetime(2020, 7, 1),
    ])
    time_day = np.array([(x - st_date).days for x in time], dtype=int)
    time_month = [1, 2] # june, july index from may
    
    votemper = 15 - depth * 0.002 - j[None, :] * 0.2 + i[None, :] * 0.1 + (time_day * 0.0002)
    vosaline = 33 + depth * 0.002 + j[None, :] * 0.2 - i[None, :] * 0.1 + (time_day * 0.0002)


    ds = xr.Dataset(
        data_vars={
            "votemper": (("lev", "profile_id"), votemper),
            "vosaline": (("lev", "profile_id"), vosaline),
            "lat": (("profile_id",), j),
            "lon": (("profile_id",), i),
            "depth": (("lev", "profile_id"), depth),
        },
        coords={
            "lev": d,
            "profile_id": profile_id,
            "t": (("profile_id",), time_month),
            "j": (("profile_id",), j),
            "i": (("profile_id",), i),
        },
    )

    return ds


def climatology(ds):
    """
    Calculate the climatology of the target grid.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Input time varying dataset.

    Returns
    -------
    xarray.Dataset
        Dataset of monthly means.
    """
    ds = ds.assign_coords(
        month=("t", ds.t.dt.strftime("%m").astype(int).data)
    )
    # calculate climatology
    ds_clim = ds.groupby('month').mean()

    # tile the climatology data back over full time series
    ds_clim_full = ds_clim.sel(month=ds.month)

    # Remove not needed time dim from variables
    for v in ["lat", "lon", "depth"]:
        ds_clim_full[v] = ds_clim_full[v].isel(t=0, drop=True)
    #ds_clim_full = ds_clim_full.drop_vars('month')
        
    return ds_clim_full