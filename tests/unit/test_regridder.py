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

from OceanOSSE.gridding.regridder_simple import SwapRegridder

    
def test_regrid(construct_ds, construct_profile_ds):
    """
    Test replacing profiles in climatology with model data.
    """
    ds_profile = construct_profile_ds
    ds = construct_ds
    ds_clim = climatology(ds)
    
    regrid_data = SwapRegridder(ds_clim)
    ds_model = regrid_data.regrid(ds_profile)

    assert ((ds_model != ds_clim) 
            & (ds_model.isel(i=3, j=5, t=31) 
            == ds_profile.isel(profile_id=0)))
    
def test_climatology(construct_ds):
    """
    Test producing a daily climatology.
    """
    ds = construct_ds
    clim = climatology(ds)
    
    ts = ds.votemper.mean(dim=["d", "j", "i"])
    clim_mean = clim.votemper.mean(dim=["d", "j", "i"])
        
    st_date1 = dt.datetime(2020, 5, 1)
    st_date2 = dt.datetime(2021, 5, 1)
    test_date1 = np.array([st_date1 + dt.timedelta(days=x) for x in range(31)])
    test_date2 = np.array([st_date2 + dt.timedelta(days=x) for x in range(31)])
    test_sec1 = np.array([(x - st_date1).days for x in test_date1])
    test_sec2 = np.array([(x - st_date1).days for x in test_date2])
    test_temp1 = 15 - (0 * 0.4) + (0 * 0.2) - (0 * 0.2) + (test_sec1 * 0.000005)
    test_temp2 = 15 - (0 * 0.4) + (0 * 0.2) - (0 * 0.2) + (test_sec2 * 0.000005)
    test_temp = np.sum(test_temp1 + test_temp2) / (2 * 31)
    clim_day = clim.votemper.sel(t='2020-05-01').isel(d=0, j=0, i=0)
    
    assert (np.isclose(ts.mean().to_numpy(), clim_mean.mean().to_numpy(), atol=1e-8)
             & (clim_day.to_numpy() == test_temp))


@pytest.fixture
def construct_ds() -> xr.Dataset:
    """
    Build a dataset for testing.
    """
    lat = np.arange(0, 8)
    lon = np.arange(0, 10)
    depth = np.arange(0, 150, 10)
    st_date = dt.datetime(2020, 5, 1)
    num_days = 730
    model_dates = np.array([st_date + dt.timedelta(days=x) for x in range(num_days)])
    model_day = np.array([x for x in range(num_days)])
    
    # Broadcast to 4D (time, depth, lat, lon)
    t, d, y, x = np.meshgrid(model_day, depth, lat, lon, indexing='ij')

    votemper = 15 - (y * 0.4) + (x * 0.2) - (d * 0.2) + (t * 0.000005)
    vosaline = 33 + (y * 0.4) - (x * 0.2) + (d * 0.2) + (t * 0.000005)
        
    # Build dataset
    ds = xr.Dataset(
        {
            "votemper": (("t", "d", "j", "i"), votemper),
            "vosaline": (("t", "d", "j", "i"), vosaline),

            "lat": (("j", "i"), y[0, 0, :, :]),
            "lon": (("j", "i"), x[0, 0, :, :]),
            "depth": (("d", "j", "i"), d[0, :, :, :]),
            "time": (("t"), t[:, 0, 0, 0])
        },
        coords={
            "d": depth,
            "j": lat,
            "i": lon,
            "t": model_dates
        },
    )
    
    return ds


@pytest.fixture
def construct_profile_ds()  -> xr.Dataset:
    d = np.arange(0, 150, 10)
    profile_id = np.arange(2)

    j = np.array([5, 6])
    i = np.array([3, 8])

    depth = np.tile(d[:, None], (1, profile_id.size))

    # Time coordinate
    st_date = dt.datetime(2020, 5, 1)
    time = np.array([
        dt.datetime(2020, 6, 1),
        dt.datetime(2020, 7, 2),
    ])
    time_day = np.array([(x - st_date).days for x in time])
    
    votemper = 15 - depth * 0.02 - j[None, :] * 0.2 + i[None, :] * 0.1 + (time_day * 0.000005)
    vosaline = 33 + depth * 0.02 + j[None, :] * 0.2 - i[None, :] * 0.1 + (time_day * 0.000005)


    ds = xr.Dataset(
        data_vars={
            "votemper": (("d", "profile_id"), votemper),
            "vosaline": (("d", "profile_id"), vosaline),
            "lat": (("profile_id",), j),
            "lon": (("profile_id",), i),
            "depth": (("d", "profile_id"), depth),
        },
        coords={
            "d": d,
            "profile_id": profile_id,
            "t": (("profile_id",), time),
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
