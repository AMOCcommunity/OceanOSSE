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
from OceanOSSE.sampling.sampler_nearest_neighbour import NNSampler

def test_sampler():
    """
    Tests for extracting a profile that falls on a model grid point.
    """
    # Build dataset
    ds = construct_ds()

    # Synthetic profile
    prof_id = np.array([0])
    profile_lon = np.array([3])
    profile_lat = np.array([5])
    profile_time = np.array([dt.datetime(2020, 5, 4)])
    profile = xr.Dataset(
        {
            "lon": (("profile_id"), profile_lon),
            "lat": (("profile_id"), profile_lat),
            "time": (("profile_id"), profile_time)
        },
        coords={
            "profile_id": prof_id,
        },
    )

    sampler = NNSampler()
    model_t = sampler.sample(ds, profile)
    
    assert (model_t.votemper.to_numpy().squeeze() == ds.votemper[3, :, 5, 3]).all()
    
    
def test_sampler_multi():
    """
    Tests for extracting multiple profiles that falls on a model grid point.
    """
    # Build dataset
    ds = construct_ds()

    # Synthetic profile
    prof_id = np.array([0, 1])
    profile_lon = np.array([3, 8])
    profile_lat = np.array([5, 6])
    profile_time = np.array([dt.datetime(2020, 5, 4), dt.datetime(2020, 8, 23)])
    profile = xr.Dataset(
        {
            "lon": (("profile_id"), profile_lon),
            "lat": (("profile_id"), profile_lat),
            "time": (("profile_id"), profile_time)
        },
        coords={
            "profile_id": prof_id,
        },
    )

    sampler = NNSampler()
    model_t = sampler.sample(ds, profile)
    
    assert (model_t.votemper.sel(profile_id=1) == ds.votemper[114, :, 6, 8]).all()
    
    
def test_sampler_nn():
    """
    Test for extracting a profile that falls between model grid points that will use nearest 
    neighbour against analytic form.
    """
    # Build dataset
    ds = construct_ds()
    
    # Synthetic profile
    prof_id = np.array([0, 1])
    profile_lon = np.array([3.5, 1.2])
    profile_lat = np.array([5.5, 2.2])
    profile_time = np.array([dt.datetime(2020, 5, 4), dt.datetime(2020, 8, 23)])
    profile = xr.Dataset(
        {
            "lon": (("profile_id"), profile_lon),
            "lat": (("profile_id"), profile_lat),
            "time": (("profile_id"), profile_time)
        },
        coords={
            "profile_id": prof_id,
        },
    )
   
    sampler = NNSampler()
    model_t = sampler.sample(ds, profile)
    
    assert ((model_t.votemper.sel(profile_id=0) == ds.votemper[3, :, 6, 4]).all() 
            & (model_t.votemper.sel(profile_id=1) == ds.votemper[114, :, 2, 1]).all())
    
    
def test_sampler_time():
    """
    Tests for extracting a profile that falls on a model grid point but 
    inbetween two time steps.
    """
    # Build dataset
    ds = construct_ds()

    # Synthetic profile
    prof_id = np.array([0])
    profile_lon = np.array([3])
    profile_lat = np.array([5])
    profile_time = np.array([dt.datetime(2020, 5, 6, 12)])
    profile = xr.Dataset(
        {
            "lon": (("profile_id"), profile_lon),
            "lat": (("profile_id"), profile_lat),
            "time": (("profile_id"), profile_time)
        },
        coords={
            "profile_id": prof_id,
        },
    )

    sampler = NNSampler()
    model_t = sampler.sample(ds, profile)
    
    assert (model_t.votemper.to_numpy().squeeze() == ds.votemper[5, :, 5, 3]).all()
    
    
def test_sampler_time_out_bounds():
    """
    Tests for extracting all profiles that are outside model time bounds.
    """
    # Build dataset
    ds = construct_ds()

    # Synthetic profile
    prof_id = np.array([0])
    profile_lon = np.array([3])
    profile_lat = np.array([5])
    profile_time = np.array([dt.datetime(2021, 5, 1)])
    profile = xr.Dataset(
        {
            "lon": (("profile_id"), profile_lon),
            "lat": (("profile_id"), profile_lat),
            "time": (("profile_id"), profile_time)
        },
        coords={
            "profile_id": prof_id,
        },
    )

    sampler = NNSampler()
    with pytest.raises(ValueError, match=r".*time bounds.") as exc_info:
        model_t = sampler.sample(ds, profile)
    
    assert exc_info.type is ValueError


def test_sampler_time_subset():
    """
    Tests for extracting profiles where some are outside model time bounds.
    """
    # Build dataset
    ds = construct_ds()

    # Synthetic profile
    prof_id = np.array([0, 1])
    profile_lon = np.array([3, 8])
    profile_lat = np.array([5, 6])
    profile_time = np.array([dt.datetime(2021, 5, 1), dt.datetime(2020, 5, 6)])
    profile = xr.Dataset(
        {
            "lon": (("profile_id"), profile_lon),
            "lat": (("profile_id"), profile_lat),
            "time": (("profile_id"), profile_time)
        },
        coords={
            "profile_id": prof_id,
        },
    )

    sampler = NNSampler()
    model_t = sampler.sample(ds, profile)
    
    assert (model_t.votemper.sel(profile_id=1) == ds.votemper[5, :, 6, 8]).all()


def construct_ds():
    """
    Build a dataset for testing.
    """
    lat = np.arange(0, 8)
    lon = np.arange(0, 10)
    depth = np.arange(0, 150, 10)
    st_date = dt.datetime(2020, 5, 1)
    num_days = 180
    model_dates = np.array([st_date + dt.timedelta(days=x) for x in range(num_days)])
    model_day = np.array([x for x in range(num_days)])
    
    # Broadcast to 4D (time, depth, lat, lon)
    t, d, y, x = np.meshgrid(model_day, depth, lat, lon, indexing='ij')

    votemper = 15 - (y * 0.4) + (x * 0.2) - (d * 0.2) + (t * 0.000005)
    
    # Build dataset
    ds = xr.Dataset(
        {
            "votemper": (("t", "d", "j", "i"), votemper),
            "lat": (("j", "i"), y[0, 0, :, :]),
            "lon": (("j", "i"), x[0, 0, :, :]),
            "depth": (("d", "j", "i"), d[0, :, :, :]),
            "time": (("t"), model_dates)
        },
        coords={
            "d": depth,
            "j": lat,
            "i": lon,
            "t": model_day
        },
    )
    
    return ds