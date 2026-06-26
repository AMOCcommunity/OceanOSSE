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
Benjamin Barton
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
    profile = xr.Dataset(
        {
            "lon": (("prof_id"), profile_lon),
            "lat": (("prof_id"), profile_lat)
        },
        coords={
            "profile_id": prof_id,
        },
    )

    sampler = NNSampler()
    model_t = sampler.sample(ds, profile)
    
    assert (model_t.votemper.to_numpy().squeeze() == ds.votemper[:, 5, 3]).all()
    
    
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
    profile = xr.Dataset(
        {
            "lon": (("profile_id"), profile_lon),
            "lat": (("profile_id"), profile_lat)
        },
        coords={
            "profile_id": prof_id,
        },
    )

    sampler = NNSampler()
    model_t = sampler.sample(ds, profile)
    
    assert (model_t.votemper.isel(profile_id=1) == ds.votemper[:, 6, 8]).all()
    
    
def test_sampler_nn():
    """
    Test for extracting a profile that falls between model grid points that will use nearest 
    neighbour against analytic form.
    """
    # Build dataset
    ds = construct_ds()
    
    # Synthetic profile
    prof_id = np.array([0, 1])
    profile_lon = np.array([3.50, 1.2])
    profile_lat = np.array([5.50, 2.2])
    profile = xr.Dataset(
        {
            "lon": (("profile_id"), profile_lon),
            "lat": (("profile_id"), profile_lat)
        },
        coords={
            "profile_id": prof_id,
        },
    )
   
    sampler = NNSampler()
    model_t = sampler.sample(ds, profile)
    print(model_t)
    
    assert ((model_t.votemper.isel(profile_id=0) == ds.votemper[:, 6, 4]).all() 
            & (model_t.votemper.isel(profile_id=1) == ds.votemper[:, 2, 1]).all())
    

def construct_ds():
    """
    Build a dataset for testing.
    """
    lat = np.arange(0, 8)
    lon = np.arange(0, 10)
    depth = np.arange(0, 150, 10)
    
    # Broadcast to 3D (depth, lat, lon)
    d, y, x = np.meshgrid(depth, lat, lon, indexing='ij')
    
    # Synthetic temperature field
    votemper =  15 - (y * 0.4) + (x * 0.2) - (d * 0.02)
    
    # Build dataset
    ds = xr.Dataset(
        {
            "votemper": (("d", "j", "i"), votemper),
            "lat": (("j", "i"), y[0, :, :]),
            "lon": (("j", "i"), x[0, :, :]),
            "depth": (("d", "j", "i"), d)
        },
        coords={
            "d": depth,
            "j": lat,
            "i": lon,
        },
    )
    
    return ds
