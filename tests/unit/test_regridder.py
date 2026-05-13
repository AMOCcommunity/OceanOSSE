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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from OceanOSSE.gridding.regridder_simple import SwapRegridder


def test_climatology():
    """
    Test producing a daily climatology.
    """
    ds = construct_ds()
    
    regrid = SwapRegridder(ds)
    clim = regrid._climatology()
    
    if 1:
        ts = ds.votemper.mean(dim=["d", "j", "i"])
        clim = clim.votemper.mean(dim=["d", "j", "i"])
        print(ts)

        ts.plot(label="Original")
        clim.plot(label="Climatology")

        plt.legend()
        plt.show()
        plt.savefig("temp_timeseries.png")
    
    #assert (model_t.votemper.to_numpy().squeeze() == ds.votemper[:, 5, 3]).all()
    assert False
    
def construct_ds():
    """
    Build a dataset for testing.
    """
    lat = np.arange(0, 8)
    lon = np.arange(0, 10)
    depth = np.arange(0, 150, 10)
    st_date = dt.datetime(2020, 5, 1)
    num_days = 730
    model_dates = np.array([st_date + dt.timedelta(days=x) for x in range(num_days)])
    model_sec = np.array([x for x in range(num_days)])
    
    # Broadcast to 4D (time, depth, lat, lon)
    t, d, y, x = np.meshgrid(model_sec, depth, lat, lon, indexing='ij')

    votemper = 15 - (y * 0.4) + (x * 0.2) - (d * 0.2) + (t * 0.000005)
        
    # Build dataset
    ds = xr.Dataset(
        {
            "votemper": (("t", "d", "j", "i"), votemper),
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