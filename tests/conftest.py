"""
conftest.py

Description:
Defines OceanOSSE pytest configuration and fixtures.

Author:
Ollie Tooth (oliver.tooth@noc.ac.uk)
"""

import pytest


@pytest.fixture()
def example_config() -> dict:
    """
    Example OceanOSSE configuration dictionary.

    Returns:
        dict: Configuration dictionary.
    """
    config = {
        "inputs": {
            "dimensions": {"time": "time_counter", "lev": "deptht", "j": "y", "i": "x"},
            "coordinates": {
                "lat": "nav_lat",
                "lon": "nav_lon",
                "depth": "deptht",
                "time": "time_counter",
            },
            "variables": {
                "thetao_con": {
                    "path": "/dssgfs01/scratch/npd/simulations/eORCA1_ERA5_v1/eORCA1_ERA5_1m_grid_T_202312-202312.nc",
                    "open_kwargs": {"engine": "netcdf4"},
                },
                "so_abs": {
                    "path": "/dssgfs01/scratch/npd/simulations/eORCA1_ERA5_v1/eORCA1_ERA5_1m_grid_T_202312-202312.nc",
                    "open_kwargs": {"engine": "netcdf4"},
                },
            },
        },
        "sampling": {
            "name": "test",
            "error_kernels": [{"name": "test", "kwargs": {"argument": None}}],
        },
        "regridding": {"name": "test", "kwargs": {"argument": None}},
        "outputs": {
            "output_dir": "~/OceanOSSE/test",
            "output_name": "OceanOSSE_TEST",
            "date_format": "M",
            "chunks": {"time_counter": 12},
            "writer_kwargs": {"unlimited_dims": "time_counter", "mode": "w"},
        },
    }

    return config
