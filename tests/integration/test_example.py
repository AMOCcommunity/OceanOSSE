"""
test_example.py

Description:
This module includes pytest integration tests
for OceanOSSE example cases.

Author:
Ollie Tooth (oliver.tooth@noc.ac.uk)
"""

import xarray as xr

from OceanOSSE.gridding.regridder import MockRegridder
from OceanOSSE.io.dataloader import NetCDFDataLoader
from OceanOSSE.io.datawriter import NetCDFDataWriter
from OceanOSSE.sampling.sampler import MockErrorKernel, MockObsSampler


# Example integration test class:
class TestOceanOSSEExample:
    def test_DataLoader_config(self, example_config: dict):
        # Initialise example DataLoader using test configuration:
        dataloader = NetCDFDataLoader.from_config(config=example_config)
        # Verify DataLoader attributes:
        assert dataloader._source == example_config["inputs"]["variables"]
        assert dataloader._dimensions == example_config["inputs"]["dimensions"]
        assert dataloader._coordinates == example_config["inputs"]["coordinates"]

    def test_DataLoader_load_data(self, example_config: dict):
        # Initialise example DataLoader using test configuration:
        dataloader = NetCDFDataLoader.from_config(config=example_config)
        # Load data using DataLoader:
        ds = dataloader.load_data()
        # Verify the integrity of xarray.Dataset:
        assert isinstance(ds, xr.Dataset)
        assert set(ds.data_vars.keys()) == set(
            example_config["inputs"]["variables"].keys()
        )
        assert set(ds.sizes.keys()) == set(
            example_config["inputs"]["dimensions"].keys()
        )
        assert set(ds.coords.keys()) == set(
            example_config["inputs"]["coordinates"].keys()
        )

    def test_Sampler_config(self, example_config: dict):
        # Initialise example Sampler using test configuration:
        sampler = MockObsSampler.from_config(config=example_config)
        # Verify Sampler attributes:
        isinstance(sampler._error_kernels[0], MockErrorKernel)

    def test_Sampler_sample(self, example_config: dict):
        # Initialise example DataLoader using test configuration:
        dataloader = NetCDFDataLoader.from_config(config=example_config)
        # Load data using DataLoader:
        ds = dataloader.load_data()

        # Initialise example Sampler using test configuration:
        sampler = MockObsSampler.from_config(config=example_config)
        # Sample observations using Sampler:
        ds_obs = sampler.sample(ds=ds)

        # Verify the integrity of sampled xarray.Dataset:
        assert isinstance(ds_obs, xr.Dataset)
        assert ds_obs.identical(ds)

    def test_Regridder_config(self, example_config: dict):
        # Initialise example Regridder using test configuration:
        regridder = MockRegridder.from_config(config=example_config)
        # Verify Regridder attributes:
        assert (
            regridder._target_grid == example_config["regridding"]["kwargs"]["argument"]
        )

    def test_Regridder_regrid(self, example_config: dict):
        # Initialise example DataLoader using test configuration:
        dataloader = NetCDFDataLoader.from_config(config=example_config)
        # Load data using DataLoader:
        ds = dataloader.load_data()

        # Initialise example Regridder using test configuration:
        regridder = MockRegridder.from_config(config=example_config)
        # Regrid data using Regridder:
        ds_regridded = regridder.regrid(ds=ds)

        # Verify the integrity of regridded xarray.Dataset:
        assert isinstance(ds_regridded, xr.Dataset)
        assert ds_regridded.identical(ds)

    def test_DataWriter_config(self, example_config: dict):
        # Initialise example DataWriter using test configuration:
        datawriter = NetCDFDataWriter.from_config(config=example_config)
        # Verify DataWriter attributes:
        assert datawriter._output_dir == example_config["outputs"]["output_dir"]
        assert datawriter._output_name == example_config["outputs"]["output_name"]
        assert datawriter._date_format == example_config["outputs"]["date_format"]
        assert datawriter._chunks == example_config["outputs"]["chunks"]
        assert datawriter._writer_kwargs == example_config["outputs"]["writer_kwargs"]
