"""
regridder.py

Description: Regridding module for OceanOSSE package.

Created By: OceanOSSE Development Team (NOC, UK)
"""

# -- Import Dependencies -- #
from __future__ import annotations

import abc
import logging
from typing import Self

import numpy as np
import pyinterp
import xarray as xr
from tqdm import tqdm

logger = logging.getLogger(__name__)


# -- Regridder Abstract Base Class -- #
class Regridder(abc.ABC):
    """
    Abstract base class for regridding synthetic ocean observations onto
    the original model grid, using methods such as objective analysis
    or interpolation.

    Parameters
    ----------
    variables : list of str or None, optional
        List of variable names to regrid.
    mask : str or None, optional
        Name of the model land-sea mask.
    """

    def __init__(
        self,
        variables: list[str] | None = None,
        mask: str | None = None,
    ) -> None:
        # -- Validate Inputs -- #
        if variables is not None and not isinstance(variables, list):
            raise TypeError("``variables`` must be a list of strings or None.")
        if mask is not None and not isinstance(mask, str):
            raise TypeError("``mask`` must be a string or None.")

        # -- Attributes -- #
        self._variables = variables
        self._mask = mask

    def __repr__(self) -> str:
        return f"{type(self).__name__}(variables={self._variables}, mask={self._mask})"

    @classmethod
    @abc.abstractmethod
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
        ...

    @abc.abstractmethod
    def regrid(
        self,
        ds_obs: xr.Dataset,
        ds_mdl: xr.Dataset,
        **kwargs
        ) -> xr.Dataset:
        """
        Regrid the synthetic ocean observation dataset onto the target model grid.

        Parameters
        ----------
        ds_obs : xarray.Dataset
            Dataset of synthetic ocean observations.
        ds_mdl : xarray.Dataset
            Dataset of time-evolving data on target model grid.
        **kwargs
            Additional keyword arguments for regridding method.

        Returns
        -------
        xarray.Dataset
            Dataset of synthetic ocean observations regridded onto target model grid.
        """
        ...


# -- Regridder Implementations -- #


class MockRegridder(Regridder):
    """
    Regridder used for testing and scaffold validation.

    Returns the original ocean model dataset unchanged.
    """

    @classmethod
    def from_config(cls, config: dict) -> Self:
        """
        Instantiate a MockRegridder from the `[regridding]` table of
        the .toml configuration file.
        """
        return cls()

    def regrid(
        self,
        ds_obs: xr.Dataset,
        ds_mdl: xr.Dataset,
        ) -> xr.Dataset:
        """
        Regrid the synthetic observation dataset onto the target grid.

        Parameters
        ----------
        ds_obs : xarray.Dataset
            Dataset of synthetic ocean observations.
        ds_mdl : xarray.Dataset
            Dataset of time-evolving data on target model grid.

        Returns
        -------
        xarray.Dataset
            Dataset of synthetic observations (unchanged from input).
        """
        logger.debug(
            "Regridding synthetic observations with MockRegridder -> returns input dataset unchanged."
        )
        logging.info(
            "--> Completed: Regridded synthetic observations with MockRegridder."
        )
        # -- Validate Inputs -- #
        if not isinstance(ds_obs, xr.Dataset):
            raise TypeError("ds_obs must be an xarray.Dataset.")
        if not isinstance(ds_mdl, xr.Dataset):
            raise TypeError("ds_mdl must be an xarray.Dataset.")

        return ds_mdl


class SwapRegridder(Regridder):
    """
    Basic Regridder to exchange climatological data with synthetic ocean
    observations defined on the original ocean model grid.

    Parameters
    ----------
    variables : list of str or None, optional
        List of variable names to regrid to ocean model grid.
    depth_max : float, optional
        Maximum depth to regrid synthetic observations onto the target grid.
        Default is 2000 m.
    """

    def __init__(
        self,
        variables: list[str] | None = None,
        mask: str | None = None,
        depth_max: int | float = 2000.0,
    ) -> None:
        # -- Initialise Regridder -- #
        super().__init__(variables=variables, mask=mask)

        # -- Validate Inputs -- #
        if not isinstance(depth_max, (int, float)):
            raise TypeError("``depth_max`` must be a float or int.")
        self._depth_max = depth_max


    @classmethod
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
        # -- Validate config dict -- #
        if not isinstance(config, dict):
            raise TypeError("config must be a dictionary.")

        # -- Instantiate SwapRegridder from config dict -- #
        source = config["inputs"].get("variables", None)
        if source is None:
            raise ValueError(
                "Missing `variables` entry in [inputs] table of config .toml file."
            )
        variables = source.keys()

        mask = config["regridding"].get("mask_name", None)
        if mask is None:
            raise ValueError(
                "Missing `mask_name` entry in [regridding] table of config .toml file."
            )

        depth_max = config["regridding"].get("depth_max", None)
        if depth_max is None:
            raise ValueError(
                "Missing `depth_max` entry in [regridding] table of config .toml file."
            )

        return cls(variables=variables, mask=mask, depth_max=depth_max)

    
    def regrid(
        self,
        ds_obs: xr.Dataset,
        ds_mdl: xr.Dataset,
        ds_clim: xr.Dataset
        ) -> xr.Dataset:
        """
        Regrid synthetic ocean observations onto the target grid.

        Parameters
        ----------
        ds_obs : xarray.Dataset
            Dataset of synthetic ocean observations.
        ds_mdl : xarray.Dataset
            Dataset of time-evolving data on target model grid.
        ds_clim : xarray.Dataset
            Dataset of climatological data on the target model grid.

        Returns
        -------
        xarray.Dataset
            Dataset of synthetic observations regridded onto target model grid.
        """
        # Initialise an empty 2-dimensional DataArray representing the model target grid:
        mask_2d = xr.full_like(ds_mdl["lon"].squeeze(), fill_value=False, dtype=bool).load()

        # Define 2-dimensional boolean mask of target model grid points to insert synthetic profiles:
        mask_2d.data[ds_obs["t"].values, ds_obs["j"].values, ds_obs["i"].values] = True

        # Define depth mask:
        mask_depth = ds_mdl['depth'] <= self._depth_max

        # Define sampling mask using 2-dimensional profile mask, depth mask, and land-sea mask:
        mask = (mask_2d & mask_depth & ds_mdl[self._mask]).transpose("time", "lev", "j", "i")

        # Define output Dataset:
        ds_out = xr.Dataset()
        for var in self._variables:
            # Assign synthetic observations to model grid where mask is True, otherwise assign climatology:
            ds_out[var] = xr.where(cond=mask, x=ds_mdl[var], y=ds_clim[var])

        return ds_out


class IDWRegridder(Regridder):
    """
    Inverse Distance Weighted (IDW) Regridder to interpolate sparse synthetic ocean
    observations onto the original ocean model grid using pyinterp.

    Parameters
    ----------
    variables : list of str or None, optional
        List of variable names to regrid to ocean model grid.
    depth_max : float, optional
        Maximum depth to regrid synthetic observations onto the target grid.
        Default is 2000 m.
    """

    def __init__(
        self,
        variables: list[str] | None = None,
        mask: str | None = None,
        depth_max: int | float = 2000.0,
        interp_kwargs: dict | None = None,
    ) -> None:
        # -- Initialise Regridder -- #
        super().__init__(variables=variables, mask=mask)

        # -- Validate Inputs -- #
        if not isinstance(depth_max, (int, float)):
            raise TypeError("``depth_max`` must be a float or int.")
        self._depth_max = depth_max

        if interp_kwargs is not None and not isinstance(interp_kwargs, dict):
            raise TypeError("``interp_kwargs`` must be a dictionary or None.")
        self._interp_kwargs = interp_kwargs


    @classmethod
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
        # -- Validate config dict -- #
        if not isinstance(config, dict):
            raise TypeError("config must be a dictionary.")

        # -- Instantiate IDWRegridder from config dict -- #
        source = config["inputs"].get("variables", None)
        if source is None:
            raise ValueError(
                "Missing `variables` entry in [inputs] table of config .toml file."
            )
        variables = list(source.keys())

        mask = config["regridding"].get("mask_name", None)
        if mask is None:
            raise ValueError(
                "Missing `mask_name` entry in [regridding] table of config .toml file."
            )

        depth_max = config["regridding"].get("depth_max", None)
        if depth_max is None:
            raise ValueError(
                "Missing `depth_max` entry in [regridding] table of config .toml file."
            )

        interp_kwargs = config["regridding"].get("interp_kwargs", None)

        return cls(variables=variables, mask=mask, depth_max=depth_max, interp_kwargs=interp_kwargs)

    
    def regrid(
        self,
        ds_obs: xr.Dataset,
        ds_mdl: xr.Dataset,
        ) -> xr.Dataset:
        """
        Regrid synthetic ocean observations onto the target grid.

        Parameters
        ----------
        ds_obs : xarray.Dataset
            Dataset of synthetic ocean observations.
        ds_mdl : xarray.Dataset
            Dataset of time-evolving data on target model grid.

        Returns
        -------
        xarray.Dataset
            Dataset of synthetic observations regridded onto target model grid.
        """
        # -- Validate Inputs -- #
        if not isinstance(ds_obs, xr.Dataset):
            raise TypeError("ds_obs must be an xarray.Dataset.")
        if not isinstance(ds_mdl, xr.Dataset):
            raise TypeError("ds_mdl must be an xarray.Dataset.")

        if self._interp_kwargs is None:
            self._interp_kwargs = {"k": None,
                                   "radius": 5,
                                   "num_threads": 0,
                                   "boundary_check": "none"
                                   }

        # -- Load Profile Data -- #
        prof_lons = ds_obs["lon"].values
        prof_lats = ds_obs["lat"].values
        prof_times = ds_obs["time"].values
        prof_values = {var: ds_obs[f"{var}_anom"].values for var in self._variables}

        # -- Load Model Grid Data -- #
        grid_lons = ds_mdl["lon"].values.flatten()
        grid_lats = ds_mdl["lat"].values.flatten()
        grid_coords = np.vstack((grid_lons, grid_lats)).T
        grid_shape = ds_mdl["lon"].shape

        logger.info("Completed: Loaded synthetic profiles and model grid data.")

        # -- Inverse Distance Weighted (IDW) Interpolation -- #
        # Identify unique profile times:
        times = np.unique(prof_times)

        # Determine maximum depth level to regrid:
        k_max = np.where(ds_mdl["depth"] <= self._depth_max)[0][-1] + 1

        # Initialise output Dataset:
        ds_out = xr.Dataset()

        # Iterate over each variable to regrid:
        for var in self._variables:
            logger.info(f"In Progress: Regridding variable `{var}` using pyinterp.inverse_distance_weighting.")
            ds_out[f"{var}_regrid"] = xr.full_like(ds_mdl[var], fill_value=np.nan)

            # Iterate over unique time-steps:
            for n in tqdm(range(len(times)),
                          desc="Interpolation Progress",
                          unit="time-step",
                          ):
                # Select only profiles at chosen time-step:
                time_mask = prof_times == times[n]
                lons = prof_lons[time_mask]
                lats = prof_lats[time_mask]
                coords = np.vstack((lons, lats)).T

                # Perform IDW interpolation at each vertical level:
                for k in range(k_max):
                    # Select profile values at chosen vertical level:
                    data = prof_values[var][time_mask, k]
                    nan_mask = ~np.isnan(data)

                    # Define R-tree configured with default WGS-84 geodetic system:
                    mesh = pyinterp.RTree3D()
                    # Add sparse synthetic profiles, excluding NaNs values:
                    mesh.packing(coords[nan_mask, :], data[nan_mask])

                    # Perform IDW interpolation onto target ocean model grid:
                    idw, _ = pyinterp.inverse_distance_weighting(
                        mesh,
                        grid_coords,
                        **self._interp_kwargs
                    )

                    ds_out[f"{var}_regrid"].data[n, k, :, :] = idw.reshape(grid_shape)

        logger.info(f"Completed: Regridded variable `{var}` using pyinterp.inverse_distance_weighting.")

        return ds_out