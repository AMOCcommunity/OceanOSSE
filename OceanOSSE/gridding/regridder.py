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

import xarray as xr

logger = logging.getLogger(__name__)


# -- Regridder Abstract Base Class -- #
class Regridder(abc.ABC):
    """
    Abstract base class for regridding synthetic ocean observations onto
    the original model grid, using methods such as objective analysis
    or interpolation.

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
        ...


# -- Regridder Implementations -- #


class MockRegridder(Regridder):
    """
    Regridder used for testing and scaffold validation.

    Returns the the synthetic observations dataset unchanged.
    """

    @classmethod
    def from_config(cls, config: dict) -> Self:
        """
        Instantiate a MockRegridder from the `[regridding]` table of
        the .toml configuration file.
        """
        return cls()

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
            Dataset of synthetic observations (unchanged from input).
        """
        logger.debug(
            "Regridding synthetic observations with MockRegridder -> returns input dataset unchanged."
        )
        logging.info(
            "--> Completed: Regridded synthetic observations with MockRegridder."
        )
        return ds
