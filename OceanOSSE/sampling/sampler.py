"""
sampler.py

Description: Sampling module for OceanOSSE package.

Created By: OceanOSSE Development Team (NOC, UK)
"""

# -- Import Dependencies -- #
from __future__ import annotations

import abc
import logging
from typing import Self

import xarray as xr

from OceanOSSE.utils import import_class

logger = logging.getLogger(__name__)


# -- Utility Functions -- #
def get_error_kernels(config: dict) -> list[ErrorKernel] | None:
    """
    Utility function to instantiate ErrorKernel instances from the `[sampling]`
    table of the .toml configuration file.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing input parameters from .toml
        configuration file.

    Returns
    -------
    list[ErrorKernel] | None
        List of initialised ErrorKernel instances, or None if no kernels are
        specified in the configuration.
    """
    error_kernels_config = config["sampling"].get("error_kernels", None)

    if error_kernels_config is not None:
        _ERROR_KERNEL_REGISTRY = {"test": MockErrorKernel}
        kernels: list[ErrorKernel] = []
        for kernel_cfg in error_kernels_config:
            if ("module" in kernel_cfg) and ("name" in kernel_cfg):
                # -- Import custom ErrorKernel class -- #
                Kernel = import_class(
                    module=kernel_cfg["module"],
                    class_name=kernel_cfg["name"],
                    class_type=ErrorKernel,
                )

            else:
                # -- Use ErrorKernel class from registry -- #
                try:
                    Kernel = _ERROR_KERNEL_REGISTRY[kernel_cfg["name"]]
                except KeyError as e:
                    raise KeyError(
                        f"ErrorKernel name '{kernel_cfg['name']}' not found in registry."
                    ) from e

            # -- Instantiate ErrorKernel from configuration -- #
            kernels.append(Kernel.from_config(config=config))

    return kernels


# -- ErrorKernel Abstract Base Class -- #
class ErrorKernel(abc.ABC):
    """
    Abstract base class for applying instrument or representation errors
    to synthetic ocean observations.

    ErrorKernel transforms a sampled xarray.Dataset by adding noise,
    applying a bias, convolving a point-spread function, etc.

    Multiple kernels can be chained by an :class:`ObsSampler` and are applied
    sequentially in declaration order.
    """

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config: dict) -> Self:
        """
        Abstract class method to instantiate an ErrorKernel from the `[sampling]`
        table of the .toml configuration file.

        This is the required constructor for all ErrorKernel subclasses - plugin
        authors must implement this method for use in OceanOSSE.

        Parameters
        ----------
        config : dict
            Configuration dictionary containing input parameters from .toml
            configuration file.

        Returns
        -------
        Self
            Initialised ErrorKernel instance.
        """
        ...

    @abc.abstractmethod
    def apply(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Abstract method to apply the error kernel to an xarray.Dataset of
        synthetic observations.

        Parameters
        ----------
        ds : xarray.Dataset
            Synthetic observations dataset produced by `ObsSampler.sample`.

        Returns
        -------
        xarray.Dataset
            Synthetic observations dataset with error applied.
        """
        ...


class MockErrorKernel(ErrorKernel):
    """
    ErrorKernel used for testing and scaffold validation.

    Returns the synthetic observations xarray.Dataset unchanged.
    """

    @classmethod
    def from_config(cls, config: dict) -> Self:
        """
        Instantiate a MockErrorKernel from the `[sampling]` table of
        the .toml configuration file.
        """
        return cls()

    def apply(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Apply the MockErrorKernel to an xarray.Dataset of synthetic observations.

        Parameters
        ----------
        ds : xarray.Dataset
            Synthetic observations dataset produced by `ObsSampler.sample`.

        Returns
        -------
        xarray.Dataset
            Synthetic observations dataset unchanged.
        """
        logger.debug(
            "Applying MockErrorKernel -> returns synthetic observations dataset unchanged."
        )
        return ds


# -- ObsSampler Abstract Base Class -- #
class ObsSampler(abc.ABC):
    """
    Abstract base class for sampling gridded ocean model output analogously
    to an ocean observing platform (e.g., Argo floats).

    Parameters
    ----------
    error_kernels : list[ErrorKernel], optional
        List of ErrorKernel instances to apply sequentially to the sampled
        synthetic observations dataset, by default None.
    """

    def __init__(self, error_kernels: list[ErrorKernel] | None = None):
        # -- Validate Inputs -- #
        if error_kernels is not None:
            if not isinstance(error_kernels, list):
                raise TypeError(
                    "`error_kernels` must be a list of ErrorKernel instances."
                )
            for n, kernel in enumerate(error_kernels):
                if not isinstance(kernel, ErrorKernel):
                    raise TypeError(f"`error_kernels[{n}]` must be an ErrorKernel.")

        # -- Class Attributes -- #
        self._error_kernels = error_kernels

    def __repr__(self) -> str:
        return f"{type(self).__name__}(error_kernels={self._error_kernels!r})"

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config: dict) -> Self:
        """
        Abstract class method to instantiate an ObsSampler from the `[sampling]`
        table of the .toml configuration file.

        This is the required constructor for all ObsSampler subclasses - plugin
        authors must implement this method for use in OceanOSSE.

        Parameters
        ----------
        config : dict
            Configuration dictionary containing input parameters from .toml
            configuration file.

        Returns
        -------
        Self
            Initialised ObsSampler instance.
        """
        ...

    @abc.abstractmethod
    def collect_samples(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Abstract method to sample a gridded xarray.Dataset of ocean model output
        to produce a synthetic observations dataset.

        This is the required sampling method for all ObsSampler subclasses -
        plugin authors must implement this method for use in OceanOSSE.

        Parameters
        ----------
        ds : xarray.Dataset
            Gridded ocean model output dataset.

        Returns
        -------
        xarray.Dataset
            Sampled synthetic observations dataset.
        """
        ...

    def apply_errors(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Apply all registered `ErrorKernel` instances to synthetic
        observations sequentially.

        If no kernels are registered, the synthetic observations
        dataset is returned unchanged.

        Parameters
        ----------
        ds : xarray.Dataset
            Synthetic observations dataset.

        Returns
        -------
        xarray.Dataset
            Synthetic observations dataset with all error kernels
            applied in order.
        """
        # -- Apply each Error Kernel sequentially -- #
        if self._error_kernels is not None:
            for kernel in self._error_kernels:
                logger.debug(f"Applying ErrorKernel --> {repr(kernel)}")
                ds = kernel.apply(ds)
            logging.info(
                "--> Completed: Applied ErrorKernels to synthetic observations."
            )

        return ds

    def sample(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Perform sampling pipeline for chosen ocean observing platform.

        Parameters
        ----------
        ds : xarray.Dataset
            Gridded ocean model dataset.

        Returns
        -------
        xarray.Dataset
            Synthetic observations dataset with errors applied.
        """
        # -- Sample the gridded ocean model output -- #
        ds_sampled = self.collect_samples(ds)
        logging.info(
            "--> Completed: Collected samples from ocean model dataset using ObsSampler."
        )

        # -- Apply error kernels sequentially to the synthetic observations -- #
        ds_obs = self.apply_errors(ds_sampled)

        return ds_obs


# -- ObsSampler Implementations -- #


class MockObsSampler(ObsSampler):
    """
    ObsSampler used for testing and scaffold validation.

    Returns the input gridded ocean model dataset unchanged as the synthetic
    observations dataset.
    """

    @classmethod
    def from_config(cls, config: dict) -> Self:
        """
        Instantiate a MockObsSampler from the `[sampling]` table of
        the .toml configuration file.
        """
        # -- Collect ErrorKernel instances from configuration -- #
        error_kernels = get_error_kernels(config=config)

        # -- Instantiate MockObsSampler with collected ErrorKernel instances -- #
        return cls(error_kernels=error_kernels or None)

    def collect_samples(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Sample a gridded xarray.Dataset of ocean model output to produce a
        synthetic observations dataset.

        Parameters
        ----------
        ds : xarray.Dataset
            Gridded ocean model output dataset.

        Returns
        -------
        xarray.Dataset
            Synthetic observations dataset (unchanged from input).
        """
        logger.debug(
            "Collecting samples with MockObsSampler -> returns input dataset unchanged."
        )
        return ds
