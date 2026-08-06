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

import numpy as np
import xarray as xr
from xarray.indexes import NDPointIndex
from xoak import SklearnGeoBallTreeAdapter

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
    def collect_samples(self, ds_mdl: xr.Dataset) -> xr.Dataset:
        """
        Abstract method to sample a gridded xarray.Dataset of ocean model output
        to produce a synthetic observations dataset.

        This is the required sampling method for all ObsSampler subclasses -
        plugin authors must implement this method for use in OceanOSSE.

        Parameters
        ----------
        ds_mdl : xarray.Dataset
            Dataset of gridded ocean model outputs.

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
            logger.info(
                "--> Completed: Applied ErrorKernels to synthetic observations."
            )

        return ds

    def sample(self, ds_mdl: xr.Dataset) -> xr.Dataset:
        """
        Perform sampling pipeline for chosen ocean observing platform.

        Parameters
        ----------
        ds_mdl : xarray.Dataset
            Dataset of gridded ocean model outputs.

        Returns
        -------
        xarray.Dataset
            Synthetic observations dataset with errors applied.
        """
        # -- Sample the gridded ocean model output -- #
        ds_sampled = self.collect_samples(ds_mdl)
        logger.info(
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

    def collect_samples(self, ds_mdl: xr.Dataset) -> xr.Dataset:
        """
        Sample a gridded xarray.Dataset of ocean model output to produce a
        synthetic observations dataset.

        Parameters
        ----------
        ds_mdl : xarray.Dataset
            Dataset of gridded ocean model outputs.

        Returns
        -------
        xarray.Dataset
            Synthetic observations dataset (unchanged from input).
        """
        logger.debug(
            "Collecting samples with MockObsSampler -> returns input dataset unchanged."
        )
        return ds_mdl


class NNSampler(ObsSampler):
    """
    Basic nearest neighbour ObsSampler to sample gridded ocean model output
    analogously to an ocean observing platform (e.g., Argo floats).

    Parameters
    ----------
    error_kernels : list[ErrorKernel], optional
        List of ErrorKernel instances to apply sequentially to the sampled
        synthetic observations dataset, by default None.
    """

    @classmethod
    def from_config(cls, config: dict) -> Self:
        """
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
        # -- Verify Input -- #
        if not isinstance(config, dict):
            raise TypeError("config must be a dictionary.")

        # -- Collect ErrorKernel instances from configuration -- #
        error_kernels = get_error_kernels(config=config)

        # -- Instantiate NNSampler with collected ErrorKernel instances -- #
        return cls(error_kernels=error_kernels or None)


    def collect_samples(
        self,
        ds_mdl : xr.Dataset,
        ds_prof: xr.Dataset,
        method: str = "ij"
        ) -> xr.Dataset:
        """
        Sample gridded ocean model outputs to produce a Dataset
        of synthetic vertical profile observations.

        Parameters
        ----------
        ds_mdl : xarray.Dataset
            Dataset of gridded ocean model outputs.
        ds_prof : xarray.Dataset
            Dataset of synthetic observation locations.
        method : str, optional
            Method for sampling the gridded ocean model outputs. Options are:
            - "ij" : Use nearest neighbour in i,j grid coordinates.
            - "geoball" : Use nearest neighbour in geospatial coordinates (lat, lon).
            Default is "ij".
            
        Returns
        -------
        xarray.Dataset
            Sampled synthetic observations dataset.
        """
        # Select only profiles within model time bounds:
        profile = self._time_bounds(ds_mdl, ds_prof)

        if method == "ij":
            # Find nearest model grid point in i,j coordinates for each profile:
            t_nn = self._find_nearest_time(ds_mdl, profile)
            i_nn, j_nn = self._find_nearest_ij(ds_mdl, profile)
            ds_synth = self._extract_locations_ij(ds_mdl, i_nn, j_nn, t_nn)
        
        elif method == "geoball":
            # Find nearest model grid point in geospatial coordinates for each profile:
            ds_mdl = self._find_nearest_geoball(ds_mdl)
            ds_synth = self._extract_locations_geoball(ds_mdl, profile)

        else:
            raise ValueError(
                f"Invalid sampling method '{method}'. Must be 'ij' or 'geoball'."
            )

        # Add integer time-index coordinate:
        t_index = np.searchsorted(ds_mdl['time'].values, ds_synth['time'].values)
        ds_synth = ds_synth.assign_coords(t=xr.DataArray(data=t_index, dims="profile_id"))
        
        return ds_synth

    
    def sample(
        self,
        ds_mdl: xr.Dataset,
        ds_prof: xr.Dataset,
        method="ij"
        ) -> xr.Dataset:
        """
        Perform sampling pipeline for chosen ocean observing platform.
        
        Parameters
        ----------
        ds_mdl : xarray.Dataset
            Dataset of gridded ocean model outputs.
        ds_prof : xarray.Dataset
            Dataset of synthetic observation locations.
        method : str, optional
            Method for sampling the gridded ocean model outputs. Options are:
            - "ij" : Use nearest neighbour in i,j grid coordinates.
            - "geoball" : Use nearest neighbour in geospatial coordinates (lat, lon).
            Default is "ij".

        Returns
        -------
        xarray.Dataset
            Synthetic observations dataset with errors applied.
        """
        # -- Sample the gridded ocean model output -- #
        ds_sampled = self.collect_samples(ds_mdl=ds_mdl, ds_prof=ds_prof, method=method)
        logger.info(
            "--> Completed: Collected samples from ocean model dataset using ObsSampler."
        )

        # -- Apply error kernels sequentially to the synthetic observations -- #
        ds_obs = self.apply_errors(ds_sampled)

        return ds_obs

    
    def _time_bounds(self, ds, profile):
        """
        Remove profiles that are out of model bounds in time.
        
        Parameters
        ----------
        ds : xarray.Dataset
            Gridded ocean model dataset.
        profile : xarray.Dataset 
            observation profile dataset

        Return
        profile : xarray.Dataset
            observation profile dataset
        """
        st_date = ds.time.min(dim="t").to_numpy()
        en_date = ds.time.max(dim="t").to_numpy()
        p_time = profile.time.to_numpy()
        
        t_index = (p_time >= st_date) & (p_time <= en_date)
        n_reject = np.sum(np.invert(t_index).astype(int))
        n_total = profile.time.size
        logger.info('Profiles rejected for being outside time bounds: {:.2f}'.format((n_reject / n_total) * 100))
        logger.info('Profiles rejected for being outside time bounds: {:.2f}%'.format((n_reject / n_total) * 100))
        logger.info(
            "--> Completed: Applied time bounds to observation profiles."
        )
        
        t_xa = xr.DataArray(t_index, coords={"profile_id": profile.coords['profile_id']})
        profile = profile.where(t_xa, drop=True)

        return profile
 
    
    def _find_nearest_ij(self, ds, profile):
        """
        Turn observation lat and lon into model index

        Parameters
        ----------
        ds : xarray.Dataset
            Gridded ocean model dataset.
        profile : xarray.Dataset observation profile dataset

        Return
        index: indicies of model in i an j
        """

        lon_sub = np.abs(ds.lon - profile.lon)
        lat_sub = np.abs(ds.lat - profile.lat)
        dist = ((lon_sub + lat_sub) / 2)
        dist = dist.stack(gridpoint=("j", "i"))
        
        # Tiny tie-break penalties to sort dist, j , i
        # Gives consitent results and 0.5 rounds up on j and down on i
        if (dist.min("gridpoint") == 0.5).any():
            score = (
                dist
                - 1e-6 * dist["j"]
                - 1e-9 * dist["i"]
                )
        else:
            score = dist
        
        nearest = score.argmin("gridpoint")
        ji = score["gridpoint"].isel(gridpoint=nearest)        

        i_nn = ji["i"]
        j_nn = ji["j"]
        i_nn = i_nn.drop_vars("gridpoint")
        j_nn = j_nn.drop_vars("gridpoint")

        return i_nn, j_nn


    def _find_nearest_time(self, ds, profile, thresh=10):
        """
        Turn observation time into model time index

        Parameters
        ----------
        ds : xarray.Dataset
            Gridded ocean model dataset.
        profile : xarray.Dataset 
            observation profile dataset
        thresh : int 
            threshold in model timesteps for a profile being out of time bounds

        Return
        index: indicies of model in time
        """
        # Time difference in microsec
        time_delta = np.abs(ds.time - profile.time)

        # Find nearest and take first occurance (i.e. round down)
        nearest = time_delta.argmin("t")
        t_near = time_delta.isel(t=nearest)

        t_nn = t_near["t"]

        # Check for out of bounds
        n_profile = len(profile.coords['profile_id'])
        for p in range(n_profile):
            ps = profile.coords['profile_id'][p].to_numpy()
            if time_delta.sel(profile_id=ps).min() > (ds.time.isel(t=1) - ds.time.isel(t=0)):
                raise ValueError("Profile time is outside model time bounds.")
        
        return t_nn

    
    def _extract_locations_ij(self, ds, i_index, j_index, t_index):
        """
        Extract a model profile at the specified model index.

        Parameters
        ----------
        ds : xarray.Dataset
            Gridded ocean model dataset.
        i_index : observation index on model grid in i direction
        j_index : observation index on model grid in j direction
        t_index : observation index in time

        Return
        xarray.Dataset
            Model profile dataset
        """

        ds_model_profile = ds.isel(i=i_index, j=j_index, t=t_index)
        
        return ds_model_profile
        

    def _find_nearest_geoball(self, ds):
        """
        Assign geoball distance indexer with lat and lon.

        Parameters
        ----------
        ds : xarray.Dataset
            Gridded ocean model dataset.

        Return
        index: indicies of model in i an j
        """

        self.lat_name = 'lat'
        self.lon_name = 'lon'
        self.time_name = 'time'
        ds = (ds.assign_coords({
                self.lat_name: ds[self.lat_name], 
                self.lon_name: ds[self.lon_name],
                self.time_name: ds[self.time_name]}).set_xindex(
                (self.lat_name, self.lon_name), 
                NDPointIndex, 
                tree_adapter_cls=SklearnGeoBallTreeAdapter))

        return ds

    def _extract_locations_geoball(self, ds, profile):
        """
        Extract a model profile at the obs profile lat and lon.

        Parameters
        ----------
        ds : xarray.Dataset
            Gridded ocean model dataset.
        profile : xarray.Dataset observation profile dataset

        Return
        xarray.Dataset
            Model profile dataset
        """
        self.prof_lat_name = 'lat'
        self.prof_lon_name = 'lon'
        self.prof_time_name = 'time'
        ds_model_profile = ds.sel({
            self.time_name: profile[self.prof_time_name],
            self.lat_name: profile[self.prof_lat_name], 
            self.lon_name: profile[self.prof_lon_name]}, 
            method='nearest')

        ds_model_profile = ds_model_profile.assign_coords(profile_id=profile["profile_id"])
        ds_model_profile = ds_model_profile.reset_coords(['lat', 'lon', 'time'])
      
        return ds_model_profile
