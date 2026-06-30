"""
dataloader.py

Description: DataLoader module for OceanOSSE package.

Created By: OceanOSSE Development Team (NOC, UK)
"""

# -- Import Dependencies -- #
from __future__ import annotations

import abc
import glob
import logging
from typing import Self

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


# -- DataLoader Abstract Base Class -- #
class DataLoader(abc.ABC):
    """
    DataLoader Base Class to load gridded ocean model data from
    persistent local filesystem or cloud object storage.

    Parameters
    ----------
    source : dict[str, dict]
        Source dictionary of gridded ocean model variables to load.
    dimensions : dict[str, str]
        Mapping of standard dimension names to input dataset dimension names.
    coordinates : dict[str, str]
        Mapping of standard coordinate names to input dataset coordinate names.

    Attributes
    ----------
    _source : dict[str, dict]
        Source dictionary of gridded ocean model variables.
    _dimensions : dict[str, str]
        Mapping of standard dimension names to input dataset dimension names.
    _coordinates : dict[str, str]
        Mapping of standard coordinate names to input dataset coordinate names.
    _table : str
        Name of the table in the .toml configuration file. Options are 'inputs'
        or 'climatology'. Default is 'inputs'.
    """

    def __init__(
        self,
        source: dict[str, dict],
        dimensions: dict[str, str],
        coordinates: dict[str, str],
        table: str = "inputs"
    ):
        # -- Verify Inputs -- #
        if not isinstance(source, dict):
            raise TypeError("``source`` must be a specfied as a dictionary.")
        if not isinstance(dimensions, dict):
            raise TypeError("``dimensions`` must be a specfied as a dictionary.")
        if not isinstance(coordinates, dict):
            raise TypeError("``coordinates`` must be a specfied as a dictionary.")
        if not isinstance(table, str):
            raise TypeError("``table`` must be a specfied as a string.")

        # -- Class Attributes -- #
        self._source = source
        self._dimensions = dimensions
        self._coordinates = coordinates
        self._table = table

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"source={self._source!r}, "
            f"dimensions={self._dimensions!r}, "
            f"coordinates={self._coordinates!r}, "
            f"table={self._table!r})"
        )

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config: dict, table: str = "inputs") -> Self:
        """
        Abstract class method to instantiate a DataLoader from the specified table
        of the .toml configuration file.

        This is the required constructor for all DataLoader subclasses - plugin
        authors must implement this method for use in OceanOSSE.

        Parameters
        ----------
        config : dict
            Configuration dictionary containing input parameters from .toml
            configuration file.
        table : str
            Name of the table in the .toml configuration file. Options are
            'inputs' or 'climatology'. Default is 'inputs'.

        Returns
        -------
        Self
            Initialised DataLoader instance.
        """
        ...

    @abc.abstractmethod
    def load_data(self) -> xr.Dataset:
        """
        Abstract method to load gridded ocean model data into a standardised xarray.Dataset.

        The returned Dataset will be validated by the validate_dataset() method.

        Returns
        -------
        xarray.Dataset
            Dataset containing standardised gridded ocean model variables.
        """
        ...

    def compute_monthly_climatology(self) -> xr.Dataset:
        """
        Compute monthly climatology from the input xarray.Dataset.

        Returns
        -------
        xarray.Dataset
            Dataset containing monthly climatology of the input variables.
        """
        # -- Load Input Dataset -- #
        ds = self.load_data()
        start_yr = ds['time'].dt.year.min().item()
        end_yr = ds['time'].dt.year.max().item()

        # -- Compute Monthly Climatology -- #
        ds_clim = ds.groupby("time.month").mean(dim="time", skipna=True)

        # -- Assign Time Bounds -- #
        # Update time bounds to reflect climatological period:
        ds_clim['time_bnds'] = xr.DataArray(
            np.zeros((12, 2), dtype='datetime64[ns]'),
            dims=('month', 'bnds'),
            coords={'month': ds_clim['month']},
        )
        ds_clim['time_bnds'].data[:, 0] = (np.datetime64(f'{start_yr}-01', 'M') + (np.timedelta64(1, 'M') * np.arange(ds['time'].size))).astype('datetime64[ns]')
        ds_clim['time_bnds'].data[:, 1] = (np.datetime64(f'{end_yr}-01', 'M') + (np.timedelta64(1, 'M') * np.arange(ds['time'].size))).astype('datetime64[ns]')

        return ds_clim

    def _standardise_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Standardise gridded ocean model xarray.Dataset by renaming dimensions and coordinates to OceanOSSE standard names.

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset with original dimension and coordinate names.

        Returns
        -------
        xarray.Dataset
            Dataset with standardised dimension and coordinate names.
        """
        # -- Rename dimensions to standard dimensions names -- #
        rename_dims = {value: key for key, value in self._dimensions.items()}
        ds = ds.rename_dims(rename_dims)

        # -- Assign standard coordinate names and drop any non-standard coordinates -- #
        ds = ds.assign_coords(
            {coord: ds[var] for coord, var in self._coordinates.items()}
        )
        drop_coords = [coord for coord in ds.coords if coord not in self._coordinates]
        ds = ds.drop_vars(drop_coords)

        return ds

    def _validate_dataset(self, ds: xr.Dataset) -> None:
        """
        Validate the standardised xarray.Dataset.

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset containing standardised gridded ocean model variables.

        Raises
        -------
        ValueError
            If the dataset does not contain the required variables or dimensions.
        """
        # -- Validate Required Dimensions -- #
        if self._table == "climatology":
            required_dims = ["month", "lev", "j", "i"]
        else:
            required_dims = ["time", "lev", "j", "i"]
        missing_dims = [dim for dim in required_dims if dim not in ds.dims]
        if missing_dims:
            raise ValueError(
                f"{type(self).__name__}: loaded dataset is missing required "
                f"dimension(s): {missing_dims}. Found dimensions: {list(ds.dims)}."
            )

        # -- Validate Required Coordinates -- #
        if self._table == "climatology":
            required_coords = ["month", "depth", "lat", "lon"]
        else:
            required_coords = ["time", "depth", "lat", "lon"]
        missing_coords = [coord for coord in required_coords if coord not in ds.coords]
        if missing_coords:
            raise ValueError(
                f"{type(self).__name__}: loaded dataset is missing required "
                f"coordinate(s): {missing_coords}. Found coordinates: {list(ds.coords)}."
            )


# -- DataLoader Implementations -- #
class NetCDFDataLoader(DataLoader):
    """
    DataLoader implementation to load gridded ocean model data from NetCDF files.

    Parameters
    ----------
    source : dict[str, dict]
        Source dictionary of gridded ocean model variables to load.
    dimensions : dict[str, str]
        Mapping of standard dimension names to input dataset dimension names.
    coordinates : dict[str, str]
        Mapping of standard coordinate names to input dataset coordinate names.
    """

    def __init__(
        self,
        source: dict[str, dict],
        dimensions: dict[str, str],
        coordinates: dict[str, str],
        table: str = "inputs",
    ) -> None:
        # -- Initialise parent DataLoader class -- #
        super().__init__(source, dimensions, coordinates, table)

    @classmethod
    def from_config(cls, config: dict, table: str = "inputs") -> Self:
        """
        Instantiate a NetCDFDataLoader from the specified table of the .toml configuration file.

        Parameters
        ----------
        config : dict
            Configuration dictionary.
        table : str, optional
            Name of the table in the .toml configuration file.
            Options are 'inputs' or 'climatology'. Default is 'inputs'.

        Returns
        -------
        NetCDFDataLoader
            Instantiated NetCDFDataLoader object.
        """
        # -- Verify Input -- #
        if not isinstance(config, dict):
            raise TypeError("config must be a dictionary.")
        if not isinstance(table, str):
            raise TypeError("table must be a string.")
        if table not in ["inputs", "climatology"]:
            raise ValueError(
                "table must be either 'inputs' or 'climatology'."
            )

        # -- Instantiate DataLoader with source dict from config -- #
        source = config[table].get("variables", None)
        if source is None:
            raise ValueError(
                f"Missing 'variables' entry in [{table}] table of config .toml file."
            )
        dimensions = config[table].get("dimensions", None)
        if dimensions is None:
            raise ValueError(
                f"Missing 'dimensions' entry in [{table}] table of config .toml file."
            )
        coordinates = config[table].get("coordinates", None)
        if coordinates is None:
            raise ValueError(
                f"Missing 'coordinates' entry in [{table}] table of config .toml file."
            )

        return cls(source=source, dimensions=dimensions, coordinates=coordinates, table=table)

    def _open_dataset(
        self,
        filepath: str,
        variables: list[str],
        open_kwargs: dict,
    ) -> xr.Dataset:
        """
        Open input variable Dataset from a netCDF file(s).

        Parameters:
        -----------
        filepath : str
            Filepath pattern to input variable netCDF file(s).
        variables : list[str]
            Name of variable(s) to load from the dataset.
        open_kwargs : dict
            Additional keyword arguments to pass to xarray.open_dataset
            or xarray.open_mfdataset.

        Returns:
        --------
        xr.Dataset
            Standardised variable Dataset.
        """
        # -- Validate Inputs -- #
        if not isinstance(filepath, str):
            raise TypeError("filepath must be a string.")
        if not isinstance(variables, list):
            raise TypeError("variables must be a list of strings.")
        if not isinstance(open_kwargs, dict):
            raise TypeError("open_kwargs must be a dictionary.")

        filepaths = glob.glob(filepath)
        if len(filepaths) == 0:
            raise FileNotFoundError(f"No files found matching filepath: {filepath}")

        # Define CFDatetimeCoder to decode time coords:
        coder = xr.coders.CFDatetimeCoder(time_unit="s")

        # -- Open input variable dataset -- #
        if len(filepaths) == 1:
            if open_kwargs is None:
                open_kwargs = {"engine": "netcdf4"}
            try:
                logging.info(
                    f"In Progress: Opening variable(s) {variables} from netCDF file '{filepath}'"
                )
                ds_var = xr.open_dataset(
                    filepaths[0], decode_times=coder, **open_kwargs
                )[variables]
                logging.info(f"--> Completed: Opened variable(s) {variables}.")

            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"Failed to open netCDF file: {filepath}"
                ) from e
        else:
            if open_kwargs is None:
                open_kwargs = {
                    "data_vars": "minimal",
                    "compat": "no_conflicts",
                    "parallel": False,
                    "engine": "netcdf4",
                }
            if variables is not None:
                open_kwargs["preprocess"] = lambda ds: ds[variables]
            try:
                logging.info(
                    f"In Progress: Opening variable(s) {variables} from multiple netCDF files '{filepath}'"
                )
                ds_var = xr.open_mfdataset(
                    filepaths, decode_times=coder, **open_kwargs
                )[variables]
                logging.info(f"--> Completed: Opened variable(s) {variables}.")

            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"Failed to open netCDF files: {filepaths}"
                ) from e

        return ds_var

    def load_data(self) -> xr.Dataset:
        """
        Load gridded ocean model data from netCDF file(s) into a standardised xarray.Dataset.
        """
        # -- Define variable names, filepaths and open_kwargs from source dict -- #
        variables = self._source.keys()
        open_kwargs_list = [
            d_var.get("open_kwargs", {}) for d_var in self._source.values()
        ]
        filepaths = [d_var.get("path", "") for d_var in self._source.values()]
        n_filepaths = len(set(filepaths))

        if n_filepaths == 1:
            # -- All Variables Share Common Filepath Pattern -- #
            # Merge open_kwargs for all variables sharing the same filepath pattern:
            open_kwargs = {}
            for d_open_kwargs in open_kwargs_list:
                open_kwargs.update(d_open_kwargs)

            # Load all variables into single xarray.Dataset:
            ds = self._open_dataset(
                filepath=filepaths[0],
                variables=list(variables),
                open_kwargs=open_kwargs,
            )
        else:
            # -- Variables Have Different Filepath Patterns -- #
            ds_list = []
            for filepath, variable, open_kwargs in zip(
                filepaths, variables, open_kwargs_list, strict=True
            ):
                ds_list.append(
                    self._open_dataset(
                        filepath=filepath, variables=[variable], open_kwargs=open_kwargs
                    )
                )
            # Merge individual variable datasets into single dataset:
            ds = xr.merge(ds_list, compat="no_conflicts", join="exact")

        # -- Standardise dataset dimensions and coordinates -- #
        ds = self._standardise_dataset(ds=ds)
        logging.info(
            f"--> Completed: Standardised dataset dimensions {list(ds.sizes.keys())} and coordinates {list(ds.coords.keys())}."
        )

        # -- Validate standardised dataset -- #
        self._validate_dataset(ds=ds)
        logging.info("--> Completed: Validated standardised dataset.")

        return ds
