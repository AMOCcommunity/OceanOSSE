"""
datawriter.py

Description: DataWriter module for OceanOSSE package.

Created By: OceanOSSE Development Team (NOC, UK)
"""

# -- Import Dependencies -- #
from __future__ import annotations

import abc
import logging
from typing import Self

import cftime
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


# -- DataWriter Abstract Base Class -- #
class DataWriter(abc.ABC):
    """
    Abstract base class for writing a processed xarray.Dataset to persistent
    storage (netCDF, Zarr, etc.).

    Parameters
    ----------
    dimensions : dict[str, str]
        Mapping of standard dimension names to input dataset dimension names.
    coordinates : dict[str, str]
        Mapping of standard coordinate names to input dataset variable names.
    output_dir : str
        Directory in which to write the output file.
    output_name : str
        Name of the output file (without extension).
    date_format : str
        Date format for time dimension in output filename.
        Options are 'Y' (YYYY), 'M' (YYYY-MM) or 'D' (YYYY-MM-DD).
    chunks : dict[str, int], optional
        Dictionary defining chunk sizes for output dataset.
        Default is None, meaning no chunking is applied.
    writer_kwargs : dict[str, any], optional
        Additional keyword arguments to pass to the underlying writing function
        (e.g. ``xarray.Dataset.to_netcdf()`` or ``xarray.Dataset.to_zarr()``).
        Default is None, meaning no additional keyword arguments are applied.

    Attributes
    ----------
    _chunks : dict[str, int] or None
        Dictionary defining chunk sizes for output dataset, or None if no chunking is applied.
    _coordinates : dict[str, str]
        Mapping of standard coordinate names to input dataset variable names.
    _date_format : str
        Date format for time dimension in output filename.
    _dimensions : dict[str, str]
        Mapping of standard dimension names to input dataset dimension names.
    _output_name : str
        Name of the output file (without extension).
    _output_dir : str
        Directory in which to write the output file.
    _writer_kwargs : dict[str, any] or None
        Additional keyword arguments to pass to the underlying writing function, or None if no additional keyword arguments are applied.
    """

    def __init__(
        self,
        dimensions: dict[str, str],
        coordinates: dict[str, str],
        output_dir: str,
        output_name: str,
        date_format: str,
        chunks: dict[str, int] | None = None,
        writer_kwargs: dict[str, any] | None = None,
    ) -> None:
        # -- Validate Input -- #
        if not isinstance(dimensions, dict):
            raise TypeError("``dimensions`` must be a specfied as a dictionary.")
        if not isinstance(coordinates, dict):
            raise TypeError("``coordinates`` must be a specfied as a dictionary.")
        if not isinstance(output_dir, str):
            raise TypeError("``output_dir`` must be a string.")
        if not isinstance(output_name, str):
            raise TypeError("``output_name`` must be a string.")
        if not isinstance(date_format, str):
            raise TypeError("``date_format`` must be a string.")
        if (chunks is not None) and not isinstance(chunks, dict):
            raise TypeError("``chunks`` must be a dict or None.")
        if (writer_kwargs is not None) and not isinstance(writer_kwargs, dict):
            raise TypeError("``writer_kwargs`` must be a dict or None.")

        # -- Class Attributes -- #
        self._dimensions = dimensions
        self._coordinates = coordinates
        self._output_dir = output_dir
        self._output_name = output_name
        self._date_format = date_format
        self._chunks = chunks
        self._writer_kwargs = writer_kwargs

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"output_dir={self._output_dir!r}, "
            f"output_name={self._output_name!r}, "
            f"date_format={self._date_format!r}, "
            f"chunks={self._chunks!r}, "
            f"writer_kwargs={self._writer_kwargs!r})"
        )

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config: dict) -> Self:
        """
        Abstract class method to instantiate a DataWriter from the `[outputs]` table
        of the .toml configuration file.

        This is the required constructor for all DataWriter subclasses - plugin
        authors must implement this method for use in OceanOSSE.

        Parameters
        ----------
        config : dict
            Configuration dictionary containing input parameters from .toml
            configuration file.

        Returns
        -------
        Self
            Initialised DataWriter instance.
        """
        ...

    @abc.abstractmethod
    def write_data(self, ds: xr.Dataset) -> str:
        """
        Abstract method to write OceanOSSE output xarray.Dataset to persistent storage.

        Parameters
        ----------
        ds : xarray.Dataset
            Processed dataset to write.

        Returns
        -------
        str
            Resolved output path where data was written (for logging).
        """
        ...

    def _reconstruct_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Reconstruct gridded ocean model xarray.Dataset from OceanOSSE output by renaming dimensions and coordinates to input names.

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset with OceanOSSE standardised dimension and coordinate names.

        Returns
        -------
        xarray.Dataset
            Dataset with original gridded ocean model dimension and coordinate names.
        """
        # -- Rename dimensions to original gridded ocean model dimension names -- #
        ds = ds.rename_dims(self._dimensions)

        # -- Assign original coordinate names and drop any standard coordinates -- #
        d_coords = {value: ds[key] for key, value in self._coordinates.items()}
        ds = ds.assign_coords(d_coords)
        drop_coords = [coord for coord in ds.coords if coord not in d_coords]
        ds = ds.drop_vars(drop_coords)

        return ds

    def _get_output_filepath(
        self,
        ds: xr.Dataset,
        output_dir: str,
        output_name: str,
        file_format: str,
        date_format: str,
    ) -> str:
        """
        Define resolved filepath to OceanOSSE output file(s).

        Parameters:
        -----------
        ds : xr.Dataset
            Output xarray Dataset.
        output_dir : str
            Directory to save output file.
        output_name : str
            Prefix of output file name.
        file_format : str
            Output file format. Options are 'netcdf' or 'zarr'.
        date_format : str
            Date format for datetime limits in output filename.
            Options are 'Y' (YYYY), 'M' (YYYY-MM) or 'D' (YYYY-MM-DD).
        """
        # -- Validate Inputs -- #
        if not isinstance(ds, xr.Dataset):
            raise TypeError("ds must be an xr.Dataset.")
        if not isinstance(output_dir, str):
            raise TypeError("output_dir must be a string.")
        if not isinstance(output_name, str):
            raise TypeError("output_name must be a string.")
        if file_format not in ["netcdf", "zarr"]:
            raise ValueError("file_format must be either 'netcdf' or 'zarr'.")

        # -- Create Date String for Output Fileapath -- #
        # Define time-limits of output dataset:
        time_limits = ds["time"].values[[0, -1]]

        # Create date string from CFTime datetime objects:
        if isinstance(time_limits[0], cftime.datetime):
            if date_format == "Y":
                fmt = "%Y"
            elif date_format == "M":
                fmt = "%Y-%m"
            elif date_format == "D":
                fmt = "%Y-%m-%d"
            else:
                raise ValueError(
                    f"Invalid date_format: '{date_format}'. Options are 'Y', 'M', 'D'."
                )
            date_str = f"{time_limits[0].strftime(fmt)}-{time_limits[1].strftime(fmt)}"

        # Create date string from numpy datetime64:
        elif isinstance(time_limits[0], np.datetime64):
            date_str = f"{np.datetime_as_string(time_limits[0], unit=date_format)}-{np.datetime_as_string(time_limits[1], unit=date_format)}"
        else:
            raise TypeError(
                f"Invalid type ({type(time_limits[0])}) for dates. Expected cftime.datetime or np.datetime64."
            )

        # -- Define Output Filepath -- #
        if file_format == "netcdf":
            output_filename = f"{output_dir}/{output_name}_{date_str}.nc"
        elif file_format == "zarr":
            output_filename = f"{output_dir}/{output_name}_{date_str}.zarr"

        return output_filename


# -- DataWriter Implementations -- #
class NetCDFDataWriter(DataWriter):
    """
    DataWriter that serialises an xarray.Dataset to a netCDF file on disk.

    Parameters
    ----------
    dimensions : dict[str, str]
        Mapping of standard dimension names to input dataset dimension names.
    coordinates : dict[str, str]
        Mapping of standard coordinate names to input dataset variable names.
    output_dir : str
        Directory in which to write the output file.
    output_name : str
        Base filename (without ``.nc`` extension).
    date_format : str
        Date format for time dimension in output filename.
        Options are 'Y' (YYYY), 'M' (YYYY-MM) or 'D' (YYYY-MM-DD).
    chunks : dict[str, int], optional
        Dictionary defining chunk sizes for output dataset.
        Default is None, meaning no chunking is applied.
    writer_kwargs : dict[str, any], optional
        Additional keyword arguments to pass to xarray.Dataset.to_netcdf.
    """

    def __init__(
        self,
        dimensions: dict[str, str],
        coordinates: dict[str, str],
        output_dir: str,
        output_name: str,
        date_format: str,
        chunks: dict[str, int] | None = None,
        writer_kwargs: dict[str, any] | None = None,
    ) -> None:
        # -- Initialise parent DataWriter class -- #
        super().__init__(
            dimensions=dimensions,
            coordinates=coordinates,
            output_dir=output_dir,
            output_name=output_name,
            date_format=date_format,
            chunks=chunks,
            writer_kwargs=writer_kwargs,
        )

    @classmethod
    def from_config(cls, config: dict) -> Self:
        """
        Instantiate a NetCDFDataWriter from the `[outputs]` table of the .toml configuration file.
        """
        # -- Verify Input -- #
        if not isinstance(config, dict):
            raise TypeError("config must be a dictionary.")

        # -- Instantiate NetCDFDataWriter from config -- #
        inputs = config["inputs"]
        outputs = config["outputs"]
        return cls(
            dimensions=inputs["dimensions"],
            coordinates=inputs["coordinates"],
            output_dir=outputs["output_dir"],
            output_name=outputs["output_name"],
            date_format=outputs["date_format"],
            chunks=outputs.get("chunks", None),
            writer_kwargs=outputs.get("writer_kwargs", None),
        )

    def write_data(self, ds: xr.Dataset) -> None:
        """
        Write OceanOSSE output xarray.Dataset to a netCDF file.

        Parameters
        ----------
        ds : xarray.Dataset
            Processed dataset to write.
        """
        # -- Validate Inputs -- #
        if not isinstance(ds, xr.Dataset):
            raise TypeError("ds must be an xr.Dataset.")
        if self._chunks is not None and not isinstance(self._chunks, dict):
            raise TypeError("chunks must be a dictionary.")
        if not isinstance(self._output_dir, str):
            raise TypeError("output_dir must be a string.")
        if not isinstance(self._output_name, str):
            raise TypeError("output_name must be a string.")
        if self._date_format not in ["Y", "M", "D"]:
            raise ValueError("date_format must be 'Y', 'M' or 'D'.")

        # -- Define Output Filepath -- #
        output_filepath = self._get_output_filepath(
            ds=ds,
            output_dir=self._output_dir,
            output_name=self._output_name,
            file_format="netcdf",
            date_format=self._date_format,
        )

        # -- Reconstruct Dataset with Original Dimension and Coordinate Names -- #
        ds = self._reconstruct_dataset(ds)

        # -- Optionally Apply Chunking -- #
        if self._chunks is not None:
            ds = ds.chunk(self._chunks)

        # -- Write Dataset to NetCDF -- #
        if self._writer_kwargs is None:
            # Default writer_kwargs for netCDF output:
            self._writer_kwargs = {
                "unlimited_dims": self._dimensions.get("time"),
                "mode": "w",
            }
        ds.to_netcdf(path=output_filepath, **self._writer_kwargs)
        logging.info(
            f"--> Completed: Written output dataset to netCDF file: {output_filepath}"
        )
