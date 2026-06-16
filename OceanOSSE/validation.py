"""
validation.py

Description: Validation functions for OceanOSSE package.

Created By: OceanOSSE Development Team (NOC, UK)
"""

# -- Import Dependencies -- #
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# -- [inputs.variables.<name>] -- #
class VariableConfig(BaseModel):
    """
    Schema for `[inputs.variables.<name>]`.

    Example
    -------
        [inputs.variables.thetao]
        path  = "/data/NEMO_thetao_*.nc"
        chunks = {time_counter = 1}
    """

    path: str
    open_kwargs: dict[str, Any] = Field(default_factory=dict)

    @field_validator("path")
    @classmethod
    def path_must_be_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("'path' must not be an empty string.")
        return v


# -- [inputs.dimensions] -- #
class DimensionsConfig(BaseModel):
    """
    Schema for `[inputs.dimensions]`.

    Example
    -------
    .. code-block:: toml

        [inputs.dimensions]
        time = "time_counter"
        lev  = "deptht"
        j    = "y"
        i    = "x"
    """

    time: str = "time"
    lev: str = "lev"
    j: str = "j"
    i: str = "i"


# -- [inputs.coordinates] -- #
class CoordinatesConfig(BaseModel):
    """
    Schema for `[inputs.coordinates]`.

    Example
    -------
    .. code-block:: toml

        [inputs.coordinates]
        time   = "time_counter"
        depth  = "deptht"
        lon    = "nav_lon"
        lat    = "nav_lat"
    """

    time: str = "time"
    depth: str = "depth"
    lon: str = "lon"
    lat: str = "lat"


# -- [inputs] -- #
class InputConfig(BaseModel):
    """
    Schema for `[inputs]`.

    The `variables` field accepts any number of named sub-tables:

    .. code-block:: toml

        [inputs.variables.thetao]
        path = "..."
        open_kwargs = { engine = "netcdf4" }

        [inputs.variables.so]
        path = "..."
        open_kwargs = { engine = "netcdf4" }
    """

    dimensions: DimensionsConfig = Field(default_factory=DimensionsConfig)
    coordinates: CoordinatesConfig = Field(default_factory=CoordinatesConfig)
    data_dir: str = ""
    format: Literal["netcdf", "zarr"] = "netcdf"
    variables: dict[str, VariableConfig]
    name: str | None = None
    module: str | None = None

    @field_validator("variables")
    @classmethod
    def variables_must_not_be_empty(cls, v: dict) -> dict:
        if not v:
            raise ValueError(
                "`[inputs.variables]` must contain at least one variable entry."
            )
        return v


# -- [sampling] -- #
class SamplingConfig(BaseModel):
    """
    Schema for `[sampling]`.

    .. code-block:: toml

        [sampling]
        name = "..."

        [[sampling.error_kernels]]
        name = "..."

        [[sampling.error_kernels]]
        module = "..."
        name = "..."
    """

    name: str = "test"
    module: str | None = None
    error_kernels: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def plugin_requires_both_module_and_name(self) -> SamplingConfig:
        if (self.module is not None) and (self.name is None):
            raise ValueError(
                "`[sampling]`: 'module' and 'name' must both be specified "
                "together for plugin loading."
            )
        return self


# -- [regridding] -- #
class RegriddingConfig(BaseModel):
    """
    Schema for `[regridding]`.

    .. code-block:: toml

        [regridding]
        module = "..."
        name = "..."
    """

    name: str = "test"
    module: str | None = None

    @model_validator(mode="after")
    def plugin_requires_both_module_and_name(self) -> RegriddingConfig:
        if (self.module is not None) and (self.name is None):
            raise ValueError(
                "``[regridding]``: 'module' and 'name' must both be specified together."
            )
        return self


# -- [outputs] -- #
class OutputConfig(BaseModel):
    format: Literal["netcdf"] = "netcdf"
    output_dir: str
    output_name: str
    date_format: Literal["Y", "M", "D"] = "Y"
    chunks: dict[str, int] = Field(default_factory=dict)
    name: str | None = None
    module: str | None = None

    @model_validator(mode="after")
    def plugin_requires_both_module_and_name(self) -> OutputConfig:
        if (self.module is not None) and (self.name is None):
            raise ValueError(
                "``[outputs]``: 'module' and 'name' must both be specified together."
            )
        return self


# -- Top-level AppConfig -- #
class AppConfig(BaseModel):
    """
    Top-level OceanOSSE configuration model.
    Validates the entire parsed .toml dict.
    """

    inputs: InputConfig
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    regridding: RegriddingConfig = Field(default_factory=RegriddingConfig)
    outputs: OutputConfig
