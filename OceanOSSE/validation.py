"""
validation.py

Description: Validation functions for OceanOSSE package.

Created By: OceanOSSE Development Team (NOC, UK)
"""

# -- Import Dependencies -- #
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# -- [<section>.variables.<name>] -- #
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


# -- [<section>.dimensions] -- #
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


# -- [<section>.coordinates] -- #
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


# -- [domain] -- #
class DomainConfig(BaseModel):
    """
    Schema for `[domain]`.

    Example
    -------
    .. code-block:: toml

        [domain]
        mask
        thkcello
        dimensions
        coordinates
        variables
    """

    format: Literal["netcdf"] = "netcdf"
    mask: str = "mask"
    thkcello: str = "thkcello"
    dimensions: DimensionsConfig = Field(default_factory=DimensionsConfig)
    coordinates: CoordinatesConfig = Field(default_factory=CoordinatesConfig)
    variables: dict[str, VariableConfig]

    @field_validator("variables")
    @classmethod
    def variables_must_not_be_empty(cls, v: dict) -> dict:
        if not v:
            raise ValueError(
                "`[domain.variables]` must contain at least one variable entry."
            )
        return v


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

    format: Literal["netcdf"] = "netcdf"
    dimensions: DimensionsConfig = Field(default_factory=DimensionsConfig)
    coordinates: CoordinatesConfig = Field(default_factory=CoordinatesConfig)
    variables: dict[str, VariableConfig]

    @field_validator("variables")
    @classmethod
    def variables_must_not_be_empty(cls, v: dict) -> dict:
        if not v:
            raise ValueError(
                "`[inputs.variables]` must contain at least one variable entry."
            )
        return v


# -- [climatology] -- #
class ClimatologyConfig(BaseModel):
    """
    Schema for `[climatology]`.

    Example
    -------
    .. code-block:: toml

        [climatology]
        read_climatology
        dimensions
        coordinates
        variables
    """

    read_climatology: bool
    format: Literal["netcdf"] = "netcdf"
    dimensions: DimensionsConfig = Field(default_factory=DimensionsConfig)
    coordinates: CoordinatesConfig = Field(default_factory=CoordinatesConfig)
    variables: dict[str, VariableConfig]

    @field_validator("variables")
    @classmethod
    def variables_must_not_be_empty(cls, v: dict) -> dict:
        if not v:
            raise ValueError(
                "`[climatology.variables]` must contain at least one variable entry."
            )
        return v


# -- [sampling] -- #
class SamplingConfig(BaseModel):
    """
    Schema for `[sampling]`.

    .. code-block:: toml

        [sampling]
        name = "..."
        sampler_kwargs = { ... }
        error_kernels = [{...}, {...}]
    """

    name: str = "test"
    module: str | None = None
    sampler_kwargs: dict[str, Any] = Field(default_factory=dict)
    error_kernels: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def plugin_requires_both_module_and_name(self) -> SamplingConfig:
        if (self.name is None) and (self.module is not None):
            raise ValueError(
                "``[sampling]``: 'name' must be specified if 'module' is provided."
            )
        return self

    @model_validator(mode="after")
    def plugin_requires_both_name_and_sampler_kwargs(self) -> SamplingConfig:
        if (self.name is None) and (self.sampler_kwargs is None):
            raise ValueError(
                "``[sampling]``: both 'name' and 'sampler_kwargs' must be specified."
            )
        return self


# -- [regridding] -- #
class RegriddingConfig(BaseModel):
    """
    Schema for `[regridding]`.

    .. code-block:: toml

        [regridding]
        name = "..."
        regridder_kwargs = { ... }
    """

    name: str = "test"
    module: str | None = None
    regridder_kwargs: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def plugin_requires_both_module_and_name(self) -> RegriddingConfig:
        if (self.name is None) and (self.module is not None):
            raise ValueError(
                "``[regridding]``: 'module' and 'name' must both be specified together."
            )
        return self

    @model_validator(mode="after")
    def plugin_requires_both_name_and_regridder_kwargs(self) -> RegriddingConfig:
        if (self.name is None) and (self.regridder_kwargs is None):
            raise ValueError(
                "``[regridding]``: both 'name' and 'regridder_kwargs' must be specified."
            )
        return self


# -- [outputs] -- #
class OutputConfig(BaseModel):
    name: str | None = None
    module: str | None = None
    format: Literal["netcdf"] = "netcdf"
    output_dir: str
    output_name: str
    date_format: Literal["Y", "M", "D"] = "Y"
    chunks: dict[str, int] = Field(default_factory=dict)
    writer_kwargs: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def plugin_requires_both_module_and_name(self) -> OutputConfig:
        if (self.module is not None) and (self.name is None):
            raise ValueError(
                "``[outputs]``: 'module' and 'name' must both be specified together."
            )
        return self

    @model_validator(mode="after")
    def plugin_requires_both_name_and_writer_kwargs(self) -> OutputConfig:
        if (self.name is None) and (self.writer_kwargs is None):
            raise ValueError(
                "``[outputs]``: both 'name' and 'writer_kwargs' must be specified."
            )
        return self


# -- Top-level AppConfig -- #
class AppConfig(BaseModel):
    """
    Top-level OceanOSSE configuration model.
    Validates the entire parsed .toml dict.
    """

    domain: DomainConfig
    inputs: InputConfig
    climatology: ClimatologyConfig
    sampling: SamplingConfig
    regridding: RegriddingConfig
    outputs: OutputConfig
