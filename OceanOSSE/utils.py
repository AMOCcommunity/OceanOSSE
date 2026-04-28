"""
utils.py

Description: Utility functions for OceanOSSE package.

Created By: OceanOSSE Development Team (NOC, UK)
"""

# -- Import Dependencies -- #
from __future__ import annotations

import importlib
import tomllib
from pathlib import Path

from OceanOSSE.validation import AppConfig


# -- Utility Functions -- #
def load_config(config_path: str) -> dict:
    """
    Load and parse an OceanOSSE configuration file.

    Parameters
    ----------
    config_path : str
        Path to the OceanOSSE ``.toml`` configuration file.

    Returns
    -------
    dict
        Parsed configuration as a nested dictionary.

    Raises
    ------
    FileNotFoundError
    tomllib.TOMLDecodeError
    """
    # -- Validate Inputs -- #
    if not isinstance(config_path, str):
        raise TypeError("``config_path`` must be a string.")
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path!r}")

    # -- Load and Parse Configuration -- #
    # Open config .toml file:
    with open(path, "rb") as f:
        cfg_data = tomllib.load(f)

    # Parse and validate config data using Pydantic models:
    config = AppConfig(**cfg_data)
    # Convert config params to dict:
    d_config = config.model_dump(mode="json")

    return d_config


def import_class(
    module: str,
    class_name: str,
    class_type: type,
) -> type:
    """
    Dynamically import a class from specified module path.

    This is used by the OceanOSSE pipeline to load third-party
    plugin classes declared in a `.toml` configuration file.

    Parameters
    ----------
    module : str
        Import path to the module, e.g. ``"my_package.samplers"``.
    class_name : str
        Name of the class to import from module, e.g. ``"ArgoSampler"``.
    class_type : type
        Expected base type of the class.

    Returns
    -------
    type
        Imported class object.

    Raises
    ------
    ModuleNotFoundError
    AttributeError
    """
    # -- Validate Inputs -- #
    if not isinstance(module, str):
        raise TypeError("``module`` must be a string.")
    if not isinstance(class_name, str):
        raise TypeError("``class_name`` must be a string.")

    # -- Dynamically import class -- #
    try:
        module = importlib.import_module(module)
    except ImportError as e:
        raise ImportError(f"Failed to import module '{module}'") from e
    try:
        Kernel = getattr(module, class_name)
    except AttributeError as e:
        raise AttributeError(
            f"Failed to import {class_type.__name__} '{class_name}' from module '{module}'"
        ) from e

    # -- Verify that class is callable -- #
    if not callable(Kernel):
        raise TypeError(f"{class_type.__name__} '{class_name}' is not callable.")

    # -- Verify that class is a subclass of the expected type -- #
    if not issubclass(Kernel, class_type):
        raise TypeError(f"'{class_name}' is not a subclass of '{class_type.__name__}'.")

    return Kernel
