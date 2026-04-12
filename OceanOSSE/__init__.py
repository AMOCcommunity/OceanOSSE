"""
OceanOSSE

Python toolbox for performing Observing System Simulation Experiments (OSSEs) in ocean general circulation models.
"""
__author__ = "Ollie Tooth (oliver.tooth@noc.ac.uk)"
__credits__ = "National Oceanography Centre (NOC), Southampton, UK"

from importlib.metadata import version as _version

from OceanOSSE import (
    cli,
    pipeline,
)

try:
    __version__ = _version("OceanOSSE")
except Exception:
    # Local copy or not installed with setuptools.
    # Disable minimum version checks on downstream libraries.
    __version__ = "9999.0.0"

__all__ = ("cli", "pipeline")