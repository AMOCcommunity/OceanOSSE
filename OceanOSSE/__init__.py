"""
OceanOSSE

Python toolbox for performing Observing System Simulation Experiments (OSSEs)
in ocean general circulation models.
"""

__author__ = "Ollie Tooth (oliver.tooth@noc.ac.uk)"
__credits__ = "National Oceanography Centre (NOC), Southampton, UK"

from importlib.metadata import version as _version

from OceanOSSE import cli, pipeline
from OceanOSSE.gridding.regridder import Regridder
from OceanOSSE.io.dataloader import DataLoader
from OceanOSSE.io.datawriter import DataWriter
from OceanOSSE.sampling.sampler import ErrorKernel, ObsSampler
from OceanOSSE.gridding.regridder_simple import SwapRegridder

try:
    __version__ = _version("OceanOSSE")
except Exception:
    __version__ = "9999.0.0"

__all__ = (
    "cli",
    "pipeline",
    "DataLoader",
    "DataWriter",
    "ErrorKernel",
    "ObsSampler",
    "Regridder",
)
