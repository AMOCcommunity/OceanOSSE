"""
cli.py

Description: Command-line interface for OceanOSSE package.

Created By: Ollie Tooth (oliver.tooth@noc.ac.uk)
"""

# -- Import dependencies -- #
import logging
import sys

import typer
from typing_extensions import Annotated, Optional

from OceanOSSE.pipeline import describe_pipeline, run_pipeline

from .__init__ import __version__

app = typer.Typer()
logger = logging.getLogger(__name__)


# -- Define CLI Functions -- #
def create_header(
    config_path: str,
    log_path: str,
) -> None:
    """
    Add OceanOSSE header to log.

    Parameters:
    -----------
    config_path : str
        Filepath to OceanOSSE config .toml file.
    log_path : str
        Filepath to OceanOSSE log file.
    """
    logger.info(
        f"""
╔══════════════════════════════════════════════════════════════╗
║                         OceanOSSE                            ║
║       Ocean Observing System Simulation Experiment Tool      ║
╠══════════════════════════════════════════════════════════════╣
  OceanOSSE Version : {__version__}
  Python Version  : {sys.version.split()[0]}
  Config File     : {config_path}
  Log File        : {log_path} 
╚══════════════════════════════════════════════════════════════╝
""",
        extra={"simple": True},
    )


def init_logging(log_path: str) -> None:
    """
    Initialise OceanOSSE logging.

    Parameters:
    -----------
    log_path : str
        Filepath to log file. If None, logs to 'ocean_osse.log'.
    """
    # === Validate Inputs === #
    if not isinstance(log_path, str):
        raise TypeError("log_path must be a string.")

    logging.basicConfig(
        format="⦿══⦿  OceanOSSE  ⦿══⦿  ║ %(levelname)10s ║ %(asctime)s ║ %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )


# === Create Typer App === #
@app.callback()
def main() -> None:
    """
    Main callback for Typer app to allow
    single run command to be defined.
    """
    pass


@app.command()
def run(
    config: Annotated[str, typer.Argument(help="Path to OceanOSSE config .toml file")],
    log: Annotated[
        Optional[str],
        typer.Option(
            help="Path to write OceanOSSE log file", rich_help_panel="Options"
        ),
    ] = "ocean_osse.log",
    dry_run: Annotated[
        Optional[bool],
        typer.Option(
            help="Describe OceanOSSE workflow without execution.",
            rich_help_panel="Options",
        ),
    ] = False,
) -> None:
    """
    Run OceanOSSE workflow defined by configuration (.toml) file in current process.
    """
    # === Initialise Logging === #
    init_logging(log_path=log)
    create_header(config_path=config, log_path=log)

    # === Run OceanOSSE === #
    args = {
        "config_file": config,
        "log_filepath": log,
    }
    if dry_run:
        describe_pipeline(args=args)
    else:
        run_pipeline(args=args)

    logging.info("✔ OceanOSSE Completed ✔")


if __name__ == "__main__":
    app()
