"""
pipeline.py

Description: Pipeline functions for OceanOSSE package.

Created By: Ollie Tooth (oliver.tooth@noc.ac.uk)
"""

# -- Import Dependencies -- #
import logging

from OceanOSSE.gridding.regridder import MockRegridder, Regridder
from OceanOSSE.io.dataloader import DataLoader, NetCDFDataLoader
from OceanOSSE.io.datawriter import DataWriter, NetCDFDataWriter
from OceanOSSE.sampling.sampler import MockObsSampler, ObsSampler
from OceanOSSE.utils import import_class, load_config

logger = logging.getLogger(__name__)

# -- Registries -- #
_DATA_LOADER_REGISTRY: dict[str, type[DataLoader]] = {"netcdf": NetCDFDataLoader}

_OBS_SAMPLER_REGISTRY: dict[str, type[ObsSampler]] = {"test": MockObsSampler}

_REGRIDDER_REGISTRY: dict[str, type[Regridder]] = {"test": MockRegridder}

_DATA_WRITER_REGISTRY: dict[str, type[DataWriter]] = {"netcdf": NetCDFDataWriter}


# -- Factory Functions -- #
def _create_DataLoader(config: dict) -> DataLoader:
    """
    Instantiate a DataLoader from the `[inputs]` table of the .toml configuration file.

    Options:
    - Built-in Registry:
        - name: "netcdf" -> NetCDFDataLoader

    - Plugins:
        - Custom DataLoader imported from `module` and `name` specified in config.
    """
    # -- Validate Inputs -- #
    if not isinstance(config, dict):
        raise TypeError("``config`` must be a dictionary.")

    # -- Instantiate DataLoader -- #
    inputs = config["inputs"]

    # 1. Plugin DataLoader:
    if (inputs.get("module") is not None) and (inputs.get("name") is not None):
        # -- Import custom DataLoader class -- #
        data_loader = import_class(
            module=inputs["module"], class_name=inputs["name"], class_type=DataLoader
        )
        logger.info(
            f"Completed: Created DataLoader from Plugin: {inputs['module']}.{inputs['name']}"
        )

    # 2. Registry DataLoader:
    else:
        # -- Use DataLoader class from registry -- #
        format = inputs.get("format", "netcdf4")
        try:
            data_loader = _DATA_LOADER_REGISTRY[format]
        except KeyError as e:
            raise KeyError(
                f"DataLoader '{format}' not found in registry. Available options: {list(_DATA_LOADER_REGISTRY.keys())}"
            ) from e
        logger.info(
            f"Completed: Created DataLoader from Registry -> {format}: {data_loader.__name__}"
        )

    return data_loader.from_config(config=config)


def _create_ObsSampler(config: dict) -> ObsSampler:
    """
    Instantiate an ObsSampler from the `[sampling]` table of the .toml configuration file.

    Options:
    - Built-in Registry:
        - name: "test" -> TestObsSampler

    - Plugins:
        - Custom ObsSampler imported from `module` and `name` specified in config.
    """
    # -- Validate Inputs -- #
    if not isinstance(config, dict):
        raise TypeError("``config`` must be a dictionary.")

    # -- Instantiate ObsSampler -- #
    sampling = config["sampling"]

    # 1. Plugin ObsSampler:
    if (sampling.get("module") is not None) and (sampling.get("name") is not None):
        # -- Import custom ObsSampler class -- #
        obs_sampler = import_class(
            module=sampling["module"],
            class_name=sampling["name"],
            class_type=ObsSampler,
        )
        logger.info(
            f"Completed: Created ObsSampler from Plugin: {sampling['module']}.{sampling['name']}"
        )

    # 2. Registry ObsSampler:
    else:
        # -- Use ObsSampler class from registry -- #
        name = sampling.get("name", "test")
        try:
            obs_sampler = _OBS_SAMPLER_REGISTRY[name]
        except KeyError as e:
            raise KeyError(
                f"ObsSampler '{name}' not found in registry. Available options: {list(_OBS_SAMPLER_REGISTRY.keys())}"
            ) from e
        logger.info(
            f"Completed: Created ObsSampler from Registry -> {name}: {obs_sampler.__name__}"
        )

    return obs_sampler.from_config(config=config)


def _create_Regridder(config: dict) -> Regridder:
    """
    Instantiate a Regridder from the `[regridding]` table of the .toml configuration file.

    Options:
    - Built-in Registry:
        - name: "test" -> TestRegridder

    - Plugins:
        - Custom Regridder imported from `module` and `name` specified in config.
    """
    # -- Validate Inputs -- #
    if not isinstance(config, dict):
        raise TypeError("``config`` must be a dictionary.")

    # -- Instantiate Regridder -- #
    regridding = config["regridding"]

    # 1. Plugin Regridder:
    if (regridding.get("module") is not None) and (regridding.get("name") is not None):
        # -- Import custom Regridder class -- #
        regridder = import_class(
            module=regridding["module"],
            class_name=regridding["name"],
            class_type=Regridder,
        )
        logger.info(
            f"Completed: Created Regridder from Plugin: {regridding['module']}.{regridding['name']}"
        )

    # 2. Registry Regridder:
    else:
        # -- Use Regridder class from registry -- #
        name = regridding.get("name", "test")
        try:
            regridder = _REGRIDDER_REGISTRY[name]
        except KeyError as e:
            raise KeyError(
                f"Regridder '{name}' not found in registry. Available options: {list(_REGRIDDER_REGISTRY.keys())}"
            ) from e
        logger.info(
            f"Completed: Created Regridder from Registry -> {name}: {regridder.__name__}"
        )

    return regridder.from_config(config=config)


def _create_DataWriter(config: dict) -> DataWriter:
    """
    Instantiate a DataWriter from the `[outputs]` table of the .toml configuration file.

    Options:
    - Built-in Registry:
        - name: "netcdf" -> NetCDFDataWriter

    - Plugins:
        - Custom DataWriter imported from `module` and `name` specified in config.
    """
    # -- Validate Inputs -- #
    if not isinstance(config, dict):
        raise TypeError("``config`` must be a dictionary.")

    # -- Instantiate DataWriter -- #
    outputs = config["outputs"]

    # 1. Plugin DataWriter:
    if (outputs.get("module") is not None) and (outputs.get("name") is not None):
        # -- Import custom DataWriter class -- #
        data_writer = import_class(
            module=outputs["module"], class_name=outputs["name"], class_type=DataWriter
        )
        logger.info(
            f"Completed: Created DataWriter from Plugin: {outputs['module']}.{outputs['name']}"
        )

    # 2. Registry DataWriter:
    else:
        # -- Use DataWriter class from registry -- #
        format = outputs.get("format", "netcdf4")
        try:
            data_writer = _DATA_WRITER_REGISTRY[format]
        except KeyError as e:
            raise KeyError(
                f"DataWriter '{format}' not found in registry. Available options: {list(_DATA_WRITER_REGISTRY.keys())}"
            ) from e
        logger.info(
            f"Completed: Created DataWriter from Registry -> {format}: {data_writer.__name__}"
        )

    return data_writer.from_config(config=config)


# -- Define Pipeline Functions -- #
def run_pipeline(args: dict) -> None:
    """
    Run OceanOSSE pipeline using specified config .ini file.

    Pipeline Steps:
    1. Instantiate DataLoader -> Load standardised ocean model dataset.
    2. Instantiate ObsSampler -> Sample synthetic ocean observations from model dataset.
    3. Instantiate Regridder -> Regrid synthetic observations onto original model grid.
    4. Instantiate DataWriter -> Write output dataset to file.

    Parameters:
    -----------
    args : dict
        Command line arguments.
    """
    # === Inputs === #
    logger.info("==== Inputs ====")
    # Load config .toml file:
    config = load_config(config_path=args["config_file"])
    logger.info(f"Completed: Read & validated config file -> {args['config_file']}")

    # Load ocean model dataset using DataLoader:
    data_loader = _create_DataLoader(config=config)
    logger.info(f"In Progress: Loading ocean model dataset using {data_loader}...")
    ds_mdl = data_loader.load_data()

    # === Sampling === #
    logger.info("==== Sampling ====")
    # Sample synthetic ocean observations using ObsSampler:
    obs_sampler = _create_ObsSampler(config=config)
    logger.info(
        f"In Progress: Sampling synthetic ocean observations using {obs_sampler}..."
    )
    ds_obs = obs_sampler.sample(ds=ds_mdl)

    # === Regridding === #
    logger.info("==== Regridding ====")
    # Regrid synthetic ocean observations using Regridder:
    regridder = _create_Regridder(config=config)
    logger.info(
        f"In Progress: Regridding synthetic ocean observations using {regridder}..."
    )
    ds_regridded = regridder.regrid(ds=ds_obs)

    # === Outputs === #
    logger.info("==== Outputs ====")
    # Write output dataset to file using DataWriter:
    data_writer = _create_DataWriter(config=config)
    logger.info(f"In Progress: Writing output dataset to file using {data_writer}...")
    data_writer.write_data(ds=ds_regridded)
    # Close all files:
    for ds in [ds_mdl, ds_obs, ds_regridded]:
        ds.close()
    logger.info("--> Completed: Closed all dataset files.")


def describe_pipeline(args: dict) -> str:
    """
    Describe & validate OceanOSSE pipeline using config.

    Parameters:
    -----------
    args : dict
        Command line arguments.

    Returns:
    --------
    str
        Description of OceanOSSE pipeline.
    """
    # === Inputs === #
    logger.info("==== Inputs ====")
    # Load config .toml file:
    config = load_config(config_path=args["config_file"])
    logger.info(f"Completed: Read & validated config file -> {args['config_file']}")

    # Load ocean model dataset using DataLoader:
    data_loader = _create_DataLoader(config=config)
    logger.info("Action: Load ocean model dataset using...")
    logger.info("* DataLoader   : %r", data_loader)

    # === Sampling === #
    logger.info("==== Sampling ====")
    # Sample synthetic ocean observations using ObsSampler:
    obs_sampler = _create_ObsSampler(config=config)
    logger.info("Action: Sample synthetic ocean observations using...")
    logger.info("* ObsSampler   : %r", obs_sampler)

    # === Regridding === #
    logger.info("==== Regridding ====")
    # Regrid synthetic ocean observations using Regridder:
    regridder = _create_Regridder(config=config)
    logger.info("Action: Regrid synthetic ocean observations using...")
    logger.info("* Regridder   : %r", regridder)

    # === Outputs === #
    logger.info("==== Outputs ====")
    # Write output dataset to file using DataWriter:
    data_writer = _create_DataWriter(config=config)
    logger.info("Action: Write output dataset to file using...")
    logger.info("* DataWriter   : %r", data_writer)
    # Define output filepath:
    outputs = config["outputs"]
    if outputs["format"] == "netcdf":
        extension = "nc"
    else:
        extension = "zarr"
    logger.info(
        f"* Output File = {outputs['output_dir']}/{outputs['output_name']}_YYYY-MM_YYYY-MM.{extension}"
    )
