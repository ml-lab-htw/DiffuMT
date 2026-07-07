import json
import logging
from abc import  ABC
from pathlib import Path
from typing import Optional


import yaml

logger = logging.getLogger(__name__)


class BaseConfig(ABC):
    @classmethod
    def load(cls, config_path: Optional[str] = None, overrides: Optional[dict] = None):
        """
        Loads a configuration, recursively handling nested BaseConfig objects.

        CHANGED: After loading, it stores the source file path on the instance
        so it can be saved back to the same location later.
        """
        logger.debug(f"Attempting to load configuration for '{cls.__name__}'...")
        data = {}
        config_path_obj = None

        if config_path:
            config_path_obj = Path(config_path)  # Use pathlib for robustness
            logger.debug(f"Loading from file: {config_path_obj}")

            if not config_path_obj.is_file():
                logger.error(f"Config file not found: {config_path_obj}")
                raise FileNotFoundError(f"Config file not found: {config_path_obj}")

            try:
                with config_path_obj.open("r") as f:
                    if config_path_obj.suffix.lower() in (".yml", ".yaml"):
                        data = yaml.safe_load(f) or {}
                        logger.debug(f"Loaded YAML data from {config_path_obj}")
                    elif config_path_obj.suffix.lower() == ".json":
                        data = json.load(f) or {}
                        logger.debug(f"Loaded JSON data from {config_path_obj}")
                    else:
                        msg = f"Unsupported config file format: '{config_path_obj.suffix}'. Use .yml, .yaml, or .json"
                        logger.error(msg)
                        raise ValueError(msg)
            except Exception as e:
                logger.error(f"Error reading config file {config_path_obj}: {e}", exc_info=True)
                raise

        if overrides:
            logger.debug(f"Applying overrides: {overrides}")
            data.update(overrides)
            logger.debug(f"Data after overrides: {data}")

        # Ensure 'save_dir' is included in the configuration data
        if 'save_dir' not in data:
            data['save_dir'] = '.models'  # Default value for save_dir

        # Create the instance from the loaded data
        instance = cls.from_dict(data)
        logger.debug(f"Created instance of '{cls.__name__}' from data.")

        # Store the source path on the instance for the new save() method
        if config_path_obj:
            instance._source_path = config_path_obj.resolve()
            logger.debug(
                f"Configuration for '{cls.__name__}' loaded successfully from {instance._source_path}."
            )
        else:
            instance._source_path = None
            logger.debug(
                f"Configuration for '{cls.__name__}' loaded/created in memory (no source file)."
            )

        return instance