import yaml
from pathlib import Path

def _load_config():
    """
    Loads the YAML configuration file from project root
    """
    # Path relative to this file: go up two levels to project root
    config_path = Path(__file__).parents[3] / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
     
CONFIG = _load_config()
