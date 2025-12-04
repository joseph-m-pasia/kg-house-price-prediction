import yaml
from pathlib import Path

def load_config():
    """
    Loads the YAML configuration file from project root
    """
    # Path relative to this file: go up two levels to project root
    config_path = Path(__file__).parents[2] / "config.yaml"

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg
