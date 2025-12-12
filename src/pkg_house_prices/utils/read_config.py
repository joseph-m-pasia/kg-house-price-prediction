from pkg_house_prices.utils.logger import logger
from pkg_house_prices.utils.project_root import PROJECT_ROOT
from pkg_house_prices.utils.config import CONFIG

# --- Helper: resolve absolute path from CONFIG ---
def read_config(*keys):
    """
    Fetch nested YAML keys and return the values
    Example: read_config("data", "train")
    """
    logger.info(f"read_config() - Reading config keys: {keys}")
    logger.info(f"read_config() - Project Root is {PROJECT_ROOT}")

    d = CONFIG
    for k in keys:
        d = d[k]
    return PROJECT_ROOT / d