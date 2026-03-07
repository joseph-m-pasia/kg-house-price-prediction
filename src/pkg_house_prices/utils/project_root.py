from pathlib import Path
from numpy import info
from pkg_house_prices.utils.logger import logger
from pkg_house_prices.utils.config import CONFIG

# --- Determine project root ---
def _get_project_root():
    """
    Returns the project root folder, works for:
    - modules (production-ready)
    - notebooks (development-ready)
    """
    logger.info("_get_project_root() - Determining project root...")
    # __file__ may exist in notebooks but represent a fake path like <ipython-input>
    if CONFIG['runtype']['prod_run']:
        logger.info(f"_get_project_root() - Running in a production environment") # True if running as production module, False if running in notebook
        return Path(__file__).parents[2]   
    else:
        logger.info(f"_get_project_root() - Running in a non-production (.g. notebook) environment")
        return Path(CONFIG['dev_absolute_path'])  # <-- set your local project root for notebooks here
    
PROJECT_ROOT = _get_project_root()                                           # <-- now PROJECT_ROOT is defined
logger.info(f"_get_project_root() - PROJECT ROOT is {PROJECT_ROOT}")



