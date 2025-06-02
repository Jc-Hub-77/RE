import os
import importlib.util
import json
import logging
from typing import Optional, Any # Added Optional, Any
from sqlalchemy.orm import Session # Added Session

from backend.models import Strategy as StrategyModel, SystemSetting # Added SystemSetting
from backend.config import settings

# Initialize logger
logger = logging.getLogger(__name__)

# Path to the directory where strategy .py files are stored.
# Adjust this path if your strategies are located elsewhere relative to this service file.
# Example: If 'utils' is in 'backend', and strategies in 'backend/strategies', then 'strategies' is correct.
STRATEGIES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'strategies')
# Ensure this path is correct and accessible by the application.
# Consider making this configurable via settings.STRATEGIES_DIR for flexibility.


def _load_strategy_class_from_db_obj(strategy_db_obj: StrategyModel):
    """Dynamically loads a strategy class from its file path stored in the DB object."""
    if not strategy_db_obj.python_code_path:
        logger.error(f"Strategy DB object ID {strategy_db_obj.id} has no python_code_path.")
        return None
    
    # Use settings.STRATEGIES_DIR if defined, otherwise fallback to local STRATEGIES_DIR
    effective_strategies_dir = getattr(settings, 'STRATEGIES_DIR', STRATEGIES_DIR)
    file_path = os.path.join(effective_strategies_dir, strategy_db_obj.python_code_path)
    
    module_name_from_path = strategy_db_obj.python_code_path.replace('.py', '').replace(os.path.sep, '.')
    
    # Try to infer class name: MyStrategyFile.py -> MyStrategyFile, or my_strategy.py -> MyStrategy
    # A 'main_class_name' field in StrategyModel is highly recommended for robustness.
    base_module_name = os.path.splitext(os.path.basename(strategy_db_obj.python_code_path))[0]
    assumed_class_name_1 = "".join(word.capitalize() for word in base_module_name.split('_'))
    assumed_class_name_2 = base_module_name # If class name is same as file name (without .py)
    
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        logger.error(f"Strategy file not found at {file_path} for strategy '{strategy_db_obj.name}'.")
        return None
        
    spec = importlib.util.spec_from_file_location(module_name_from_path, file_path)
    if spec is None or spec.loader is None:
        logger.error(f"Could not load spec or loader for strategy module {module_name_from_path} at {file_path}.")
        return None
    
    strategy_module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(strategy_module)
        
        loaded_class = None
        if hasattr(strategy_module, assumed_class_name_1):
            loaded_class = getattr(strategy_module, assumed_class_name_1)
        elif hasattr(strategy_module, assumed_class_name_2):
             loaded_class = getattr(strategy_module, assumed_class_name_2)
        elif hasattr(strategy_module, "Strategy"): # Common fallback
            loaded_class = getattr(strategy_module, "Strategy")

        if not loaded_class: # Fallback: iterate through module attributes
            for attr_name in dir(strategy_module):
                attr = getattr(strategy_module, attr_name)
                if isinstance(attr, type) and attr_name.endswith("Strategy") and attr_name != "BaseStrategy":
                    logger.info(f"Found potential strategy class by convention: {attr_name} in {file_path}")
                    loaded_class = attr
                    break 
            if not loaded_class:
                logger.error(f"Could not find a suitable strategy class in {file_path} for '{strategy_db_obj.name}'. Tried {assumed_class_name_1}, {assumed_class_name_2}, 'Strategy'.")
                return None
        
        logger.info(f"Successfully loaded strategy class '{loaded_class.__name__}' from {file_path}.")
        return loaded_class
    except Exception as e:
        logger.error(f"Error loading strategy module {module_name_from_path} for '{strategy_db_obj.name}': {e}", exc_info=True)
        return None

def get_system_setting(db_session: Session, key: str, default_value: Optional[Any] = None) -> Optional[str]:
    """
    Retrieves a system setting value from the database.

    Args:
        db_session: The SQLAlchemy database session.
        key: The key of the setting to retrieve.
        default_value: The value to return if the key is not found.

    Returns:
        The value of the setting as a string if found, otherwise the default_value.
    """
    try:
        setting = db_session.query(SystemSetting).filter(SystemSetting.key == key).first()
        if setting:
            return setting.value
        else:
            logger.debug(f"System setting for key '{key}' not found. Returning default value: {default_value}")
            return default_value
    except Exception as e:
        logger.error(f"Error retrieving system setting for key '{key}': {e}", exc_info=True)
        return default_value
