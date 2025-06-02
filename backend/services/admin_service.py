# backend/services/admin_service.py
import datetime
import os
import logging
from typing import Optional # Ensure Optional is imported
from sqlalchemy.orm import Session
from sqlalchemy import desc, or_ # For count, desc, or_
import sqlalchemy 
import sqlalchemy.types 
import json
import importlib.util # Added for strategy validation

from backend.models import User, Strategy, UserStrategySubscription, PaymentTransaction, ApiKey, SystemSetting # Added SystemSetting
from backend.services import live_trading_service # Added live_trading_service
from backend.config import settings
from fastapi import HTTPException # Added HTTPException

# Initialize logger
logger = logging.getLogger(__name__)

# --- Helper for Payment Options Validation ---
def _validate_payment_options_json(payment_options_json_str: Optional[str]) -> Optional[list]:
    """
    Validates the structure and types of payment_options_json.
    Raises HTTPException if validation fails.
    Returns the parsed list if valid, or None if input is None or empty.
    """
    if not payment_options_json_str: # Handles None or empty string
        return None

    try:
        parsed_options = json.loads(payment_options_json_str)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format for payment options.")

    if not isinstance(parsed_options, list):
        raise HTTPException(status_code=400, detail="Payment options must be a list.")

    if not parsed_options: # Empty list is acceptable
        return []

    for option in parsed_options:
        if not isinstance(option, dict):
            raise HTTPException(status_code=400, detail="Each payment option must be a dictionary.")
        
        if 'months' not in option or 'price_usd' not in option:
            raise HTTPException(status_code=400, detail="Each payment option must have 'months' and 'price_usd' keys.")
        
        if not isinstance(option['months'], int):
            raise HTTPException(status_code=400, detail=f"Payment option 'months' (value: {option['months']}) must be an integer.")
        
        if not isinstance(option['price_usd'], (int, float)):
            raise HTTPException(status_code=400, detail=f"Payment option 'price_usd' (value: {option['price_usd']}) must be a number (integer or float).")
        
        # 'description' is optional, no specific validation needed here unless constraints apply (e.g., max length)
        
    return parsed_options


# --- Admin User Management ---
def list_all_users(db_session: Session, page: int = 1, per_page: int = 20, search_term: str = None, sort_by: str = "id", sort_order: str = "asc"):
    """Lists all users with pagination, search, and sorting."""
    query = db_session.query(User)
    
    if search_term:
        search_filter = f"%{search_term}%"
        query = query.filter(
            or_(
                User.username.ilike(search_filter), 
                User.email.ilike(search_filter),
                User.id.cast(sqlalchemy.types.String).ilike(search_filter) # Search by ID
            )
        )
    
    # Sorting
    sort_column = getattr(User, sort_by, User.id) 
    if sort_order.lower() == "desc":
        query = query.order_by(desc(sort_column))
    else:
        query = query.order_by(sort_column) 

    total_users = query.count()
    users_data = query.offset((page - 1) * per_page).limit(per_page).all()
    
    return {
        "status": "success",
        "users": [{
            "id": u.id, "username": u.username, "email": u.email, 
            "is_admin": u.is_admin, 
            "is_active": u.is_active, 
            "email_verified": u.email_verified,
            "created_at": u.created_at.isoformat() if u.created_at else None,
            "profile_full_name": u.profile.full_name if u.profile else None 
        } for u in users_data],
        "total_users": total_users,
        "page": page,
        "per_page": per_page,
        "total_pages": (total_users + per_page - 1) // per_page if per_page > 0 else 0
    }

def set_user_admin_status(db_session: Session, user_id: int, make_admin: bool):
    user = db_session.query(User).filter(User.id == user_id).first()
    if not user:
        return {"status": "error", "message": "User not found."}
    
    if not make_admin and user.is_admin:
        admin_count = db_session.query(User).filter(User.is_admin == True).count()
        if admin_count <= 1:
            return {"status": "error", "message": "Cannot remove the last admin account."}

    user.is_admin = make_admin
    try:
        db_session.commit()
        logger.info(f"Admin: User {user_id} admin status set to {make_admin}.")
        return {"status": "success", "message": f"User {user_id} admin status updated to {make_admin}."}
    except Exception as e:
        db_session.rollback()
        logger.error(f"Error setting admin status for user {user_id}: {e}", exc_info=True)
        return {"status": "error", "message": f"Database error: {e}"}

def toggle_user_active_status(db_session: Session, user_id: int, activate: bool):
    """Toggles the active status of a user."""
    user = db_session.query(User).filter(User.id == user_id).first()
    if not user:
        logger.warning(f"Admin: Attempted to toggle active status for non-existent user ID: {user_id}")
        return {"status": "error", "message": "User not found."}

    user.is_active = activate

    try:
        db_session.commit()
        status_message = "activated" if activate else "deactivated"
        logger.info(f"Admin: User {user_id} has been {status_message}.")
        return {"status": "success", "message": f"User {user_id} {status_message} successfully."}
    except Exception as e:
        db_session.rollback()
        logger.error(f"Admin: Error toggling active status for user {user_id}: {e}", exc_info=True)
        return {"status": "error", "message": f"Database error: {e}"}

def toggle_user_email_verified(db_session: Session, user_id: int, set_verified_status: bool): # Renamed 'activate' to 'set_verified_status' for clarity
    """
    Admin function to manually set a user's email verification status.
    """
    user = db_session.query(User).filter(User.id == user_id).first()
    if not user:
        logger.warning(f"Admin: Attempted to toggle email verified status for non-existent user ID: {user_id}")
        return {"status": "error", "message": "User not found."}

    user.email_verified = set_verified_status
    if set_verified_status: # If marking as verified, clear any pending verification token
        user.email_verification_token = None
        user.email_verification_token_expires_at = None
    
    # If marking as unverified, should we generate a new token and send an email?
    # For now, this function just sets the status. Sending a new verification email could be a separate admin action.

    try:
        db_session.commit()
        status_message = "verified" if set_verified_status else "unverified"
        logger.info(f"Admin: Email for user {user_id} has been marked as {status_message}.")
        return {"status": "success", "message": f"User's email marked as {status_message} successfully."}
    except Exception as e:
        db_session.rollback()
        logger.error(f"Admin: Error toggling email verified status for user {user_id}: {e}", exc_info=True)
        return {"status": "error", "message": f"Database error: {e}"}


# --- Admin Strategy Management ---
def list_all_strategies_admin(db_session: Session):
    try:
        strategies = db_session.query(Strategy).order_by(Strategy.name).all()
        return {
            "status": "success", 
            "strategies": [
                {
                    "id": s.id, "name": s.name, "description": s.description, 
                    "python_code_path": s.python_code_path, 
                    "default_parameters": s.default_parameters,
                    "category": s.category, "risk_level": s.risk_level,
                    "is_active": s.is_active,
                    "created_at": s.created_at.isoformat() if s.created_at else None
                } for s in strategies
            ]
        }
    except Exception as e:
        logger.error(f"Error listing strategies for admin: {e}", exc_info=True)
        return {"status": "error", "message": "Could not retrieve strategies."}


def add_new_strategy_admin(db_session: Session, name: str, description: str, python_code_path: str, 
                           default_parameters: str, category: str, risk_level: str,
                           payment_options_json: Optional[str] = None):
    existing_strategy = db_session.query(Strategy).filter(Strategy.name == name).first()
    if existing_strategy:
        return {"status": "error", "message": f"Strategy with name '{name}' already exists."}
    
    if not settings.STRATEGIES_DIR:
        logger.error("Admin: STRATEGIES_DIR is not configured in settings. Cannot add strategy.")
        return {"status": "error", "message": "System configuration error: STRATEGIES_DIR not set."}
    effective_strategies_dir = settings.STRATEGIES_DIR

    full_path = os.path.join(effective_strategies_dir, python_code_path)
    
    if not os.path.exists(full_path) or not os.path.isfile(full_path) or not python_code_path.endswith(".py"):
        logger.warning(f"Admin: Attempted to add strategy with invalid path: {full_path} (based on python_code_path: {python_code_path})")
        return {"status": "error", "message": f"Strategy file not found or invalid at path: {python_code_path}"}

    # Validate strategy file content
    try:
        module_name = os.path.splitext(os.path.basename(python_code_path))[0] # Get filename without .py
        
        spec = importlib.util.spec_from_file_location(module_name, full_path)
        if spec is None or spec.loader is None: # Check both spec and spec.loader
            logger.warning(f"Admin: Could not create module spec or loader for strategy: {full_path}")
            return {"status": "error", "message": f"Could not load strategy module from path: {python_code_path}"}
        
        strategy_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(strategy_module)

        StrategyClass = None
        # Common convention: Class name is CamelCase version of file name or simply "Strategy"
        expected_class_name_1 = "".join(word.capitalize() for word in module_name.split('_')) # my_strategy -> MyStrategy
        expected_class_name_2 = "Strategy" # General fallback

        if hasattr(strategy_module, expected_class_name_1):
            StrategyClass = getattr(strategy_module, expected_class_name_1)
        elif hasattr(strategy_module, expected_class_name_2):
            StrategyClass = getattr(strategy_module, expected_class_name_2)
        
        if StrategyClass is None:
            logger.warning(f"Admin: Strategy module {python_code_path} does not contain a recognized Strategy class (e.g., {expected_class_name_1} or {expected_class_name_2}).")
            return {"status": "error", "message": "Strategy module does not conform to expected class naming convention."}
        
        # Check for required methods (adjust as per your BaseStrategy or expected interface)
        required_methods = ['run_backtest', 'execute_live_signal'] 
        for method_name in required_methods:
            if not (hasattr(StrategyClass, method_name) and callable(getattr(StrategyClass, method_name))):
                logger.warning(f"Admin: Strategy class in {python_code_path} does not have required method: {method_name}.")
                return {"status": "error", "message": f"Strategy class does not have required method: {method_name}."}
    except Exception as e:
        logger.error(f"Admin: Error validating strategy file {python_code_path}: {e}", exc_info=True)
        return {"status": "error", "message": f"Error validating strategy file: {str(e)}"}

    try:
        json.loads(default_parameters)
    except json.JSONDecodeError:
        logger.warning(f"Admin: Attempted to add strategy with invalid JSON parameters: {default_parameters}")
        return {"status": "error", "message": "Default parameters must be valid JSON."}

    # Validate payment_options_json using the helper
    # The helper will raise HTTPException if validation fails, which will be caught by FastAPI framework.
    # If it returns successfully, we don't need to store its return value here explicitly,
    # as the original payment_options_json string is stored in the model.
    # However, if we wanted to store the cleaned/parsed version, we could assign it.
    # For now, just validate. The actual string is passed to the model.
    try:
        _validate_payment_options_json(payment_options_json)
    except HTTPException as e:
        logger.warning(f"Admin: Validation failed for payment_options_json during strategy creation. Detail: {e.detail}")
        return {"status": "error", "message": e.detail} # Return error structure consistent with others

    new_strategy = Strategy(
        name=name, 
        description=description, 
        python_code_path=python_code_path, # Store relative path
        default_parameters=default_parameters,
        category=category,
        risk_level=risk_level,
        is_active=True,
        payment_options_json=payment_options_json
    )
    try:
        db_session.add(new_strategy)
        db_session.commit()
        db_session.refresh(new_strategy)
        logger.info(f"Admin: New strategy '{name}' added with ID {new_strategy.id}.")
        return {"status": "success", "message": "Strategy added successfully.", "strategy_id": new_strategy.id}
    except Exception as e:
        db_session.rollback()
        logger.error(f"Error adding new strategy '{name}': {e}", exc_info=True)
        return {"status": "error", "message": f"Database error while adding strategy: {e}"}

def update_strategy_admin(db_session: Session, strategy_id: int, updates: dict):
    strategy = db_session.query(Strategy).filter(Strategy.id == strategy_id).first()
    if not strategy:
        return {"status": "error", "message": "Strategy not found."}

    allowed_fields = ["name", "description", "python_code_path", "default_parameters", "category", "risk_level", "is_active", "payment_options_json"]
    updated_count = 0
    for key, value in updates.items():
        if key in allowed_fields:
            if key == "name" and value != strategy.name:
                existing_strategy = db_session.query(Strategy).filter(Strategy.name == value, Strategy.id != strategy_id).first()
                if existing_strategy:
                    return {"status": "error", "message": f"Another strategy with name '{value}' already exists."}
            
            if key == "python_code_path" and value != strategy.python_code_path:
                if not settings.STRATEGIES_DIR:
                    logger.error("Admin: STRATEGIES_DIR is not configured in settings. Cannot update strategy path.")
                    return {"status": "error", "message": "System configuration error: STRATEGIES_DIR not set."}
                effective_strategies_dir = settings.STRATEGIES_DIR
                
                full_path = os.path.join(effective_strategies_dir, value)
                if not os.path.exists(full_path) or not os.path.isfile(full_path) or not value.endswith(".py"):
                    logger.warning(f"Admin: Attempted to update strategy with invalid path: {full_path}")
                    return {"status": "error", "message": f"New strategy file not found or invalid at path: {value}"}
                
                # Add full validation for the new strategy file content
                try:
                    module_name = os.path.splitext(os.path.basename(value))[0]
                    spec = importlib.util.spec_from_file_location(module_name, full_path)
                    if spec is None or spec.loader is None:
                        logger.warning(f"Admin: Could not create module spec or loader for updated strategy path: {full_path}")
                        return {"status": "error", "message": f"Could not load strategy module from updated path: {value}"}
                    
                    strategy_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(strategy_module)

                    StrategyClass = None
                    expected_class_name_1 = "".join(word.capitalize() for word in module_name.split('_'))
                    expected_class_name_2 = "Strategy"
                    if hasattr(strategy_module, expected_class_name_1): StrategyClass = getattr(strategy_module, expected_class_name_1)
                    elif hasattr(strategy_module, expected_class_name_2): StrategyClass = getattr(strategy_module, expected_class_name_2)
                    
                    if StrategyClass is None:
                        logger.warning(f"Admin: Updated strategy module {value} does not contain a recognized Strategy class.")
                        return {"status": "error", "message": "Updated strategy module does not conform to expected class naming convention."}
                    
                    required_methods = ['run_backtest', 'execute_live_signal']
                    for method_name in required_methods:
                        if not (hasattr(StrategyClass, method_name) and callable(getattr(StrategyClass, method_name))):
                            logger.warning(f"Admin: Strategy class in updated {value} does not have required method: {method_name}.")
                            return {"status": "error", "message": f"Updated strategy class does not have required method: {method_name}."}
                except Exception as e:
                    logger.error(f"Admin: Error validating updated strategy file {value}: {e}", exc_info=True)
                    return {"status": "error", "message": f"Error validating updated strategy file: {str(e)}"}

            if key == "payment_options_json": # value can be None, empty string, or JSON string
                try:
                    # _validate_payment_options_json handles None, empty string, and validation.
                    # It will raise HTTPException if validation fails.
                    _validate_payment_options_json(value) 
                except HTTPException as e:
                    logger.warning(f"Admin: Validation failed for payment_options_json during strategy update for ID {strategy_id}. Detail: {e.detail}")
                    return {"status": "error", "message": e.detail} # Return error structure
                # If validation passes, setattr will handle it (value could be None, empty str, or valid JSON str)
            
            setattr(strategy, key, value)
            updated_count +=1
    
    if updated_count == 0:
        return {"status": "info", "message": "No valid fields provided for update."}

    try:
        db_session.commit()
        logger.info(f"Admin: Strategy {strategy_id} updated.")
        return {"status": "success", "message": "Strategy updated successfully."}
    except Exception as e:
        db_session.rollback()
        logger.error(f"Error updating strategy {strategy_id}: {e}", exc_info=True)
        return {"status": "error", "message": f"Database error while updating strategy: {e}"}


# --- Admin Subscriptions & Payments Overview ---
def list_all_subscriptions_admin(db_session: Session, page: int = 1, per_page: int = 20, user_id: Optional[int] = None, strategy_id: Optional[int] = None, is_active: Optional[bool] = None):
    """Lists all user strategy subscriptions with pagination and filtering."""
    try:
        query = db_session.query(UserStrategySubscription).join(User).join(Strategy).outerjoin(ApiKey) # outerjoin for ApiKey

        if user_id is not None:
            query = query.filter(UserStrategySubscription.user_id == user_id)
        if strategy_id is not None:
            query = query.filter(UserStrategySubscription.strategy_id == strategy_id)
        if is_active is not None:
            query = query.filter(UserStrategySubscription.is_active == is_active)
        
        total_subscriptions = query.count()
        # Order by subscription ID descending by default for recent items first
        subscriptions_data = query.order_by(desc(UserStrategySubscription.id)).offset((page - 1) * per_page).limit(per_page).all()

        subscriptions_list = []
        for sub in subscriptions_data:
            subscriptions_list.append({
                "id": sub.id,
                "user_id": sub.user_id,
                "username": sub.user.username if sub.user else None,
                "strategy_id": sub.strategy_id,
                "strategy_name": sub.strategy.name if sub.strategy else None,
                "api_key_id": sub.api_key_id,
                "api_key_label": sub.api_key.label if sub.api_key else None, 
                "is_active": sub.is_active,
                "subscribed_at": sub.subscribed_at.isoformat() if sub.subscribed_at else None,
                "expires_at": sub.expires_at.isoformat() if sub.expires_at else None, 
                "custom_parameters": sub.custom_parameters, # This is already a JSON string from the model
                "status_message": sub.status_message,
                "celery_task_id": sub.celery_task_id # Added celery_task_id
            })

        return {
            "status": "success",
            "subscriptions": subscriptions_list,
            "total_subscriptions": total_subscriptions,
            "page": page,
            "per_page": per_page,
            "total_pages": (total_subscriptions + per_page - 1) // per_page if per_page > 0 else 0
        }
    except Exception as e:
        logger.error(f"Admin: Error listing all subscriptions (page {page}): {e}", exc_info=True)
        return {"status": "error", "message": f"Could not retrieve subscriptions: {e}"}

def list_all_payments_admin(db_session: Session, page: int = 1, per_page: int = 20):
    """Lists all payment transactions with pagination."""
    try:
        query = db_session.query(PaymentTransaction).join(User) # Assuming User is always linked

        total_payments = query.count()
        # Order by payment ID descending for recent items first
        payments_data = query.order_by(desc(PaymentTransaction.id)).offset((page - 1) * per_page).limit(per_page).all()

        payments_list = []
        for payment in payments_data:
            payments_list.append({
                "id": payment.id,
                "user_id": payment.user_id,
                "username": payment.user.username if payment.user else None,
                "usd_equivalent": float(payment.usd_equivalent) if payment.usd_equivalent is not None else None,
                "crypto_currency": payment.crypto_currency,
                "status": payment.status,
                "gateway_transaction_id": payment.gateway_transaction_id,
                "payment_gateway": payment.payment_gateway,
                "created_at": payment.created_at.isoformat() if payment.created_at else None,
                "updated_at": payment.updated_at.isoformat() if payment.updated_at else None,
                "description": payment.description,
                "user_strategy_subscription_id": payment.user_strategy_subscription_id
            })

        return {
            "status": "success",
            "payments": payments_list,
            "total_payments": total_payments,
            "page": page,
            "per_page": per_page,
            "total_pages": (total_payments + per_page - 1) // per_page if per_page > 0 else 0
        }
    except Exception as e:
        logger.error(f"Admin: Error listing all payments (page {page}): {e}", exc_info=True)
        return {"status": "error", "message": f"Could not retrieve payments: {e}"}

def get_total_revenue(db_session: Session):
    """Calculates the total revenue from completed payment transactions."""
    try:
        # Ensure the column name matches the model (usd_equivalent)
        total_revenue = db_session.query(sqlalchemy.func.sum(PaymentTransaction.usd_equivalent)).filter(
            PaymentTransaction.status == "completed"
        ).scalar()
        return total_revenue if total_revenue is not None else 0.0
    except Exception as e:
        logger.error(f"Admin: Error calculating total revenue: {e}", exc_info=True)
        return 0.0 # Return 0 or handle error as appropriate

# --- Admin Site Settings Management (Conceptual) ---
def get_site_settings_admin(): 
    settings_dict = {
       "PROJECT_NAME": settings.PROJECT_NAME,
       "PROJECT_VERSION": settings.PROJECT_VERSION,
       "DATABASE_URL_CONFIGURED": bool(settings.DATABASE_URL), 
       "JWT_SECRET_KEY_SET": settings.JWT_SECRET_KEY != "a_very_secure_default_secret_key_please_change_me", 
       "API_ENCRYPTION_KEY_SET": bool(settings.API_ENCRYPTION_KEY),
       "SMTP_HOST": settings.SMTP_HOST or "Not Set",
       "EMAILS_FROM_EMAIL": settings.EMAILS_FROM_EMAIL or "Not Set",
       "FRONTEND_URL": settings.FRONTEND_URL,
       "ALLOWED_ORIGINS": settings.ALLOWED_ORIGINS,
       "REFERRAL_COMMISSION_RATE": settings.REFERRAL_COMMISSION_RATE,
       "COINBASE_COMMERCE_API_KEY_SET": bool(settings.COINBASE_COMMERCE_API_KEY),
       "ENVIRONMENT": os.getenv("ENVIRONMENT", "Not Set")
    }
    return {"status": "success", "settings": settings_dict} # This can be kept for read-only view of config.py settings

def get_all_system_settings_admin(db_session: Session):
    """Retrieves all system settings from the database."""
    try:
        settings_db = db_session.query(SystemSetting).all()
        settings_list = [
            {"key": s.key, "value": s.value, "description": s.description, "updated_at": s.updated_at.isoformat()}
            for s in settings_db
        ]
        return {"status": "success", "system_settings": settings_list}
    except Exception as e:
        logger.exception(f"Error retrieving all system settings: {e}")
        return {"status": "error", "message": "Could not retrieve system settings."}

def update_system_setting_admin(db_session: Session, setting_key: str, new_value: str, description: Optional[str] = None, performing_admin_id: Optional[int] = None):
    """Updates or creates a system setting in the database."""
    
    # Basic validation for known numeric settings
    known_numeric_settings = ["MAX_BACKTEST_DAYS_SYSTEM", "DEFAULT_BACKTEST_INITIAL_CAPITAL"]
    known_float_settings = ["REFERRAL_COMMISSION_RATE", "DEFAULT_BACKTEST_INITIAL_CAPITAL"] # REFERRAL_COMMISSION_RATE handled by its own function

    if setting_key in known_numeric_settings and setting_key not in known_float_settings:
        try:
            int(new_value)
        except ValueError:
            return {"status": "error", "message": f"Setting '{setting_key}' must be an integer. Received: '{new_value}'."}
    
    if setting_key in known_float_settings:
        try:
            val = float(new_value)
            if setting_key == "REFERRAL_COMMISSION_RATE" and not (0 < val < 1): # Specific validation for this key
                 return {"status": "error", "message": "Referral commission rate must be between 0 and 1 (exclusive)."}
        except ValueError:
            return {"status": "error", "message": f"Setting '{setting_key}' must be a float. Received: '{new_value}'."}

    try:
        setting = db_session.query(SystemSetting).filter(SystemSetting.key == setting_key).first()
        if setting:
            setting.value = new_value
            if description is not None: # Allow updating description
                setting.description = description
            setting.updated_at = datetime.datetime.utcnow()
            action = "updated"
        else:
            setting = SystemSetting(
                key=setting_key,
                value=new_value,
                description=description if description is not None else f"System setting for {setting_key}",
                updated_at=datetime.datetime.utcnow()
            )
            db_session.add(setting)
            action = "created"
        
        db_session.commit()
        logger.info(f"Admin (ID: {performing_admin_id or 'Unknown'}) {action} system setting '{setting_key}' to value '{new_value}'.")
        return {"status": "success", "message": f"System setting '{setting_key}' {action} successfully."}
    except Exception as e:
        db_session.rollback()
        logger.exception(f"Error {action} system setting '{setting_key}': {e}")
        return {"status": "error", "message": f"Database error while {action} system setting."}

def admin_update_referral_commission_rate(db_session: Session, new_rate: float, performing_admin_id: int):
    """
    Updates the global referral commission rate in system settings.
    """
    if not (0 < new_rate < 1): # Rate should be like 0.1 for 10%. Max < 100%
        logger.warning(f"Admin ID {performing_admin_id} attempt to set invalid referral commission rate: {new_rate}.")
        return {"status": "error", "message": "New rate must be between 0 and 1 (e.g., 0.1 for 10%)."}

    setting_key = "referral_commission_rate"
    try:
        setting = db_session.query(SystemSetting).filter(SystemSetting.key == setting_key).first()
        if setting:
            setting.value = str(new_rate)
            setting.updated_at = datetime.datetime.utcnow()
            logger.info(f"Admin ID {performing_admin_id} updated existing referral commission rate to: {new_rate}.")
        else:
            setting = SystemSetting(
                key=setting_key,
                value=str(new_rate),
                description="Global referral commission rate. E.g., 0.1 means 10%.",
                updated_at=datetime.datetime.utcnow()
            )
            db_session.add(setting)
            logger.info(f"Admin ID {performing_admin_id} created new referral commission rate setting: {new_rate}.")
        
        db_session.commit()
        return {"status": "success", "message": f"Referral commission rate successfully updated to {new_rate:.2f}."}
    except Exception as e:
        db_session.rollback()
        logger.error(f"Admin ID {performing_admin_id}: Error updating referral commission rate to {new_rate}: {e}", exc_info=True)
        return {"status": "error", "message": "Database error while updating referral commission rate."}


# --- Admin Subscription Restart ---
def restart_strategy_subscription(db_session: Session, subscription_id: int, admin_user_id: int):
    """
    Admin function to restart a strategy subscription.
    This function now delegates the core logic to live_trading_service.restart_strategy_admin.
    """
    logger.info(f"Admin User ID: {admin_user_id} initiated restart for subscription ID: {subscription_id} via admin_service, delegating to live_trading_service.")

    # Call the centralized restart function in live_trading_service
    # This function is assumed to handle all logic including fetching subscription, error handling, stopping, and starting.
    # It should also handle logging for its own steps.
    # We pass admin_user_id for logging/auditing within restart_strategy_admin.
    restart_result = live_trading_service.restart_strategy_admin(
        db_session=db_session, 
        subscription_id=subscription_id, 
        admin_id=admin_user_id
    )

    if restart_result["status"] == "success":
        logger.info(f"Admin service call to restart subscription ID {subscription_id} completed successfully. Result: {restart_result.get('message')}")
    else:
        logger.error(f"Admin service call to restart subscription ID {subscription_id} failed. Reason: {restart_result.get('message')}")

    return restart_result

def get_dashboard_summary(db_session: Session):
    """
    Aggregates data for the admin dashboard summary.
    """
    try:
        total_users = db_session.query(User).count()
        total_revenue = get_total_revenue(db_session) # Uses existing helper

        now = datetime.datetime.utcnow()
        active_subscriptions = db_session.query(UserStrategySubscription).filter(
            UserStrategySubscription.is_active == True,
            UserStrategySubscription.expires_at > now # Ensure subscription is not expired
        ).count()
        
        # Alternative for active_subscriptions if you also want to include those with no expiry:
        # active_subscriptions = db_session.query(UserStrategySubscription).filter(
        #     UserStrategySubscription.is_active == True,
        #     or_(
        #         UserStrategySubscription.expires_at == None,
        #         UserStrategySubscription.expires_at > now
        #     )
        # ).count()


        total_strategies = db_session.query(Strategy).count()

        # Placeholder for recent activities - this would require a more complex activity logging system
        recent_activities = [
            "User 'john_doe' registered.",
            "Strategy 'EMA Crossover' was subscribed to by user 'jane_doe'.",
            "Payment of $19.99 received from user 'john_doe'."
        ] # Replace with actual activity logging if implemented

        summary_data = {
            "totalUsers": total_users,
            "totalRevenue": total_revenue,
            "activeSubscriptions": active_subscriptions,
            "totalStrategies": total_strategies,
            "recentActivities": recent_activities # Added placeholder
        }
        return {"status": "success", "summary": summary_data}
    except Exception as e:
        logger.error(f"Error generating admin dashboard summary: {e}", exc_info=True)
        return {"status": "error", "message": "Could not generate dashboard summary."}

def delete_strategy_admin(db_session: Session, strategy_id: int):
    """Deletes a strategy from the database."""
    strategy = db_session.query(Strategy).filter(Strategy.id == strategy_id).first()
    if not strategy:
        return {"status": "error", "message": "Strategy not found."}

    # Optional: Check for active subscriptions before deleting
    active_subscriptions_count = db_session.query(UserStrategySubscription).filter(
        UserStrategySubscription.strategy_id == strategy_id,
        UserStrategySubscription.is_active == True
    ).count()

    if active_subscriptions_count > 0:
        logger.warning(f"Attempt to delete strategy ID {strategy_id} which has {active_subscriptions_count} active subscriptions.")
        # Depending on policy, could prevent deletion or just log. For now, allowing deletion.
        # To prevent: return {"status": "error", "message": f"Strategy has {active_subscriptions_count} active subscriptions. Cannot delete."}
    
    try:
        db_session.delete(strategy)
        db_session.commit()
        logger.info(f"Admin: Strategy ID {strategy_id} ('{strategy.name}') deleted successfully.")
        return {"status": "success", "message": "Strategy deleted successfully."}
    except Exception as e: # Consider SQLAlchemyError for more specific DB errors
        db_session.rollback()
        logger.exception(f"Error deleting strategy ID {strategy_id}: {e}")
        return {"status": "error", "message": f"Database error while deleting strategy: {str(e)}"}

def get_subscription_details_admin(db_session: Session, subscription_id: int):
    """Retrieves detailed information for a specific user subscription for admin view."""
    try:
        subscription = db_session.query(UserStrategySubscription)\
            .join(User, UserStrategySubscription.user_id == User.id)\
            .join(Strategy, UserStrategySubscription.strategy_id == Strategy.id)\
            .outerjoin(ApiKey, UserStrategySubscription.api_key_id == ApiKey.id)\
            .filter(UserStrategySubscription.id == subscription_id)\
            .first()

        if not subscription:
            return {"status": "error", "message": "Subscription not found."}

        # Attempt to parse custom_parameters JSON string into a dict
        custom_parameters_dict = None
        if subscription.custom_parameters:
            try:
                custom_parameters_dict = json.loads(subscription.custom_parameters)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse custom_parameters JSON for subscription {subscription.id}: {subscription.custom_parameters}", exc_info=True)
                custom_parameters_dict = {"error": "Could not parse parameters"}
        
        # Constructing the detailed response
        subscription_details = {
            "id": subscription.id,
            "user_id": subscription.user_id,
            "username": subscription.user.username if subscription.user else None,
            "strategy_id": subscription.strategy_id,
            "strategy_name": subscription.strategy.name if subscription.strategy else None,
            "api_key_id": subscription.api_key_id,
            "api_key_label": subscription.api_key.label if subscription.api_key else None,
            "is_active": subscription.is_active,
            "subscribed_at": subscription.subscribed_at.isoformat() if subscription.subscribed_at else None,
            "expires_at": subscription.expires_at.isoformat() if subscription.expires_at else None,
            "custom_parameters": custom_parameters_dict, # Parsed dict or error
            "status_message": subscription.status_message,
            "celery_task_id": subscription.celery_task_id
        }
        return {"status": "success", "subscription": subscription_details}
    except Exception as e:
        logger.exception(f"Error retrieving subscription details for admin (ID: {subscription_id}): {e}")
        return {"status": "error", "message": "Could not retrieve subscription details."}
