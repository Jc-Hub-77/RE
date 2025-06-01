# backend/services/admin_service.py
import datetime
import os
import logging
from sqlalchemy.orm import Session
from sqlalchemy import desc, or_ # For count, desc, or_
import sqlalchemy 
import sqlalchemy.types 
import json
import importlib.util # Added for strategy validation

from backend.models import User, Strategy, UserStrategySubscription, PaymentTransaction, ApiKey 
from backend.config import settings

# Initialize logger
logger = logging.getLogger(__name__)

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
                           default_parameters: str, category: str, risk_level: str):
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

    new_strategy = Strategy(
        name=name, 
        description=description, 
        python_code_path=python_code_path, # Store relative path
        default_parameters=default_parameters,
        category=category,
        risk_level=risk_level,
        is_active=True 
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

    allowed_fields = ["name", "description", "python_code_path", "default_parameters", "category", "risk_level", "is_active"]
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
def list_all_subscriptions_admin(db_session: Session, page: int = 1, per_page: int = 20):
    """Lists all user strategy subscriptions with pagination."""
    try:
        query = db_session.query(UserStrategySubscription).join(User).join(Strategy).outerjoin(ApiKey) # outerjoin for ApiKey

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
    return {"status": "success", "settings": settings_dict}

def update_site_setting_admin(setting_key: str, setting_value: str):
    # This remains conceptual as modifying .env or runtime os.environ is complex and risky via API.
    # Such changes should ideally trigger a configuration reload or app restart,
    # which is beyond the scope of this function.
    sensitive_keys = [
        "DATABASE_URL", "JWT_SECRET_KEY", "SMTP_PASSWORD", 
        "API_ENCRYPTION_KEY", "COINBASE_COMMERCE_API_KEY", "COINBASE_COMMERCE_WEBHOOK_SECRET"
    ]
    if setting_key in sensitive_keys:
         logger.warning(f"Admin: Attempt to update sensitive setting '{setting_key}' via API was blocked.")
         return {"status": "error", "message": f"Setting '{setting_key}' is sensitive and cannot be updated via API for security reasons."}
    
    # Example: For non-sensitive, known settings that might be stored in DB or a mutable config object (not .env)
    # if setting_key == "SOME_NON_SENSITIVE_SETTING":
    #    try:
    #        # Update logic here (e.g., save to a DB table for settings)
    #        logger.info(f"Admin: Site setting '{setting_key}' updated to '{setting_value}'.")
    #        return {"status": "success", "message": f"Setting '{setting_key}' updated."}
    #    except Exception as e:
    #        logger.error(f"Admin: Error updating site setting '{setting_key}': {e}", exc_info=True)
    #        return {"status": "error", "message": f"Could not update setting '{setting_key}': {e}"}

    logger.info(f"Admin: Simulated attempt to update site setting '{setting_key}' to '{setting_value}'. This is not implemented for direct .env modification.")
    return {"status": "info_simulated", "message": f"Updating setting '{setting_key}' is conceptual. Direct modification of environment variables at runtime is not supported through this function."}
