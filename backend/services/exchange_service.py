# backend/services/exchange_service.py
import os
from typing import Optional, Dict, Any # Added Dict, Any
import ccxt
import json
import datetime
import logging 
import time # Added for fetch_historical_data rate limit handling
import pandas as pd # Added for fetch_historical_data

from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy.orm import Session
from backend.models import ApiKey, User, UserStrategySubscription # Added UserStrategySubscription
from backend.config import settings 

# Initialize logger
logger = logging.getLogger(__name__)

cipher_suite = None
if settings.API_ENCRYPTION_KEY: 
    try:
        cipher_suite = Fernet(settings.API_ENCRYPTION_KEY.encode() if isinstance(settings.API_ENCRYPTION_KEY, str) else settings.API_ENCRYPTION_KEY)
        logger.info("API Key encryption cipher initialized (using configured API_ENCRYPTION_KEY).")
    except Exception as e: 
        logger.critical(f"Could not initialize Fernet cipher with configured API_ENCRYPTION_KEY: {e}. API key encryption/decryption will fail.", exc_info=True)
        cipher_suite = None 
else:
    cipher_suite = None
    logger.critical("No API_ENCRYPTION_KEY found in settings. API key encryption/decryption will fail.")

def _encrypt_data(data: str) -> str:
    if not cipher_suite: raise ValueError("Encryption cipher not initialized. Cannot encrypt.")
    return cipher_suite.encrypt(data.encode()).decode()

def _decrypt_data(encrypted_data: str) -> str:
    if not cipher_suite: raise ValueError("Encryption cipher not initialized. Cannot decrypt.")
    try:
        return cipher_suite.decrypt(encrypted_data.encode()).decode()
    except InvalidToken:
        logger.error("Decryption failed. Invalid token or key mismatch for data: %s", encrypted_data[:20] + "...") 
        raise ValueError("Decryption failed. Ensure encryption key is correct and data is not corrupted.")

SUPPORTED_EXCHANGES = [exc.lower() for exc in ccxt.exchanges]

def add_exchange_api_key(db_session: Session, user_id: int, exchange_name: str, 
                         api_key_public: str, secret_key: str, passphrase: Optional[str] = None, label: Optional[str] = None):
    exchange_name_lower = exchange_name.lower()
    if exchange_name_lower not in SUPPORTED_EXCHANGES:
        return {"status": "error", "message": f"Exchange '{exchange_name}' is not supported."}

    user = db_session.query(User).filter(User.id == user_id).first()
    if not user: return {"status": "error", "message": "User not found."}

    if label: 
        existing_key_with_label = db_session.query(ApiKey).filter(
            ApiKey.user_id == user_id, ApiKey.exchange_name == exchange_name_lower, ApiKey.label == label
        ).first()
        if existing_key_with_label:
            return {"status": "error", "message": f"An API key with the label '{label}' already exists for {exchange_name}."}
    
    if not cipher_suite:
        return {"status": "error", "message": "System error: Encryption service not available."}

    try:
        encrypted_api_key_val = _encrypt_data(api_key_public)
        encrypted_secret_val = _encrypt_data(secret_key)
        encrypted_passphrase_val = _encrypt_data(passphrase) if passphrase else None
    except ValueError as e:
        logger.error(f"Encryption error during add_exchange_api_key: {e}", exc_info=True)
        return {"status": "error", "message": f"Encryption error: {e}"}

    api_key_preview = api_key_public[:4] + "..." + api_key_public[-4:] if len(api_key_public) > 8 else api_key_public
    effective_label = label if label else f"{exchange_name.capitalize()} Key - {datetime.datetime.utcnow().strftime('%Y%m%d%H%M')}"

    new_api_key_entry = ApiKey(
        user_id=user_id,
        exchange_name=exchange_name_lower,
        label=effective_label,
        api_key_public_preview=api_key_preview,
        encrypted_api_key=encrypted_api_key_val,
        encrypted_secret_key=encrypted_secret_val,
        encrypted_passphrase=encrypted_passphrase_val,
        status="pending_verification",
        created_at=datetime.datetime.utcnow()
    )
    
    try:
        db_session.add(new_api_key_entry)
        db_session.commit()
        db_session.refresh(new_api_key_entry)
        return {
            "status": "success", 
            "message": f"API key for {exchange_name} added with label '{effective_label}'. Please test connectivity.",
            "api_key_id": new_api_key_entry.id
        }
    except Exception as e:
        db_session.rollback()
        logger.error(f"Error adding API key to DB: {e}", exc_info=True)
        return {"status": "error", "message": "Database error while adding API key."}

def get_user_exchange_api_keys_display(db_session: Session, user_id: int):
    api_keys_query = db_session.query(ApiKey).filter(ApiKey.user_id == user_id).order_by(ApiKey.label).all()
    
    keys_display = []
    for key_entry in api_keys_query:
        keys_display.append({
            "id": key_entry.id,
            "exchange_name": key_entry.exchange_name.capitalize(),
            "label": key_entry.label,
            "api_key_preview": key_entry.api_key_public_preview,
            "status": key_entry.status,
            "status_message": key_entry.status_message,
            "last_tested_at": key_entry.last_tested_at.isoformat() if key_entry.last_tested_at else None,
            "created_at": key_entry.created_at.isoformat() if key_entry.created_at else None
        })
    return {"status": "success", "keys": keys_display}

def remove_exchange_api_key(db_session: Session, user_id: int, api_key_id: int):
    key_to_delete = db_session.query(ApiKey).filter(ApiKey.id == api_key_id, ApiKey.user_id == user_id).first()
    if not key_to_delete:
        return {"status": "error", "message": "API key not found or access denied."}
    
    active_subs_count = db_session.query(UserStrategySubscription).filter(
        UserStrategySubscription.api_key_id == api_key_id,
        UserStrategySubscription.is_active == True
    ).count()
    if active_subs_count > 0:
        return {"status": "error", "message": f"Cannot delete API key. It is used by {active_subs_count} active strategy subscription(s)."}

    try:
        db_session.delete(key_to_delete)
        db_session.commit()
        logger.info(f"API Key ID {api_key_id} for user {user_id} removed successfully.")
        return {"status": "success", "message": "API key removed successfully."}
    except Exception as e:
        db_session.rollback()
        logger.error(f"Error removing API key {api_key_id} for user {user_id}: {e}", exc_info=True)
        return {"status": "error", "message": "Database error while removing API key."}

def test_api_connectivity(db_session: Session, user_id: int, api_key_id: int):
    key_entry = db_session.query(ApiKey).filter(ApiKey.id == api_key_id, ApiKey.user_id == user_id).first()
    if not key_entry:
        return {"status": "error", "message": "API key not found or access denied."}

    if not cipher_suite:
        key_entry.status = "error_system"
        key_entry.status_message = "System error: Encryption service not available for testing."
        db_session.commit()
        return {"status": "error_system", "message": key_entry.status_message}

    key_entry.status = "testing"
    key_entry.status_message = "Attempting to connect to exchange..."
    key_entry.last_tested_at = datetime.datetime.utcnow()
    db_session.commit() 

    try:
        decrypted_api_key = _decrypt_data(key_entry.encrypted_api_key)
        decrypted_secret_key = _decrypt_data(key_entry.encrypted_secret_key)
        decrypted_passphrase = _decrypt_data(key_entry.encrypted_passphrase) if key_entry.encrypted_passphrase else None
    except ValueError as e:
        key_entry.status = "error_decryption"
        key_entry.status_message = f"API key decryption failed: {e}. Please re-add the key."
        db_session.commit()
        return {"status": key_entry.status, "message": key_entry.status_message}
    
    exchange_name_lower = key_entry.exchange_name.lower()
    if not hasattr(ccxt, exchange_name_lower):
        key_entry.status = "error_exchange_unsupported"
        key_entry.status_message = f"Exchange '{exchange_name_lower}' is not supported by CCXT."
        db_session.commit()
        return {"status": key_entry.status, "message": key_entry.status_message}

    exchange_class = getattr(ccxt, exchange_name_lower)
    config = {'apiKey': decrypted_api_key, 'secret': decrypted_secret_key, 'options': {'adjustForTimeDifference': True}, 'enableRateLimit': True}
    if decrypted_passphrase: config['password'] = decrypted_passphrase

    exchange = exchange_class(config)
    
    try:
        markets = exchange.fetch_markets() 
        if markets:
            key_entry.status = "active"
            key_entry.status_message = "API connection successful. Market data fetched."
        else:
            key_entry.status = "error_test_failed"
            key_entry.status_message = "API connection test failed: No market data returned."
            
    except ccxt.AuthenticationError as e:
        key_entry.status = "error_authentication"
        key_entry.status_message = f"Authentication failed: {str(e)}. Check API key permissions and validity."
    except ccxt.NetworkError as e:
        key_entry.status = "error_network" 
        key_entry.status_message = f"Network error: {str(e)}. Check connectivity or try again later."
    except ccxt.ExchangeError as e: 
        key_entry.status = "error_exchange"
        key_entry.status_message = f"Exchange error: {str(e)}."
    except Exception as e: 
        key_entry.status = "error_unknown"
        key_entry.status_message = f"An unexpected error occurred: {str(e)}."
    
    try:
        db_session.commit()
    except Exception as e_db:
        db_session.rollback()
        logger.error(f"DB error updating API key {api_key_id} status after test: {e_db}", exc_info=True)
        return {"status": key_entry.status, "message": key_entry.status_message + f" (DB status update failed: {e_db})"}
        
    return {"status": key_entry.status, "message": key_entry.status_message, "api_key_id": key_entry.id}

def get_exchange_client(db_session: Session, api_key_id: int, user_id: int) -> Optional[ccxt.Exchange]:
    key_entry = db_session.query(ApiKey).filter(ApiKey.id == api_key_id, ApiKey.user_id == user_id).first()
    if not key_entry or key_entry.status != "active":
        logger.warning(f"Cannot get exchange client: API key ID {api_key_id} not found, not active, or doesn't belong to user {user_id}.")
        return None
    
    if not cipher_suite:
        logger.error("Cannot get exchange client: Encryption service not available.")
        return None

    try:
        decrypted_api_key = _decrypt_data(key_entry.encrypted_api_key)
        decrypted_secret_key = _decrypt_data(key_entry.encrypted_secret_key)
        decrypted_passphrase = _decrypt_data(key_entry.encrypted_passphrase) if key_entry.encrypted_passphrase else None
    except ValueError:
        logger.error(f"Decryption failed for API key ID {api_key_id}. Cannot initialize client.", exc_info=True)
        return None

    exchange_name_lower = key_entry.exchange_name.lower()
    if not hasattr(ccxt, exchange_name_lower):
        logger.error(f"Exchange '{exchange_name_lower}' not supported by CCXT.")
        return None

    exchange_class = getattr(ccxt, exchange_name_lower)
    config = {'apiKey': decrypted_api_key, 'secret': decrypted_secret_key, 'options': {'adjustForTimeDifference': True}, 'enableRateLimit': True}
    if decrypted_passphrase: config['password'] = decrypted_passphrase
    
    try:
        client = exchange_class(config)
        return client
    except Exception as e:
        logger.error(f"Failed to initialize CCXT client for {exchange_name_lower} with API key ID {api_key_id}: {e}", exc_info=True)
        return None

# --- Core Exchange Interaction Methods ---

def fetch_account_balance(exchange: ccxt.Exchange) -> Dict[str, Any]:
    """Fetches account balance from the exchange."""
    try:
        balance = exchange.fetch_balance()
        logger.info(f"Successfully fetched balance from {exchange.id}.")
        return {"status": "success", "balance": balance}
    except ccxt.AuthenticationError as e:
        logger.error(f"Authentication failed on {exchange.id} while fetching balance: {e}", exc_info=True)
        return {"status": "error", "message": f"Authentication failed: {e}"}
    except ccxt.NetworkError as e:
        logger.error(f"Network error on {exchange.id} while fetching balance: {e}", exc_info=True)
        return {"status": "error", "message": f"Network error: {e}"}
    except ccxt.ExchangeError as e:
        logger.error(f"Exchange error on {exchange.id} while fetching balance: {e}", exc_info=True)
        return {"status": "error", "message": f"Exchange error: {e}"}
    except Exception as e:
        logger.error(f"Unexpected error on {exchange.id} while fetching balance: {e}", exc_info=True)
        return {"status": "error", "message": f"An unexpected error occurred: {e}"}

def create_exchange_order(exchange: ccxt.Exchange, symbol: str, order_type: str, side: str, 
                          amount: float, price: Optional[float] = None, params: Optional[dict] = None) -> Dict[str, Any]:
    """Creates an order on the exchange."""
    try:
        if not exchange.has['createOrder']:
            logger.error(f"Exchange {exchange.id} does not support createOrder.")
            return {"status": "error", "message": f"Exchange {exchange.id} does not support creating orders via API."}

        order_params = params if params else {}
        order = exchange.create_order(symbol, order_type, side, amount, price, order_params)
        logger.info(f"Successfully created {side} {order_type} order for {amount} {symbol} on {exchange.id}. Order ID: {order.get('id')}")
        return {"status": "success", "order": order}
    except ccxt.AuthenticationError as e:
        logger.error(f"Authentication failed on {exchange.id} creating order: {e}", exc_info=True)
        return {"status": "error", "message": f"Authentication failed: {e}"}
    except ccxt.InsufficientFunds as e:
        logger.error(f"Insufficient funds on {exchange.id} for order ({symbol}, {amount}): {e}", exc_info=True)
        return {"status": "error", "message": f"Insufficient funds: {e}"}
    except ccxt.InvalidOrder as e: # E.g. price/amount precision, market closed
        logger.error(f"Invalid order on {exchange.id} ({symbol}, {amount}, {price}): {e}", exc_info=True)
        return {"status": "error", "message": f"Invalid order: {e}"}
    except ccxt.NetworkError as e:
        logger.error(f"Network error on {exchange.id} creating order: {e}", exc_info=True)
        return {"status": "error", "message": f"Network error: {e}"}
    except ccxt.ExchangeError as e: # More general exchange errors
        logger.error(f"Exchange error on {exchange.id} creating order: {e}", exc_info=True)
        return {"status": "error", "message": f"Exchange error: {e}"}
    except Exception as e:
        logger.error(f"Unexpected error on {exchange.id} creating order: {e}", exc_info=True)
        return {"status": "error", "message": f"An unexpected error occurred: {e}"}

def fetch_exchange_order_status(exchange: ccxt.Exchange, order_id: str, symbol: Optional[str] = None) -> Dict[str, Any]:
    """Fetches the status of a specific order from the exchange."""
    try:
        if not exchange.has['fetchOrder']:
            logger.error(f"Exchange {exchange.id} does not support fetchOrder.")
            return {"status": "error", "message": f"Exchange {exchange.id} does not support fetching order status."}
        
        order = exchange.fetch_order(order_id, symbol)
        logger.info(f"Successfully fetched status for order ID {order_id} on {exchange.id}.")
        return {"status": "success", "order": order}
    except ccxt.OrderNotFound as e:
        logger.warning(f"Order ID {order_id} not found on {exchange.id}: {e}", exc_info=True)
        return {"status": "error", "message": f"Order not found: {e}"}
    except ccxt.AuthenticationError as e:
        logger.error(f"Authentication failed on {exchange.id} fetching order {order_id}: {e}", exc_info=True)
        return {"status": "error", "message": f"Authentication failed: {e}"}
    except ccxt.NetworkError as e:
        logger.error(f"Network error on {exchange.id} fetching order {order_id}: {e}", exc_info=True)
        return {"status": "error", "message": f"Network error: {e}"}
    except ccxt.ExchangeError as e:
        logger.error(f"Exchange error on {exchange.id} fetching order {order_id}: {e}", exc_info=True)
        return {"status": "error", "message": f"Exchange error: {e}"}
    except Exception as e:
        logger.error(f"Unexpected error on {exchange.id} fetching order {order_id}: {e}", exc_info=True)
        return {"status": "error", "message": f"An unexpected error occurred: {e}"}

def cancel_exchange_order(exchange: ccxt.Exchange, order_id: str, symbol: Optional[str] = None) -> Dict[str, Any]:
    """Cancels an order on the exchange."""
    try:
        if not exchange.has['cancelOrder']:
            logger.error(f"Exchange {exchange.id} does not support cancelOrder.")
            return {"status": "error", "message": f"Exchange {exchange.id} does not support canceling orders."}

        response = exchange.cancel_order(order_id, symbol)
        logger.info(f"Successfully sent cancel request for order ID {order_id} on {exchange.id}.")
        # CCXT cancel_order response varies; some return order structure, some just status/id.
        # We return the raw response from ccxt.
        return {"status": "success", "response": response}
    except ccxt.OrderNotFound as e: # If order already filled or doesn't exist
        logger.warning(f"Order ID {order_id} not found for cancellation on {exchange.id} (possibly already filled/canceled): {e}", exc_info=True)
        return {"status": "error", "message": f"Order not found for cancellation (possibly already filled/canceled): {e}"}
    except ccxt.InvalidOrder as e: # If order cannot be cancelled (e.g. already cancelled/filled)
         logger.warning(f"Cannot cancel order ID {order_id} on {exchange.id} (e.g. already filled/canceled): {e}", exc_info=True)
         return {"status": "error", "message": f"Cannot cancel order (already filled/canceled): {e}"}
    except ccxt.AuthenticationError as e:
        logger.error(f"Authentication failed on {exchange.id} canceling order {order_id}: {e}", exc_info=True)
        return {"status": "error", "message": f"Authentication failed: {e}"}
    except ccxt.NetworkError as e:
        logger.error(f"Network error on {exchange.id} canceling order {order_id}: {e}", exc_info=True)
        return {"status": "error", "message": f"Network error: {e}"}
    except ccxt.ExchangeError as e:
        logger.error(f"Exchange error on {exchange.id} canceling order {order_id}: {e}", exc_info=True)
        return {"status": "error", "message": f"Exchange error: {e}"}
    except Exception as e:
        logger.error(f"Unexpected error on {exchange.id} canceling order {order_id}: {e}", exc_info=True)
        return {"status": "error", "message": f"An unexpected error occurred: {e}"}

# --- End of Core Exchange Interaction Methods ---

def fetch_historical_data(exchange_id: str, symbol: str, timeframe: str, start_date: datetime.datetime, end_date: datetime.datetime):
    """
    Fetches historical OHLCV data for a given symbol and timeframe from an exchange.
    Returns data as a pandas DataFrame.
    """
    exchange_id_lower = exchange_id.lower()
    if exchange_id_lower not in SUPPORTED_EXCHANGES:
        logger.error(f"Exchange '{exchange_id}' is not supported for historical data fetching.")
        return pd.DataFrame() # Return empty DataFrame

    try:
        exchange_class = getattr(ccxt, exchange_id_lower)
        exchange = exchange_class({'enableRateLimit': True})
        logger.info(f"Initialized CCXT exchange '{exchange_id}' for historical data.")
    except Exception as e:
        logger.error(f"Failed to initialize CCXT exchange '{exchange_id}' for historical data: {e}", exc_info=True)
        return pd.DataFrame()

    all_ohlcv = []
    since_ms = int(start_date.timestamp() * 1000)
    end_ms = int(end_date.timestamp() * 1000)
    limit = 1000 

    logger.info(f"Fetching historical data for {symbol}@{timeframe} on {exchange_id} from {start_date} to {end_date}.")

    while since_ms < end_ms:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since_ms, limit)
            if not ohlcv:
                logger.info(f"No more historical data found for {symbol}@{timeframe} from {exchange_id} starting at {datetime.datetime.fromtimestamp(since_ms / 1000)}.")
                break 

            all_ohlcv.extend(ohlcv)
            last_timestamp = ohlcv[-1][0]
            
            # Ensure we are making progress to avoid infinite loops on problematic data
            if last_timestamp < since_ms : # Should not happen if data is correctly ordered
                 logger.warning(f"Fetched data is older than 'since' timestamp. Breaking to avoid loop. Last: {last_timestamp}, Since: {since_ms}")
                 break
            since_ms = last_timestamp + exchange.parse_timeframe(timeframe) * 1000
            
            # Optional delay: time.sleep(exchange.rateLimit / 1000)
        except ccxt.RateLimitExceeded as e:
            logger.warning(f"Rate limit exceeded while fetching historical data from {exchange_id}. Retrying after delay: {e.args[0] if e.args else e}", exc_info=False) # Log only message for RL
            time.sleep(max(exchange.rateLimit / 1000, 1)) # Ensure at least 1s sleep
            continue 
        except ccxt.NetworkError as e: # More specific network error
            logger.error(f"Network error fetching historical data for {symbol}@{timeframe} on {exchange_id}: {e}", exc_info=True)
            time.sleep(5) # Wait longer for network issues
            continue # Optionally retry for network issues
        except ccxt.BaseError as e: # Other CCXT errors
            logger.error(f"CCXT error fetching historical data for {symbol}@{timeframe} on {exchange_id}: {e}", exc_info=True)
            break
        except Exception as e:
            logger.error(f"Unexpected error fetching historical data for {symbol}@{timeframe} on {exchange_id}: {e}", exc_info=True)
            break

    if not all_ohlcv:
        logger.warning(f"No historical data fetched for {symbol}@{timeframe} on {exchange_id} in the specified range.")
        return pd.DataFrame()

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[df.index <= end_date] # Ensure data does not exceed end_date
    # Remove duplicates that might occur if exchange returns overlapping ranges
    df = df[~df.index.duplicated(keep='first')] 
    
    logger.info(f"Successfully fetched {len(df)} historical data points for {symbol}@{timeframe} on {exchange_id}.")
    return df
