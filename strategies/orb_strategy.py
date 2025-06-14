import logging
import numpy as np
import ccxt 
from decimal import Decimal, ROUND_DOWN, ROUND_UP, getcontext
from datetime import datetime, time, date, timedelta # Ensure time, date, timedelta are imported
import pytz
import pandas 
import json # For custom_data
from typing import Optional, Dict, Any, List # For type hinting

from sqlalchemy.orm import Session
from backend.models import Position, Order, UserStrategySubscription
from backend import strategy_utils

# Ensure high precision for Decimal calculations
getcontext().prec = 18

class OpeningRangeBreakoutStrategy:

    @staticmethod
    def get_parameters_definition():
        return {
            "trading_pair": {"type": "str", "label": "Trading Pair (e.g., BTC/USDT)", "default": "BTC/USDT"},
            "kline_interval": {"type": "str", "label": "Kline Interval for Breakout Logic", "default": "5m", "choices": ["1m", "3m", "5m", "15m", "1h"]},
            "orb_candle_hour_local": {"type": "int", "label": "ORB Candle Hour (Local Timezone)", "default": 9, "min": 0, "max": 23},
            "orb_candle_minute_local": {"type": "int", "label": "ORB Candle Minute (Local Timezone)", "default": 15, "min": 0, "max": 59, "step": 1},
            "orb_candle_timezone": {"type": "str", "label": "Timezone for ORB Candle Time", "default": "America/New_York"},
            "orb_kline_interval": {"type": "str", "label": "Kline Interval for ORB Candle", "default": "15m"},
            "stop_loss_pct_from_orb": {"type": "float", "label": "Stop Loss % from ORB Level", "default": 0.5},
            "take_profit_pct_from_orb": {"type": "float", "label": "Take Profit % from ORB Level", "default": 1.5},
            "order_quantity_usd": {"type": "float", "label": "Order Quantity (USD Notional)", "default": 100.0},
            "leverage": {"type": "int", "label": "Leverage", "default": 10},
            # New parameters
            "session_eod_close_hour_local": {"type": "int", "default": 23, "min": 0, "max": 23, "label": "EOD Close Hour (Local)"},
            "session_eod_close_minute_local": {"type": "int", "default": 59, "min": 0, "max": 59, "label": "EOD Close Minute (Local)"},
            "order_fill_timeout_seconds": {"type": "int", "default": 60, "min": 10, "max": 300, "label": "Order Fill Timeout (s)"},
            "order_fill_check_interval_seconds": {"type": "int", "default": 3, "min": 1, "max": 30, "label": "Order Fill Check Interval (s)"},
        }

    def __init__(self, db_session: Session, user_sub_obj: UserStrategySubscription, strategy_params: dict, exchange_ccxt, logger_obj=None): # Renamed logger
        self.db_session = db_session
        self.user_sub_obj = user_sub_obj
        self.params = strategy_params
        self.exchange_ccxt = exchange_ccxt
        self.logger = logger_obj if logger_obj else logging.getLogger(__name__)
        self.name = "OpeningRangeBreakoutStrategy"

        # Load parameters
        self.trading_pair = self.params.get("trading_pair", "BTC/USDT")
        self.kline_interval = self.params.get("kline_interval", "5m")
        self.orb_kline_interval = self.params.get("orb_kline_interval", "15m")
        self.orb_hour_local = int(self.params.get("orb_candle_hour_local", 9))
        self.orb_minute_local = int(self.params.get("orb_candle_minute_local", 15))
        self.orb_candle_timezone_str = self.params.get("orb_candle_timezone", "America/New_York")
        try:
            self.orb_candle_tz = pytz.timezone(self.orb_candle_timezone_str)
        except pytz.exceptions.UnknownTimeZoneError:
            self.logger.error(f"Unknown ORB candle timezone: {self.orb_candle_timezone_str}. Defaulting to UTC.")
            self.orb_candle_tz = pytz.utc
        self.stop_loss_pct = Decimal(str(self.params.get("stop_loss_pct_from_orb", "0.5"))) / Decimal("100")
        self.take_profit_pct = Decimal(str(self.params.get("take_profit_pct_from_orb", "1.5"))) / Decimal("100")
        self.order_quantity_usd = Decimal(str(self.params.get("order_quantity_usd", "100.0")))
        self.leverage = int(self.params.get("leverage", 10))

        # Initialize new parameters
        self.session_eod_close_hour_local = int(self.params.get("session_eod_close_hour_local", 23))
        self.session_eod_close_minute_local = int(self.params.get("session_eod_close_minute_local", 59))
        self.order_fill_timeout_seconds = int(self.params.get("order_fill_timeout_seconds", 60))
        self.order_fill_check_interval_seconds = int(self.params.get("order_fill_check_interval_seconds", 3))

        # ORB state
        self.orb_high: Optional[Decimal] = None
        self.orb_low: Optional[Decimal] = None
        self.orb_levels_set_for_utc_date: Optional[date] = None
        
        # Position and Order State
        self.active_position_db_id: Optional[int] = None
        self.active_position_side: Optional[str] = None
        self.current_pos_entry_price: Optional[Decimal] = None
        self.current_pos_qty: Decimal = Decimal("0")
        self.active_sl_tp_exchange_ids: Dict[str, Optional[str]] = {} # {'sl_id': sl_exchange_id, 'tp_id': tp_exchange_id}
        self.sl_order_db_id: Optional[int] = None
        self.tp_order_db_id: Optional[int] = None

        self.price_precision_str: Optional[str] = None
        self.quantity_precision_str: Optional[str] = None
        self._precisions_fetched_ = False

        self._fetch_market_precision()
        self._load_persistent_position_state()
        self._set_leverage()
        self.logger.info(f"{self.name} initialized for {self.trading_pair}, UserSubID {self.user_sub_obj.id}. PosID: {self.active_position_db_id}")

    @classmethod
    def validate_parameters(cls, params: dict) -> dict:
        """Validates strategy-specific parameters."""
        definition = cls.get_parameters_definition()
        validated_params = {}
        _logger = logging.getLogger(__name__) # Use a logger instance

        for key, def_value in definition.items():
            val_type_str = def_value.get("type")
            choices = def_value.get("choices") 
            min_val = def_value.get("min")
            max_val = def_value.get("max")
            default_val = def_value.get("default")

            user_val = params.get(key)

            if user_val is None: # Parameter not provided by user
                if default_val is not None:
                    user_val = default_val # Apply default
                else:
                    raise ValueError(f"Required parameter '{key}' is missing and has no default.")
            
            # Type checking and coercion
            if val_type_str == "int":
                try:
                    user_val = int(user_val)
                except (ValueError, TypeError):
                    raise ValueError(f"Parameter '{key}' must be an integer. Got: {user_val}")
            elif val_type_str == "float":
                try:
                    user_val = float(user_val)
                except (ValueError, TypeError):
                    raise ValueError(f"Parameter '{key}' must be a float. Got: {user_val}")
            elif val_type_str == "str": 
                if not isinstance(user_val, str):
                    raise ValueError(f"Parameter '{key}' must be a string. Got: {user_val}")
                # Specific validation for timezone string if desired
                if key == "orb_candle_timezone":
                    try:
                        pytz.timezone(user_val)
                    except pytz.exceptions.UnknownTimeZoneError:
                        raise ValueError(f"Invalid timezone string for '{key}': {user_val}")
            elif val_type_str == "bool": # Not used in this strategy's definition
                if not isinstance(user_val, bool):
                    if str(user_val).lower() in ['true', 'yes', '1']: user_val = True
                    elif str(user_val).lower() in ['false', 'no', '0']: user_val = False
                    else: raise ValueError(f"Parameter '{key}' must be a boolean. Got: {user_val}")
            
            # Choice validation
            if choices and user_val not in choices:
                raise ValueError(f"Parameter '{key}' value '{user_val}' is not in valid choices: {choices}")
            
            # Min/Max validation
            if val_type_str in ["int", "float"]:
                if min_val is not None and user_val < min_val:
                    raise ValueError(f"Parameter '{key}' value {user_val} is less than min {min_val}.")
                if max_val is not None and user_val > max_val:
                    raise ValueError(f"Parameter '{key}' value {user_val} is greater than max {max_val}.")
            
            validated_params[key] = user_val

        # Check for unknown parameters
        for key_param in params:
            if key_param not in definition:
                _logger.warning(f"Unknown parameter '{key_param}' provided for {cls.__name__}. It will be ignored.")
        
        return validated_params

    def _reset_sl_tp_state(self):
        self.active_sl_tp_exchange_ids = {}
        self.sl_order_db_id = None
        self.tp_order_db_id = None

    def _reset_internal_trade_state(self):
        self.active_position_db_id = None
        self.active_position_side = None
        self.current_pos_entry_price = None
        self.current_pos_qty = Decimal("0")
        self._reset_sl_tp_state()
        self.logger.info(f"[{self.name}-{self.trading_pair}] Internal trade state reset.")

    def _load_persistent_position_state(self):
        if not self.db_session or not self.user_sub_obj:
            self.logger.warning(f"[{self.name}-{self.trading_pair}] DB session/user_sub_obj not available for loading state.")
            self._reset_internal_trade_state()
            return

        open_pos = strategy_utils.get_open_strategy_position(self.db_session, self.user_sub_obj.id, self.trading_pair)
        if open_pos:
            self.logger.info(f"[{self.name}-{self.trading_pair}] Loading state for open position ID: {open_pos.id}")
            self.active_position_db_id = open_pos.id
            self.active_position_side = open_pos.side
            self.current_pos_entry_price = Decimal(str(open_pos.entry_price)) if open_pos.entry_price is not None else None
            self.current_pos_qty = Decimal(str(open_pos.amount)) if open_pos.amount is not None else Decimal("0")
            
            if open_pos.custom_data:
                try:
                    state_data = json.loads(open_pos.custom_data)
                    self.active_sl_tp_exchange_ids = state_data.get('active_sl_tp_exchange_ids', {})
                    self.sl_order_db_id = state_data.get('sl_order_db_id')
                    self.tp_order_db_id = state_data.get('tp_order_db_id')
                except json.JSONDecodeError:
                    self.logger.error(f"[{self.name}-{self.trading_pair}] Error decoding custom_data for pos {open_pos.id}. Querying open orders as fallback.")
                    self._query_and_set_open_sl_tp_orders_from_db() # Fallback
            else:
                self._query_and_set_open_sl_tp_orders_from_db() # Fallback if no custom_data
            self.logger.info(f"[{self.name}-{self.trading_pair}] Loaded state: PosID {self.active_position_db_id}, Side {self.active_position_side}, SL ExchID {self.active_sl_tp_exchange_ids.get('sl_id')}, TP ExchID {self.active_sl_tp_exchange_ids.get('tp_id')}")
        else:
            self.logger.info(f"[{self.name}-{self.trading_pair}] No active persistent position found.")
            self._reset_internal_trade_state()
            
    def _query_and_set_open_sl_tp_orders_from_db(self):
        """Helper to query open SL/TP orders from DB if not in custom_data."""
        self._reset_sl_tp_state() # Start fresh
        sl_orders = strategy_utils.get_open_orders_for_subscription(self.db_session, self.user_sub_obj.id, self.trading_pair, order_type='STOP_MARKET') # Standardize type
        if sl_orders: self.active_sl_tp_exchange_ids['sl_id'] = sl_orders[0].order_id; self.sl_order_db_id = sl_orders[0].id
        
        tp_orders = strategy_utils.get_open_orders_for_subscription(self.db_session, self.user_sub_obj.id, self.trading_pair, order_type='TAKE_PROFIT_MARKET') # Standardize type
        if tp_orders: self.active_sl_tp_exchange_ids['tp_id'] = tp_orders[0].order_id; self.tp_order_db_id = tp_orders[0].id
        self.logger.info(f"[{self.name}-{self.trading_pair}] Queried open SL/TP orders. SL ExchID: {self.active_sl_tp_exchange_ids.get('sl_id')}, TP ExchID: {self.active_sl_tp_exchange_ids.get('tp_id')}")


    def _save_position_custom_state(self):
        if not self.active_position_db_id or not self.db_session:
            self.logger.debug(f"[{self.name}-{self.trading_pair}] No active position DB ID or DB session to save custom state.")
            return
        pos_to_update = self.db_session.query(Position).filter(Position.id == self.active_position_db_id).first()
        if pos_to_update:
            state_data = {
                'active_sl_tp_exchange_ids': self.active_sl_tp_exchange_ids,
                'sl_order_db_id': self.sl_order_db_id,
                'tp_order_db_id': self.tp_order_db_id
            }
            pos_to_update.custom_data = json.dumps(state_data)
            pos_to_update.updated_at = datetime.utcnow()
            try:
                self.db_session.commit()
                self.logger.info(f"[{self.name}-{self.trading_pair}] Saved custom state (SL/TP IDs) for PosID {self.active_position_db_id}")
            except Exception as e:
                self.logger.error(f"[{self.name}-{self.trading_pair}] Error saving custom state for PosID {self.active_position_db_id}: {e}", exc_info=True)
                self.db_session.rollback()
        else:
             self.logger.warning(f"[{self.name}-{self.trading_pair}] Position {self.active_position_db_id} not found to save custom state.")


    def _fetch_market_precision(self): # Same as original
        if not self._precisions_fetched_:
            try:
                self.exchange_ccxt.load_markets(True)
                market = self.exchange_ccxt.markets[self.trading_pair]
                self.quantity_precision_str = str(market['precision']['amount'])
                self.price_precision_str = str(market['precision']['price'])
                self._precisions_fetched_ = True
                self.logger.info(f"Precision for {self.trading_pair}: Qty Prec Str={self.quantity_precision_str}, Price Prec Str={self.price_precision_str}")
            except Exception as e:
                self.logger.error(f"Error fetching precision for {self.trading_pair}: {e}. Using defaults.", exc_info=True)
                self.quantity_precision_str = "0.00001"; self.price_precision_str = "0.01"    

    def _get_decimal_places(self, precision_str: Optional[str]) -> int: # Same as original
        if precision_str is None: self.logger.warning("Precision string is None, using default 8."); return 8
        try:
            if 'e-' in precision_str.lower():
                num_val = float(precision_str); precision_str = format(num_val, f'.{abs(int(precision_str.split("e-")[1]))}f')
            d_prec = Decimal(precision_str)
            if d_prec.as_tuple().exponent < 0: return abs(d_prec.as_tuple().exponent)
            return 0 
        except Exception as e: self.logger.warning(f"Could not parse precision string '{precision_str}'. Error: {e}. Using default 8."); return 8

    def _format_quantity(self, quantity: Decimal) -> str: # Same as original
        places = self._get_decimal_places(self.quantity_precision_str)
        return str(quantity.quantize(Decimal('1e-' + str(places)), rounding=ROUND_DOWN))

    def _format_price(self, price: Decimal) -> str: # Same as original
        places = self._get_decimal_places(self.price_precision_str)
        return str(price.quantize(Decimal('1e-' + str(places)), rounding=ROUND_NEAREST))

    def _set_leverage(self): # Same as original
        try:
            if hasattr(self.exchange_ccxt, 'set_leverage'):
                symbol_for_leverage = self.trading_pair.split(':')[0] if ':' in self.trading_pair else self.trading_pair
                self.exchange_ccxt.set_leverage(self.leverage, symbol_for_leverage)
                self.logger.info(f"Leverage set to {self.leverage}x for {symbol_for_leverage}")
        except Exception as e: self.logger.warning(f"Could not set leverage for {self.trading_pair}: {e}")

    def _reset_daily_orb_state(self): # Same as original
        self.orb_high = None; self.orb_low = None; self.orb_levels_set_for_utc_date = None
        self.logger.info("Daily ORB state reset.")

    def _determine_and_set_orb_levels(self, current_utc_datetime: datetime): # Removed unused params
        # Same as original, ensure datetime usage is correct
        target_local_time = time(self.orb_hour_local, self.orb_minute_local)
        current_candle_tz_datetime = current_utc_datetime.astimezone(self.orb_candle_tz)
        est_date_for_orb_candle = current_candle_tz_datetime.date()
        orb_candle_dt_local = self.orb_candle_tz.localize(datetime.combine(est_date_for_orb_candle, target_local_time))
        orb_candle_open_utc = orb_candle_dt_local.astimezone(pytz.utc)

        if orb_candle_open_utc > current_utc_datetime:
            self.logger.info(f"ORB candle time {orb_candle_open_utc.strftime('%Y-%m-%d %H:%M:%S %Z')} for {est_date_for_orb_candle.strftime('%Y-%m-%d %Z')} has not occurred yet.")
            return False # Cannot set today's ORB levels yet

        orb_candle_timestamp_ms = int(orb_candle_open_utc.timestamp() * 1000)
        self.logger.info(f"Attempting to fetch ORB candle for {self.trading_pair} at {orb_candle_open_utc.strftime('%Y-%m-%d %H:%M:%S %Z')} (Interval: {self.orb_kline_interval})")
        try:
            ohlcv = self.exchange_ccxt.fetch_ohlcv(self.trading_pair, self.orb_kline_interval, since=orb_candle_timestamp_ms, limit=1)
            if ohlcv and len(ohlcv) > 0 and ohlcv[0][0] == orb_candle_timestamp_ms:
                self.orb_high = Decimal(str(ohlcv[0][2])); self.orb_low = Decimal(str(ohlcv[0][3]))
                self.orb_levels_set_for_utc_date = orb_candle_open_utc.date()
                self.logger.info(f"ORB levels set for {self.orb_levels_set_for_utc_date}: High={self.orb_high}, Low={self.orb_low}"); return True
            else: self.logger.warning(f"Could not fetch matching ORB candle.")
        except Exception as e: self.logger.error(f"Error fetching ORB candle: {e}", exc_info=True)
        self._reset_daily_orb_state(); return False
        
    def _await_order_fill(self, exchange_order_id: str) -> Optional[Dict]: # Added return type hint
        # Using instance parameters for timeout and interval
        start_time = time.time() # Ensure 'time' is imported as 'time' not 'datetime.time'
        self.logger.info(f"[{self.name}-{self.trading_pair}] Awaiting fill for order {exchange_order_id} (timeout: {self.order_fill_timeout_seconds}s)")
        while time.time() - start_time < self.order_fill_timeout_seconds:
            try:
                order = self.exchange_ccxt.fetch_order(exchange_order_id, self.trading_pair)
                if order['status'] == 'closed':
                    self.logger.info(f"Order {exchange_order_id} filled.")
                    return order
                elif order['status'] in ['canceled', 'rejected', 'expired']:
                    self.logger.warning(f"Order {exchange_order_id} is {order['status']}. Not waiting further.")
                    return order
                self.logger.info(f"Order {exchange_order_id} status is {order['status']}. Attempt "
                                 f"{int((time.time() - start_time) / self.order_fill_check_interval_seconds) + 1}"
                                 f"/{int(self.order_fill_timeout_seconds / self.order_fill_check_interval_seconds)}. Waiting...")
                time.sleep(self.order_fill_check_interval_seconds)
            except ccxt.OrderNotFound:
                self.logger.warning(f"Order {exchange_order_id} not found. Retrying.")
            except Exception as e:
                self.logger.error(f"Error fetching order {exchange_order_id}: {e}. Retrying.", exc_info=True)
        self.logger.warning(f"Timeout for order {exchange_order_id}. Final check.")
        try:
            final_status = self.exchange_ccxt.fetch_order(exchange_order_id, self.trading_pair)
            self.logger.info(f"Final status for order {exchange_order_id}: {final_status['status']}")
            return final_status
        except Exception as e:
            self.logger.error(f"Final check for order {exchange_order_id} failed: {e}", exc_info=True)
        return None

    def _place_order(self, order_type: str, side: str, quantity: Decimal, price: Optional[Decimal]=None, params: Optional[Dict]=None) -> Optional[Order]:
        db_order = None
        try:
            formatted_qty_str = self._format_quantity(quantity)
            formatted_price_str = self._format_price(price) if price else None
            
            db_order = strategy_utils.create_strategy_order_in_db(
                self.db_session, self.user_sub_obj.id, self.trading_pair, order_type.lower(), side, 
                float(formatted_qty_str), float(formatted_price_str) if formatted_price_str else None, 
                status='pending_exchange', notes=f"ORB {order_type} {side}"
            )
            if not db_order: self.logger.error(f"Failed to create DB order for {side} {quantity} {self.trading_pair}."); return None

            exchange_order = self.exchange_ccxt.create_order(self.trading_pair, order_type, side, float(formatted_qty_str), float(formatted_price_str) if formatted_price_str else None, params)
            
            updated_db_order = strategy_utils.update_strategy_order_in_db(
                self.db_session, order_db_id=db_order.id, 
                updates={'order_id': exchange_order.get('id'), 'status': 'open', 'raw_order_data': json.dumps(exchange_order)}
            )
            self.logger.info(f"Order placed: {side} {order_type} {formatted_qty_str} {self.trading_pair}. ExchID: {exchange_order.get('id')}, DB_ID: {db_order.id}")
            return updated_db_order
        except Exception as e:
            self.logger.error(f"Failed to place {side} {order_type} for {self.trading_pair}: {e}", exc_info=True)
            if db_order and db_order.id: strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=db_order.id, updates={'status': 'error_on_exchange', 'status_message': str(e)[:255]})
        return None

    def _cancel_active_sl_tp_orders(self):
        for order_key, exchange_order_id in list(self.active_sl_tp_exchange_ids.items()):
            if exchange_order_id:
                db_id_to_use = self.sl_order_db_id if order_key == 'sl_id' else (self.tp_order_db_id if order_key == 'tp_id' else None)
                try:
                    self.logger.info(f"Canceling {order_key} order {exchange_order_id} (DB ID: {db_id_to_use})")
                    self.exchange_ccxt.cancel_order(exchange_order_id, self.trading_pair)
                    if db_id_to_use: strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=db_id_to_use, updates={'status': 'canceled'})
                    else: strategy_utils.update_strategy_order_in_db(self.db_session, exchange_order_id=exchange_order_id, subscription_id=self.user_sub_obj.id, symbol=self.trading_pair, updates={'status': 'canceled'})
                except ccxt.OrderNotFound:
                     self.logger.warning(f"{order_key} order {exchange_order_id} not found for cancellation.")
                     if db_id_to_use: strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=db_id_to_use, updates={'status': 'not_found_on_cancel'})
                     else: strategy_utils.update_strategy_order_in_db(self.db_session, exchange_order_id=exchange_order_id, subscription_id=self.user_sub_obj.id, symbol=self.trading_pair, updates={'status': 'not_found_on_cancel'})
                except Exception as e: self.logger.error(f"Error cancelling {order_key} order {exchange_order_id}: {e}")
        self._reset_sl_tp_state()
        self._save_position_custom_state() # Save cleared SL/TP info

    def _sync_exchange_position_state(self, current_price: Decimal): # Added current_price for context
        if not self.active_position_db_id or not self.db_session: return False
        
        position_closed_event = False
        sl_exchange_id = self.active_sl_tp_exchange_ids.get('sl_id')
        tp_exchange_id = self.active_sl_tp_exchange_ids.get('tp_id')

        if self.sl_order_db_id and sl_exchange_id:
            try:
                sl_details = self.exchange_ccxt.fetch_order(sl_exchange_id, self.trading_pair)
                if sl_details['status'] == 'closed':
                    self.logger.info(f"SL order {sl_exchange_id} filled.")
                    strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=self.sl_order_db_id, updates={'status': 'closed', 'price': sl_details.get('average'), 'filled': sl_details.get('filled'), 'closed_at': datetime.utcnow(), 'raw_order_data': json.dumps(sl_details)})
                    strategy_utils.close_strategy_position_in_db(self.db_session, self.active_position_db_id, Decimal(str(sl_details['average'])), Decimal(str(sl_details['filled'])), f"Closed by SL {sl_exchange_id}")
                    position_closed_event = True
                    if tp_exchange_id: self._cancel_specific_order(tp_exchange_id, self.tp_order_db_id, "TP (SL Hit)")
            except ccxt.OrderNotFound: self.logger.warning(f"SL order {sl_exchange_id} not found in sync."); strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=self.sl_order_db_id, updates={'status': 'not_found_on_sync'}); self.active_sl_tp_exchange_ids['sl_id'] = None; self.sl_order_db_id = None
            except Exception as e: self.logger.error(f"Error syncing SL order {sl_exchange_id}: {e}")

        if not position_closed_event and self.tp_order_db_id and tp_exchange_id:
            try:
                tp_details = self.exchange_ccxt.fetch_order(tp_exchange_id, self.trading_pair)
                if tp_details['status'] == 'closed':
                    self.logger.info(f"TP order {tp_exchange_id} filled.")
                    strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=self.tp_order_db_id, updates={'status': 'closed', 'price': tp_details.get('average'), 'filled': tp_details.get('filled'), 'closed_at': datetime.utcnow(), 'raw_order_data': json.dumps(tp_details)})
                    strategy_utils.close_strategy_position_in_db(self.db_session, self.active_position_db_id, Decimal(str(tp_details['average'])), Decimal(str(tp_details['filled'])), f"Closed by TP {tp_exchange_id}")
                    position_closed_event = True
                    if sl_exchange_id: self._cancel_specific_order(sl_exchange_id, self.sl_order_db_id, "SL (TP Hit)")
            except ccxt.OrderNotFound: self.logger.warning(f"TP order {tp_exchange_id} not found in sync."); strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=self.tp_order_db_id, updates={'status': 'not_found_on_sync'}); self.active_sl_tp_exchange_ids['tp_id'] = None; self.tp_order_db_id = None
            except Exception as e: self.logger.error(f"Error syncing TP order {tp_exchange_id}: {e}")

        if position_closed_event:
            self._reset_internal_trade_state()
            self._save_position_custom_state() # Save cleared state
            return True
        return False
        
    def _cancel_specific_order(self, exchange_order_id: str, db_order_id: Optional[int], reason_prefix: str):
        # Same as in macd_forecast_mtf_strategy
        try:
            self.exchange_ccxt.cancel_order(exchange_order_id, self.trading_pair)
            self.logger.info(f"Canceled {reason_prefix} order {exchange_order_id} on exchange.")
            updates = {'status': 'canceled', 'status_message': f'Canceled: {reason_prefix}'}
            if db_order_id: strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=db_order_id, updates=updates)
            else: strategy_utils.update_strategy_order_in_db(self.db_session, exchange_order_id=exchange_order_id, subscription_id=self.user_sub_obj.id, symbol=self.trading_pair, updates=updates)
        except ccxt.OrderNotFound:
            self.logger.warning(f"{reason_prefix} order {exchange_order_id} not found on exchange for cancellation.")
            updates_not_found = {'status': 'not_found_on_cancel', 'status_message': f'Not found on cancel: {reason_prefix}'}
            if db_order_id: strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=db_order_id, updates=updates_not_found)
            else: strategy_utils.update_strategy_order_in_db(self.db_session, exchange_order_id=exchange_order_id, subscription_id=self.user_sub_obj.id, symbol=self.trading_pair, updates=updates_not_found)
        except Exception as e: self.logger.error(f"Failed to cancel {reason_prefix} order {exchange_order_id}: {e}", exc_info=True)


    def _process_trading_logic(self, ohlcv_df: pandas.DataFrame):
        if len(ohlcv_df) < 4: self.logger.warning("Not enough data for ORB breakout check."); return
        current_high = Decimal(str(ohlcv_df['high'].iloc[-1])); current_low = Decimal(str(ohlcv_df['low'].iloc[-1]))
        prev_high = Decimal(str(ohlcv_df['high'].iloc[-2])); prev_low = Decimal(str(ohlcv_df['low'].iloc[-2]))
        two_ago_close = Decimal(str(ohlcv_df['close'].iloc[-3])); two_ago_high = Decimal(str(ohlcv_df['high'].iloc[-3])); two_ago_low = Decimal(str(ohlcv_df['low'].iloc[-3]))
        three_ago_close = Decimal(str(ohlcv_df['close'].iloc[-4]))

        if not self.orb_high or not self.orb_low: self.logger.info("ORB levels not set. Holding."); return
        if self.active_position_side: self.logger.debug(f"Already in {self.active_position_side} position. Monitoring."); return
        
        crossed_over_s_high = (three_ago_close <= self.orb_high and two_ago_close > self.orb_high)
        buy_cond_final = (crossed_over_s_high and prev_high > two_ago_high and current_high > prev_high)
        crossed_under_s_low = (three_ago_close >= self.orb_low and two_ago_close < self.orb_low)
        sell_cond_final = (crossed_under_s_low and prev_low < two_ago_low and current_low < prev_low)

        entry_price_signal = Decimal(str(ohlcv_df['close'].iloc[-1]))
        if entry_price_signal == Decimal("0"): self.logger.warning("Entry price is zero."); return
        
        qty_to_trade = self.order_quantity_usd / entry_price_signal # Note: Leverage applied by exchange

        if buy_cond_final or sell_cond_final:
            side = "long" if buy_cond_final else "short"
            self.logger.info(f"ORB {side.upper()} Signal for {self.trading_pair} at {entry_price_signal}")
            
            entry_order_db = self._place_order('market', side, qty_to_trade)
            if entry_order_db and entry_order_db.order_id:
                filled_entry = self._await_order_fill(entry_order_db.order_id)
                if filled_entry and filled_entry['status'] == 'closed':
                    actual_entry_price = Decimal(str(filled_entry['average']))
                    actual_filled_qty = Decimal(str(filled_entry['filled']))
                    strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=entry_order_db.id, updates={'status': 'closed', 'price': float(actual_entry_price), 'filled': float(actual_filled_qty), 'cost': float(actual_entry_price * actual_filled_qty), 'closed_at': datetime.utcnow()})

                    new_pos_db = strategy_utils.create_strategy_position_in_db(self.db_session, self.user_sub_obj.id, self.trading_pair, str(self.exchange_ccxt.id), side, float(actual_filled_qty), float(actual_entry_price), status_message="ORB Entry")
                    if new_pos_db:
                        self.active_position_db_id = new_pos_db.id; self.active_position_side = side
                        self.current_pos_entry_price = actual_entry_price; self.current_pos_qty = actual_filled_qty
                        
                        orb_level_ref = self.orb_high if side == "long" else self.orb_low
                        sl_price = orb_level_ref * (Decimal('1') - self.stop_loss_pct) if side == "long" else orb_level_ref * (Decimal('1') + self.stop_loss_pct)
                        tp_price = orb_level_ref * (Decimal('1') + self.take_profit_pct) if side == "long" else orb_level_ref * (Decimal('1') - self.take_profit_pct)

                        sl_order_db = self._place_order('STOP_MARKET', 'sell' if side == 'long' else 'buy', actual_filled_qty, params={'stopPrice': self._format_price(sl_price), 'reduceOnly':True})
                        if sl_order_db: self.sl_order_db_id = sl_order_db.id; self.active_sl_tp_exchange_ids['sl_id'] = sl_order_db.order_id
                        
                        tp_order_db = self._place_order('TAKE_PROFIT_MARKET', 'sell' if side == 'long' else 'buy', actual_filled_qty, params={'stopPrice': self._format_price(tp_price), 'reduceOnly':True})
                        if tp_order_db: self.tp_order_db_id = tp_order_db.id; self.active_sl_tp_exchange_ids['tp_id'] = tp_order_db.order_id
                        
                        self._save_position_custom_state()
                    else: self.logger.error("Failed to create position in DB after ORB entry.")
                else: self.logger.error(f"ORB Entry order {entry_order_db.order_id} did not fill."); strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=entry_order_db.id, updates={'status': filled_entry.get('status','fill_check_failed') if filled_entry else 'fill_check_failed'})
            else: self.logger.error("Failed to place ORB entry order or create DB record.")

    def _perform_eod_close_if_needed(self, current_utc_dt: datetime):
        # Check if it's EOD in the ORB candle's timezone
        target_eod_local_time = time(self.session_eod_close_hour_local, self.session_eod_close_minute_local, 0)
        current_candle_tz_datetime = current_utc_dt.astimezone(self.orb_candle_tz)

        if current_candle_tz_datetime.time() >= target_eod_local_time:
            if self.active_position_db_id:
                self.logger.info(f"[{self.name}-{self.trading_pair}] EOD detected in {self.orb_candle_timezone_str} at or after {target_eod_local_time.strftime('%H:%M')}. Closing active position {self.active_position_db_id}.")
                self._cancel_active_sl_tp_orders() # Cancel SL/TP first
                
                # Fetch current price for closing PnL estimation (market order will fill at varying price)
                close_price_estimation = Decimal(str(self.exchange_ccxt.fetch_ticker(self.trading_pair)['last']))
                
                close_order_db = self._place_order('market', 'sell' if self.active_position_side == 'long' else 'buy', self.current_pos_qty, params={'reduceOnly': True})
                if close_order_db and close_order_db.order_id:
                    filled_eod_close = self._await_order_fill(close_order_db.order_id)
                    if filled_eod_close and filled_eod_close['status'] == 'closed':
                        actual_close_price = Decimal(str(filled_eod_close['average']))
                        actual_filled_qty = Decimal(str(filled_eod_close['filled']))
                        strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=close_order_db.id, updates={'status':'closed', 'price': float(actual_close_price), 'filled': float(actual_filled_qty), 'closed_at': datetime.utcnow()})
                        strategy_utils.close_strategy_position_in_db(self.db_session, self.active_position_db_id, actual_close_price, actual_filled_qty, "EOD Close")
                    else: self.logger.error(f"EOD market close order {close_order_db.order_id} failed to fill or status unknown.")
                else: self.logger.error("Failed to place EOD market close order.")
                
                self._reset_internal_trade_state() # Reset all position state
                self._save_position_custom_state() # Save the cleared state for custom_data
            
            self._reset_daily_orb_state() # Reset ORB levels for the next day
            return True # EOD close was processed (or attempted)
        return False


    def execute_live_signal(self): # Removed market_data_df, htf_df as they are fetched internally
        self.logger.info(f"Executing {self.name} for {self.trading_pair} UserSubID {self.user_sub_obj.id}")
        
        current_utc_dt = datetime.utcnow().replace(tzinfo=pytz.utc)
        
        # Sync SL/TP hits from exchange before any other logic
        try:
            current_price_for_sync = Decimal(str(self.exchange_ccxt.fetch_ticker(self.trading_pair)['last']))
            if self._sync_exchange_position_state(current_price_for_sync):
                self.logger.info(f"[{self.name}-{self.trading_pair}] Position closed by SL/TP sync. Cycle ended.")
                return
        except Exception as e:
            self.logger.error(f"[{self.name}-{self.trading_pair}] Error during pre-logic SL/TP sync or price fetch: {e}", exc_info=True)
            return # Avoid proceeding with potentially stale state

        # EOD close logic (if applicable for the day)
        if self._perform_eod_close_if_needed(current_utc_dt):
            return # EOD processing happened, end cycle

        # Determine ORB levels if not set for the current UTC date
        if not self.orb_high or not self.orb_low or self.orb_levels_set_for_utc_date != current_utc_dt.date():
            if not self._determine_and_set_orb_levels(current_utc_dt):
                self.logger.info("Failed to set ORB levels or ORB time not yet passed. Waiting.")
                return

        if not self.orb_high or not self.orb_low: # Double check after attempt
            self.logger.info("ORB levels still not available after determination attempt. Waiting.")
            return

        try:
            limit_needed = 4 # For ORB breakout logic based on last 4 candles
            ohlcv_data = self.exchange_ccxt.fetch_ohlcv(self.trading_pair, timeframe=self.kline_interval, limit=limit_needed)
            if not ohlcv_data or len(ohlcv_data) < limit_needed:
                self.logger.warning(f"Not enough OHLCV data for {self.trading_pair} on {self.kline_interval}. Got {len(ohlcv_data)}")
                return

            ohlcv_df = pandas.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            # No need to convert timestamp to datetime if using iloc and assuming latest data is last
            
            self._process_trading_logic(ohlcv_df)

        except Exception as e:
            self.logger.error(f"Error in ORB execute_live_signal: {e}", exc_info=True)

    def run_backtest(self, historical_data_feed, current_simulated_time_utc):
        # Backtest logic remains simulation-based
        self.logger.info(f"Running backtest for {self.name} on {self.trading_pair} at {current_simulated_time_utc}...")
        current_utc_dt = current_simulated_time_utc

        if self.orb_levels_set_for_utc_date != current_utc_dt.date():
            self._reset_daily_orb_state()
            # EOD close for backtest (simulated)
            if self.active_position_side: # If position carried overnight
                 self.logger.info(f"[BACKTEST] Closing overnight {self.active_position_side} position for {self.trading_pair} due to date change.")
                 # Simulate PnL calculation if needed for backtest results
                 self._reset_internal_trade_state() # Reset for backtest state
            
            self._determine_and_set_orb_levels(current_utc_dt, historical_data_feed)

        if not self.orb_high or not self.orb_low:
            return {"action": "HOLD", "reason": "ORB levels not set for simulated day"}

        limit_needed = 4
        ohlcv_df = historical_data_feed.get_ohlcv(self.trading_pair, self.kline_interval, limit_needed, end_time_utc=current_simulated_time_utc)

        if ohlcv_df is None or ohlcv_df.empty or len(ohlcv_df) < limit_needed:
            return {"action": "HOLD", "reason": f"Insufficient OHLCV data for breakout check in backtest at {current_simulated_time_utc}"}
        
        # Convert to Decimal for backtest consistency if _process_trading_logic expects it
        for col in ['open', 'high', 'low', 'close']:
            ohlcv_df[col] = ohlcv_df[col].apply(lambda x: Decimal(str(x)))

        return self._process_trading_logic(ohlcv_df, is_backtest=True, current_simulated_time_utc=current_simulated_time_utc, historical_data_feed=historical_data_feed)

```
