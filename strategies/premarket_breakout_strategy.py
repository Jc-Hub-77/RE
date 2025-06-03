import logging
from datetime import datetime, time, date, timedelta
import pytz
import numpy as np # May not be strictly needed if all TA is via price action
import ccxt
from decimal import Decimal, ROUND_DOWN, ROUND_UP
import time as time_module 
import json 
# import asyncio # Removed asyncio
from sqlalchemy.orm import Session # Added for type hinting if not already there from other strategies
from backend.models import Position, Order, UserStrategySubscription # Added for type hinting

from backend import strategy_utils 
# from backend.models import Order, Position # For type hinting if needed by utils

class PreMarketBreakout:

    @staticmethod
    def get_parameters_definition():
        return {
            "trading_pair": {"type": "str", "label": "Trading Pair (e.g., BTC/USDT:USDT)", "default": "BTC/USDT:USDT"},
            "leverage": {"type": "int", "label": "Leverage", "default": 5, "min": 1, "max": 100},
            "stop_loss_percent": {"type": "float", "label": "Stop Loss % (from entry)", "default": 0.005, "min": 0.001, "max": 0.1, "step": 0.0001}, 
            "take_profit_percent": {"type": "float", "label": "Take Profit % (from entry)", "default": 0.01, "min": 0.001, "max": 0.2, "step": 0.0001}, 
            "risk_allocation_percent": {"type": "float", "label": "Risk Allocation % (of equity per trade)", "default": 0.02, "min": 0.001, "max": 0.1, "step": 0.001}, 
            "kline_interval_for_levels": {"type": "str", "label": "Kline Interval for Pre-Market Levels", "default": "5m", "choices": ["1m", "3m", "5m", "15m"]},
            "kline_interval_for_breakout": {"type": "str", "label": "Kline Interval for Breakout Signal", "default": "1m", "choices": ["1m", "3m", "5m"]},
            "max_entry_deviation_percent": {"type": "float", "label": "Max Entry Price Deviation % (from breakout level)", "default": 0.001, "min": 0.0, "max": 0.01, "step": 0.0001}, 
            "pre_market_start_time_est": {"type": "str", "label": "Pre-Market Start Time (EST HH:MM)", "default": "07:30"},
            "pre_market_end_time_est": {"type": "str", "label": "Pre-Market End Time (EST HH:MM)", "default": "09:29"},
            "market_open_time_est": {"type": "str", "label": "Market Open Time (EST HH:MM)", "default": "09:30"},
            "trading_session_end_time_est": {"type": "str", "label": "Trading Session End Time (EST HH:MM)", "default": "15:55"}, 
            "est_timezone": {"type": "str", "label": "EST Timezone Name", "default": "US/Eastern", "choices": ["US/Eastern", "America/New_York"]},
            # New parameters for order fill
            "order_fill_max_retries": {"type": "int", "default": 10, "min": 1, "max": 30, "label": "Order Fill Max Retries"},
            "order_fill_delay_seconds": {"type": "int", "default": 2, "min": 1, "max": 10, "label": "Order Fill Delay (s)"},
        }

    def __init__(self, db_session, user_sub_obj, strategy_params: dict, exchange_ccxt, logger=None):
        self.db_session = db_session
        self.user_sub_obj = user_sub_obj
        self.params = strategy_params
        self.exchange_ccxt = exchange_ccxt
        self.logger = logger if logger else logging.getLogger(__name__)
        self.name = "PreMarketBreakout"

        self.trading_pair = self.params.get("trading_pair", "BTC/USDT:USDT")
        self.leverage = int(self.params.get("leverage", 5))
        self.stop_loss_percent = Decimal(str(self.params.get("stop_loss_percent", "0.005")))
        self.take_profit_percent = Decimal(str(self.params.get("take_profit_percent", "0.01")))
        self.risk_allocation_percent = Decimal(str(self.params.get("risk_allocation_percent", "0.02")))
        self.kline_interval_for_levels = self.params.get("kline_interval_for_levels", "5m")
        self.kline_interval_for_breakout = self.params.get("kline_interval_for_breakout", "1m")
        self.max_entry_deviation_percent = Decimal(str(self.params.get("max_entry_deviation_percent", "0.001")))
        self.est_timezone_str = self.params.get("est_timezone", "US/Eastern")
        self.est_tz = pytz.timezone(self.est_timezone_str)
        try:
            self.pre_market_start_time = datetime.strptime(self.params.get("pre_market_start_time_est", "07:30"), '%H:%M').time()
            self.pre_market_end_time = datetime.strptime(self.params.get("pre_market_end_time_est", "09:29"), '%H:%M').time()
            self.market_open_time = datetime.strptime(self.params.get("market_open_time_est", "09:30"), '%H:%M').time()
            self.trading_session_end_time = datetime.strptime(self.params.get("trading_session_end_time_est", "15:55"), '%H:%M').time()
        except ValueError as e:
            self.logger.error(f"Error parsing time parameters: {e}. Using defaults.")
            self.pre_market_start_time, self.pre_market_end_time, self.market_open_time, self.trading_session_end_time = time(7,30), time(9,29), time(9,30), time(15,55)

        self.premarket_high, self.premarket_low = None, None
        self.max_deviation_high_entry, self.max_deviation_low_entry = None, None
        self.initialized_for_session_date, self.last_trade_candle_timestamp = None, None
        self.target_notional_usdt_for_trade = Decimal("0.0")
        
        self.active_position_db_id, self.active_position_side = None, None
        self.current_pos_entry_price, self.current_pos_qty = Decimal("0"), Decimal("0")
        self.sl_order_db_id, self.tp_order_db_id = None, None
        self.active_sl_exchange_id, self.active_tp_exchange_id = None, None
        
        # Initialize order fill parameters
        self.order_fill_max_retries = int(self.params.get("order_fill_max_retries", 10)) # Defaulting here if not in params
        self.order_fill_delay_seconds = int(self.params.get("order_fill_delay_seconds", 2)) # Defaulting here if not in params

        self.quantity_precision, self.price_precision = None, None
        self._fetch_market_precision()
        try:
            if hasattr(self.exchange_ccxt, 'set_leverage') and self.trading_pair.endswith(':USDT'): 
                self.exchange_ccxt.set_leverage(self.leverage, self.trading_pair)
                self.logger.info(f"Leverage set to {self.leverage}x for {self.trading_pair}")
        except Exception as e: self.logger.error(f"Failed to set leverage: {e}")
        self.logger.info(f"{self.name} initialized for UserSubID {self.user_sub_obj.id if self.user_sub_obj else 'N/A'}")

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
                
                if key == "est_timezone": # Specific validation for timezone string
                    try:
                        pytz.timezone(user_val)
                    except pytz.exceptions.UnknownTimeZoneError:
                        raise ValueError(f"Invalid timezone string for '{key}': {user_val}")
                elif key.endswith("_time_est"): # Specific validation for HH:MM time strings
                    try:
                        datetime.strptime(user_val, '%H:%M')
                    except ValueError:
                        raise ValueError(f"Parameter '{key}' must be in HH:MM format. Got: {user_val}")

            elif val_type_str == "bool": # Not in current definition but good to have
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

    def _monitor_order_fill_sync(self, exchange_order_id: str): # Uses instance vars for retries/delay
        self.logger.info(f"Monitoring fill for order ID: {exchange_order_id} (retries: {self.order_fill_max_retries}, delay: {self.order_fill_delay_seconds}s)")
        for attempt in range(self.order_fill_max_retries):
            try:
                order_status = self.exchange_ccxt.fetch_order(exchange_order_id, self.trading_pair)
                if order_status['status'] == 'closed':
                    self.logger.info(f"Order {exchange_order_id} confirmed filled: {order_status}")
                    return order_status
                elif order_status['status'] in ['canceled', 'rejected', 'expired']:
                    self.logger.warning(f"Order {exchange_order_id} terminal but not filled: {order_status['status']}")
                    return order_status # Return the terminal status
                self.logger.info(f"Order {exchange_order_id} status: {order_status['status']}. Attempt {attempt+1}/{self.order_fill_max_retries}.")
            except ccxt.OrderNotFound:
                self.logger.warning(f"Order {exchange_order_id} not found (attempt {attempt+1}).")
            except Exception as e:
                self.logger.error(f"Error fetching order {exchange_order_id}: {e}", exc_info=True)
            time_module.sleep(self.order_fill_delay_seconds) 
        self.logger.warning(f"Order {exchange_order_id} did not fill after {self.order_fill_max_retries} retries. Returning last known status or None.")
        try: # Final attempt to get status
            return self.exchange_ccxt.fetch_order(exchange_order_id, self.trading_pair)
        except Exception as e:
            self.logger.error(f"Final check for order {exchange_order_id} failed: {e}", exc_info=True)
        return None

    def _place_order_with_sl_tp(self, side, quantity_decimal: Decimal, entry_price_decimal: Decimal, sl_price_decimal: Decimal, tp_price_decimal: Decimal, current_simulated_time_utc=None): # Made sync
        formatted_qty_float = float(self._format_quantity(quantity_decimal))
        sl_price_str = str(self._format_price(sl_price_decimal))
        tp_price_str = str(self._format_price(tp_price_decimal))
        entry_price_float = float(entry_price_decimal)

        action_details = {"action": "OPEN_ORDER", "side": side, "symbol": self.trading_pair, "qty": formatted_qty_float, "entry_approx": entry_price_float, "sl": float(sl_price_str), "tp": float(tp_price_str), "timestamp_utc": current_simulated_time_utc or datetime.utcnow()}
        self.logger.info(f"Attempting to place order: {action_details}")

        if current_simulated_time_utc:
            self.logger.info(f"[BACKTEST] Simulating {side} order...")
            self.active_position_side = side
            self.current_pos_entry_price = entry_price_decimal
            self.current_pos_qty = quantity_decimal
            return {"status": "simulated_open", "order_id": f"sim_{datetime.utcnow().timestamp()}", **action_details}

        db_market_order, db_sl_order, db_tp_order = None, None, None
        market_order_receipt, sl_order_receipt, tp_order_receipt = None, None, None
        exchange_market_order_id = None
        
        try:
            self._cancel_all_open_orders(current_simulated_time_utc=current_simulated_time_utc) # Made sync, pass param

            db_market_order = strategy_utils.create_strategy_order_in_db(
                self.db_session, self.user_sub_obj.id, self.trading_pair, 'MARKET', side, formatted_qty_float, entry_price_float, 'pending_exchange_creation'
            )
            if not db_market_order: raise Exception("DB market order creation failed.")

            market_order_receipt = self.exchange_ccxt.create_order(self.trading_pair, 'MARKET', 'buy' if side == 'long' else 'sell', formatted_qty_float)
            exchange_market_order_id = market_order_receipt.get('id')
            self.logger.info(f"Market order submitted. ID: {exchange_market_order_id}, Receipt: {market_order_receipt}")
            strategy_utils.update_strategy_order_in_db(self.db_session, db_market_order.id, updates={'order_id': exchange_market_order_id, 'status': 'submitted_to_exchange', 'raw_order_data': json.dumps(market_order_receipt)})

            filled_order_details = self._monitor_order_fill_sync(exchange_market_order_id) # Changed call
            if not filled_order_details or filled_order_details.get('status') != 'closed':
                strategy_utils.update_strategy_order_in_db(self.db_session, db_market_order.id, updates={'status': 'fill_check_failed', 'status_message': 'Order fill timeout or failure.'})
                return {"status": "error", "message": "Market order fill confirmation failed.", **action_details}

            actual_filled_qty = Decimal(str(filled_order_details.get('filled', formatted_qty_float)))
            actual_avg_price = Decimal(str(filled_order_details.get('average', entry_price_float)))
            actual_cost = Decimal(str(filled_order_details.get('cost', float(actual_filled_qty * actual_avg_price))))
            strategy_utils.update_strategy_order_in_db(self.db_session, db_market_order.id, updates={'status': 'closed', 'filled': float(actual_filled_qty), 'price': float(actual_avg_price), 'cost': float(actual_cost), 'closed_at': datetime.utcnow(), 'raw_order_data': json.dumps(filled_order_details)})

            db_position = strategy_utils.create_strategy_position_in_db(
                self.db_session, self.user_sub_obj.id, self.trading_pair, self.exchange_ccxt.id, side, float(actual_filled_qty), float(actual_avg_price), "Position opened; SL/TP pending."
            )
            if not db_position: raise Exception("DB position creation failed after fill.")
            
            self.active_position_db_id = db_position.id
            self.active_position_side = side
            self.current_pos_entry_price = actual_avg_price
            self.current_pos_qty = actual_filled_qty
            self.logger.info(f"Position DB ID: {db_position.id}. Side: {side}, Qty: {actual_filled_qty}, Entry: {actual_avg_price}")

            # SL Order
            db_sl_order = strategy_utils.create_strategy_order_in_db(self.db_session, self.user_sub_obj.id, self.trading_pair, 'STOP_MARKET', 'sell' if side == 'long' else 'buy', float(actual_filled_qty), float(sl_price_decimal), 'pending_exchange_creation', notes=f"SL for pos {db_position.id}")
            if db_sl_order:
                try:
                    sl_order_receipt = self.exchange_ccxt.create_order(self.trading_pair, 'STOP_MARKET', 'sell' if side == 'long' else 'buy', float(actual_filled_qty), params={'stopPrice': sl_price_str, 'reduceOnly': True})
                    self.active_sl_exchange_id = sl_order_receipt.get('id')
                    strategy_utils.update_strategy_order_in_db(self.db_session, db_sl_order.id, updates={'order_id': self.active_sl_exchange_id, 'status': 'open', 'raw_order_data': json.dumps(sl_order_receipt)})
                    self.sl_order_db_id = db_sl_order.id
                except Exception as e_sl: self.logger.error(f"SL order placement error: {e_sl}", exc_info=True); strategy_utils.update_strategy_order_in_db(self.db_session, db_sl_order.id, updates={'status': 'creation_failed', 'status_message': str(e_sl)})
            
            # TP Order
            db_tp_order = strategy_utils.create_strategy_order_in_db(self.db_session, self.user_sub_obj.id, self.trading_pair, 'TAKE_PROFIT_MARKET', 'sell' if side == 'long' else 'buy', float(actual_filled_qty), float(tp_price_decimal), 'pending_exchange_creation', notes=f"TP for pos {db_position.id}")
            if db_tp_order:
                try:
                    tp_order_receipt = self.exchange_ccxt.create_order(self.trading_pair, 'TAKE_PROFIT_MARKET', 'sell' if side == 'long' else 'buy', float(actual_filled_qty), params={'stopPrice': tp_price_str, 'reduceOnly': True})
                    self.active_tp_exchange_id = tp_order_receipt.get('id')
                    strategy_utils.update_strategy_order_in_db(self.db_session, db_tp_order.id, updates={'order_id': self.active_tp_exchange_id, 'status': 'open', 'raw_order_data': json.dumps(tp_order_receipt)})
                    self.tp_order_db_id = db_tp_order.id
                except Exception as e_tp: self.logger.error(f"TP order placement error: {e_tp}", exc_info=True); strategy_utils.update_strategy_order_in_db(self.db_session, db_tp_order.id, updates={'status': 'creation_failed', 'status_message': str(e_tp)})

            if self.active_position_db_id: # Persist linked SL/TP order DB IDs
                pos_status_msg = f"SL DB ID: {self.sl_order_db_id}, TP DB ID: {self.tp_order_db_id}."
                strategy_utils.update_strategy_position_in_db(self.db_session, self.active_position_db_id, updates={'status_message': pos_status_msg}) # Assuming this util exists

            return {"status": "live_orders_placed", "market_order_id": exchange_market_order_id, "sl_order_id": self.active_sl_exchange_id, "tp_order_id": self.active_tp_exchange_id, "db_position_id": self.active_position_db_id, **action_details}

        except Exception as e:
            self.logger.error(f"Order sequence error: {e}", exc_info=True)
            if db_market_order and db_market_order.id and (not market_order_receipt or market_order_receipt.get('status') != 'closed'): # If market order wasn't confirmed filled
                strategy_utils.update_strategy_order_in_db(self.db_session, db_market_order.id, updates={'status': 'error_before_fill', 'status_message': str(e)})
            return {"status": "error", "message": str(e), **action_details}

    def _close_position(self, current_side, current_qty_decimal: Decimal, current_price_for_closing_decimal: Decimal, reason="signal", current_simulated_time_utc=None): # Made sync
        formatted_qty = self._format_quantity(current_qty_decimal)
        action_details = {
            "action": "CLOSE_POSITION", "side_closed": current_side, "symbol": self.trading_pair,
            "qty": float(formatted_qty), "price_approx": float(current_price_for_closing_decimal), "reason": reason,
            "timestamp_utc": current_simulated_time_utc or datetime.utcnow()
        }
        self.logger.info(f"Closing position: {action_details}")

        if current_simulated_time_utc: # Backtesting
            self.logger.info(f"[BACKTEST] Simulating CLOSE {current_side} position...")
            self.active_position_db_id = None; self.active_position_side = None; self.current_pos_entry_price = Decimal("0"); self.current_pos_qty = Decimal("0")
            self.sl_order_db_id = None; self.tp_order_db_id = None; self.active_sl_exchange_id = None; self.active_tp_exchange_id = None
            return {"status": "simulated_close", **action_details}

        db_close_order = None
        try:
            self._cancel_all_open_orders(current_simulated_time_utc=current_simulated_time_utc) # Made sync
            ccxt_side = 'sell' if current_side == 'long' else 'buy'
            db_close_order = strategy_utils.create_strategy_order_in_db(self.db_session, self.user_sub_obj.id, self.trading_pair, 'MARKET_CLOSE', ccxt_side, float(formatted_qty), float(current_price_for_closing_decimal), 'pending_exchange_creation', notes=f"Close for pos {self.active_position_db_id}, reason: {reason}")
            if not db_close_order: raise Exception("DB close order creation failed.")

            close_order_receipt = self.exchange_ccxt.create_order(self.trading_pair, 'MARKET', ccxt_side, float(formatted_qty), params={'reduceOnly': True})
            exchange_close_order_id = close_order_receipt.get('id')
            self.logger.info(f"Close order submitted. ID: {exchange_close_order_id}")
            strategy_utils.update_strategy_order_in_db(self.db_session, db_close_order.id, updates={'order_id': exchange_close_order_id, 'status': 'submitted_to_exchange', 'raw_order_data': json.dumps(close_order_receipt)})
            
            filled_close_details = self._monitor_order_fill_sync(exchange_close_order_id) # Changed call
            final_close_price = current_price_for_closing_decimal
            if filled_close_details and filled_close_details.get('status') == 'closed':
                final_close_price = Decimal(str(filled_close_details.get('average', current_price_for_closing_decimal)))
                strategy_utils.update_strategy_order_in_db(self.db_session, db_close_order.id, updates={'status': 'closed', 'filled': float(filled_close_details.get('filled', formatted_qty)), 'price': float(final_close_price), 'cost': float(filled_close_details.get('cost', float(formatted_qty * final_close_price))), 'closed_at': datetime.utcnow(), 'raw_order_data': json.dumps(filled_close_details)})
            else:
                self.logger.warning(f"Close order {exchange_close_order_id} fill uncertain."); strategy_utils.update_strategy_order_in_db(self.db_session, db_close_order.id, updates={'status': 'fill_check_failed'})

            if self.active_position_db_id:
                pnl = None
                if self.current_pos_entry_price > 0 and current_qty_decimal > 0: pnl = (final_close_price - self.current_pos_entry_price) * current_qty_decimal if current_side == 'long' else (self.current_pos_entry_price - final_close_price) * current_qty_decimal
                strategy_utils.close_strategy_position_in_db(self.db_session, self.active_position_db_id, float(final_close_price), float(formatted_qty), reason, float(pnl) if pnl is not None else None)
            
            self.active_position_db_id = None; self.active_position_side = None; self.current_pos_entry_price = Decimal("0"); self.current_pos_qty = Decimal("0")
            self.sl_order_db_id = None; self.tp_order_db_id = None; self.active_sl_exchange_id = None; self.active_tp_exchange_id = None
            return {"status": "live_close_order_placed", "close_order_id": exchange_close_order_id, **action_details}
        except Exception as e:
            self.logger.error(f"Error closing live position: {e}", exc_info=True)
            if db_close_order and db_close_order.id: strategy_utils.update_strategy_order_in_db(self.db_session, db_close_order.id, updates={'status': 'error', 'status_message': str(e)})
            return {"status": "error", "message": str(e), **action_details}

    def _cancel_all_open_orders(self, current_simulated_time_utc=None): # Made sync
        if current_simulated_time_utc: 
            self.logger.info(f"[BACKTEST] Simulating cancel all open orders for {self.trading_pair}")
            return {"status": "simulated_cancel_all"}
        try:
            self.exchange_ccxt.cancel_all_orders(self.trading_pair)
            self.logger.info(f"Cancelled all open orders for {self.trading_pair}")
            if self.sl_order_db_id: strategy_utils.update_strategy_order_in_db(self.db_session, self.sl_order_db_id, updates={'status': 'cancelled', 'status_message': 'Cancelled by strategy.'}); self.sl_order_db_id=None; self.active_sl_exchange_id=None
            if self.tp_order_db_id: strategy_utils.update_strategy_order_in_db(self.db_session, self.tp_order_db_id, updates={'status': 'cancelled', 'status_message': 'Cancelled by strategy.'}); self.tp_order_db_id=None; self.active_tp_exchange_id=None
            return {"status": "live_orders_cancelled"}
        except Exception as e:
            self.logger.error(f"Error cancelling orders: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def _eod_close_logic(self, current_est_datetime, current_price_for_closing_decimal: Decimal, current_simulated_time_utc=None): # Made sync
        self.logger.info(f"EOD check at {current_est_datetime.strftime('%Y-%m-%d %H:%M:%S')} EST")
        self._get_current_position_details(historical_data_feed=current_simulated_time_utc is not None) 
        if self.active_position_side is not None: 
            self.logger.info(f"EOD: Open position (Side: {self.active_position_side}). Closing.")
            self._close_position(self.active_position_side, self.current_pos_qty, current_price_for_closing_decimal, "EOD", current_simulated_time_utc) # Removed await
        self._cancel_all_open_orders(current_simulated_time_utc=current_simulated_time_utc) # Removed await
        self._reset_daily_state() 
        return {"action": "EOD_CLOSE_RESET"}

    def execute_live_signal(self, market_data_df=None): # Made sync
        now_est = self._get_current_est_datetime()
        now_est_time = now_est.time()
        if now_est_time >= self.market_open_time and (self.initialized_for_session_date is None or self.initialized_for_session_date != now_est.date()):
            if not self._initialize_session_levels(now_est): return {"action": "HOLD", "reason": "Failed level initialization"}
        if not self.premarket_high or not self.premarket_low: 
            if now_est_time < self.market_open_time and now_est_time >= self.pre_market_start_time: self.logger.info(f"Pre-market period. Levels not yet finalized.")
            elif now_est_time >= self.market_open_time: self.logger.warning(f"Market open but levels not set."); self._initialize_session_levels(now_est) # Removed await
            else: self.logger.info(f"Outside trading session hours.")
            return {"action": "HOLD", "reason": "Levels not set or outside trading hours"}
        if now_est_time >= self.trading_session_end_time:
            try: eod_close_price = Decimal(str(self.exchange_ccxt.fetch_ticker(self.trading_pair)['last'])); return self._eod_close_logic(now_est, eod_close_price) # Removed await
            except Exception as e: self.logger.error(f"Failed EOD price fetch: {e}"); return {"action": "ERROR", "reason": "Failed EOD price fetch"}
        try:
            ohlcv_breakout = self.exchange_ccxt.fetch_ohlcv(self.trading_pair, timeframe=self.kline_interval_for_breakout, limit=2)
            if not ohlcv_breakout or len(ohlcv_breakout) < 2: return {"action": "HOLD", "reason": "Insufficient breakout kline data"}
            last_closed_candle = ohlcv_breakout[-2]; last_closed_candle_close = Decimal(str(last_closed_candle[4])); last_closed_candle_timestamp_ms = last_closed_candle[0]
            if self.last_trade_candle_timestamp == last_closed_candle_timestamp_ms: return {"action": "HOLD", "reason": "Already processed this candle"}
        except Exception as e: self.logger.error(f"Kline fetch error: {e}", exc_info=True); return {"action": "ERROR", "reason": "Kline fetch error"}
        self._get_current_position_details() 
        action_taken_this_cycle = False; trade_result = None
        if self.active_position_side is None: 
            entry_price = last_closed_candle_close 
            if entry_price > self.premarket_high and entry_price <= self.max_deviation_high_entry:
                self.logger.info(f"LONG Breakout: {entry_price} > {self.premarket_high}")
                sl = entry_price * (Decimal("1") - self.stop_loss_percent); tp = entry_price * (Decimal("1") + self.take_profit_percent)
            if self.target_notional_usdt_for_trade > 0 and entry_price > 0: quantity = self.target_notional_usdt_for_trade / entry_price; trade_result = self._place_order_with_sl_tp('long', quantity, entry_price, sl, tp); action_taken_this_cycle = True # Removed await
            elif entry_price < self.premarket_low and entry_price >= self.max_deviation_low_entry:
                self.logger.info(f"SHORT Breakout: {entry_price} < {self.premarket_low}")
                sl = entry_price * (Decimal("1") + self.stop_loss_percent); tp = entry_price * (Decimal("1") - self.take_profit_percent)
                if self.target_notional_usdt_for_trade > 0 and entry_price > 0: quantity = self.target_notional_usdt_for_trade / entry_price; trade_result = self._place_order_with_sl_tp('short', quantity, entry_price, sl, tp); action_taken_this_cycle = True # Removed await
        else: self.logger.info(f"Position open ({self.active_position_side}). Monitoring.")
        if action_taken_this_cycle: self.last_trade_candle_timestamp = last_closed_candle_timestamp_ms; return trade_result if trade_result else {"action": "ERROR", "reason": "Trade placement failed"}
        return {"action": "HOLD", "reason": "No signal or position exists"}

    def run_backtest(self, historical_data_feed, current_simulated_time_utc: datetime): # Made sync
        now_est = self._get_current_est_datetime(current_simulated_time_utc)
        now_est_time = now_est.time()
        current_price_for_eval_decimal = Decimal(str(historical_data_feed.get_current_price(self.trading_pair, current_simulated_time_utc)))
        if now_est_time >= self.market_open_time and (self.initialized_for_session_date is None or self.initialized_for_session_date != now_est.date()):
            if not self._initialize_session_levels(now_est, historical_data_feed): return {"action": "HOLD", "reason": "Backtest: Failed level initialization"}
        if not self.premarket_high or not self.premarket_low:
            if now_est_time < self.market_open_time and now_est_time >= self.pre_market_start_time: self.logger.info(f"[BT] Pre-market. Levels not final.")
            elif now_est_time >= self.market_open_time: self.logger.warning(f"[BT] Market open, levels not set."); self._initialize_session_levels(now_est, historical_data_feed) # Removed await
            else: self.logger.info(f"[BT] Outside trading hours.")
            return {"action": "HOLD", "reason": "Backtest: Levels not set/outside hours"}
        if now_est_time >= self.trading_session_end_time: return self._eod_close_logic(now_est, current_price_for_eval_decimal, current_simulated_time_utc) # Removed await
        try:
            interval_sec = self.exchange_ccxt.parse_timeframe(self.kline_interval_for_breakout)
            since_utc_ms = int((current_simulated_time_utc - timedelta(minutes=self.exchange_ccxt.parse_timeframe(self.kline_interval_for_breakout)*5/60)).timestamp()*1000) # type: ignore
            ohlcv = historical_data_feed.get_ohlcv(self.trading_pair, self.kline_interval_for_breakout, since_utc_ms, 10, int(current_simulated_time_utc.timestamp()*1000))
            if not ohlcv: return {"action": "HOLD", "reason": "Backtest: No kline data"}
            target_ts_ms = int((current_simulated_time_utc - timedelta(seconds=interval_sec)).timestamp()*1000)
            candle = next((c for c in reversed(ohlcv) if c[0] == target_ts_ms), None)
            if not candle: return {"action": "HOLD", "reason": "Backtest: Specific kline not found"}
            close = Decimal(str(candle[4])); ts_ms = candle[0]
            if self.last_trade_candle_timestamp == ts_ms: return {"action": "HOLD", "reason": "Backtest: Already processed candle"}
        except Exception as e: self.logger.error(f"[BT] Kline error: {e}", exc_info=True); return {"action": "ERROR", "reason": "Backtest: Kline error"}
        self._get_current_position_details(historical_data_feed)
        action_taken, trade_action = False, {"action": "HOLD"}
        if self.active_position_side is None:
            entry = close
            if entry > self.premarket_high and entry <= self.max_deviation_high_entry:
                sl = entry*(1-self.stop_loss_percent); tp = entry*(1+self.take_profit_percent)
            if self.target_notional_usdt_for_trade > 0 and entry_price > 0: qty = self.target_notional_usdt_for_trade/entry; trade_action = self._place_order_with_sl_tp('long',qty,entry,sl,tp,current_simulated_time_utc); action_taken=True # Removed await
            elif entry < self.premarket_low and entry >= self.max_deviation_low_entry:
                sl = entry*(1+self.stop_loss_percent); tp = entry*(1-self.take_profit_percent)
                if self.target_notional_usdt_for_trade > 0 and entry > 0: qty = self.target_notional_usdt_for_trade/entry; trade_action = self._place_order_with_sl_tp('short',qty,entry,sl,tp,current_simulated_time_utc); action_taken=True # Removed await
        if action_taken: self.last_trade_candle_timestamp = ts_ms
        return trade_action
