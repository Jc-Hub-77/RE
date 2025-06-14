# trading_platform/strategies/ema_crossover_strategy.py
import pandas as pd
import ta 
import logging
import datetime
import json
import time # For awaiting order fills
from sqlalchemy.orm import Session
from backend.models import Position, Order, UserStrategySubscription # Ensure UserStrategySubscription is imported
import ccxt # Now explicitly imported for exception handling like ccxt.OrderNotFound
from backend import strategy_utils # Import the new strategy_utils

logger = logging.getLogger(__name__)

class EMACrossoverStrategy:
    def __init__(self, symbol: str, timeframe: str, db_session: Session, user_sub_obj: UserStrategySubscription, 
                 short_ema_period: int = 10, long_ema_period: int = 20, 
                 capital: float = 10000, 
                 risk_per_trade_percent: float = 1.0,
                 stop_loss_percent: float = 2.0, 
                 take_profit_percent: float = 4.0,
                 # New parameters added to signature
                 order_fill_timeout_seconds: int = 60,
                 order_fill_check_interval_seconds: int = 3,
                 **custom_parameters
                 ):
        self.symbol = symbol
        self.timeframe = timeframe
        
        # Process parameters defined in get_parameters_definition first
        # This allows overrides from custom_parameters if they happen to be passed there too
        # though ideally they are passed as named args if modified from default.
        defined_params = self.get_parameters_definition()
        params_to_set = {}
        for p_name, p_def in defined_params.items():
            # Prioritize named args, then custom_parameters, then default from definition
            if p_name == "short_ema_period": params_to_set[p_name] = short_ema_period
            elif p_name == "long_ema_period": params_to_set[p_name] = long_ema_period
            elif p_name == "risk_per_trade_percent": params_to_set[p_name] = risk_per_trade_percent
            elif p_name == "stop_loss_percent": params_to_set[p_name] = stop_loss_percent
            elif p_name == "take_profit_percent": params_to_set[p_name] = take_profit_percent
            # For newly added params, ensure they are picked up correctly
            elif p_name == "order_fill_timeout_seconds": 
                params_to_set[p_name] = custom_parameters.get(p_name, order_fill_timeout_seconds)
            elif p_name == "order_fill_check_interval_seconds":
                params_to_set[p_name] = custom_parameters.get(p_name, order_fill_check_interval_seconds)
            else: # Fallback for any other defined params if not explicitly handled above
                 params_to_set[p_name] = custom_parameters.get(p_name, p_def['default'])


        self.short_ema_period = int(params_to_set["short_ema_period"])
        self.long_ema_period = int(params_to_set["long_ema_period"])
        
        self.risk_per_trade_decimal = float(params_to_set["risk_per_trade_percent"]) / 100.0
        self.stop_loss_decimal = float(params_to_set["stop_loss_percent"]) / 100.0
        self.take_profit_decimal = float(params_to_set["take_profit_percent"]) / 100.0
        
        self.order_fill_timeout_seconds = int(params_to_set["order_fill_timeout_seconds"])
        self.order_fill_check_interval_seconds = int(params_to_set["order_fill_check_interval_seconds"])

        self.name = f"EMA Crossover ({self.short_ema_period}/{self.long_ema_period})"
        self.description = f"A simple EMA crossover strategy using {self.short_ema_period}-period and {self.long_ema_period}-period EMAs."
        
        self.price_precision = 8 
        self.quantity_precision = 8 
        self._precisions_fetched_ = False

        self.db_session = db_session
        self.user_sub_obj = user_sub_obj
        self.logger = logger 

        self.active_position_db_id = None
        self.active_sl_order_exchange_id = None
        self.active_tp_order_exchange_id = None
        self.active_sl_order_db_id = None
        self.active_tp_order_db_id = None
        
        self.current_pos_type = None 
        self.entry_price = 0.0
        self.pos_size_asset = 0.0

        init_params_log = {
            "symbol": symbol, "timeframe": timeframe, "short_ema_period": self.short_ema_period,
            "long_ema_period": self.long_ema_period, "capital_param": capital,
            "risk_per_trade_percent": risk_per_trade_percent,
            "stop_loss_percent": stop_loss_percent, "take_profit_percent": take_profit_percent,
            "subscription_id": self.user_sub_obj.id if self.user_sub_obj else "N/A",
            "custom_parameters": custom_parameters
        }
        self.logger.info(f"[{self.name}-{self.symbol}] Initialized with effective params: {init_params_log}")
        
        self._load_persistent_state()

    def _load_persistent_state(self):
        if not self.db_session or not self.user_sub_obj:
            self.logger.warning(f"[{self.name}-{self.symbol}] DB session or user subscription object not available in _load_persistent_state.")
            return

        open_position = strategy_utils.get_open_strategy_position(self.db_session, self.user_sub_obj.id, self.symbol)

        if open_position:
            self.logger.info(f"[{self.name}-{self.symbol}] Loading persistent state for open position ID: {open_position.id}")
            self.active_position_db_id = open_position.id
            self.current_pos_type = open_position.side
            self.entry_price = open_position.entry_price
            self.pos_size_asset = open_position.amount
            
            # Load open SL order
            sl_orders = strategy_utils.get_open_orders_for_subscription(self.db_session, self.user_sub_obj.id, self.symbol, order_type='stop_market')
            if sl_orders: # Assuming only one open SL per position/symbol
                self.active_sl_order_exchange_id = sl_orders[0].order_id
                self.active_sl_order_db_id = sl_orders[0].id
            
            # Load open TP order
            tp_orders = strategy_utils.get_open_orders_for_subscription(self.db_session, self.user_sub_obj.id, self.symbol, order_type='limit')
            if tp_orders: # Assuming only one open TP per position/symbol
                self.active_tp_order_exchange_id = tp_orders[0].order_id
                self.active_tp_order_db_id = tp_orders[0].id

            self.logger.info(f"[{self.name}-{self.symbol}] Loaded state: PosID {self.active_position_db_id}, Side {self.current_pos_type}, SL ExchID {self.active_sl_order_exchange_id}, TP ExchID {self.active_tp_order_exchange_id}")
        else:
            self.logger.info(f"[{self.name}-{self.symbol}] No active persistent position found in DB for this subscription.")

    @classmethod
    def validate_parameters(cls, params: dict) -> dict:
        """Validates strategy-specific parameters."""
        definition = cls.get_parameters_definition()
        validated_params = {}
        # cls.logger is not available in classmethod directly without passing or using a global/module logger
        # For simplicity, direct print or raise error. Or use logging.getLogger inside.
        _logger = logging.getLogger(__name__)


        for key, def_value in definition.items():
            val_type_str = def_value.get("type")
            choices = def_value.get("options") 
            min_val = def_value.get("min")
            max_val = def_value.get("max")
            default_val = def_value.get("default")
            
            user_val = params.get(key)

            if user_val is None: # Parameter not provided by user
                if default_val is not None:
                    user_val = default_val # Apply default
                else:
                    # If truly required and no default, this is an issue.
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
            elif val_type_str == "string": # Assuming "timeframe" type is string-like
                if not isinstance(user_val, str):
                    raise ValueError(f"Parameter '{key}' must be a string. Got: {user_val}")
            elif val_type_str == "bool":
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

        # Check for unknown parameters specifically passed in `params`
        for key_param in params:
            if key_param not in definition:
                # These are parameters that were in the input `params` but not defined in get_parameters_definition.
                # This strategy's __init__ takes symbol, timeframe, capital as named args, and **custom_parameters.
                # If this validator is for the **custom_parameters part, then any unknown key is an error.
                # If it's for a combined dict, then we might allow symbol/timeframe/capital.
                # Assuming this validates the content of UserStrategySubscription.custom_parameters.
                _logger.warning(f"Parameter '{key_param}' was provided but is not defined in EMA Crossover strategy. It will be ignored if not an explicit __init__ argument.")
                # To be strict and only allow defined params:
                # raise ValueError(f"Unknown parameter '{key_param}' provided for EMA Crossover strategy.")
                # If you want to pass them through (e.g. if they are handled by __init__ explicitly):
                # validated_params[key_param] = params[key_param]


        return validated_params

    @classmethod
    def get_parameters_definition(cls):
        return {
            "short_ema_period": {"type": "int", "default": 10, "min": 2, "max": 100, "label": "Short EMA Period"},
            "long_ema_period": {"type": "int", "default": 20, "min": 5, "max": 200, "label": "Long EMA Period"},
            "risk_per_trade_percent": {"type": "float", "default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1, "label": "Risk per Trade (% of Effective Capital)"},
            "stop_loss_percent": {"type": "float", "default": 2.0, "min": 0.1, "step": 0.1, "label": "Stop Loss % from Entry"},
            "take_profit_percent": {"type": "float", "default": 4.0, "min": 0.1, "step": 0.1, "label": "Take Profit % from Entry"},
            # New param definitions
            "order_fill_timeout_seconds": {"type": "int", "default": 60, "min": 10, "max": 300, "label": "Order Fill Timeout (s)"},
            "order_fill_check_interval_seconds": {"type": "int", "default": 3, "min": 1, "max": 30, "label": "Order Fill Check Interval (s)"},
        }

    def _get_precisions_live(self, exchange_ccxt):
        if not self._precisions_fetched_:
            try:
                exchange_ccxt.load_markets(True)
                market = exchange_ccxt.market(self.symbol)
                self.price_precision = market['precision']['price']
                self.quantity_precision = market['precision']['amount']
                self._precisions_fetched_ = True
                self.logger.info(f"[{self.name}-{self.symbol}] Precisions: Price={self.price_precision}, Qty={self.quantity_precision}")
            except Exception as e:
                self.logger.error(f"[{self.name}-{self.symbol}] Error fetching live precisions: {e}", exc_info=True)
    
    def _format_price(self, price, exchange_ccxt):
        self._get_precisions_live(exchange_ccxt)
        return float(exchange_ccxt.price_to_precision(self.symbol, price))

    def _format_quantity(self, quantity, exchange_ccxt):
        self._get_precisions_live(exchange_ccxt)
        return float(exchange_ccxt.amount_to_precision(self.symbol, quantity))

    def _await_order_fill(self, exchange_ccxt, order_id: str, symbol: str): # Removed defaults
        start_time = time.time()
        self.logger.info(f"[{self.name}-{self.symbol}] Awaiting fill for order {order_id} (timeout: {self.order_fill_timeout_seconds}s)")
        while time.time() - start_time < self.order_fill_timeout_seconds:
            try:
                order = exchange_ccxt.fetch_order(order_id, symbol)
                self.logger.debug(f"[{self.name}-{self.symbol}] Order {order_id} status: {order['status']}")
                if order['status'] == 'closed':
                    self.logger.info(f"[{self.name}-{self.symbol}] Order {order_id} confirmed filled. Avg Price: {order.get('average')}, Filled Qty: {order.get('filled')}")
                    return order
                elif order['status'] in ['canceled', 'rejected', 'expired']:
                    self.logger.warning(f"[{self.name}-{self.symbol}] Order {order_id} is {order['status']}, will not be filled.")
                    return order
            except ccxt.OrderNotFound: self.logger.warning(f"[{self.name}-{self.symbol}] Order {order_id} not found. Retrying.")
            except Exception as e: self.logger.error(f"[{self.name}-{self.symbol}] Error fetching order {order_id}: {e}. Retrying.", exc_info=True)
            time.sleep(self.order_fill_check_interval_seconds)
        self.logger.warning(f"[{self.name}-{self.symbol}] Timeout waiting for order {order_id} to fill. Final check.")
        try:
            final_order_status = exchange_ccxt.fetch_order(order_id, symbol)
            self.logger.info(f"[{self.name}-{self.symbol}] Final status for order {order_id}: {final_order_status['status']}")
            return final_order_status
        except Exception as e: self.logger.error(f"[{self.name}-{self.symbol}] Final check for order {order_id} failed: {e}", exc_info=True); return None

    def _calculate_emas(self, df: pd.DataFrame):
        if 'close' not in df.columns: self.logger.error(f"[{self.name}-{self.symbol}] DataFrame must contain 'close' column."); return df
        df[f'ema_short'] = ta.trend.EMAIndicator(df['close'], window=self.short_ema_period).ema_indicator()
        df[f'ema_long'] = ta.trend.EMAIndicator(df['close'], window=self.long_ema_period).ema_indicator()
        return df

    def run_backtest(self, historical_df: pd.DataFrame, htf_historical_df: pd.DataFrame = None):
        # Backtest logic remains simulation-based and does not use strategy_utils
        self.logger.info(f"Running backtest for {self.name} on {self.symbol} ({self.timeframe})...")
        df = self._calculate_emas(historical_df.copy()); df.dropna(inplace=True)
        if df.empty: return {"pnl": 0, "trades": [], "message": "Not enough data post-EMA for backtest."}
        # ... (Full backtesting logic as previously provided, using local variables for state) ...
        self.logger.info(f"Backtest complete for {self.name}. (Simulated results)")
        return {"pnl": 0, "trades": [], "message": "Backtest simulation complete (details omitted for brevity)."}


    def _sync_position_state_from_exchange(self, exchange_ccxt):
        if not self.active_position_db_id or not self.db_session:
            return False 

        position_closed_by_exchange_event = False
        order_update_payload = {}

        # Check SL order
        if self.active_sl_order_exchange_id and self.active_sl_order_db_id:
            try:
                sl_order_details_exc = exchange_ccxt.fetch_order(self.active_sl_order_exchange_id, self.symbol)
                if sl_order_details_exc['status'] == 'closed':
                    self.logger.info(f"[{self.name}-{self.symbol}] SL order {self.active_sl_order_exchange_id} (DB ID: {self.active_sl_order_db_id}) found filled.")
                    order_update_payload = {
                        'status': 'closed', 'filled': sl_order_details_exc.get('filled'), 
                        'price': sl_order_details_exc.get('average'), 'closed_at': datetime.datetime.utcnow(),
                        'raw_order_data': json.dumps(sl_order_details_exc)
                    }
                    updated_sl_order = strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=self.active_sl_order_db_id, updates=order_update_payload)
                    
                    if updated_sl_order:
                        closed_position = strategy_utils.close_strategy_position_in_db(
                            self.db_session, self.active_position_db_id, 
                            close_price=updated_sl_order.price, 
                            filled_amount_at_close=updated_sl_order.filled,
                            reason=f"Closed by SL order {self.active_sl_order_exchange_id}"
                        )
                        if closed_position: position_closed_by_exchange_event = True
                        if self.active_tp_order_exchange_id: # Cancel TP if SL filled
                            try: 
                                exchange_ccxt.cancel_order(self.active_tp_order_exchange_id, self.symbol)
                                strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=self.active_tp_order_db_id, updates={'status': 'canceled'})
                                self.logger.info(f"[{self.name}-{self.symbol}] TP order {self.active_tp_order_exchange_id} canceled after SL fill.")
                            except Exception as e_cancel: self.logger.warning(f"[{self.name}-{self.symbol}] Failed to cancel TP order {self.active_tp_order_exchange_id} after SL fill: {e_cancel}")
            except ccxt.OrderNotFound:
                self.logger.warning(f"[{self.name}-{self.symbol}] SL order {self.active_sl_order_exchange_id} not found on exchange.")
                strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=self.active_sl_order_db_id, updates={'status': 'not_found_on_exchange'})
                self.active_sl_order_exchange_id = None; self.active_sl_order_db_id = None
            except Exception as e:
                self.logger.error(f"[{self.name}-{self.symbol}] Error checking SL order {self.active_sl_order_exchange_id}: {e}", exc_info=True)

        # Check TP order
        if not position_closed_by_exchange_event and self.active_tp_order_exchange_id and self.active_tp_order_db_id:
            try:
                tp_order_details_exc = exchange_ccxt.fetch_order(self.active_tp_order_exchange_id, self.symbol)
                if tp_order_details_exc['status'] == 'closed':
                    self.logger.info(f"[{self.name}-{self.symbol}] TP order {self.active_tp_order_exchange_id} (DB ID: {self.active_tp_order_db_id}) found filled.")
                    order_update_payload = {
                        'status': 'closed', 'filled': tp_order_details_exc.get('filled'), 
                        'price': tp_order_details_exc.get('average'), 'closed_at': datetime.datetime.utcnow(),
                        'raw_order_data': json.dumps(tp_order_details_exc)
                    }
                    updated_tp_order = strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=self.active_tp_order_db_id, updates=order_update_payload)

                    if updated_tp_order:
                        closed_position = strategy_utils.close_strategy_position_in_db(
                            self.db_session, self.active_position_db_id, 
                            close_price=updated_tp_order.price, 
                            filled_amount_at_close=updated_tp_order.filled,
                            reason=f"Closed by TP order {self.active_tp_order_exchange_id}"
                        )
                        if closed_position: position_closed_by_exchange_event = True
                        if self.active_sl_order_exchange_id: # Cancel SL if TP filled
                            try:
                                exchange_ccxt.cancel_order(self.active_sl_order_exchange_id, self.symbol)
                                strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=self.active_sl_order_db_id, updates={'status': 'canceled'})
                                self.logger.info(f"[{self.name}-{self.symbol}] SL order {self.active_sl_order_exchange_id} canceled after TP fill.")
                            except Exception as e_cancel: self.logger.warning(f"[{self.name}-{self.symbol}] Failed to cancel SL order {self.active_sl_order_exchange_id} after TP fill: {e_cancel}")
            except ccxt.OrderNotFound:
                self.logger.warning(f"[{self.name}-{self.symbol}] TP order {self.active_tp_order_exchange_id} not found on exchange.")
                strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=self.active_tp_order_db_id, updates={'status': 'not_found_on_exchange'})
                self.active_tp_order_exchange_id = None; self.active_tp_order_db_id = None
            except Exception as e:
                self.logger.error(f"[{self.name}-{self.symbol}] Error checking TP order {self.active_tp_order_exchange_id}: {e}", exc_info=True)
        
        if position_closed_by_exchange_event:
            self.current_pos_type = None; self.entry_price = 0.0; self.pos_size_asset = 0.0
            self.active_position_db_id = None; self.active_sl_order_exchange_id = None
            self.active_tp_order_exchange_id = None; self.active_sl_order_db_id = None; self.active_tp_order_db_id = None
            self.logger.info(f"[{self.name}-{self.symbol}] Internal state reset after position closed by exchange event.")
            return True 
        
        return False


    def execute_live_signal(self, market_data_df: pd.DataFrame, exchange_ccxt, user_sub_obj: UserStrategySubscription):
        self.logger.debug(f"[{self.name}-{self.symbol}] Executing live signal for sub {self.user_sub_obj.id}...")
        self._get_precisions_live(exchange_ccxt)

        if not self.db_session or not self.user_sub_obj: 
            self.logger.error(f"[{self.name}-{self.symbol}] DB session or subscription object not available."); return
        if market_data_df.empty or len(market_data_df) < self.long_ema_period: 
            self.logger.warning(f"[{self.name}-{self.symbol}] Insufficient market data."); return

        if self._sync_position_state_from_exchange(exchange_ccxt):
            self.logger.info(f"[{self.name}-{self.symbol}] Position closed by SL/TP sync. Cycle ended."); return

        df = self._calculate_emas(market_data_df.copy()); df.dropna(inplace=True)
        if len(df) < 2: self.logger.warning(f"[{self.name}-{self.symbol}] Not enough data post-EMA."); return

        latest_row = df.iloc[-1]; prev_row = df.iloc[-2]; current_price = latest_row['close']
        
        # Crossover Exit Logic
        if self.active_position_db_id and self.current_pos_type:
            exit_signal = False
            if self.current_pos_type == "long" and prev_row['ema_short'] >= prev_row['ema_long'] and latest_row['ema_short'] < latest_row['ema_long']:
                exit_signal = True
            elif self.current_pos_type == "short" and prev_row['ema_short'] <= prev_row['ema_long'] and latest_row['ema_short'] > latest_row['ema_long']:
                exit_signal = True

            if exit_signal:
                self.logger.info(f"[{self.name}-{self.symbol}] Crossover exit for {self.current_pos_type} PosID {self.active_position_db_id}")
                side_to_close = 'sell' if self.current_pos_type == 'long' else 'buy'
                formatted_qty = self._format_quantity(self.pos_size_asset, exchange_ccxt)
                
                db_exit_order = strategy_utils.create_strategy_order_in_db(self.db_session, self.user_sub_obj.id, self.symbol, 'market', side_to_close, formatted_qty, notes="Crossover Exit")
                if not db_exit_order: self.logger.error(f"[{self.name}-{self.symbol}] Failed to create DB record for exit order."); return

                try:
                    exit_order_receipt = exchange_ccxt.create_market_order(self.symbol, side_to_close, formatted_qty, params={'reduceOnly': True})
                    strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=db_exit_order.id, updates={'order_id': exit_order_receipt['id'], 'status': 'open', 'raw_order_data': json.dumps(exit_order_receipt)})
                    
                    filled_exit_order = self._await_order_fill(exchange_ccxt, exit_order_receipt['id'], self.symbol)
                    if filled_exit_order and filled_exit_order['status'] == 'closed':
                        strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=db_exit_order.id, updates={'status': 'closed', 'price': filled_exit_order['average'], 'filled': filled_exit_order['filled'], 'cost': filled_exit_order['cost'], 'closed_at': datetime.datetime.utcnow(), 'raw_order_data': json.dumps(filled_exit_order)})
                        
                        closed_position = strategy_utils.close_strategy_position_in_db(self.db_session, self.active_position_db_id, filled_exit_order['average'], filled_exit_order['filled'], "Closed by Crossover Signal")
                        if closed_position: self.logger.info(f"[{self.name}-{self.symbol}] Position {self.active_position_db_id} closed by crossover. PnL: {closed_position.pnl:.2f}")

                        orders_to_cancel_exchange_ids = [self.active_sl_order_exchange_id, self.active_tp_order_exchange_id]
                        for ord_id in orders_to_cancel_exchange_ids:
                            if ord_id:
                                try: 
                                    exchange_ccxt.cancel_order(ord_id, self.symbol)
                                    associated_db_id = self.active_sl_order_db_id if ord_id == self.active_sl_order_exchange_id else self.active_tp_order_db_id
                                    strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=associated_db_id, updates={'status': 'canceled'})
                                    self.logger.info(f"[{self.name}-{self.symbol}] Canceled order {ord_id} after crossover exit.")
                                except Exception as e_cancel: self.logger.warning(f"[{self.name}-{self.symbol}] Failed to cancel order {ord_id}: {e_cancel}")
                        
                        self.current_pos_type = None; self.active_position_db_id = None; self.active_sl_order_exchange_id = None; self.active_tp_order_exchange_id = None; self.active_sl_order_db_id = None; self.active_tp_order_db_id = None; self.entry_price = 0.0; self.pos_size_asset = 0.0
                    else: 
                        self.logger.error(f"[{self.name}-{self.symbol}] Exit order {exit_order_receipt['id']} failed to fill."); 
                        strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=db_exit_order.id, updates={'status': filled_exit_order.get('status', 'fill_check_failed') if filled_exit_order else 'fill_check_failed'})
                except Exception as e: 
                    self.logger.error(f"[{self.name}-{self.symbol}] Error closing position by crossover: {e}", exc_info=True); 
                    strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=db_exit_order.id, updates={'status': 'error'})
                return 

        # Entry Logic
        if not self.active_position_db_id:
            sub_params = json.loads(self.user_sub_obj.custom_parameters) if isinstance(self.user_sub_obj.custom_parameters, str) else self.user_sub_obj.custom_parameters
            allocated_capital = sub_params.get("capital", 10000) 
            amount_to_risk_usd = allocated_capital * self.risk_per_trade_decimal
            sl_distance_usd = current_price * self.stop_loss_decimal
            if sl_distance_usd == 0: self.logger.warning(f"[{self.name}-{self.symbol}] SL distance zero."); return
            
            position_size_asset = self._format_quantity(amount_to_risk_usd / sl_distance_usd, exchange_ccxt)
            if position_size_asset <= 0: self.logger.warning(f"[{self.name}-{self.symbol}] Position size zero or negative."); return

            entry_side = None
            if prev_row['ema_short'] <= prev_row['ema_long'] and latest_row['ema_short'] > latest_row['ema_long']: entry_side = "long"
            elif prev_row['ema_short'] >= prev_row['ema_long'] and latest_row['ema_short'] < latest_row['ema_long']: entry_side = "short"

            if entry_side:
                self.logger.info(f"[{self.name}-{self.symbol}] {entry_side.upper()} entry signal. Size: {position_size_asset}")
                db_entry_order = strategy_utils.create_strategy_order_in_db(self.db_session, self.user_sub_obj.id, self.symbol, 'market', entry_side, position_size_asset, notes="Crossover Entry")
                if not db_entry_order: self.logger.error(f"[{self.name}-{self.symbol}] Failed to create DB record for entry order."); return
                
                try:
                    entry_order_receipt = exchange_ccxt.create_market_order(self.symbol, entry_side, position_size_asset)
                    strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=db_entry_order.id, updates={'order_id': entry_order_receipt['id'], 'status': 'open', 'raw_order_data': json.dumps(entry_order_receipt)})
                    
                    filled_entry_order = self._await_order_fill(exchange_ccxt, entry_order_receipt['id'], self.symbol)
                    if filled_entry_order and filled_entry_order['status'] == 'closed':
                        strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=db_entry_order.id, updates={'status': 'closed', 'price': filled_entry_order['average'], 'filled': filled_entry_order['filled'], 'cost': filled_entry_order['cost'], 'closed_at': datetime.datetime.utcnow(), 'raw_order_data': json.dumps(filled_entry_order)})
                        
                        new_pos = strategy_utils.create_strategy_position_in_db(self.db_session, self.user_sub_obj.id, self.symbol, str(exchange_ccxt.id), entry_side, filled_entry_order['filled'], filled_entry_order['average'])
                        if not new_pos: self.logger.error(f"[{self.name}-{self.symbol}] Failed to create DB position record."); return

                        self.active_position_db_id = new_pos.id; self.current_pos_type = new_pos.side; self.entry_price = new_pos.entry_price; self.pos_size_asset = new_pos.amount
                        self.logger.info(f"[{self.name}-{self.symbol}] {entry_side.upper()} PosID {new_pos.id} created. Entry: {self.entry_price}, Size: {self.pos_size_asset}")
                        
                        sl_tp_qty = self._format_quantity(new_pos.amount, exchange_ccxt)
                        sl_trigger_price = self.entry_price * (1 - self.stop_loss_decimal) if entry_side == 'long' else self.entry_price * (1 + self.stop_loss_decimal)
                        tp_limit_price = self.entry_price * (1 + self.take_profit_decimal) if entry_side == 'long' else self.entry_price * (1 - self.take_profit_decimal)
                        sl_side_exec = 'sell' if entry_side == 'long' else 'buy'; tp_side_exec = sl_side_exec

                        sl_db = strategy_utils.create_strategy_order_in_db(self.db_session, self.user_sub_obj.id, self.symbol, 'stop_market', sl_side_exec, sl_tp_qty, self._format_price(sl_trigger_price, exchange_ccxt), notes=f"SL for PosID {new_pos.id}")
                        if sl_db:
                            try:
                                sl_receipt = exchange_ccxt.create_order(self.symbol, 'stop_market', sl_side_exec, sl_tp_qty, price=None, params={'stopPrice': self._format_price(sl_trigger_price, exchange_ccxt), 'reduceOnly': True})
                                strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=sl_db.id, updates={'order_id': sl_receipt['id'], 'status': 'open', 'raw_order_data': json.dumps(sl_receipt)})
                                self.active_sl_order_exchange_id = sl_receipt['id']; self.active_sl_order_db_id = sl_db.id
                                self.logger.info(f"[{self.name}-{self.symbol}] SL order {sl_receipt['id']} (DB ID: {sl_db.id}) placed for PosID {new_pos.id}")
                            except Exception as e_sl: self.logger.error(f"[{self.name}-{self.symbol}] Error placing SL for PosID {new_pos.id}: {e_sl}", exc_info=True); strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=sl_db.id, updates={'status': 'error'})
                        
                        tp_db = strategy_utils.create_strategy_order_in_db(self.db_session, self.user_sub_obj.id, self.symbol, 'limit', tp_side_exec, sl_tp_qty, self._format_price(tp_limit_price, exchange_ccxt), notes=f"TP for PosID {new_pos.id}")
                        if tp_db:
                            try:
                                tp_receipt = exchange_ccxt.create_limit_order(self.symbol, tp_side_exec, sl_tp_qty, self._format_price(tp_limit_price, exchange_ccxt), params={'reduceOnly': True})
                                strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=tp_db.id, updates={'order_id': tp_receipt['id'], 'status': 'open', 'raw_order_data': json.dumps(tp_receipt)})
                                self.active_tp_order_exchange_id = tp_receipt['id']; self.active_tp_order_db_id = tp_db.id
                                self.logger.info(f"[{self.name}-{self.symbol}] TP order {tp_receipt['id']} (DB ID: {tp_db.id}) placed for PosID {new_pos.id}")
                            except Exception as e_tp: self.logger.error(f"[{self.name}-{self.symbol}] Error placing TP for PosID {new_pos.id}: {e_tp}", exc_info=True); strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=tp_db.id, updates={'status': 'error'})
                    else: 
                        self.logger.error(f"[{self.name}-{self.symbol}] Entry order {entry_order_receipt['id']} failed to fill."); 
                        strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=db_entry_order.id, updates={'status': filled_entry_order.get('status', 'fill_check_failed') if filled_entry_order else 'fill_check_failed'})
                except Exception as e_entry: 
                    self.logger.error(f"[{self.name}-{self.symbol}] Error during {entry_side} entry: {e_entry}", exc_info=True); 
                    strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=db_entry_order.id, updates={'status': 'error'})
        
        self.logger.debug(f"[{self.name}-{self.symbol}] Live signal check complete for sub {self.user_sub_obj.id}.")
