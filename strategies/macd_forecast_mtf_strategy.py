import logging
import numpy as np
import talib
import pandas as pd # Corrected import
from decimal import Decimal, ROUND_DOWN, ROUND_UP, ROUND_NEAREST, getcontext # Added ROUND_NEAREST
import json 
from datetime import datetime 
from typing import Optional, Dict, Any, List 

from sqlalchemy.orm import Session
from backend.models import Position, Order, UserStrategySubscription
from backend import strategy_utils

# Ensure high precision for Decimal calculations
getcontext().prec = 18

class MACDForecastMTFStrategy:

    @staticmethod
    def get_parameters_definition():
        return {
            "trading_pair": {"type": "str", "label": "Trading Pair", "default": "BTC/USDT"},
            "htf_interval": {"type": "str", "label": "Higher Timeframe (Trend)", "default": "4h", "choices": ["1h", "2h", "4h", "6h", "12h", "1d"]},
            "chart_interval": {"type": "str", "label": "Chart Kline Interval", "default": "15m", "choices": ["1m", "3m", "5m", "15m", "30m", "1h"]},
            "macd_fast_len": {"type": "int", "label": "MACD Fast Length", "default": 12, "min": 2},
            "macd_slow_len": {"type": "int", "label": "MACD Slow Length", "default": 26, "min": 2},
            "macd_signal_len": {"type": "int", "label": "MACD Signal Length", "default": 9, "min": 2},
            "macd_trend_determination": {"type": "str", "label": "MACD Trend Determination", "default": "MACD vs Signal", "choices": ["MACD vs Zero", "MACD vs Signal"]},
            "order_quantity_usd": {"type": "float", "label": "Order Quantity (USD Notional)", "default": 100.0, "min": 1.0},
            "leverage": {"type": "int", "label": "Leverage", "default": 10, "min": 1, "max": 100},
            "use_stop_loss": {"type": "bool", "label": "Use Stop Loss", "default": True},
            "stop_loss_pct": {"type": "float", "label": "Stop Loss % (from entry)", "default": 2.0, "min": 0.1, "step": 0.1},
            "use_take_profit": {"type": "bool", "label": "Use Take Profit", "default": True},
            "take_profit_pct": {"type": "float", "label": "Take Profit % (from entry)", "default": 4.0, "min": 0.1, "step": 0.1},
            "forecast_max_memory": {"type": "int", "label": "Forecast Max Memory (per bar index)", "default": 50, "min": 2}, # Will be ignored in backtest
            "forecast_length_bars": {"type": "int", "label": "Forecast Projection Length (bars)", "default": 100, "min": 1}, # Will be ignored in backtest
            "forecast_upper_percentile": {"type": "int", "label": "Forecast Upper Percentile", "default": 80, "min": 51, "max": 99}, # Will be ignored in backtest
            "forecast_mid_percentile": {"type": "int", "label": "Forecast Mid Percentile", "default": 50, "min": 1, "max": 99}, # Will be ignored in backtest
            "forecast_lower_percentile": {"type": "int", "label": "Forecast Lower Percentile", "default": 20, "min": 1, "max": 49}, # Will be ignored in backtest
            # New parameters
            "order_fill_max_retries": {"type": "int", "default": 5, "min": 1, "max": 20, "label": "Order Fill Max Retries"},
            "order_fill_delay_seconds": {"type": "int", "default": 2, "min": 1, "max": 10, "label": "Order Fill Delay (s)"},
            "data_fetch_buffer": {"type": "int", "default": 50, "min": 0, "max": 200, "label": "Data Fetch Buffer (candles)"},
        }

    def __init__(self, db_session: Optional[Session], user_sub_obj: Optional[UserStrategySubscription], strategy_params: dict, exchange_ccxt: Optional[Any], logger_obj=None):
        self.db_session = db_session
        self.user_sub_obj = user_sub_obj
        self.params = strategy_params
        self.exchange_ccxt = exchange_ccxt
        self.logger = logger_obj if logger_obj else logging.getLogger(__name__)
        self.name = "MACDForecastMTFStrategy"
        
        # Initialize params for both live and backtest scenarios
        self._initialize_parameters(strategy_params)

        # Live trading specific initializations
        if self.db_session and self.user_sub_obj and self.exchange_ccxt:
            self.capital_param = Decimal(str(self.params.get("capital", "10000"))) 
            self.active_position_db_id: Optional[int] = None
            self.forecast_memory: Dict[int, List[List[float]]] = {1: [], 0: []} 
            self.current_uptrend_idx: int = 0
            self.current_downtrend_idx: int = 0
            self.current_uptrend_init_price: Decimal = Decimal("0")
            self.current_downtrend_init_price: Decimal = Decimal("0")
            self.is_prev_chart_uptrend: Optional[bool] = None
            self.active_position_side: Optional[str] = None
            self.position_entry_price: Optional[Decimal] = None
            self.position_qty: Decimal = Decimal("0")
            self.active_sl_tp_orders: Dict[str, Optional[str]] = {}

            self._fetch_market_precision()
            self._load_strategy_and_position_state()
            self._set_leverage()
            self.logger.info(f"{self.name} initialized for LIVE trading {self.trading_pair}, UserSubID {self.user_sub_obj.id}. Loaded state for PosID: {self.active_position_db_id}")
        else:
            # For backtesting or scenarios where DB/exchange is not available
            self.logger.info(f"{self.name} initialized for BACKTESTING or without DB/Exchange context.")
            # Backtesting will manage its own state locally within the run_backtest method

    def _initialize_parameters(self, params_override: dict):
        """Helper to set parameters from defaults or overrides."""
        defaults = self.get_parameters_definition()
        current_params = {k: v['default'] for k, v in defaults.items()}
        current_params.update(params_override) # Apply overrides

        self.trading_pair = current_params.get("trading_pair")
        self.htf_interval = current_params.get("htf_interval")
        self.chart_interval = current_params.get("chart_interval")
        self.macd_fast_len = int(current_params.get("macd_fast_len"))
        self.macd_slow_len = int(current_params.get("macd_slow_len"))
        self.macd_signal_len = int(current_params.get("macd_signal_len"))
        self.macd_trend_determination = current_params.get("macd_trend_determination")
        
        self.order_quantity_usd = Decimal(str(current_params.get("order_quantity_usd")))
        self.leverage = int(current_params.get("leverage"))
        self.use_stop_loss = current_params.get("use_stop_loss")
        self.stop_loss_pct = Decimal(str(current_params.get("stop_loss_pct"))) / Decimal("100")
        self.use_take_profit = current_params.get("use_take_profit")
        self.take_profit_pct = Decimal(str(current_params.get("take_profit_pct"))) / Decimal("100")

        # Forecast params (used in live, ignored in this backtest version)
        self.forecast_max_memory = int(current_params.get("forecast_max_memory", 50))
        self.forecast_length_bars = int(current_params.get("forecast_length_bars", 100))
        self.forecast_upper_percentile = int(current_params.get("forecast_upper_percentile", 80))
        self.forecast_mid_percentile = int(current_params.get("forecast_mid_percentile", 50))
        self.forecast_lower_percentile = int(current_params.get("forecast_lower_percentile", 20))
        
        # Store merged params if needed for other parts of the class
        self.params = current_params
        
        # Initialize new parameters from current_params (which includes defaults and overrides)
        self.order_fill_max_retries = int(current_params.get("order_fill_max_retries"))
        self.order_fill_delay_seconds = int(current_params.get("order_fill_delay_seconds"))
        self.data_fetch_buffer = int(current_params.get("data_fetch_buffer"))


    @classmethod
    def validate_parameters(cls, params: dict) -> dict:
        """Validates strategy-specific parameters."""
        definition = cls.get_parameters_definition()
        validated_params = {}
        _logger = logging.getLogger(__name__) # Use a logger instance

        for key, def_value in definition.items():
            val_type_str = def_value.get("type")
            choices = def_value.get("choices") # Corrected from "options"
            min_val = def_value.get("min")
            max_val = def_value.get("max")
            default_val = def_value.get("default")

            user_val = params.get(key)

            if user_val is None: # Parameter not provided by user
                if default_val is not None:
                    user_val = default_val # Apply default
                else:
                    # This implies a required parameter (no default) is missing.
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
            elif val_type_str == "str": # Handles "str" and interval types like "1h", "15m"
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
                # This strategy's __init__ takes strategy_params and applies them via _initialize_parameters.
                # So, any key in params not in definition is an unknown parameter.
                _logger.warning(f"Unknown parameter '{key_param}' provided for {cls.__name__}. It will be ignored.")
                # To be strict: raise ValueError(f"Unknown parameter '{key_param}' provided for {cls.__name__}.")
        
        return validated_params

    def _load_strategy_and_position_state(self):
        if not self.db_session or not self.user_sub_obj:
            self.logger.warning(f"[{self.name}-{self.trading_pair}] DB session or user_sub_obj not available. Cannot load state.")
            return

        position = strategy_utils.get_open_strategy_position(self.db_session, self.user_sub_obj.id, self.trading_pair)
        if position:
            self.logger.info(f"[{self.name}-{self.trading_pair}] Loading persistent state for open position ID: {position.id}")
            self.active_position_db_id = position.id
            self.active_position_side = position.side
            self.position_entry_price = Decimal(str(position.entry_price)) if position.entry_price is not None else None
            self.position_qty = Decimal(str(position.amount)) if position.amount is not None else Decimal("0")

            if position.custom_data:
                try:
                    state = json.loads(position.custom_data)
                    fm_loaded = state.get("forecast_memory", {1: [], 0: []})
                    self.forecast_memory = {int(k): v for k, v in fm_loaded.items()} if isinstance(fm_loaded, dict) else {1: [], 0: []}
                    self.current_uptrend_idx = state.get("current_uptrend_idx", 0)
                    self.current_downtrend_idx = state.get("current_downtrend_idx", 0)
                    self.current_uptrend_init_price = Decimal(state.get("current_uptrend_init_price", "0"))
                    self.current_downtrend_init_price = Decimal(state.get("current_downtrend_init_price", "0"))
                    self.active_sl_tp_orders = state.get("active_sl_tp_orders", {})
                    self.is_prev_chart_uptrend = state.get("is_prev_chart_uptrend")
                    self.logger.info(f"[{self.name}-{self.trading_pair}] Loaded custom state from DB: {state}")
                except json.JSONDecodeError as e:
                    self.logger.error(f"[{self.name}-{self.trading_pair}] Error parsing custom_data JSON for pos {position.id}: {e}. Starting with default strategy state.")
                    self._initialize_default_strategy_state()
            else:
                self.logger.info(f"[{self.name}-{self.trading_pair}] No custom_data found for pos {position.id}. Initializing default strategy state.")
                self._initialize_default_strategy_state()
        else:
            self.logger.info(f"[{self.name}-{self.trading_pair}] No active persistent position found. Initializing all states to default.")
            self.active_position_db_id = None
            self._initialize_default_strategy_state()
            self.active_position_side = None
            self.position_entry_price = None
            self.position_qty = Decimal("0")

    def _initialize_default_strategy_state(self):
        self.forecast_memory = {1: [], 0: []}
        self.current_uptrend_idx = 0
        self.current_downtrend_idx = 0
        self.current_uptrend_init_price = Decimal("0")
        self.current_downtrend_init_price = Decimal("0")
        self.active_sl_tp_orders = {}
        self.is_prev_chart_uptrend = None

    def _save_persistent_state(self):
        if not self.db_session or not self.active_position_db_id:
            self.logger.debug(f"[{self.name}-{self.trading_pair}] DB session/active_pos_id not set. State not saved (normal if no position or backtesting).")
            return

        pos_to_update = self.db_session.query(Position).filter(Position.id == self.active_position_db_id).first()
        if not pos_to_update:
            self.logger.error(f"[{self.name}-{self.trading_pair}] Position with ID {self.active_position_db_id} not found in DB to save state.")
            return

        state_data = {
            "forecast_memory": self.forecast_memory,
            "current_uptrend_idx": self.current_uptrend_idx,
            "current_downtrend_idx": self.current_downtrend_idx,
            "current_uptrend_init_price": str(self.current_uptrend_init_price),
            "current_downtrend_init_price": str(self.current_downtrend_init_price),
            "active_sl_tp_orders": self.active_sl_tp_orders,
            "is_prev_chart_uptrend": self.is_prev_chart_uptrend
        }
        pos_to_update.custom_data = json.dumps(state_data)
        pos_to_update.updated_at = datetime.utcnow()
        try:
            self.db_session.commit()
            self.logger.info(f"[{self.name}-{self.trading_pair}] Successfully saved persistent state to Position ID {self.active_position_db_id}.")
        except Exception as e:
            self.logger.error(f"[{self.name}-{self.trading_pair}] Error saving persistent state for Position ID {self.active_position_db_id}: {e}", exc_info=True)
            self.db_session.rollback()

    def _fetch_market_precision(self):
        try:
            self.exchange_ccxt.load_markets()
            market = self.exchange_ccxt.markets[self.trading_pair]
            self.quantity_precision_str = str(market['precision']['amount'])
            self.price_precision_str = str(market['precision']['price'])
        except Exception as e:
            self.logger.error(f"Error fetching precision for {self.trading_pair}: {e}. Using defaults.")
            self.quantity_precision_str = "0.00001" 
            self.price_precision_str = "0.01"

    def _get_decimal_places(self, precision_str: str) -> int:
        if precision_str is None: return 8 
        try:
            if 'e-' in precision_str.lower():
                return abs(int(precision_str.split("e-")[1]))
            d_prec = Decimal(precision_str)
            return abs(d_prec.as_tuple().exponent) if d_prec.as_tuple().exponent < 0 else 0
        except Exception: return 8

    def _format_quantity(self, quantity: Decimal) -> str:
        places = self._get_decimal_places(self.quantity_precision_str)
        return str(quantity.quantize(Decimal('1e-' + str(places)), rounding=ROUND_DOWN))

    def _format_price(self, price: Decimal) -> str:
        places = self._get_decimal_places(self.price_precision_str)
        return str(price.quantize(Decimal('1e-' + str(places)), rounding=ROUND_NEAREST))

    def _set_leverage(self):
        try:
            if hasattr(self.exchange_ccxt, 'set_leverage'):
                symbol_for_leverage = self.trading_pair.split(':')[0] if ':' in self.trading_pair else self.trading_pair
                self.exchange_ccxt.set_leverage(self.leverage, symbol_for_leverage)
                self.logger.info(f"Leverage set to {self.leverage}x for {symbol_for_leverage}")
        except Exception as e: self.logger.warning(f"Could not set leverage for {self.trading_pair}: {e}")

    def _get_ohlcv_data(self, interval, limit):
        return self.exchange_ccxt.fetch_ohlcv(self.trading_pair, interval, limit=limit)

    def _calculate_macd_values(self, ohlcv_df_or_series, fast_len=None, slow_len=None, signal_len=None):
        # Allow overriding MACD params for flexibility, e.g. if backtest params differ
        fast = fast_len if fast_len else self.macd_fast_len
        slow = slow_len if slow_len else self.macd_slow_len
        signal = signal_len if signal_len else self.macd_signal_len

        if isinstance(ohlcv_df_or_series, pd.DataFrame):
            close_prices = ohlcv_df_or_series['close'].to_numpy(dtype=float)
        elif isinstance(ohlcv_df_or_series, pd.Series):
            close_prices = ohlcv_df_or_series.to_numpy(dtype=float)
        else:
            self.logger.error("Invalid data type for MACD calculation. Expected DataFrame or Series.")
            return None, None, None
            
        if len(close_prices) < slow: return None, None, None
        macd, signal_line, hist = talib.MACD(close_prices, fastperiod=fast, slowperiod=slow, signalperiod=signal)
        return macd, signal_line, hist

    def _determine_trend(self, macd_value, signal_value):
        if self.macd_trend_determination == "MACD vs Zero":
            is_bullish = macd_value > 0; is_bearish = macd_value < 0
        else: 
            is_bullish = macd_value > signal_value; is_bearish = macd_value < signal_value
        return is_bullish, is_bearish

    def _populate_memory(self, current_trend_type, current_trend_bar_idx, current_trend_init_price, current_close_price):
        price_deviation = float(current_close_price - current_trend_init_price)
        if current_trend_type not in self.forecast_memory: self.forecast_memory[current_trend_type] = []
        while len(self.forecast_memory[current_trend_type]) <= current_trend_bar_idx: self.forecast_memory[current_trend_type].append([])
        self.forecast_memory[current_trend_type][current_trend_bar_idx].insert(0, price_deviation)
        if len(self.forecast_memory[current_trend_type][current_trend_bar_idx]) > self.forecast_max_memory:
            self.forecast_memory[current_trend_type][current_trend_bar_idx].pop()

    def _calculate_forecast_bands(self, trend_about_to_start_type, new_trend_init_price):
        forecast_bands = []
        if trend_about_to_start_type not in self.forecast_memory or not self.forecast_memory[trend_about_to_start_type]:
            return forecast_bands
        historical_segments = self.forecast_memory[trend_about_to_start_type]
        for bar_offset in range(self.forecast_length_bars):
            if bar_offset < len(historical_segments):
                deviations = historical_segments[bar_offset]
                if len(deviations) > 1:
                    forecast_bands.append({
                        'bar_offset': bar_offset,
                        'lower': float(new_trend_init_price + Decimal(str(np.percentile(deviations, self.forecast_lower_percentile)))),
                        'mid': float(new_trend_init_price + Decimal(str(np.percentile(deviations, self.forecast_mid_percentile)))),
                        'upper': float(new_trend_init_price + Decimal(str(np.percentile(deviations, self.forecast_upper_percentile))))
                    })
        return forecast_bands
    
    def _place_order(self, symbol:str, order_type:str, side:str, quantity:Decimal, price:Optional[Decimal]=None, params:Optional[Dict]=None) -> Optional[Dict]:
        # This is a LIVE trading method, uses DB and exchange
        if not self.db_session or not self.user_sub_obj or not self.exchange_ccxt:
            self.logger.error("Attempted to place live order without DB/Exchange context. Aborting.")
            return None
        # ... (rest of the _place_order method from original, unchanged)
        db_order = None
        try:
            formatted_qty_str = self._format_quantity(quantity)
            formatted_price_str = self._format_price(price) if price else None
            
            db_order = strategy_utils.create_strategy_order_in_db(
                self.db_session, self.user_sub_obj.id, symbol, order_type.lower(), side, 
                float(formatted_qty_str), float(formatted_price_str) if formatted_price_str else None, 
                status='pending_exchange', notes=f"Strategy {self.name}"
            )
            if not db_order: 
                self.logger.error(f"Failed to create initial DB order record for {side} {quantity} {symbol}. Aborting exchange order."); return None

            exchange_order = self.exchange_ccxt.create_order(symbol, order_type, side, float(formatted_qty_str), float(formatted_price_str) if formatted_price_str else None, params)
            
            strategy_utils.update_strategy_order_in_db(
                self.db_session, order_db_id=db_order.id, 
                updates={'order_id': exchange_order.get('id'), 'status': 'open', 'raw_order_data': json.dumps(exchange_order)}
            )
            self.logger.info(f"Order placed: {side} {order_type} {formatted_qty_str} {symbol} at {formatted_price_str if formatted_price_str else 'Market'}. ExchOrderID: {exchange_order.get('id')}, DBOrderID: {db_order.id}")
            return exchange_order
        except Exception as e:
            self.logger.error(f"Failed to place {side} {order_type} for {symbol}: {e}", exc_info=True)
            if db_order and db_order.id: 
                strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=db_order.id, updates={'status': 'error_on_exchange', 'status_message': str(e)[:255]})
        return None

    def _await_order_fill(self, exchange, order_id, symbol, max_retries=5, delay_seconds=2):
        # Simplified await logic for live trading context, using instance parameters
        for attempt in range(self.order_fill_max_retries):
            try:
                order = exchange.fetch_order(order_id, symbol)
                if order['status'] == 'closed':
                    return order
                elif order['status'] == 'canceled' or order['status'] == 'rejected':
                    self.logger.warning(f"Order {order_id} is {order['status']}. Not waiting further.")
                    return order
                self.logger.info(f"Order {order_id} status is {order['status']}. Attempt {attempt+1}/{self.order_fill_max_retries}. Waiting...")
                import time # Ensure time is imported if not already at top level
                time.sleep(self.order_fill_delay_seconds) 
            except Exception as e:
                self.logger.error(f"Error fetching order {order_id}: {e}. Attempt {attempt+1}/{self.order_fill_max_retries}.")
                if attempt == self.order_fill_max_retries - 1: return None # Failed to fetch after retries
        return None # Default if loop finishes

    def _cancel_active_sl_tp_orders_in_db(self):
        # This is a LIVE trading method
        if not self.db_session or not self.user_sub_obj or not self.exchange_ccxt: return
        # ... (rest of the method from original, unchanged)
        for order_key, exchange_order_id in self.active_sl_tp_orders.items():
            if exchange_order_id:
                try:
                    self.logger.info(f"[{self.name}-{self.trading_pair}] Canceling {order_key} order {exchange_order_id} on exchange.")
                    self.exchange_ccxt.cancel_order(exchange_order_id, self.trading_pair)
                    strategy_utils.update_strategy_order_in_db(
                        self.db_session, exchange_order_id=exchange_order_id, 
                        subscription_id=self.user_sub_obj.id, symbol=self.trading_pair, 
                        updates={'status': 'canceled'}
                    )
                except Exception as e: # Catch more specific errors like OrderNotFound
                    self.logger.error(f"[{self.name}-{self.trading_pair}] Failed to cancel {order_key} order {exchange_order_id}: {e}")
        self.active_sl_tp_orders = {}

    def _close_all_positions(self, current_market_price: Decimal):
        # This is a LIVE trading method
        if not self.db_session or not self.user_sub_obj or not self.exchange_ccxt: return
        # ... (rest of the method from original, unchanged, ensure _await_order_fill is available)
        if self.active_position_side and self.active_position_db_id:
            close_side = 'sell' if self.active_position_side == 'long' else 'buy'
            self.logger.info(f"Closing all positions for {self.trading_pair} (Side: {self.active_position_side}, Qty: {self.position_qty}). Market Price: {current_market_price}")
            close_order_receipt = self._place_order(self.trading_pair, 'market', close_side, self.position_qty, params={'reduceOnly': True})
            if close_order_receipt:
                filled_close_order = self._await_order_fill(self.exchange_ccxt, close_order_receipt['id'], self.trading_pair)
                if filled_close_order and filled_close_order['status'] == 'closed':
                    strategy_utils.update_strategy_order_in_db(self.db_session, exchange_order_id=filled_close_order['id'], subscription_id=self.user_sub_obj.id, symbol=self.trading_pair, updates={'status': 'closed', 'price': filled_close_order['average'], 'filled': filled_close_order['filled'], 'cost': filled_close_order['cost'], 'closed_at': datetime.utcnow()})
                    strategy_utils.close_strategy_position_in_db(self.db_session, self.active_position_db_id, Decimal(str(filled_close_order['average'])), Decimal(str(filled_close_order['filled'])), "Close All Signal Triggered")
                else: # Handle non-fill or error
                    self.logger.error(f"Market close order {close_order_receipt['id']} failed to fill or status unknown.")
                    # Update DB based on available info
            self._cancel_active_sl_tp_orders_in_db()
            self.active_position_side = None; self.position_entry_price = None; self.position_qty = Decimal("0"); self.active_position_db_id = None
            self._initialize_default_strategy_state(); self._save_persistent_state()
        else: self.logger.info(f"[{self.name}-{self.trading_pair}] No active position to close.")


    def _sync_exchange_position_state(self, current_price: Decimal):
        # This is a LIVE trading method
        if not self.db_session or not self.user_sub_obj or not self.exchange_ccxt or not self.active_position_db_id or not self.active_sl_tp_orders:
             return False
        # ... (rest of the method from original, unchanged)
        position_closed_by_exchange_event = False
        sl_id = self.active_sl_tp_orders.get('sl_id')
        tp_id = self.active_sl_tp_orders.get('tp_id')
        # SL Check
        if sl_id:
            try:
                sl_order_details = self.exchange_ccxt.fetch_order(sl_id, self.trading_pair)
                if sl_order_details['status'] == 'closed':
                    self.logger.info(f"SL order {sl_id} filled."); # DB updates
                    strategy_utils.close_strategy_position_in_db(self.db_session, self.active_position_db_id, Decimal(str(sl_order_details['average'])), Decimal(str(sl_order_details['filled'])), f"Closed by SL {sl_id}")
                    position_closed_by_exchange_event = True
                    if tp_id: self.exchange_ccxt.cancel_order(tp_id, self.trading_pair) # Cancel TP
            except Exception as e: self.logger.error(f"Error checking SL {sl_id}: {e}")
        # TP Check (if not closed by SL)
        if not position_closed_by_exchange_event and tp_id:
            try:
                tp_order_details = self.exchange_ccxt.fetch_order(tp_id, self.trading_pair)
                if tp_order_details['status'] == 'closed':
                    self.logger.info(f"TP order {tp_id} filled."); # DB updates
                    strategy_utils.close_strategy_position_in_db(self.db_session, self.active_position_db_id, Decimal(str(tp_order_details['average'])), Decimal(str(tp_order_details['filled'])), f"Closed by TP {tp_id}")
                    position_closed_by_exchange_event = True
                    if sl_id: self.exchange_ccxt.cancel_order(sl_id, self.trading_pair) # Cancel SL
            except Exception as e: self.logger.error(f"Error checking TP {tp_id}: {e}")

        if position_closed_by_exchange_event:
            self.active_position_side = None; self.position_entry_price = None; self.position_qty = Decimal("0")
            self.active_sl_tp_orders = {}; self.active_position_db_id = None
            self._initialize_default_strategy_state(); self._save_persistent_state()
            return True
        return False

    def _process_trading_logic(self, chart_ohlcv_df, htf_ohlcv_df):
        # This is a LIVE trading method
        if not self.db_session or not self.user_sub_obj or not self.exchange_ccxt: return
        # ... (rest of the method from original, largely unchanged but ensure it uses instance parameters like self.leverage etc.)
        # ... and ensure _await_order_fill is integrated after placing entry order
        chart_macd_vals, chart_signal_vals, _ = self._calculate_macd_values(chart_ohlcv_df)
        htf_macd_vals, htf_signal_vals, _ = self._calculate_macd_values(htf_ohlcv_df)

        if chart_macd_vals is None or htf_macd_vals is None or len(chart_macd_vals) < 2 or len(htf_macd_vals) < 1:
            self.logger.warning("Not enough data for MACD in _process_trading_logic."); return

        latest_chart_macd = chart_macd_vals[-1]; latest_chart_signal = chart_signal_vals[-1]
        prev_chart_macd = chart_macd_vals[-2]; prev_chart_signal = chart_signal_vals[-2]
        latest_htf_macd = htf_macd_vals[-1]; latest_htf_signal = htf_signal_vals[-1]

        is_chart_uptrend, is_chart_downtrend = self._determine_trend(latest_chart_macd, latest_chart_signal)
        is_htf_uptrend, _ = self._determine_trend(latest_htf_macd, latest_htf_signal) # HTF only for trend direction

        chart_trigger_up = prev_chart_macd <= prev_chart_signal and latest_chart_macd > latest_chart_signal
        chart_trigger_down = prev_chart_macd >= prev_chart_signal and latest_chart_macd < latest_chart_signal
        current_close_price = Decimal(str(chart_ohlcv_df['close'].iloc[-1]))

        # Forecasting logic (can be kept for live, but backtest will ignore)
        if is_chart_uptrend:
            if not self.is_prev_chart_uptrend:
                self.current_uptrend_init_price = current_close_price; self.current_uptrend_idx = 0
                if chart_trigger_up: self._calculate_forecast_bands(1, self.current_uptrend_init_price) # Trend type 1 for up
            else: self.current_uptrend_idx += 1
            self._populate_memory(1, self.current_uptrend_idx, self.current_uptrend_init_price, current_close_price)
        
        if is_chart_downtrend: # Corrected from `elif` to `if` to handle transitions correctly
            if self.is_prev_chart_uptrend is None or self.is_prev_chart_uptrend: # If changed from up or undefined
                self.current_downtrend_init_price = current_close_price; self.current_downtrend_idx = 0
                if chart_trigger_down: self._calculate_forecast_bands(0, self.current_downtrend_init_price) # Trend type 0 for down
            else: self.current_downtrend_idx += 1
            self._populate_memory(0, self.current_downtrend_idx, self.current_downtrend_init_price, current_close_price)
        self.is_prev_chart_uptrend = is_chart_uptrend


        long_condition = chart_trigger_up and is_chart_uptrend and is_htf_uptrend
        short_condition = chart_trigger_down and is_chart_downtrend and not is_htf_uptrend # Short only if HTF is not bullish (neutral or bearish)

        if (long_condition and self.active_position_side == 'short') or \
           (short_condition and self.active_position_side == 'long'):
            self._close_all_positions(current_close_price)
        
        action_taken_this_cycle = False
        if not self.active_position_side:
            entry_price = current_close_price
            if entry_price == Decimal("0"): return
            
            qty_to_trade = (self.order_quantity_usd / entry_price) # Base asset quantity, leverage applied by exchange

            if long_condition or short_condition:
                entry_side = "long" if long_condition else "short"
                entry_order_receipt = self._place_order(self.trading_pair, 'market', entry_side, qty_to_trade)
                if entry_order_receipt:
                    filled_entry_order = self._await_order_fill(self.exchange_ccxt, entry_order_receipt['id'], self.trading_pair)
                    if filled_entry_order and filled_entry_order['status'] == 'closed':
                        actual_entry_price = Decimal(str(filled_entry_order['average']))
                        actual_filled_qty = Decimal(str(filled_entry_order['filled']))
                        # DB updates for order and new position
                        new_pos_db = strategy_utils.create_strategy_position_in_db(self.db_session, self.user_sub_obj.id, self.trading_pair, str(self.exchange_ccxt.id), entry_side, float(actual_filled_qty), float(actual_entry_price))
                        if new_pos_db:
                            self.active_position_db_id = new_pos_db.id; self.active_position_side = entry_side
                            self.position_entry_price = actual_entry_price; self.position_qty = actual_filled_qty
                            # Place SL/TP orders
                            sl_price = None; tp_price = None
                            if self.use_stop_loss: sl_price = self.position_entry_price * (Decimal('1') - self.stop_loss_pct) if entry_side == 'long' else self.position_entry_price * (Decimal('1') + self.stop_loss_pct)
                            if self.use_take_profit: tp_price = self.position_entry_price * (Decimal('1') + self.take_profit_pct) if entry_side == 'long' else self.position_entry_price * (Decimal('1') - self.take_profit_pct)
                            if sl_price:
                                sl_ord = self._place_order(self.trading_pair, 'STOP_MARKET', 'sell' if entry_side == 'long' else 'buy', self.position_qty, params={'stopPrice': self._format_price(sl_price), 'reduceOnly': True})
                                if sl_ord: self.active_sl_tp_orders['sl_id'] = sl_ord.get('id')
                            if tp_price:
                                tp_ord = self._place_order(self.trading_pair, 'TAKE_PROFIT_MARKET', 'sell' if entry_side == 'long' else 'buy', self.position_qty, params={'stopPrice': self._format_price(tp_price), 'reduceOnly': True})
                                if tp_ord: self.active_sl_tp_orders['tp_id'] = tp_ord.get('id')
                            action_taken_this_cycle = True
                    else: # Handle failed entry fill
                        self.logger.error(f"Entry order {entry_order_receipt['id']} did not fill or status unknown.")
        
        if action_taken_this_cycle or self.current_uptrend_idx == 0 or self.current_downtrend_idx == 0:
            self._save_persistent_state()


    def execute_live_signal(self):
        self.logger.info(f"Executing {self.name} for {self.trading_pair} UserSubID {self.user_sub_obj.id}")
        if not self.db_session or not self.user_sub_obj or not self.exchange_ccxt:
            self.logger.error("Cannot execute live signal: DB session, user subscription, or exchange_ccxt not available.")
            return

        self._load_strategy_and_position_state() # Ensure fresh state

        try:
            current_ticker = self.exchange_ccxt.fetch_ticker(self.trading_pair)
            current_close_price = Decimal(str(current_ticker['last']))
            if self._sync_exchange_position_state(current_close_price):
                self.logger.info(f"[{self.name}-{self.trading_pair}] Position closed by SL/TP sync. Cycle ended.")
                return 
        except Exception as e:
            self.logger.error(f"[{self.name}-{self.trading_pair}] Error during pre-logic sync/price fetch: {e}", exc_info=True)
            return

        chart_limit = max(self.macd_slow_len, self.macd_signal_len) + self.forecast_max_memory + self.data_fetch_buffer 
        htf_limit = max(self.macd_slow_len, self.macd_signal_len) + self.data_fetch_buffer

        try:
            chart_ohlcv_list = self._get_ohlcv_data(self.chart_interval, chart_limit)
            htf_ohlcv_list = self._get_ohlcv_data(self.htf_interval, htf_limit)

            min_chart_len = max(self.macd_slow_len, self.macd_signal_len) + 2 
            min_htf_len = max(self.macd_slow_len, self.macd_signal_len) + 1

            if not chart_ohlcv_list or len(chart_ohlcv_list) < min_chart_len or \
               not htf_ohlcv_list or len(htf_ohlcv_list) < min_htf_len:
                self.logger.warning("Insufficient OHLCV for MACD calc in execute_live_signal.")
                return

            chart_ohlcv_df = pd.DataFrame(chart_ohlcv_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            chart_ohlcv_df['timestamp'] = pd.to_datetime(chart_ohlcv_df['timestamp'], unit='ms')
            
            htf_ohlcv_df = pd.DataFrame(htf_ohlcv_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            htf_ohlcv_df['timestamp'] = pd.to_datetime(htf_ohlcv_df['timestamp'], unit='ms')
            
            self._process_trading_logic(chart_ohlcv_df, htf_ohlcv_df)

        except Exception as e:
            self.logger.error(f"Error in {self.name} execute_live_signal main: {e}", exc_info=True)

    def run_backtest(self, ohlcv_data: pd.DataFrame, initial_capital: float, params: dict) -> Dict[str, Any]:
        self.logger.info(f"[{self.name}-{self.trading_pair}] Starting backtest...")
        self._initialize_parameters(params) # Apply backtest-specific parameters
        
        self.logger.warning("This MACDForecastMTFStrategy backtest is SIMPLIFIED and does NOT include the forecasting component.")

        # Ensure ohlcv_data (chart interval) has a datetime index and correct columns
        if not isinstance(ohlcv_data.index, pd.DatetimeIndex):
            ohlcv_data['timestamp'] = pd.to_datetime(ohlcv_data['timestamp'])
            ohlcv_data.set_index('timestamp', inplace=True)
        
        # Resample chart_interval data to HTF
        # Pandas resample rule needs to be derived from htf_interval string
        resample_rule_map = {'1m':'1T','3m':'3T','5m':'5T','15m':'15T','30m':'30T','1h':'1H','2h':'2H','4h':'4H','6h':'6H','12h':'12H','1d':'1D'}
        resample_rule = resample_rule_map.get(self.htf_interval, self.htf_interval) # Default to value if not in map (e.g. '1D')
        
        htf_ohlcv_df = ohlcv_data['close'].resample(resample_rule).ohlc().dropna()
        htf_ohlcv_df.columns = ['open', 'high', 'low', 'close'] # Ensure correct column names after resample

        # Calculate MACD for chart and HTF
        chart_macd, chart_signal, _ = self._calculate_macd_values(ohlcv_data['close'])
        htf_macd, htf_signal, _ = self._calculate_macd_values(htf_ohlcv_df['close'])

        if chart_macd is None or htf_macd is None:
            return {"pnl": 0, "trades_log": [], "equity_curve": [], "message": "Not enough data for MACD calculation."}

        # Align HTF data to chart data (forward fill)
        # Create full DataFrame for HTF MACD values aligned with chart_ohlcv_df's index
        htf_indicators = pd.DataFrame(index=htf_ohlcv_df.index)
        htf_indicators['htf_macd'] = htf_macd
        htf_indicators['htf_signal'] = htf_signal
        
        # Reindex and ffill, then merge. This ensures correct alignment.
        aligned_htf_indicators = htf_indicators.reindex(ohlcv_data.index, method='ffill')
        
        # Combine all data into a single DataFrame for easier iteration
        df = ohlcv_data.copy()
        df['chart_macd'] = chart_macd
        df['chart_signal'] = chart_signal
        df = df.merge(aligned_htf_indicators, left_index=True, right_index=True, how='left')

        # Initialize portfolio
        equity = Decimal(str(initial_capital))
        trades_log = []
        equity_curve = [{'timestamp': df.index[0].isoformat(), 'equity': float(equity)}] # Ensure df.index[0] is valid

        bt_position_side: Optional[str] = None
        bt_entry_price = Decimal("0")
        bt_position_qty = Decimal("0")
        bt_sl_price = Decimal("0")
        bt_tp_price = Decimal("0")

        # Loop through chart data
        start_index = max(self.macd_slow_len + self.macd_signal_len, 1) # Ensure previous bar exists
        start_index = df.index.get_loc(df.dropna(subset=['chart_macd', 'chart_signal', 'htf_macd', 'htf_signal']).index[0]) if len(df.dropna(subset=['chart_macd', 'chart_signal', 'htf_macd', 'htf_signal'])) > 0 else len(df)
        start_index = max(start_index, 1)


        for i in range(start_index, len(df)):
            current_bar = df.iloc[i]
            prev_bar = df.iloc[i-1]
            current_timestamp = current_bar.name.isoformat() # df.index is DatetimeIndex
            current_price = Decimal(str(current_bar['close']))
            current_low = Decimal(str(current_bar['low']))
            current_high = Decimal(str(current_bar['high']))

            # MACD values from combined DataFrame
            current_chart_macd = current_bar['chart_macd']; current_chart_signal = current_bar['chart_signal']
            prev_chart_macd = prev_bar['chart_macd']; prev_chart_signal = prev_bar['chart_signal']
            current_htf_macd = current_bar['htf_macd']; current_htf_signal = current_bar['htf_signal']

            if pd.isna(current_chart_macd) or pd.isna(current_chart_signal) or \
               pd.isna(prev_chart_macd) or pd.isna(prev_chart_signal) or \
               pd.isna(current_htf_macd) or pd.isna(current_htf_signal):
                equity_curve.append({'timestamp': current_timestamp, 'equity': float(equity)})
                continue

            is_chart_uptrend, _ = self._determine_trend(current_chart_macd, current_chart_signal)
            is_htf_bullish, _ = self._determine_trend(current_htf_macd, current_htf_signal)

            chart_trigger_up = prev_chart_macd <= prev_chart_signal and current_chart_macd > current_chart_signal
            chart_trigger_down = prev_chart_macd >= prev_chart_signal and current_chart_macd < current_chart_signal
            
            # Exit Logic
            if bt_position_side:
                exit_trade = False; exit_price = Decimal("0"); exit_reason = ""
                if bt_position_side == 'long':
                    if self.use_stop_loss and current_low <= bt_sl_price: exit_price = bt_sl_price; exit_reason = "SL"; exit_trade = True
                    elif self.use_take_profit and current_high >= bt_tp_price: exit_price = bt_tp_price; exit_reason = "TP"; exit_trade = True
                    elif chart_trigger_down: exit_price = current_price; exit_reason = "Chart MACD Reversal"; exit_trade = True
                elif bt_position_side == 'short':
                    if self.use_stop_loss and current_high >= bt_sl_price: exit_price = bt_sl_price; exit_reason = "SL"; exit_trade = True
                    elif self.use_take_profit and current_low <= bt_tp_price: exit_price = bt_tp_price; exit_reason = "TP"; exit_trade = True
                    elif chart_trigger_up: exit_price = current_price; exit_reason = "Chart MACD Reversal"; exit_trade = True
                
                if exit_trade:
                    pnl = (exit_price - bt_entry_price) * bt_position_qty if bt_position_side == 'long' else (bt_entry_price - exit_price) * bt_position_qty
                    pnl_leveraged = pnl * Decimal(str(self.leverage))
                    equity += pnl_leveraged
                    trades_log.append({
                        'timestamp': current_timestamp, 'type': 'exit ' + bt_position_side,
                        'price': float(exit_price), 'quantity': float(bt_position_qty),
                        'pnl_realized': float(pnl_leveraged), 'reason': exit_reason, 'equity': float(equity)
                    })
                    bt_position_side = None; bt_position_qty = Decimal("0")

            # Entry Logic
            if not bt_position_side:
                entry_signal: Optional[str] = None
                if chart_trigger_up and is_chart_uptrend and is_htf_bullish:
                    entry_signal = 'long'
                elif chart_trigger_down and not is_chart_uptrend and not is_htf_bullish: # enter short if chart is downtrend and htf is bearish
                    entry_signal = 'short'
                
                if entry_signal:
                    bt_entry_price = current_price
                    bt_position_qty = self.order_quantity_usd / bt_entry_price # Base asset quantity for notional USD
                    
                    if bt_position_qty > Decimal("0"):
                        bt_position_side = entry_signal
                        if self.use_stop_loss:
                            bt_sl_price = bt_entry_price * (Decimal('1') - self.stop_loss_pct) if entry_signal == 'long' else bt_entry_price * (Decimal('1') + self.stop_loss_pct)
                        if self.use_take_profit:
                            bt_tp_price = bt_entry_price * (Decimal('1') + self.take_profit_pct) if entry_signal == 'long' else bt_entry_price * (Decimal('1') - self.take_profit_pct)
                        
                        trades_log.append({
                            'timestamp': current_timestamp, 'type': 'entry ' + entry_signal,
                            'price': float(bt_entry_price), 'quantity': float(bt_position_qty),
                            'pnl_realized': 0.0, 'reason': 'MACD MTF Signal', 'equity': float(equity)
                        })
                    else:
                        self.logger.warning(f"[BT] Calculated position qty is zero or negative. Skipping trade.")
            
            equity_curve.append({'timestamp': current_timestamp, 'equity': float(equity)})

        final_pnl = equity - Decimal(str(initial_capital))
        final_pnl_percent = (final_pnl / Decimal(str(initial_capital))) * Decimal("100") if initial_capital > 0 else Decimal("0")
        
        # Basic stats
        total_trades = len([t for t in trades_log if 'entry' in t['type']])
        winning_trades = sum(1 for t in trades_log if t['pnl_realized'] > 0)
        losing_trades = sum(1 for t in trades_log if t['pnl_realized'] < 0)

        self.logger.info(f"[{self.name}-{self.trading_pair}] Backtest finished. Final PnL: {final_pnl:.2f} ({final_pnl_percent:.2f}%)")
        
        return {
            "pnl": float(final_pnl), "pnl_percentage": float(final_pnl_percent),
            "total_trades": total_trades, "winning_trades": winning_trades, "losing_trades": losing_trades,
            "sharpe_ratio": 0.0, "max_drawdown": 0.0, # Placeholder for now
            "trades_log": trades_log, "equity_curve": equity_curve,
            "message": "Backtest completed (simplified, no forecasting).",
            "initial_capital": float(initial_capital), "final_equity": float(equity)
        }

# Example of how it might be called by a backtesting engine (not part of the class itself):
# if __name__ == '__main__':
#     # Create a dummy logger
#     logging.basicConfig(level=logging.INFO)
#     logger = logging.getLogger("BacktestRunner")

#     # Sample OHLCV data (replace with actual data loading)
#     # Timestamps should be datetime objects, ensure 'close', 'low', 'high', 'open', 'volume' columns
#     data = {
#         'timestamp': pd.to_datetime(['2023-01-01 00:00', '2023-01-01 00:15', '2023-01-01 00:30', '2023-01-01 00:45', '2023-01-01 01:00'] * 100), # Needs more data
#         'open': np.random.rand(500) * 10 + 20000,
#         'high': np.random.rand(500) * 15 + 20050,
#         'low': np.random.rand(500) * 10 + 19950,
#         'close': np.random.rand(500) * 10 + 20000,
#         'volume': np.random.rand(500) * 100
#     }
#     sample_ohlcv_df = pd.DataFrame(data)
#     sample_ohlcv_df.set_index('timestamp', inplace=True)
    
#     # Ensure enough data for typical MACD (e.g., 26 periods slow + 9 signal = 35) and HTF resampling
#     # For 4h HTF from 15m chart, you need at least 16 bars for one HTF bar.
#     # So, need at least (35 periods for HTF MACD) * (16 bars of chart per HTF bar) = 560 chart bars.
#     # Let's generate more data for a meaningful run
#     num_bars = 1000
#     base_time = datetime(2023, 1, 1)
#     time_deltas = [pd.Timedelta(minutes=15*i) for i in range(num_bars)]
#     timestamps = [base_time + delta for delta in time_deltas]
    
#     data_large = {
#         'open': np.random.normal(loc=20000, scale=50, size=num_bars),
#         'high': np.random.normal(loc=20050, scale=50, size=num_bars),
#         'low': np.random.normal(loc=19950, scale=50, size=num_bars),
#         'close': np.random.normal(loc=20000, scale=50, size=num_bars),
#         'volume': np.random.uniform(1, 100, size=num_bars)
#     }
#     sample_ohlcv_df_large = pd.DataFrame(data_large, index=pd.DatetimeIndex(timestamps))
#     # Ensure high > low and open/close within high/low
#     sample_ohlcv_df_large['high'] = sample_ohlcv_df_large[['open', 'close']].max(axis=1) + np.random.uniform(0, 10, size=num_bars)
#     sample_ohlcv_df_large['low'] = sample_ohlcv_df_large[['open', 'close']].min(axis=1) - np.random.uniform(0, 10, size=num_bars)


#     # Strategy parameters for backtest
#     bt_params = {
#         "trading_pair": "BTC/USDT", # Informational for logging
#         "htf_interval": "1h",     # Higher timeframe
#         "chart_interval": "15m",  # Informational, data is assumed to be this
#         "macd_fast_len": 12,
#         "macd_slow_len": 26,
#         "macd_signal_len": 9,
#         "order_quantity_usd": 100.0, # Fixed notional USD per trade
#         "leverage": 10,
#         "use_stop_loss": True,
#         "stop_loss_pct": 1.0, # 1% SL
#         "use_take_profit": True,
#         "take_profit_pct": 2.0 # 2% TP
#     }

#     # Instantiate strategy (no DB/exchange needed for this backtest method)
#     strategy_instance = MACDForecastMTFStrategy(db_session=None, user_sub_obj=None, 
#                                                 strategy_params=bt_params, # Pass merged params here
#                                                 exchange_ccxt=None, logger_obj=logger)
    
#     # Run backtest
#     backtest_results = strategy_instance.run_backtest(ohlcv_data=sample_ohlcv_df_large.copy(), # Pass copy
#                                                       initial_capital=10000.0, 
#                                                       params=bt_params) # Pass specific backtest params
    
#     logger.info(f"Backtest Results: {backtest_results['message']}")
#     logger.info(f"Initial Capital: {backtest_results['initial_capital']:.2f}")
#     logger.info(f"Final Equity: {backtest_results['final_equity']:.2f}")
#     logger.info(f"PNL: {backtest_results['pnl']:.2f} ({backtest_results['pnl_percentage']:.2f}%)")
#     logger.info(f"Total Trades: {backtest_results['total_trades']}")
#     logger.info(f"Winning Trades: {backtest_results['winning_trades']}")
#     logger.info(f"Losing Trades: {backtest_results['losing_trades']}")
    
#     # Optionally plot equity curve if matplotlib is available
#     # import matplotlib.pyplot as plt
#     # equity_df = pd.DataFrame(backtest_results['equity_curve'])
#     # equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
#     # equity_df.set_index('timestamp', inplace=True)
#     # equity_df['equity'].plot(title="Equity Curve")
#     # plt.show()

#     # logger.info("Trades Log:")
#     # for trade in backtest_results['trades_log']:
#     #     logger.info(trade)
```
