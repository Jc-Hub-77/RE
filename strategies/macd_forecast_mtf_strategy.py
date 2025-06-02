import logging
import numpy as np
import talib
import pandas
import ccxt 
from decimal import Decimal, ROUND_DOWN, ROUND_UP, getcontext
import json 
from datetime import datetime # Correct import for datetime.utcnow
from typing import Optional, Dict, Any, List # For type hinting

from sqlalchemy.orm import Session
from backend.models import Position, Order, UserStrategySubscription
from backend import strategy_utils

# Ensure high precision for Decimal calculations
getcontext().prec = 18 # Standard for crypto, adjust if needed

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
            "forecast_max_memory": {"type": "int", "label": "Forecast Max Memory (per bar index)", "default": 50, "min": 2},
            "forecast_length_bars": {"type": "int", "label": "Forecast Projection Length (bars)", "default": 100, "min": 1},
            "forecast_upper_percentile": {"type": "int", "label": "Forecast Upper Percentile", "default": 80, "min": 51, "max": 99},
            "forecast_mid_percentile": {"type": "int", "label": "Forecast Mid Percentile", "default": 50, "min": 1, "max": 99},
            "forecast_lower_percentile": {"type": "int", "label": "Forecast Lower Percentile", "default": 20, "min": 1, "max": 49},
        }

    def __init__(self, db_session: Session, user_sub_obj: UserStrategySubscription, strategy_params: dict, exchange_ccxt, logger_obj=None): # Renamed logger to logger_obj to avoid conflict
        self.db_session = db_session
        self.user_sub_obj = user_sub_obj
        self.params = strategy_params
        self.exchange_ccxt = exchange_ccxt
        self.logger = logger_obj if logger_obj else logging.getLogger(__name__) # Use passed logger or module default
        self.name = "MACDForecastMTFStrategy"
        self.capital_param = Decimal(str(self.params.get("capital", "10000"))) # Default capital if not in sub params

        self.trading_pair = self.params.get("trading_pair", "BTC/USDT")
        self.htf_interval = self.params.get("htf_interval", "4h")
        self.chart_interval = self.params.get("chart_interval", "15m")
        self.macd_fast_len = int(self.params.get("macd_fast_len", 12))
        self.macd_slow_len = int(self.params.get("macd_slow_len", 26))
        self.macd_signal_len = int(self.params.get("macd_signal_len", 9))
        self.macd_trend_determination = self.params.get("macd_trend_determination", "MACD vs Signal")
        
        self.order_quantity_usd = Decimal(str(self.params.get("order_quantity_usd", "100.0")))
        self.leverage = int(self.params.get("leverage", 10))
        self.use_stop_loss = self.params.get("use_stop_loss", True)
        self.stop_loss_pct = Decimal(str(self.params.get("stop_loss_pct", "2.0"))) / Decimal("100")
        self.use_take_profit = self.params.get("use_take_profit", True)
        self.take_profit_pct = Decimal(str(self.params.get("take_profit_pct", "4.0"))) / Decimal("100")

        self.forecast_max_memory = int(self.params.get("forecast_max_memory", 50))
        self.forecast_length_bars = int(self.params.get("forecast_length_bars", 100))
        self.forecast_upper_percentile = int(self.params.get("forecast_upper_percentile", 80))
        self.forecast_mid_percentile = int(self.params.get("forecast_mid_percentile", 50))
        self.forecast_lower_percentile = int(self.params.get("forecast_lower_percentile", 20))

        # State variables
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
        self.active_sl_tp_orders: Dict[str, Optional[str]] = {} # {'sl_id': exchange_order_id, 'tp_id': exchange_order_id}

        self._fetch_market_precision() # Needs to be called before any formatting
        self._load_strategy_and_position_state() # Load state
        self._set_leverage()
        self.logger.info(f"{self.name} initialized for {self.trading_pair}, UserSubID {self.user_sub_obj.id}. Loaded state for PosID: {self.active_position_db_id}")

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
                    # Load forecast memory, ensuring keys are integers
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
                    # Initialize to defaults if parsing fails
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
        if not self.db_session:
            self.logger.warning(f"[{self.name}-{self.trading_pair}] DB session not available. Cannot save state.")
            return

        if self.active_position_db_id is None:
            self.logger.info(f"[{self.name}-{self.trading_pair}] No active position DB ID. State not saved to Position.custom_data (normal if no position).")
            # Potentially save non-position-specific state to UserStrategySubscription.strategy_state_json if needed
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

    def _fetch_market_precision(self): # Renamed from _get_precisions_live to avoid conflict with EMA strategy if merged
        # Same as original
        try:
            self.exchange_ccxt.load_markets()
            market = self.exchange_ccxt.markets[self.trading_pair]
            self.quantity_precision_str = str(market['precision']['amount'])
            self.price_precision_str = str(market['precision']['price'])
            self.logger.info(f"Precision for {self.trading_pair}: Qty Prec Str={self.quantity_precision_str}, Price Prec Str={self.price_precision_str}")
        except Exception as e:
            self.logger.error(f"Error fetching precision for {self.trading_pair}: {e}. Using defaults.")
            self.quantity_precision_str = "0.00001" # Default for crypto like BTC
            self.price_precision_str = "0.01"    # Default for USDT pairs

    def _get_decimal_places(self, precision_str: str) -> int:
        # Same as original
        if precision_str is None: return 8 
        try:
            if 'e-' in precision_str.lower():
                num_val = float(precision_str)
                precision_str = format(num_val, f'.{abs(int(precision_str.split("e-")[1]))}f')
            d_prec = Decimal(precision_str)
            if d_prec.as_tuple().exponent < 0: return abs(d_prec.as_tuple().exponent)
            return 0 
        except Exception as e: self.logger.warning(f"Could not parse precision string '{precision_str}'. Error: {e}. Using default 8."); return 8

    def _format_quantity(self, quantity: Decimal) -> str:
        # Same as original
        places = self._get_decimal_places(self.quantity_precision_str)
        return str(quantity.quantize(Decimal('1e-' + str(places)), rounding=ROUND_DOWN))

    def _format_price(self, price: Decimal) -> str:
        # Same as original
        places = self._get_decimal_places(self.price_precision_str)
        # Use ROUND_NEAREST for price usually, or as per exchange rules
        return str(price.quantize(Decimal('1e-' + str(places)), rounding=ROUND_NEAREST))


    def _set_leverage(self):
        # Same as original
        try:
            if hasattr(self.exchange_ccxt, 'set_leverage'):
                symbol_for_leverage = self.trading_pair.split(':')[0] if ':' in self.trading_pair else self.trading_pair
                self.exchange_ccxt.set_leverage(self.leverage, symbol_for_leverage)
                self.logger.info(f"Leverage set to {self.leverage}x for {symbol_for_leverage}")
        except Exception as e: self.logger.warning(f"Could not set leverage for {self.trading_pair}: {e}")


    def _get_ohlcv_data(self, interval, limit): # Removed unused params
        # Simplified for live trading
        return self.exchange_ccxt.fetch_ohlcv(self.trading_pair, interval, limit=limit)

    def _calculate_macd_values(self, ohlcv_df):
        # Same as original
        if ohlcv_df is None or ohlcv_df.empty or len(ohlcv_df) < self.macd_slow_len: return None, None, None
        close_prices = ohlcv_df['close'].to_numpy(dtype=float)
        macd, signal, hist = talib.MACD(close_prices, fastperiod=self.macd_fast_len, slowperiod=self.macd_slow_len, signalperiod=self.macd_signal_len)
        return macd, signal, hist

    def _determine_trend(self, macd_value, signal_value):
        # Same as original
        if self.macd_trend_determination == "MACD vs Zero":
            is_bullish = macd_value > 0; is_bearish = macd_value < 0
        else: 
            is_bullish = macd_value > signal_value; is_bearish = macd_value < signal_value
        return is_bullish, is_bearish

    def _populate_memory(self, current_trend_type, current_trend_bar_idx, current_trend_init_price, current_close_price):
        # Same as original, ensure Decimal to float for JSON
        price_deviation = float(current_close_price - current_trend_init_price)
        if current_trend_type not in self.forecast_memory: self.forecast_memory[current_trend_type] = []
        while len(self.forecast_memory[current_trend_type]) <= current_trend_bar_idx: self.forecast_memory[current_trend_type].append([])
        self.forecast_memory[current_trend_type][current_trend_bar_idx].insert(0, price_deviation)
        if len(self.forecast_memory[current_trend_type][current_trend_bar_idx]) > self.forecast_max_memory:
            self.forecast_memory[current_trend_type][current_trend_bar_idx].pop()

    def _calculate_forecast_bands(self, trend_about_to_start_type, new_trend_init_price):
        # Same as original, ensure Decimal to float for JSON
        forecast_bands = []
        if trend_about_to_start_type not in self.forecast_memory or not self.forecast_memory[trend_about_to_start_type]:
            self.logger.info(f"No historical data for trend type {trend_about_to_start_type}."); return forecast_bands
        historical_segments = self.forecast_memory[trend_about_to_start_type]
        max_historical_trend_len = len(historical_segments)
        for bar_offset in range(self.forecast_length_bars):
            if bar_offset < max_historical_trend_len:
                deviations = historical_segments[bar_offset]
                if len(deviations) > 1:
                    forecast_bands.append({
                        'bar_offset': bar_offset,
                        'lower': float(new_trend_init_price + Decimal(str(np.percentile(deviations, self.forecast_lower_percentile)))),
                        'mid': float(new_trend_init_price + Decimal(str(np.percentile(deviations, self.forecast_mid_percentile)))),
                        'upper': float(new_trend_init_price + Decimal(str(np.percentile(deviations, self.forecast_upper_percentile))))
                    })
        if forecast_bands: self.logger.info(f"Calculated {len(forecast_bands)} forecast points. First: {forecast_bands[0] if forecast_bands else 'N/A'}")
        return forecast_bands
    
    def _place_order(self, symbol:str, order_type:str, side:str, quantity:Decimal, price:Optional[Decimal]=None, params:Optional[Dict]=None) -> Optional[Dict]:
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
            if db_order and db_order.id: # If DB order was created but exchange failed
                strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=db_order.id, updates={'status': 'error_on_exchange', 'status_message': str(e)[:255]})
        return None

    def _cancel_active_sl_tp_orders_in_db(self):
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
                    self.logger.info(f"[{self.name}-{self.trading_pair}] Successfully canceled {order_key} order {exchange_order_id} and updated DB.")
                except ccxt.OrderNotFound:
                     self.logger.warning(f"[{self.name}-{self.trading_pair}] {order_key} order {exchange_order_id} not found on exchange, likely already filled or canceled.")
                     strategy_utils.update_strategy_order_in_db(self.db_session, exchange_order_id=exchange_order_id, subscription_id=self.user_sub_obj.id, symbol=self.trading_pair, updates={'status': 'not_found_on_cancel'})
                except Exception as e:
                    self.logger.error(f"[{self.name}-{self.trading_pair}] Failed to cancel {order_key} order {exchange_order_id}: {e}", exc_info=True)
                    # Optionally update DB order status to 'cancel_failed'
        self.active_sl_tp_orders = {}


    def _close_all_positions(self, current_market_price: Decimal):
        if self.active_position_side and self.active_position_db_id:
            close_side = 'sell' if self.active_position_side == 'long' else 'buy'
            self.logger.info(f"Closing all positions for {self.trading_pair} (Side: {self.active_position_side}, Qty: {self.position_qty}). Market Price: {current_market_price}")
            
            # Place market order to close the position
            close_order_receipt = self._place_order(self.trading_pair, 'market', close_side, self.position_qty, params={'reduceOnly': True})
            
            if close_order_receipt: # _place_order now handles initial DB logging
                # Await fill for market order (important for accurate PnL)
                filled_close_order = self._await_order_fill(self.exchange_ccxt, close_order_receipt['id'], self.trading_pair)
                if filled_close_order and filled_close_order['status'] == 'closed':
                    strategy_utils.update_strategy_order_in_db(self.db_session, exchange_order_id=filled_close_order['id'], subscription_id=self.user_sub_obj.id, symbol=self.trading_pair, updates={'status': 'closed', 'price': filled_close_order['average'], 'filled': filled_close_order['filled'], 'cost': filled_close_order['cost'], 'closed_at': datetime.utcnow()})
                    
                    strategy_utils.close_strategy_position_in_db(
                        self.db_session, self.active_position_db_id, 
                        close_price=Decimal(str(filled_close_order['average'])), 
                        filled_amount_at_close=Decimal(str(filled_close_order['filled'])), 
                        reason="Close All Signal Triggered"
                    )
                else:
                    self.logger.error(f"[{self.name}-{self.trading_pair}] Market close order {close_order_receipt['id']} failed to fill or status unknown. Manual check required.")
                    # Update DB order with available status
                    if filled_close_order:
                         strategy_utils.update_strategy_order_in_db(self.db_session, exchange_order_id=close_order_receipt['id'], subscription_id=self.user_sub_obj.id, symbol=self.trading_pair, updates={'status': filled_close_order.get('status', 'fill_check_failed')})
                    else: # If await_order_fill returned None
                         strategy_utils.update_strategy_order_in_db(self.db_session, exchange_order_id=close_order_receipt['id'], subscription_id=self.user_sub_obj.id, symbol=self.trading_pair, updates={'status': 'fill_await_failed'})

            self._cancel_active_sl_tp_orders_in_db() # Cancel any existing SL/TP orders

            self.active_position_side = None
            self.position_entry_price = None
            self.position_qty = Decimal("0")
            self.active_position_db_id = None # Reset after closing
            self._initialize_default_strategy_state() # Reset forecast memory etc.
            self._save_persistent_state() # Save the reset state (or clear custom_data for the closed position)
        else:
            self.logger.info(f"[{self.name}-{self.trading_pair}] No active position to close.")


    def _sync_exchange_position_state(self, current_price: Decimal): # Added current_price for context
        if not self.active_position_db_id or not self.db_session or not self.active_sl_tp_orders:
            return False 

        position_closed_by_exchange_event = False
        
        sl_id = self.active_sl_tp_orders.get('sl_id')
        tp_id = self.active_sl_tp_orders.get('tp_id')

        if sl_id:
            try:
                sl_order_details = self.exchange_ccxt.fetch_order(sl_id, self.trading_pair)
                if sl_order_details['status'] == 'closed':
                    self.logger.info(f"[{self.name}-{self.trading_pair}] SL order {sl_id} found filled on exchange.")
                    strategy_utils.update_strategy_order_in_db(self.db_session, exchange_order_id=sl_id, subscription_id=self.user_sub_obj.id, symbol=self.trading_pair, updates={'status': 'closed', 'price': sl_order_details.get('average'), 'filled': sl_order_details.get('filled'), 'closed_at': datetime.utcnow(), 'raw_order_data': json.dumps(sl_order_details)})
                    
                    strategy_utils.close_strategy_position_in_db(self.db_session, self.active_position_db_id, Decimal(str(sl_order_details['average'])), Decimal(str(sl_order_details['filled'])), f"Closed by SL order {sl_id}")
                    position_closed_by_exchange_event = True
                    if tp_id: # Cancel TP
                        try: self.exchange_ccxt.cancel_order(tp_id, self.trading_pair); strategy_utils.update_strategy_order_in_db(self.db_session, exchange_order_id=tp_id, subscription_id=self.user_sub_obj.id, symbol=self.trading_pair, updates={'status': 'canceled'})
                        except Exception as e_c: self.logger.warning(f"Failed to cancel TP {tp_id} after SL fill: {e_c}")
            except ccxt.OrderNotFound: 
                self.logger.warning(f"SL order {sl_id} not found on exchange."); self.active_sl_tp_orders['sl_id'] = None
                strategy_utils.update_strategy_order_in_db(self.db_session, exchange_order_id=sl_id, subscription_id=self.user_sub_obj.id, symbol=self.trading_pair, updates={'status': 'not_found_on_sync'})
            except Exception as e: self.logger.error(f"Error checking SL order {sl_id}: {e}", exc_info=True)

        if not position_closed_by_exchange_event and tp_id:
            try:
                tp_order_details = self.exchange_ccxt.fetch_order(tp_id, self.trading_pair)
                if tp_order_details['status'] == 'closed':
                    self.logger.info(f"[{self.name}-{self.trading_pair}] TP order {tp_id} found filled on exchange.")
                    strategy_utils.update_strategy_order_in_db(self.db_session, exchange_order_id=tp_id, subscription_id=self.user_sub_obj.id, symbol=self.trading_pair, updates={'status': 'closed', 'price': tp_order_details.get('average'), 'filled': tp_order_details.get('filled'), 'closed_at': datetime.utcnow(), 'raw_order_data': json.dumps(tp_order_details)})

                    strategy_utils.close_strategy_position_in_db(self.db_session, self.active_position_db_id, Decimal(str(tp_order_details['average'])), Decimal(str(tp_order_details['filled'])), f"Closed by TP order {tp_id}")
                    position_closed_by_exchange_event = True
                    if sl_id: # Cancel SL
                        try: self.exchange_ccxt.cancel_order(sl_id, self.trading_pair); strategy_utils.update_strategy_order_in_db(self.db_session, exchange_order_id=sl_id, subscription_id=self.user_sub_obj.id, symbol=self.trading_pair, updates={'status': 'canceled'})
                        except Exception as e_c: self.logger.warning(f"Failed to cancel SL {sl_id} after TP fill: {e_c}")
            except ccxt.OrderNotFound: 
                self.logger.warning(f"TP order {tp_id} not found on exchange."); self.active_sl_tp_orders['tp_id'] = None
                strategy_utils.update_strategy_order_in_db(self.db_session, exchange_order_id=tp_id, subscription_id=self.user_sub_obj.id, symbol=self.trading_pair, updates={'status': 'not_found_on_sync'})
            except Exception as e: self.logger.error(f"Error checking TP order {tp_id}: {e}", exc_info=True)

        if position_closed_by_exchange_event:
            self.active_position_side = None; self.position_entry_price = None; self.position_qty = Decimal("0")
            self.active_sl_tp_orders = {}; self.active_position_db_id = None
            self._initialize_default_strategy_state() # Reset forecast memory etc.
            self._save_persistent_state() # Save reset state (or clear custom_data)
            return True
        return False


    def _process_trading_logic(self, chart_ohlcv_df, htf_ohlcv_df):
        # Uses self.active_position_side, self.position_entry_price, self.position_qty
        # Updates these and self.active_sl_tp_orders, calls self._save_persistent_state()
        chart_macd_vals, chart_signal_vals, _ = self._calculate_macd_values(chart_ohlcv_df)
        htf_macd_vals, htf_signal_vals, _ = self._calculate_macd_values(htf_ohlcv_df)

        if chart_macd_vals is None or htf_macd_vals is None or len(chart_macd_vals) < 2 or len(htf_macd_vals) < 1:
            self.logger.warning("Not enough data for MACD calculations in _process_trading_logic."); return

        latest_chart_macd = chart_macd_vals[-1]; latest_chart_signal = chart_signal_vals[-1]
        prev_chart_macd = chart_macd_vals[-2]; prev_chart_signal = chart_signal_vals[-2]
        latest_htf_macd = htf_macd_vals[-1]; latest_htf_signal = htf_signal_vals[-1]

        is_chart_uptrend, is_chart_downtrend = self._determine_trend(latest_chart_macd, latest_chart_signal)
        is_htf_uptrend, is_htf_downtrend = self._determine_trend(latest_htf_macd, latest_htf_signal) # Corrected variable names

        chart_trigger_up = prev_chart_macd <= prev_chart_signal and latest_chart_macd > latest_chart_signal
        chart_trigger_down = prev_chart_macd >= prev_chart_signal and latest_chart_macd < latest_chart_signal
        current_close_price = Decimal(str(chart_ohlcv_df['close'].iloc[-1]))

        if is_chart_uptrend:
            if not self.is_prev_chart_uptrend:
                self.current_uptrend_init_price = current_close_price; self.current_uptrend_idx = 0
                self.logger.info(f"Chart uptrend started. Init price: {self.current_uptrend_init_price}, Index: {self.current_uptrend_idx}")
                if chart_trigger_up: self._calculate_forecast_bands(1, self.current_uptrend_init_price)
            else: self.current_uptrend_idx += 1
            self._populate_memory(1, self.current_uptrend_idx, self.current_uptrend_init_price, current_close_price)
        
        if is_chart_downtrend:
            if self.is_prev_chart_uptrend is None or self.is_prev_chart_uptrend:
                self.current_downtrend_init_price = current_close_price; self.current_downtrend_idx = 0
                self.logger.info(f"Chart downtrend started. Init price: {self.current_downtrend_init_price}, Index: {self.current_downtrend_idx}")
                if chart_trigger_down: self._calculate_forecast_bands(0, self.current_downtrend_init_price)
            else: self.current_downtrend_idx += 1
            self._populate_memory(0, self.current_downtrend_idx, self.current_downtrend_init_price, current_close_price)
        self.is_prev_chart_uptrend = is_chart_uptrend

        long_condition = chart_trigger_up and is_chart_uptrend and is_htf_uptrend
        short_condition = chart_trigger_down and is_chart_downtrend and is_htf_downtrend

        if (long_condition and self.active_position_side == 'short') or \
           (short_condition and self.active_position_side == 'long'):
            self.logger.info("Signal opposite to current position. Closing position first.")
            self._close_all_positions(current_close_price) # Pass current price for PnL on close
        
        action_taken_this_cycle = False
        if not self.active_position_side: # If no active position (either initially or after closing one)
            entry_price = current_close_price
            if entry_price == Decimal("0"): self.logger.warning("Entry price zero, cannot size."); return
            
            # Use capital from subscription parameters if available, else from init
            sub_params = json.loads(self.user_sub_obj.custom_parameters) if isinstance(self.user_sub_obj.custom_parameters, str) else self.user_sub_obj.custom_parameters
            allocated_capital = Decimal(str(sub_params.get("capital", self.capital_param)))
            qty_to_trade = (self.order_quantity_usd / entry_price) * self.leverage # Simplified, consider full notional

            if long_condition or short_condition:
                entry_side = "long" if long_condition else "short"
                self.logger.info(f"Trading Signal: ENTER {entry_side.upper()} for {self.trading_pair} at {entry_price}")
                entry_order_receipt = self._place_order(self.trading_pair, 'market', entry_side, qty_to_trade)
                
                if entry_order_receipt:
                    # Assuming market order fills quickly, use current_price as approx entry for now
                    # Await fill would be better for precise entry price
                    filled_entry_order = self._await_order_fill(self.exchange_ccxt, entry_order_receipt['id'], self.trading_pair)
                    if filled_entry_order and filled_entry_order['status'] == 'closed':
                        actual_entry_price = Decimal(str(filled_entry_order['average']))
                        actual_filled_qty = Decimal(str(filled_entry_order['filled']))

                        strategy_utils.update_strategy_order_in_db(self.db_session, exchange_order_id=filled_entry_order['id'], subscription_id=self.user_sub_obj.id, symbol=self.trading_pair, updates={'status':'closed', 'price':float(actual_entry_price), 'filled':float(actual_filled_qty), 'cost': float(actual_entry_price * actual_filled_qty), 'closed_at': datetime.utcnow()})

                        new_pos_db = strategy_utils.create_strategy_position_in_db(
                            self.db_session, self.user_sub_obj.id, self.trading_pair, 
                            str(self.exchange_ccxt.id), entry_side, float(actual_filled_qty), float(actual_entry_price)
                        )
                        if new_pos_db:
                            self.active_position_db_id = new_pos_db.id
                            self.active_position_side = entry_side
                            self.position_entry_price = actual_entry_price
                            self.position_qty = actual_filled_qty
                            self.logger.info(f"Position {entry_side.upper()} created. DB ID: {new_pos_db.id}, Entry: {self.position_entry_price}, Qty: {self.position_qty}")

                            sl_price = None; tp_price = None
                            if self.use_stop_loss: sl_price = self.position_entry_price * (Decimal('1') - self.stop_loss_pct) if entry_side == 'long' else self.position_entry_price * (Decimal('1') + self.stop_loss_pct)
                            if self.use_take_profit: tp_price = self.position_entry_price * (Decimal('1') + self.take_profit_pct) if entry_side == 'long' else self.position_entry_price * (Decimal('1') - self.take_profit_pct)
                            
                            if sl_price:
                                sl_ord_receipt = self._place_order(self.trading_pair, 'STOP_MARKET', 'sell' if entry_side == 'long' else 'buy', self.position_qty, params={'stopPrice': self._format_price(sl_price), 'reduceOnly': True})
                                if sl_ord_receipt: self.active_sl_tp_orders['sl_id'] = sl_ord_receipt.get('id')
                            if tp_price:
                                tp_ord_receipt = self._place_order(self.trading_pair, 'TAKE_PROFIT_MARKET', 'sell' if entry_side == 'long' else 'buy', self.position_qty, params={'stopPrice': self._format_price(tp_price), 'reduceOnly': True})
                                if tp_ord_receipt: self.active_sl_tp_orders['tp_id'] = tp_ord_receipt.get('id')
                            action_taken_this_cycle = True
                    else:
                         self.logger.error(f"Entry order {entry_order_receipt['id']} did not fill or status unknown.")
                         strategy_utils.update_strategy_order_in_db(self.db_session, exchange_order_id=entry_order_receipt['id'], subscription_id=self.user_sub_obj.id, symbol=self.trading_pair, updates={'status': filled_entry_order.get('status', 'fill_check_failed') if filled_entry_order else 'fill_check_failed'})


        if action_taken_this_cycle or self.current_uptrend_idx == 0 or self.current_downtrend_idx == 0 :
            self._save_persistent_state() # Save state if new position, or if new trend started


    def execute_live_signal(self): # Removed args, they are instance members now
        self.logger.info(f"Executing {self.name} for {self.trading_pair} UserSubID {self.user_sub_obj.id}")
        
        # Load state at the beginning of each execution cycle
        # This ensures that if the task restarts, it picks up the latest persisted state.
        # However, for a single continuous run, __init__ loads it once.
        # If the Celery task calls this method repeatedly on the same strategy instance,
        # _load_strategy_and_position_state() might be better here.
        # For now, assuming Celery task creates a new strategy instance for each "job" or manages instance lifecycle.
        # If execute_live_signal is called multiple times on the same instance, uncommenting below might be needed.
        # self._load_strategy_and_position_state() 

        if self.is_prev_chart_uptrend is None and self.active_position_side is None: # If still seems uninitialized
             self._load_strategy_and_position_state() # Attempt a reload

        try:
            current_close_price = Decimal(str(self.exchange_ccxt.fetch_ticker(self.trading_pair)['last']))
            if self._sync_exchange_position_state(current_close_price): # Pass current price for context
                self.logger.info(f"[{self.name}-{self.trading_pair}] Position closed by SL/TP sync. Cycle ended.")
                return # Position state changed, end cycle.
        except Exception as e:
            self.logger.error(f"[{self.name}-{self.trading_pair}] Error during pre-logic sync or price fetch: {e}", exc_info=True)
            return


        chart_limit = max(self.macd_slow_len, self.macd_signal_len) + self.forecast_max_memory + 50 # Ensure enough data
        htf_limit = max(self.macd_slow_len, self.macd_signal_len) + 50

        try:
            chart_ohlcv_list = self._get_ohlcv_data(self.chart_interval, chart_limit)
            htf_ohlcv_list = self._get_ohlcv_data(self.htf_interval, htf_limit)

            min_chart_len = max(self.macd_slow_len, self.macd_signal_len) +2 
            min_htf_len = max(self.macd_slow_len, self.macd_signal_len) +1

            if not chart_ohlcv_list or len(chart_ohlcv_list) < min_chart_len or \
               not htf_ohlcv_list or len(htf_ohlcv_list) < min_htf_len:
                self.logger.warning("Insufficient OHLCV data for MACD calc in execute_live_signal.")
                return

            chart_ohlcv_df = pandas.DataFrame(chart_ohlcv_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            chart_ohlcv_df['timestamp'] = pandas.to_datetime(chart_ohlcv_df['timestamp'], unit='ms')
            
            htf_ohlcv_df = pandas.DataFrame(htf_ohlcv_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            htf_ohlcv_df['timestamp'] = pandas.to_datetime(htf_ohlcv_df['timestamp'], unit='ms')
            
            self._process_trading_logic(chart_ohlcv_df, htf_ohlcv_df)

        except Exception as e:
            self.logger.error(f"Error in {self.name} execute_live_signal main try block: {e}", exc_info=True)

    def run_backtest(self, historical_data_feed, current_simulated_time_utc):
        # Backtest logic remains simulation-based and does not use self.db_session or strategy_utils
        # It should use its own local state variables for simulation.
        self.logger.info(f"Running backtest for {self.name} on {self.trading_pair} at {current_simulated_time_utc}...")
        
        chart_limit = max(self.macd_slow_len, self.macd_signal_len) + self.forecast_max_memory + 50
        htf_limit = max(self.macd_slow_len, self.macd_signal_len) + 50

        chart_ohlcv_df = historical_data_feed.get_ohlcv(self.trading_pair, self.chart_interval, chart_limit, end_time_utc=current_simulated_time_utc)
        htf_ohlcv_df = historical_data_feed.get_ohlcv(self.trading_pair, self.htf_interval, htf_limit, end_time_utc=current_simulated_time_utc)

        min_chart_len = max(self.macd_slow_len, self.macd_signal_len) +2
        min_htf_len = max(self.macd_slow_len, self.macd_signal_len) +1

        if chart_ohlcv_df is None or chart_ohlcv_df.empty or htf_ohlcv_df is None or htf_ohlcv_df.empty or \
           len(chart_ohlcv_df) < min_chart_len or len(htf_ohlcv_df) < min_htf_len:
            return {"action": "HOLD", "reason": "Insufficient historical data for MACD in backtest"}

        # Pass a copy of relevant parameters to _process_trading_logic for backtesting context
        # This is a simplified call; _process_trading_logic might need more context or be adapted for backtesting
        # if its live version heavily relies on instance state not applicable to backtest simulation.
        # For this refactor, we assume _process_trading_logic can be called with is_backtest=True
        # and will manage its own simulated state.
        
        # The original _process_trading_logic was designed for live.
        # A proper backtest would need a separate simulation loop or significant adaptation
        # of _process_trading_logic to manage simulated orders and state.
        # For now, returning a placeholder as the focus is on live trading refactor.
        self.logger.warning("Backtesting for MACDForecastMTFStrategy needs a dedicated simulation loop. Returning placeholder.")
        return {"action": "HOLD", "reason": "Backtest simulation for this strategy is complex and not fully implemented in this refactor."}

```
