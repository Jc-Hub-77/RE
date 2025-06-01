import logging
import numpy as np
import talib
import pandas
import ccxt # Though self.exchange_ccxt is provided
from decimal import Decimal, ROUND_DOWN, ROUND_UP, getcontext
import json # For serializing/deserializing forecast memory and state
from datetime import datetime

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
            "forecast_max_memory": {"type": "int", "label": "Forecast Max Memory (per bar index)", "default": 50, "min": 2},
            "forecast_length_bars": {"type": "int", "label": "Forecast Projection Length (bars)", "default": 100, "min": 1},
            "forecast_upper_percentile": {"type": "int", "label": "Forecast Upper Percentile", "default": 80, "min": 51, "max": 99},
            "forecast_mid_percentile": {"type": "int", "label": "Forecast Mid Percentile", "default": 50, "min": 1, "max": 99},
            "forecast_lower_percentile": {"type": "int", "label": "Forecast Lower Percentile", "default": 20, "min": 1, "max": 49},
        }

    def __init__(self, db_session, user_sub_obj, strategy_params: dict, exchange_ccxt, logger=None):
        self.db_session = db_session
        self.user_sub_obj = user_sub_obj
        self.params = strategy_params
        self.exchange_ccxt = exchange_ccxt
        self.logger = logger if logger else logging.getLogger(__name__)
        self.name = "MACDForecastMTFStrategy"

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

        # State variables for forecasting
        self.forecast_memory = {1: [], 0: []}  # Keys: 1 for uptrend data, 0 for downtrend data
                                              # Values: List of lists. memory[type][idx_in_trend] = [price_deviations]
        self.current_uptrend_idx = 0
        self.current_downtrend_idx = 0
        self.current_uptrend_init_price = Decimal("0")
        self.current_downtrend_init_price = Decimal("0")
        self.is_prev_chart_uptrend = None # Stores the chart trend from the previous tick

        # Active position state
        self.active_position_side = None  # 'long', 'short', or None
        self.position_entry_price = None
        self.position_qty = Decimal("0")
        self.active_sl_tp_orders = {} # {'sl_id': id, 'tp_id': id}

        self._load_persistent_state() # Attempt to load forecast_memory and other states

        self._fetch_market_precision()
        self._set_leverage()
        self.logger.info(f"{self.name} initialized for {self.trading_pair}, UserSubID {self.user_sub_obj.id}")

    def _load_persistent_state(self):
        self.logger.info("DB_TODO: Load persistent state (forecast_memory, trend indices, init prices, active position) from database.")
        # Example:
        # state_json = self.db_session.query(UserStrategySubscription.strategy_state_json)        #                        .filter(UserStrategySubscription.id == self.user_sub_obj.id).scalar()
        # if state_json:
        #     try:
        #         state = json.loads(state_json)
        #         self.forecast_memory = {int(k): v for k, v in state.get("forecast_memory", {1:[], 0:[]}).items()} # Ensure keys are int
        #         self.current_uptrend_idx = state.get("current_uptrend_idx", 0)
        #         # ... load other state vars ...
        #         # Convert Decimal strings back to Decimal
        #         self.current_uptrend_init_price = Decimal(state.get("current_uptrend_init_price", "0"))
        #         self.current_downtrend_init_price = Decimal(state.get("current_downtrend_init_price", "0"))
        #         self.active_position_side = state.get("active_position_side")
        #         self.position_entry_price = Decimal(state.get("position_entry_price", "0")) if self.active_position_side else None
        #         self.position_qty = Decimal(state.get("position_qty", "0")) if self.active_position_side else Decimal("0")
        #         self.active_sl_tp_orders = state.get("active_sl_tp_orders", {})
        #         self.is_prev_chart_uptrend = state.get("is_prev_chart_uptrend")
        #         self.logger.info("Successfully loaded persistent state.")
        #     except Exception as e:
        #         self.logger.error(f"Error loading/parsing persistent state: {e}. Starting with fresh state.")
        # else:
        #     self.logger.info("No persistent state found. Starting with fresh state.")
        pass


    def _save_persistent_state(self):
        self.logger.info("DB_TODO: Save persistent state (forecast_memory, trend indices, init prices, active position) to database.")
        # Example:
        # state = {
        #     "forecast_memory": self.forecast_memory,
        #     "current_uptrend_idx": self.current_uptrend_idx,
        #     "current_downtrend_idx": self.current_downtrend_idx,
        #     "current_uptrend_init_price": str(self.current_uptrend_init_price), # Convert Decimals to str
        #     "current_downtrend_init_price": str(self.current_downtrend_init_price),
        #     "active_position_side": self.active_position_side,
        #     "position_entry_price": str(self.position_entry_price) if self.position_entry_price else None,
        #     "position_qty": str(self.position_qty) if self.position_qty else "0",
        #     "active_sl_tp_orders": self.active_sl_tp_orders,
        #     "is_prev_chart_uptrend": self.is_prev_chart_uptrend
        # }
        # try:
        #     state_json = json.dumps(state)
        #     # self.db_session.query(UserStrategySubscription).filter(UserStrategySubscription.id == self.user_sub_obj.id)        #     #               .update({"strategy_state_json": state_json})
        #     # self.db_session.commit()
        #     self.logger.info("Successfully saved persistent state.")
        # except Exception as e:
        #     self.logger.error(f"Error saving persistent state: {e}")
        #     # self.db_session.rollback() # If using db_session directly
        pass

    def _fetch_market_precision(self):
        try:
            self.exchange_ccxt.load_markets()
            market = self.exchange_ccxt.markets[self.trading_pair]
            self.quantity_precision_str = str(market['precision']['amount'])
            self.price_precision_str = str(market['precision']['price'])
            self.logger.info(f"Precision for {self.trading_pair}: Qty Prec Str={self.quantity_precision_str}, Price Prec Str={self.price_precision_str}")
        except Exception as e:
            self.logger.error(f"Error fetching precision for {self.trading_pair}: {e}. Using defaults.")
            self.quantity_precision_str = "0.00001"
            self.price_precision_str = "0.01"

    def _get_decimal_places(self, precision_str):
        if precision_str is None: return 8 # Default if None
        try:
            # Convert scientific notation like "1e-5" to "0.00001"
            if 'e-' in precision_str.lower():
                num_val = float(precision_str)
                precision_str = format(num_val, f'.{abs(int(precision_str.split("e-")[1]))}f')

            d_prec = Decimal(precision_str)
            if d_prec.as_tuple().exponent < 0:
                return abs(d_prec.as_tuple().exponent)
            return 0 # If precision is integer like 1, 10
        except Exception as e:
            self.logger.warning(f"Could not parse precision string '{precision_str}'. Error: {e}. Using default 8.")
            return 8


    def _format_quantity(self, quantity: Decimal):
        places = self._get_decimal_places(self.quantity_precision_str)
        return str(quantity.quantize(Decimal('1e-' + str(places)), rounding=ROUND_DOWN))

    def _format_price(self, price: Decimal):
        places = self._get_decimal_places(self.price_precision_str)
        return str(price.quantize(Decimal('1e-' + str(places)), rounding=ROUND_NEAREST))

    def _set_leverage(self):
        try:
            if hasattr(self.exchange_ccxt, 'set_leverage'):
                symbol_for_leverage = self.trading_pair.split(':')[0] if ':' in self.trading_pair else self.trading_pair
                self.exchange_ccxt.set_leverage(self.leverage, symbol_for_leverage)
                self.logger.info(f"Leverage set to {self.leverage}x for {symbol_for_leverage}")
        except Exception as e:
            self.logger.warning(f"Could not set leverage for {self.trading_pair}: {e}")


    def _get_ohlcv_data(self, interval, limit, is_backtest, historical_data_feed=None, end_time_utc=None):
        if is_backtest and historical_data_feed:
            return historical_data_feed.get_ohlcv(self.trading_pair, interval, limit=limit, end_time_utc=end_time_utc)
        else:
            return self.exchange_ccxt.fetch_ohlcv(self.trading_pair, interval, limit=limit)


    def _calculate_macd_values(self, ohlcv_df):
        if ohlcv_df is None or ohlcv_df.empty or len(ohlcv_df) < self.macd_slow_len:
            return None, None, None
        close_prices = ohlcv_df['close'].to_numpy(dtype=float)
        macd, signal, hist = talib.MACD(close_prices,
                                        fastperiod=self.macd_fast_len,
                                        slowperiod=self.macd_slow_len,
                                        signalperiod=self.macd_signal_len)
        return macd, signal, hist

    def _determine_trend(self, macd_value, signal_value):
        if self.macd_trend_determination == "MACD vs Zero":
            is_bullish = macd_value > 0
            is_bearish = macd_value < 0
        else: # MACD vs Signal
            is_bullish = macd_value > signal_value
            is_bearish = macd_value < signal_value
        return is_bullish, is_bearish

    def _populate_memory(self, current_trend_type, current_trend_bar_idx, current_trend_init_price, current_close_price):
        price_deviation = float(current_close_price - current_trend_init_price)

        if current_trend_type not in self.forecast_memory:
            self.forecast_memory[current_trend_type] = []

        while len(self.forecast_memory[current_trend_type]) <= current_trend_bar_idx:
            self.forecast_memory[current_trend_type].append([])
        
        self.forecast_memory[current_trend_type][current_trend_bar_idx].insert(0, price_deviation)

        if len(self.forecast_memory[current_trend_type][current_trend_bar_idx]) > self.forecast_max_memory:
            self.forecast_memory[current_trend_type][current_trend_bar_idx].pop()


    def _calculate_forecast_bands(self, trend_about_to_start_type, new_trend_init_price):
        forecast_bands = []
        
        if trend_about_to_start_type not in self.forecast_memory or not self.forecast_memory[trend_about_to_start_type]:
            self.logger.info(f"No historical data in forecast_memory for trend type {trend_about_to_start_type} to make forecast.")
            return forecast_bands

        historical_segments = self.forecast_memory[trend_about_to_start_type]
        max_historical_trend_len = len(historical_segments)

        for bar_offset in range(self.forecast_length_bars):
            if bar_offset < max_historical_trend_len:
                deviations_for_this_bar_idx = historical_segments[bar_offset]
                if len(deviations_for_this_bar_idx) > 1:
                    lower_dev = np.percentile(deviations_for_this_bar_idx, self.forecast_lower_percentile)
                    mid_dev = np.percentile(deviations_for_this_bar_idx, self.forecast_mid_percentile)
                    upper_dev = np.percentile(deviations_for_this_bar_idx, self.forecast_upper_percentile)

                    forecast_bands.append({
                        'bar_offset': bar_offset,
                        'lower': float(new_trend_init_price + Decimal(str(lower_dev))),
                        'mid': float(new_trend_init_price + Decimal(str(mid_dev))),
                        'upper': float(new_trend_init_price + Decimal(str(upper_dev)))
                    })

        if forecast_bands: self.logger.info(f"Calculated {len(forecast_bands)} forecast points. First point: {forecast_bands[0] if forecast_bands else 'N/A'}")
        return forecast_bands


    def _get_current_position_details(self, symbol, historical_data_feed=None, is_backtest=False):
        if self.active_position_side:
            return {'side': self.active_position_side, 'qty': self.position_qty, 'entry_price': self.position_entry_price}
        
        if is_backtest and historical_data_feed:
             return historical_data_feed.get_current_position(symbol)

        if not is_backtest:
             self.logger.info("DB_TODO: Query DB for position details if self.active_position_side is None")
             # Example live query:
             # live_pos = query_db_for_pos(self.db_session, self.user_sub_obj.id, symbol)
             # if live_pos: update self.active_position_side, self.position_entry_price, self.position_qty
             # return live_pos
             pass
        
        return None

    def _place_order(self, symbol, order_type, side, quantity, price=None, params=None, current_simulated_time_utc=None):
        if current_simulated_time_utc:
             self.logger.info(f"BACKTEST_SIM: Place {side} {order_type} for {quantity} {symbol} at {price if price else 'Market'}")
             return {"id": f"sim_{datetime.utcnow().timestamp()}", "status": "open", "simulated": True, "price": price, "amount": quantity}
        try:
            formatted_qty = self._format_quantity(quantity)
            formatted_price = self._format_price(price) if price else None
            order = self.exchange_ccxt.create_order(symbol, order_type, side, float(formatted_qty), float(formatted_price) if formatted_price else None, params)
            self.logger.info(f"Order placed: {side} {order_type} {formatted_qty} {symbol} at {formatted_price if formatted_price else 'Market'}. OrderID: {order.get('id')}")
            # TODO: Record order in DB
            return order
        except Exception as e:
            self.logger.error(f"Failed to place {side} {order_type} for {symbol}: {e}", exc_info=True)
        return None

    def _close_all_positions(self, symbol, current_simulated_time_utc=None, historical_data_feed=None, is_backtest=False):
        pos_details = self._get_current_position_details(symbol, historical_data_feed, is_backtest)
        if pos_details:
            close_side = 'sell' if pos_details['side'] == 'long' else 'buy'
            self.logger.info(f"Closing all positions for {symbol} (side: {pos_details['side']}, qty: {pos_details['qty']}).")
            self._place_order(symbol, 'MARKET', close_side, pos_details['qty'], params={'reduceOnly': True}, current_simulated_time_utc=current_simulated_time_utc)
            # TODO: DB Update for position close.
            # self._cancel_active_sl_tp_orders(symbol, current_simulated_time_utc) # Assumes this method exists
            self.active_position_side = None
            self.position_entry_price = None
            self.position_qty = Decimal("0")
            self.active_sl_tp_orders = {}


    def _process_trading_logic(self, chart_ohlcv_df, htf_ohlcv_df, is_backtest=False, current_simulated_time_utc=None, historical_data_feed=None):
        chart_macd_vals, chart_signal_vals, _ = self._calculate_macd_values(chart_ohlcv_df)
        htf_macd_vals, htf_signal_vals, _ = self._calculate_macd_values(htf_ohlcv_df)

        if chart_macd_vals is None or htf_macd_vals is None or len(chart_macd_vals) < 2 or len(htf_macd_vals) < 1:
            self.logger.warning("Not enough data for MACD calculations.")
            return {"action": "HOLD", "reason": "MACD calc error"} if is_backtest else None
        
        latest_chart_macd = chart_macd_vals[-1]
        latest_chart_signal = chart_signal_vals[-1]
        prev_chart_macd = chart_macd_vals[-2]
        prev_chart_signal = chart_signal_vals[-2]
        latest_htf_macd = htf_macd_vals[-1]
        latest_htf_signal = htf_signal_vals[-1]

        is_chart_uptrend, is_chart_downtrend = self._determine_trend(latest_chart_macd, latest_chart_signal)
        is_htf_uptrend, _ = self._determine_trend(latest_htf_macd, latest_htf_signal)
        _, is_htf_downtrend = self._determine_trend(latest_htf_macd, latest_htf_signal)

        chart_trigger_up = prev_chart_macd <= prev_chart_signal and latest_chart_macd > latest_chart_signal
        chart_trigger_down = prev_chart_macd >= prev_chart_signal and latest_chart_macd < latest_chart_signal

        current_close_price = Decimal(str(chart_ohlcv_df['close'].iloc[-1]))

        if is_chart_uptrend:
            if not self.is_prev_chart_uptrend:
                self.current_uptrend_init_price = current_close_price
                self.current_uptrend_idx = 0
                self.logger.info(f"Chart uptrend started. Init price: {self.current_uptrend_init_price}, Index: {self.current_uptrend_idx}")
                if chart_trigger_up:
                    self._calculate_forecast_bands(1, self.current_uptrend_init_price)
            else:
                self.current_uptrend_idx += 1
            self._populate_memory(1, self.current_uptrend_idx, self.current_uptrend_init_price, current_close_price)
        
        if is_chart_downtrend:
            if self.is_prev_chart_uptrend is None or self.is_prev_chart_uptrend:
                self.current_downtrend_init_price = current_close_price
                self.current_downtrend_idx = 0
                self.logger.info(f"Chart downtrend started. Init price: {self.current_downtrend_init_price}, Index: {self.current_downtrend_idx}")
                if chart_trigger_down:
                    self._calculate_forecast_bands(0, self.current_downtrend_init_price)
            else:
                self.current_downtrend_idx += 1
            self._populate_memory(0, self.current_downtrend_idx, self.current_downtrend_init_price, current_close_price)

        self.is_prev_chart_uptrend = is_chart_uptrend

        long_condition = chart_trigger_up and is_chart_uptrend and is_htf_uptrend
        short_condition = chart_trigger_down and is_chart_downtrend and is_htf_downtrend

        # Update position details before making decisions
        current_pos_details = self._get_current_position_details(self.trading_pair, historical_data_feed, is_backtest)
        if current_pos_details and self.active_position_side is None: # Adopt if strategy doesn't know about it
            self.active_position_side = current_pos_details['side']
            self.position_entry_price = current_pos_details['entry_price']
            self.position_qty = current_pos_details['qty']
            self.logger.info(f"Adopted position found: {self.active_position_side} Qty: {self.position_qty}")


        if (long_condition and self.active_position_side == 'short') or \
           (short_condition and self.active_position_side == 'long'):
            self.logger.info("Signal opposite to current position. Closing position first.")
            self._close_all_positions(self.trading_pair, current_simulated_time_utc, historical_data_feed, is_backtest)
        
        action_taken_this_cycle = False
        if not self.active_position_side:
            entry_price = current_close_price
            if entry_price == Decimal("0"): # Avoid division by zero
                self.logger.warning("Entry price is zero, cannot calculate quantity.")
                return {"action": "HOLD", "reason": "Entry price is zero"} if is_backtest else None
            qty_to_trade = self.order_quantity_usd / entry_price

            sl_price = None
            tp_price = None

            if long_condition:
                self.logger.info(f"Trading Signal: ENTER LONG for {self.trading_pair} at {entry_price}")
                entry_order = self._place_order(self.trading_pair, 'MARKET', 'buy', qty_to_trade, current_simulated_time_utc=current_simulated_time_utc)
                if entry_order or is_backtest:
                    self.active_position_side = 'long'
                    self.position_entry_price = entry_price
                    self.position_qty = qty_to_trade
                    self.logger.info(f"DB_TODO: Create Position LONG {self.trading_pair}")
                    if self.use_stop_loss: sl_price = entry_price * (Decimal('1') - self.stop_loss_pct)
                    if self.use_take_profit: tp_price = entry_price * (Decimal('1') + self.take_profit_pct)
                    if sl_price:
                        sl_ord = self._place_order(self.trading_pair, 'STOP_MARKET', 'sell', qty_to_trade, params={'stopPrice': self._format_price(sl_price), 'reduceOnly': True}, current_simulated_time_utc=current_simulated_time_utc)
                        if sl_ord: self.active_sl_tp_orders['sl_id'] = sl_ord.get('id')
                    if tp_price:
                        tp_ord = self._place_order(self.trading_pair, 'TAKE_PROFIT_MARKET', 'sell', qty_to_trade, params={'stopPrice': self._format_price(tp_price), 'reduceOnly': True}, current_simulated_time_utc=current_simulated_time_utc)
                        if tp_ord: self.active_sl_tp_orders['tp_id'] = tp_ord.get('id')
                    action_taken_this_cycle = True
                    if is_backtest: return {"action": "ENTER_LONG", "price": float(entry_price), "qty": float(qty_to_trade), "sl": float(sl_price) if sl_price else None, "tp": float(tp_price) if tp_price else None}

            elif short_condition:
                self.logger.info(f"Trading Signal: ENTER SHORT for {self.trading_pair} at {entry_price}")
                entry_order = self._place_order(self.trading_pair, 'MARKET', 'sell', qty_to_trade, current_simulated_time_utc=current_simulated_time_utc)
                if entry_order or is_backtest: # If order placed or simulating
                    self.active_position_side = 'short'
                    self.position_entry_price = entry_price
                    self.position_qty = qty_to_trade
                    self.logger.info(f"DB_TODO: Create Position SHORT {self.trading_pair}")
                    if self.use_stop_loss: sl_price = entry_price * (Decimal('1') + self.stop_loss_pct)
                    if self.use_take_profit: tp_price = entry_price * (Decimal('1') - self.take_profit_pct)
                    if sl_price:
                        sl_ord = self._place_order(self.trading_pair, 'STOP_MARKET', 'buy', qty_to_trade, params={'stopPrice': self._format_price(sl_price), 'reduceOnly': True}, current_simulated_time_utc=current_simulated_time_utc)
                        if sl_ord: self.active_sl_tp_orders['sl_id'] = sl_ord.get('id')
                    if tp_price:
                        tp_ord = self._place_order(self.trading_pair, 'TAKE_PROFIT_MARKET', 'buy', qty_to_trade, params={'stopPrice': self._format_price(tp_price), 'reduceOnly': True}, current_simulated_time_utc=current_simulated_time_utc)
                        if tp_ord: self.active_sl_tp_orders['tp_id'] = tp_ord.get('id')
                    action_taken_this_cycle = True
                    if is_backtest: return {"action": "ENTER_SHORT", "price": float(entry_price), "qty": float(qty_to_trade), "sl": float(sl_price) if sl_price else None, "tp": float(tp_price) if tp_price else None}

        if action_taken_this_cycle or self.current_uptrend_idx == 0 or self.current_downtrend_idx == 0 :
            self._save_persistent_state()

        if is_backtest and not action_taken_this_cycle:
            return {"action": "HOLD", "reason": "No entry signal or already in position"}
        return None


    def execute_live_signal(self, market_data_df=None):
        self.logger.info(f"Executing {self.name} for {self.trading_pair} UserSubID {self.user_sub_obj.id}")

        if not self.trading_pair or self.trading_pair == "UNSPECIFIED/USDT":
             self.trading_pair = self.params.get("trading_pair", "BTC/USDT")
             if not self.trading_pair or self.trading_pair == "UNSPECIFIED/USDT":
                  self.logger.error("Trading pair not specified for this strategy instance.")
                  return

        if self.is_prev_chart_uptrend is None and self.active_position_side is None: # Simple check if state seems uninitialized and no known active pos
            self._load_persistent_state()

        # TODO: Check if a live position was closed by SL/TP on exchange by querying active orders/positions.
        # If self.active_position_side is set, but fetch_positions shows no such position, reset state.
        # This is important for live trading to correctly re-sync with exchange state.
        # Example:
        # actual_live_pos = self.exchange_ccxt.fetch_positions([self.trading_pair]) # Simplified
        # if not actual_live_pos_found and self.active_position_side:
        #    self.logger.info("Position seems closed on exchange. Resetting state.")
        #    self._reset_state_after_external_close() # New method to handle this cleanup
        #    self._save_persistent_state()


        chart_limit = max(self.macd_slow_len, self.macd_signal_len) + self.forecast_max_memory + 50
        htf_limit = max(self.macd_slow_len, self.macd_signal_len) + 50

        try:
            chart_ohlcv_list = self._get_ohlcv_data(self.chart_interval, chart_limit, is_backtest=False)
            htf_ohlcv_list = self._get_ohlcv_data(self.htf_interval, htf_limit, is_backtest=False)

            min_chart_len = max(self.macd_slow_len, self.macd_signal_len) +2 # Need at least 2 for prev values
            min_htf_len = max(self.macd_slow_len, self.macd_signal_len) +1

            if not chart_ohlcv_list or len(chart_ohlcv_list) < min_chart_len or \
               not htf_ohlcv_list or len(htf_ohlcv_list) < min_htf_len:
                self.logger.warning("Insufficient OHLCV data for chart or HTF for MACD calc.")
                return

            chart_ohlcv_df = pandas.DataFrame(chart_ohlcv_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            chart_ohlcv_df['timestamp'] = pandas.to_datetime(chart_ohlcv_df['timestamp'], unit='ms')
            # chart_ohlcv_df.set_index('timestamp', inplace=True) # Not strictly needed if using iloc

            htf_ohlcv_df = pandas.DataFrame(htf_ohlcv_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            htf_ohlcv_df['timestamp'] = pandas.to_datetime(htf_ohlcv_df['timestamp'], unit='ms')
            # htf_ohlcv_df.set_index('timestamp', inplace=True)

            self._process_trading_logic(chart_ohlcv_df, htf_ohlcv_df, is_backtest=False)

        except Exception as e:
            self.logger.error(f"Error in {self.name} execute_live_signal: {e}", exc_info=True)


    def run_backtest(self, historical_data_feed, current_simulated_time_utc):
        chart_limit = max(self.macd_slow_len, self.macd_signal_len) + self.forecast_max_memory + 50
        htf_limit = max(self.macd_slow_len, self.macd_signal_len) + 50

        chart_ohlcv_df = historical_data_feed.get_ohlcv(self.trading_pair, self.chart_interval, chart_limit, end_time_utc=current_simulated_time_utc)
        htf_ohlcv_df = historical_data_feed.get_ohlcv(self.trading_pair, self.htf_interval, htf_limit, end_time_utc=current_simulated_time_utc)

        min_chart_len = max(self.macd_slow_len, self.macd_signal_len) +2
        min_htf_len = max(self.macd_slow_len, self.macd_signal_len) +1

        if chart_ohlcv_df is None or chart_ohlcv_df.empty or htf_ohlcv_df is None or htf_ohlcv_df.empty or \
           len(chart_ohlcv_df) < min_chart_len or len(htf_ohlcv_df) < min_htf_len:
            return {"action": "HOLD", "reason": "Insufficient historical data for MACD"}

        return self._process_trading_logic(chart_ohlcv_df, htf_ohlcv_df, is_backtest=True,
                                           current_simulated_time_utc=current_simulated_time_utc,
                                           historical_data_feed=historical_data_feed)

```
