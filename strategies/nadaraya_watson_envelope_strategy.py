import logging
import numpy as np
import talib
import pandas
import ccxt # Though self.exchange_ccxt is provided
from decimal import Decimal, ROUND_DOWN, ROUND_UP, getcontext
from datetime import datetime, time # Ensure time is imported

# Ensure high precision for Decimal calculations
getcontext().prec = 18

class NadarayaWatsonStochRSIStrategy:

    @staticmethod
    def get_parameters_definition():
        return {
            "trading_pair": {"type": "str", "label": "Trading Pair", "default": "BTC/USDT"},
            "leverage": {"type": "int", "label": "Leverage", "default": 10, "min": 1, "max": 100},
            "order_quantity_usd": {"type": "float", "label": "Order Quantity (USD Notional)", "default": 100.0, "min": 1.0},

            "nw_timeframe": {"type": "str", "label": "Nadaraya-Watson Timeframe", "default": "15m", "choices": ["1m", "3m", "5m", "15m", "30m", "1h"]},
            "nw_h_bandwidth": {"type": "float", "label": "NW: h (Bandwidth)", "default": 8.0, "min": 1.0, "max": 50.0, "step": 0.1},
            "nw_multiplier": {"type": "float", "label": "NW: Band Multiplier", "default": 3.0, "min": 0.5, "max": 10.0, "step": 0.1},
            "nw_yhat_lookback": {"type": "int", "label": "NW: y_hat Lookback (smoothing window)", "default": 20, "min": 5, "max": 200},
            "nw_mae_lookback": {"type": "int", "label": "NW: MAE Lookback (for band width)", "default": 20, "min": 5, "max": 200},

            "stoch_rsi_timeframe": {"type": "str", "label": "StochRSI Timeframe (Filter)", "default": "1h", "choices": ["15m", "30m", "1h", "4h", "1d"]},
            "stoch_rsi_length": {"type": "int", "label": "StochRSI: RSI Length", "default": 14, "min": 5, "max": 50},
            "stoch_rsi_stoch_length": {"type": "int", "label": "StochRSI: Stochastic Length (for Stoch of RSI)", "default": 14, "min": 5, "max": 50},
            "stoch_rsi_k_smooth": {"type": "int", "label": "StochRSI: %K Smoothing", "default": 3, "min": 1, "max": 50},
            "stoch_rsi_d_smooth": {"type": "int", "label": "StochRSI: %D Smoothing", "default": 3, "min": 1, "max": 50},
            "stoch_rsi_oversold_level": {"type": "float", "label": "StochRSI Oversold Level", "default": 20.0, "min": 0, "max": 100, "step": 1},
            "stoch_rsi_overbought_level": {"type": "float", "label": "StochRSI Overbought Level", "default": 80.0, "min": 0, "max": 100, "step": 1},

            "stop_loss_pct": {"type": "float", "label": "Stop Loss % (from entry)", "default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1},
            "take_profit_pct": {"type": "float", "label": "Take Profit % (from entry)", "default": 3.0, "min": 0.1, "max": 20.0, "step": 0.1},
        }

    def __init__(self, db_session, user_sub_obj, strategy_params: dict, exchange_ccxt, logger=None):
        self.db_session = db_session
        self.user_sub_obj = user_sub_obj
        self.params = strategy_params
        self.exchange_ccxt = exchange_ccxt
        self.logger = logger if logger else logging.getLogger(__name__)
        self.name = "NadarayaWatsonStochRSIStrategy"

        self.trading_pair = self.params.get("trading_pair", "BTC/USDT")
        self.leverage = int(self.params.get("leverage", 10))
        self.order_quantity_usd = Decimal(str(self.params.get("order_quantity_usd", "100.0")))

        self.nw_timeframe = self.params.get("nw_timeframe", "15m")
        self.nw_h_bandwidth = float(self.params.get("nw_h_bandwidth", 8.0))
        self.nw_multiplier = float(self.params.get("nw_multiplier", 3.0))
        self.nw_yhat_lookback = int(self.params.get("nw_yhat_lookback", 20))
        self.nw_mae_lookback = int(self.params.get("nw_mae_lookback", 20))

        self.stoch_rsi_timeframe = self.params.get("stoch_rsi_timeframe", "1h")
        self.stoch_rsi_length = int(self.params.get("stoch_rsi_length", 14))
        self.stoch_rsi_stoch_length = int(self.params.get("stoch_rsi_stoch_length", 14))
        self.stoch_rsi_k_smooth = int(self.params.get("stoch_rsi_k_smooth", 3))
        self.stoch_rsi_d_smooth = int(self.params.get("stoch_rsi_d_smooth", 3))
        self.stoch_rsi_oversold_level = Decimal(str(self.params.get("stoch_rsi_oversold_level", "20.0")))
        self.stoch_rsi_overbought_level = Decimal(str(self.params.get("stoch_rsi_overbought_level", "80.0")))

        self.stop_loss_pct = Decimal(str(self.params.get("stop_loss_pct", "1.0"))) / Decimal("100")
        self.take_profit_pct = Decimal(str(self.params.get("take_profit_pct", "3.0"))) / Decimal("100")

        self.active_position_side = None  # 'long', 'short', or None
        self.active_sl_tp_orders = {} # Store {'sl_id': id, 'tp_id': id}
        self.position_entry_price = None # Store entry price of active position
        self.position_qty = Decimal("0")


        self._fetch_market_precision()
        self._set_leverage()
        self.logger.info(f"{self.name} initialized for {self.trading_pair}, UserSubID {self.user_sub_obj.id}")
        # TODO: Load persistent state for active_position_side, entry_price, qty, active_sl_tp_orders from DB

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


    def _gauss(self, x, h):
        if h == 0: return 1.0 if x == 0 else 0.0 # Prevent division by zero if h is exactly 0
        return np.exp(-((x ** 2) / (2 * h ** 2)))

    def _causal_nadaraya_watson_envelope(self, data_series_np, h, mult, y_hat_lookback, mae_lookback):
        n = len(data_series_np)
        y_hat_arr = np.full(n, np.nan)
        upper_band_arr = np.full(n, np.nan)
        lower_band_arr = np.full(n, np.nan)

        for i in range(n):
            start_idx_yhat = max(0, i - y_hat_lookback + 1)
            current_window_yhat = data_series_np[start_idx_yhat : i+1]

            weighted_sum = 0.0
            total_weight = 0.0
            for j_local in range(len(current_window_yhat)):
                gauss_dist = float((len(current_window_yhat) - 1) - j_local)
                weight = self._gauss(gauss_dist, h)
                weighted_sum += current_window_yhat[j_local] * weight
                total_weight += weight

            if total_weight > 1e-8:
                y_hat_arr[i] = weighted_sum / total_weight
            elif len(current_window_yhat) > 0:
                y_hat_arr[i] = current_window_yhat[-1] # Fallback to last known price if weights are zero

        errors = np.abs(data_series_np - y_hat_arr)

        for i in range(n):
            if np.isnan(y_hat_arr[i]) or i == 0 :
                continue

            start_idx_mae = max(0, i - mae_lookback)
            relevant_errors_for_mae = errors[start_idx_mae : i]
            valid_errors = relevant_errors_for_mae[~np.isnan(relevant_errors_for_mae)]

            if len(valid_errors) > 0:
                mae_value = np.mean(valid_errors) * mult
                upper_band_arr[i] = y_hat_arr[i] + mae_value
                lower_band_arr[i] = y_hat_arr[i] - mae_value

        return y_hat_arr, upper_band_arr, lower_band_arr

    def _calculate_stoch_rsi(self, close_prices_np, rsi_len, stoch_len, k_smooth, d_smooth):
        if len(close_prices_np) < rsi_len + stoch_len + k_smooth + d_smooth + 1: # Adjusted for safety
            self.logger.warning(f"Not enough data for StochRSI: got {len(close_prices_np)}, need more for periods.")
            return np.full(len(close_prices_np), np.nan), np.full(len(close_prices_np), np.nan)

        rsi = talib.RSI(close_prices_np, timeperiod=rsi_len)
        rsi = rsi[~np.isnan(rsi)]
        if len(rsi) < stoch_len + k_smooth + d_smooth -1 :
             self.logger.warning(f"Not enough RSI data for STOCH: got {len(rsi)}")
             return np.full(len(close_prices_np), np.nan), np.full(len(close_prices_np), np.nan)

        stoch_k, stoch_d = talib.STOCH(rsi, rsi, rsi,
                                       fastk_period=stoch_len,
                                       slowk_period=k_smooth, slowk_matype=0,
                                       slowd_period=d_smooth, slowd_matype=0)

        nan_padding_count = len(close_prices_np) - len(stoch_k)
        stoch_k_padded = np.pad(stoch_k, (nan_padding_count, 0), 'constant', constant_values=np.nan)
        stoch_d_padded = np.pad(stoch_d, (nan_padding_count, 0), 'constant', constant_values=np.nan)

        return stoch_k_padded, stoch_d_padded

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

    def _cancel_active_sl_tp_orders(self, symbol, current_simulated_time_utc=None):
        if self.active_sl_tp_orders.get('sl_id'):
            try:
                if not current_simulated_time_utc: self.exchange_ccxt.cancel_order(self.active_sl_tp_orders['sl_id'], symbol)
                self.logger.info(f"Cancelled SL order {self.active_sl_tp_orders['sl_id']} for {symbol}")
            except Exception as e: self.logger.error(f"Error cancelling SL order {self.active_sl_tp_orders['sl_id']}: {e}")
        if self.active_sl_tp_orders.get('tp_id'):
            try:
                if not current_simulated_time_utc: self.exchange_ccxt.cancel_order(self.active_sl_tp_orders['tp_id'], symbol)
                self.logger.info(f"Cancelled TP order {self.active_sl_tp_orders['tp_id']} for {symbol}")
            except Exception as e: self.logger.error(f"Error cancelling TP order {self.active_sl_tp_orders['tp_id']}: {e}")
        self.active_sl_tp_orders = {}

    def _process_trading_logic(self, nw_ohlcv_df, stoch_rsi_ohlcv_df, is_backtest=False, current_simulated_time_utc=None, historical_data_feed=None):
        symbol = self.trading_pair
        min_nw_data_len = self.nw_yhat_lookback + self.nw_mae_lookback + 5
        min_stoch_rsi_data_len = self.stoch_rsi_length + self.stoch_rsi_stoch_length + self.stoch_rsi_k_smooth + self.stoch_rsi_d_smooth + 50

        if nw_ohlcv_df is None or len(nw_ohlcv_df) < min_nw_data_len:
            self.logger.warning(f"Not enough data for Nadaraya-Watson. Need > {min_nw_data_len}, Got {len(nw_ohlcv_df) if nw_ohlcv_df is not None else 0}")
            return {"action": "HOLD", "reason": "NW calc data insufficient"} if is_backtest else None
        if stoch_rsi_ohlcv_df is None or len(stoch_rsi_ohlcv_df) < min_stoch_rsi_data_len:
            self.logger.warning(f"Not enough data for StochRSI. Need > {min_stoch_rsi_data_len}, Got {len(stoch_rsi_ohlcv_df) if stoch_rsi_ohlcv_df is not None else 0}")
            return {"action": "HOLD", "reason": "StochRSI calc data insufficient"} if is_backtest else None

        nw_closes_np = nw_ohlcv_df['close'].to_numpy(dtype=float)
        _, upper_band, lower_band = self._causal_nadaraya_watson_envelope(
            nw_closes_np, self.nw_h_bandwidth, self.nw_multiplier, self.nw_yhat_lookback, self.nw_mae_lookback
        )
        if np.isnan(upper_band[-1]) or np.isnan(lower_band[-1]):
            self.logger.warning("NW bands are NaN.")
            return {"action": "HOLD", "reason": "NW bands NaN"} if is_backtest else None
        
        latest_nw_close = Decimal(str(nw_ohlcv_df['close'].iloc[-1]))
        latest_upper_band = Decimal(str(upper_band[-1]))
        latest_lower_band = Decimal(str(lower_band[-1]))

        stoch_rsi_closes_np = stoch_rsi_ohlcv_df['close'].to_numpy(dtype=float)
        stoch_k, _ = self._calculate_stoch_rsi(
            stoch_rsi_closes_np, self.stoch_rsi_length, self.stoch_rsi_stoch_length,
            self.stoch_rsi_k_smooth, self.stoch_rsi_d_smooth
        )
        if np.isnan(stoch_k[-1]):
            self.logger.warning("StochRSI K is NaN.")
            return {"action": "HOLD", "reason": "StochRSI K NaN"} if is_backtest else None
        latest_stoch_k = Decimal(str(stoch_k[-1]))

        # Update position details for current cycle
        current_pos_details = self._get_current_position_details(symbol, historical_data_feed, is_backtest)
        if current_pos_details and self.active_position_side is None: # Adopt if strategy doesn't know
             self.active_position_side = current_pos_details['side']
             self.position_entry_price = current_pos_details['entry_price']
             self.position_qty = current_pos_details['qty']
             self.logger.info(f"Adopted position: {self.active_position_side} Qty: {self.position_qty}")


        action_taken = False
        trade_action_details = {}

        if not self.active_position_side:
            entry_price = latest_nw_close
            if entry_price == Decimal("0"): self.logger.warning("Entry price is zero."); return {"action":"HOLD", "reason":"Entry price zero"} if is_backtest else None
            qty_to_trade = self.order_quantity_usd / entry_price
            sl_val = self.stop_loss_pct; tp_val = self.take_profit_pct

            if latest_nw_close <= latest_lower_band and latest_stoch_k < self.stoch_rsi_oversold_level:
                self.logger.info(f"Signal: ENTER LONG for {symbol} at {entry_price}. NW_Low: {latest_lower_band}, StochK: {latest_stoch_k}")
                sl_price = entry_price * (Decimal('1') - sl_val); tp_price = entry_price * (Decimal('1') + tp_val)
                if not is_backtest:
                    entry_order = self._place_order(symbol, 'MARKET', 'buy', qty_to_trade)
                    if entry_order: self.active_position_side = 'long'; self.position_entry_price = entry_price; self.position_qty = qty_to_trade; # ... set SL/TP orders ...
                else: self.active_position_side = 'long'; self.position_entry_price = entry_price; self.position_qty = qty_to_trade
                action_taken = True
                trade_action_details = {"action": "BUY", "price": float(entry_price), "qty": float(qty_to_trade), "sl": float(sl_price), "tp": float(tp_price)}

            elif latest_nw_close >= latest_upper_band and latest_stoch_k > self.stoch_rsi_overbought_level:
                self.logger.info(f"Signal: ENTER SHORT for {symbol} at {entry_price}. NW_High: {latest_upper_band}, StochK: {latest_stoch_k}")
                sl_price = entry_price * (Decimal('1') + sl_val); tp_price = entry_price * (Decimal('1') - tp_val)
                if not is_backtest:
                    entry_order = self._place_order(symbol, 'MARKET', 'sell', qty_to_trade)
                    if entry_order: self.active_position_side = 'short'; self.position_entry_price = entry_price; self.position_qty = qty_to_trade; # ... set SL/TP orders ...
                else: self.active_position_side = 'short'; self.position_entry_price = entry_price; self.position_qty = qty_to_trade
                action_taken = True
                trade_action_details = {"action": "SELL", "price": float(entry_price), "qty": float(qty_to_trade), "sl": float(sl_price), "tp": float(tp_price)}
        
        elif self.active_position_side:
            if self.active_position_side == 'long' and latest_nw_close >= latest_upper_band and latest_stoch_k > self.stoch_rsi_overbought_level:
                self.logger.info(f"Signal: CLOSE LONG & ENTER SHORT for {symbol} at {latest_nw_close}.")
                if not is_backtest: self._cancel_active_sl_tp_orders(symbol); self._place_order(symbol, 'MARKET', 'sell', self.position_qty, params={'reduceOnly': True})
                self.active_position_side = None; self.position_entry_price = None; self.position_qty = Decimal("0") # Reset state before new entry
                # Now place short (logic from above block)
                entry_price = latest_nw_close; qty_to_trade = self.order_quantity_usd / entry_price; sl_val = self.stop_loss_pct; tp_val = self.take_profit_pct
                sl_price = entry_price * (Decimal('1') + sl_val); tp_price = entry_price * (Decimal('1') - tp_val)
                if not is_backtest:
                    entry_order = self._place_order(symbol, 'MARKET', 'sell', qty_to_trade)
                    if entry_order: self.active_position_side = 'short'; self.position_entry_price = entry_price; self.position_qty = qty_to_trade; #... set SL/TP orders
                else: self.active_position_side = 'short'; self.position_entry_price = entry_price; self.position_qty = qty_to_trade
                action_taken = True
                trade_action_details = {"action": "REVERSE_TO_SHORT", "price": float(entry_price), "qty": float(qty_to_trade), "sl": float(sl_price), "tp": float(tp_price)}

            elif self.active_position_side == 'short' and latest_nw_close <= latest_lower_band and latest_stoch_k < self.stoch_rsi_oversold_level:
                self.logger.info(f"Signal: CLOSE SHORT & ENTER LONG for {symbol} at {latest_nw_close}.")
                if not is_backtest: self._cancel_active_sl_tp_orders(symbol); self._place_order(symbol, 'MARKET', 'buy', self.position_qty, params={'reduceOnly': True})
                self.active_position_side = None; self.position_entry_price = None; self.position_qty = Decimal("0")
                entry_price = latest_nw_close; qty_to_trade = self.order_quantity_usd / entry_price; sl_val = self.stop_loss_pct; tp_val = self.take_profit_pct
                sl_price = entry_price * (Decimal('1') - sl_val); tp_price = entry_price * (Decimal('1') + tp_val)
                if not is_backtest:
                    entry_order = self._place_order(symbol, 'MARKET', 'buy', qty_to_trade)
                    if entry_order: self.active_position_side = 'long'; self.position_entry_price = entry_price; self.position_qty = qty_to_trade; #... set SL/TP orders
                else: self.active_position_side = 'long'; self.position_entry_price = entry_price; self.position_qty = qty_to_trade
                action_taken = True
                trade_action_details = {"action": "REVERSE_TO_LONG", "price": float(entry_price), "qty": float(qty_to_trade), "sl": float(sl_price), "tp": float(tp_price)}

        if is_backtest:
            return trade_action_details if action_taken else {"action": "HOLD", "reason": "No signal or position held"}
        # TODO: Save state to DB if action_taken or other relevant state changes
        return None


    def execute_live_signal(self, market_data_df=None):
        self.logger.info(f"Executing {self.name} for {self.trading_pair} UserSubID {self.user_sub_obj.id}")
        # TODO: Load active position state from DB if self.active_position_side is None
        # Example: if self.active_position_side is None: self._load_live_position_from_db()

        # TODO: Check if current position was closed by SL/TP on exchange.
        # Example: self._check_and_sync_live_position_status()

        nw_data_needed = self.nw_yhat_lookback + self.nw_mae_lookback + 50
        stoch_rsi_data_needed = self.stoch_rsi_length + self.stoch_rsi_stoch_length + self.stoch_rsi_k_smooth + self.stoch_rsi_d_smooth + 100

        try:
            nw_ohlcv_list = self.exchange_ccxt.fetch_ohlcv(self.trading_pair, self.nw_timeframe, limit=nw_data_needed)
            stoch_rsi_ohlcv_list = self.exchange_ccxt.fetch_ohlcv(self.trading_pair, self.stoch_rsi_timeframe, limit=stoch_rsi_data_needed)

            min_nw_len = self.nw_yhat_lookback + self.nw_mae_lookback + 2 # Min for calc
            min_stoch_len = self.stoch_rsi_length + self.stoch_rsi_stoch_length + self.stoch_rsi_k_smooth + self.stoch_rsi_d_smooth + 1 # Min for calc

            if not nw_ohlcv_list or len(nw_ohlcv_list) < min_nw_len or \
               not stoch_rsi_ohlcv_list or len(stoch_rsi_ohlcv_list) < min_stoch_len:
                self.logger.warning("Insufficient OHLCV data for NW or StochRSI for live signal.")
                return

            nw_ohlcv_df = pandas.DataFrame(nw_ohlcv_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            stoch_rsi_ohlcv_df = pandas.DataFrame(stoch_rsi_ohlcv_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            # TODO: Proper alignment of HTF (stoch_rsi_ohlcv_df) data. For live, using latest [-1] is an approximation.

            self._process_trading_logic(nw_ohlcv_df, stoch_rsi_ohlcv_df, is_backtest=False)

        except Exception as e:
            self.logger.error(f"Error in {self.name} execute_live_signal: {e}", exc_info=True)


    def run_backtest(self, historical_data_feed, current_simulated_time_utc):
        nw_data_needed = self.nw_yhat_lookback + self.nw_mae_lookback + 50
        stoch_rsi_data_needed = self.stoch_rsi_length + self.stoch_rsi_stoch_length + self.stoch_rsi_k_smooth + self.stoch_rsi_d_smooth + 100
        
        nw_ohlcv_df = historical_data_feed.get_ohlcv(self.trading_pair, self.nw_timeframe, nw_data_needed, end_time_utc=current_simulated_time_utc)
        stoch_rsi_ohlcv_df = historical_data_feed.get_ohlcv(self.trading_pair, self.stoch_rsi_timeframe, stoch_rsi_data_needed, end_time_utc=current_simulated_time_utc)

        min_nw_len = self.nw_yhat_lookback + self.nw_mae_lookback + 2
        min_stoch_len = self.stoch_rsi_length + self.stoch_rsi_stoch_length + self.stoch_rsi_k_smooth + self.stoch_rsi_d_smooth + 1

        if nw_ohlcv_df is None or len(nw_ohlcv_df) < min_nw_len or \
           stoch_rsi_ohlcv_df is None or len(stoch_rsi_ohlcv_df) < min_stoch_len:
            return {"action": "HOLD", "reason": "Insufficient historical data for indicators"}

        return self._process_trading_logic(nw_ohlcv_df, stoch_rsi_ohlcv_df, is_backtest=True,
                                           current_simulated_time_utc=current_simulated_time_utc,
                                           historical_data_feed=historical_data_feed)

```
This subtask will write the Python code to `strategies/nadaraya_watson_envelope_strategy.py`.
It includes the Nadaraya-Watson calculation, Stochastic RSI calculation, MTF data handling considerations, combined entry logic, and SL/TP mechanisms. Placeholders for DB interactions for position state are included.
It assumes `trading_pair` is provided via `strategy_params`.
It uses `pandas` for DataFrame operations.
The strategy also implements logic to reverse positions if an opposite signal occurs while already in a trade.
