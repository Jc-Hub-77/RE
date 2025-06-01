import logging
import numpy as np
import talib
import ccxt
from decimal import Decimal, ROUND_DOWN, ROUND_UP, getcontext
from datetime import datetime, time # Ensure time is imported

# Ensure high precision for Decimal calculations
getcontext().prec = 18 # Set precision for Decimal

# Need to ensure pandas is imported if using DataFrames
import pandas

class TheOrphanStrategy:

    @staticmethod
    def get_parameters_definition():
        return {
            "bb_length": {"type": "int", "label": "BB Length", "default": 24, "min": 5, "max": 100},
            "bb_stdev": {"type": "float", "label": "BB StdDev", "default": 2.1, "min": 0.5, "max": 5.0, "step": 0.1},
            "trend_ema_period": {"type": "int", "label": "Trend EMA Period", "default": 365, "min": 50, "max": 500},
            "vol_filter_stdev_length": {"type": "int", "label": "Volatility STDEV Length", "default": 15, "min": 5, "max": 50},
            "vol_filter_sma_length": {"type": "int", "label": "Volatility STDEV's SMA Length", "default": 28, "min": 5, "max": 100},
            "stop_loss_pct": {"type": "float", "label": "Stop Loss %", "default": 2.0, "min": 0.1, "max": 20.0, "step": 0.1},
            "take_profit_pct": {"type": "float", "label": "Take Profit %", "default": 9.0, "min": 0.1, "max": 50.0, "step": 0.1},
            "trailing_stop_activation_pct": {"type": "float", "label": "Trailing Stop Activation % Profit", "default": 0.5, "min": 0.1, "max": 10.0, "step": 0.1},
            "trailing_stop_offset_pct": {"type": "float", "label": "Trailing Stop Offset %", "default": 0.5, "min": 0.1, "max": 10.0, "step": 0.1},
            "kline_interval": {"type": "str", "label": "Kline Interval", "default": "1h", "choices": ["15m", "30m", "1h", "4h", "1d"]},
            "leverage": {"type": "int", "label": "Leverage", "default": 10, "min": 1, "max": 100},
            "order_quantity_usd": {"type": "float", "label": "Order Quantity (USD Notional)", "default": 100.0, "min": 1.0}
        }

    def __init__(self, db_session, user_sub_obj, strategy_params: dict, exchange_ccxt, logger=None):
        self.db_session = db_session
        self.user_sub_obj = user_sub_obj
        self.params = strategy_params
        self.exchange_ccxt = exchange_ccxt
        self.logger = logger if logger else logging.getLogger(__name__)
        self.name = "TheOrphanStrategy"
        self.trading_pair = self.params.get("trading_pair", "UNSPECIFIED/USDT") # Should be set by platform via user subscription

        self.bb_length = int(self.params.get("bb_length", 24))
        self.bb_stdev = float(self.params.get("bb_stdev", 2.1))
        self.trend_ema_period = int(self.params.get("trend_ema_period", 365))
        self.vol_filter_stdev_length = int(self.params.get("vol_filter_stdev_length", 15))
        self.vol_filter_sma_length = int(self.params.get("vol_filter_sma_length", 28))
        
        self.stop_loss_pct = Decimal(str(self.params.get("stop_loss_pct", "2.0"))) / Decimal("100")
        self.take_profit_pct = Decimal(str(self.params.get("take_profit_pct", "9.0"))) / Decimal("100")
        self.trailing_stop_activation_pct = Decimal(str(self.params.get("trailing_stop_activation_pct", "0.5"))) / Decimal("100")
        self.trailing_stop_offset_pct = Decimal(str(self.params.get("trailing_stop_offset_pct", "0.5"))) / Decimal("100")
        
        self.kline_interval = self.params.get("kline_interval", "1h")
        self.leverage = int(self.params.get("leverage", 10))
        self.order_quantity_usd = Decimal(str(self.params.get("order_quantity_usd", "100.0")))

        # State variables
        self.position_entry_price = None
        self.position_side = None
        self.position_qty = Decimal("0")
        self.high_water_mark = None
        self.low_water_mark = None
        self.trailing_stop_active = False
        self.current_trailing_stop_price = None
        self.active_stop_loss_order_id = None
        self.active_take_profit_order_id = None

        # TODO: Load persistent state from DB if resuming an active position for this user_sub_obj.id
        # E.g., query Position table for this subscription_id and populate the above state vars.
        # For now, assumes fresh start or state is managed externally and passed in if needed.
        # self._load_persistent_state()

        self._fetch_market_precision()
        self._set_leverage()
        self.logger.info(f"{self.name} initialized for {self.trading_pair} with UserSubID {self.user_sub_obj.id}")

    def _fetch_market_precision(self):
        try:
            market = self.exchange_ccxt.markets[self.trading_pair]
            self.quantity_precision_str = market['precision']['amount']
            self.price_precision_str = market['precision']['price']
        except Exception as e:
            self.logger.error(f"Error fetching precision for {self.trading_pair}: {e}. Using defaults.")
            self.quantity_precision_str = "0.001" # Default, adjust as needed
            self.price_precision_str = "0.01"   # Default, adjust as needed

    def _get_decimal_places(self, precision_str):
        # Handles cases where precision_str might be None or not a valid number string
        if precision_str is None: return 8 # Default decimal places
        try:
            return abs(Decimal(str(precision_str)).as_tuple().exponent)
        except:
            self.logger.warning(f"Could not parse precision string '{precision_str}'. Using default 8.")
            return 8


    def _format_quantity(self, quantity: Decimal):
        places = self._get_decimal_places(self.quantity_precision_str)
        return str(quantity.quantize(Decimal('1e-' + str(places)), rounding=ROUND_DOWN))

    def _format_price(self, price: Decimal):
        places = self._get_decimal_places(self.price_precision_str)
        return str(price.quantize(Decimal('1e-' + str(places)), rounding=ROUND_DOWN)) # Or ROUND_NEAREST for price

    def _set_leverage(self):
        try:
            if hasattr(self.exchange_ccxt, 'set_leverage'):
                self.exchange_ccxt.set_leverage(self.leverage, self.trading_pair)
                self.logger.info(f"Leverage set to {self.leverage}x for {self.trading_pair}")
        except Exception as e:
            self.logger.warning(f"Could not set leverage for {self.trading_pair}: {e}")

    def _calculate_indicators(self, ohlcv_df):
        indicators = {}
        close_prices = ohlcv_df['close'].to_numpy(dtype=float)

        # Bollinger Bands
        upper_bb, middle_bb, lower_bb = talib.BBANDS(close_prices, timeperiod=self.bb_length, nbdevup=self.bb_stdev, nbdevdn=self.bb_stdev, matype=0)
        indicators['upper_bb'] = upper_bb
        indicators['middle_bb'] = middle_bb
        indicators['lower_bb'] = lower_bb

        # Trend Filter EMA
        indicators['trend_ema'] = talib.EMA(close_prices, timeperiod=self.trend_ema_period)

        # Volatility Filter
        vol_std = talib.STDDEV(close_prices, timeperiod=self.vol_filter_stdev_length)
        vol_sma_of_std = talib.SMA(vol_std, timeperiod=self.vol_filter_sma_length)
        indicators['vol_cond'] = vol_std > vol_sma_of_std # Boolean series

        return indicators

    def _get_current_position_details(self, symbol, historical_data_feed=None):
        # In a live scenario, this should prioritize fetching from the platform's DB
        # for positions associated with self.user_sub_obj.id
        self.logger.info(f"DB_TODO: Query database for active position for {symbol} and sub_id {self.user_sub_obj.id}")
        # If DB has a position, load self.position_entry_price, self.position_side, self.position_qty,
        # self.high_water_mark, self.low_water_mark, self.trailing_stop_active, self.current_trailing_stop_price,
        # self.active_stop_loss_order_id, self.active_take_profit_order_id here.

        # Fallback to CCXT for a simplified check (less robust for multi-bot environments)
        if historical_data_feed: # Backtesting
            return historical_data_feed.get_current_position(symbol)

        try:
            positions = self.exchange_ccxt.fetch_positions([symbol])
            for p in positions:
                if p['symbol'] == symbol:
                    qty = Decimal(p.get('contracts', '0')) # Futures often use 'contracts'
                    if qty == Decimal('0') and 'info' in p and 'positionAmt' in p['info']: # Binance specific
                        qty = Decimal(str(p['info']['positionAmt']))

                    if qty != Decimal('0'):
                        side = 'long' if qty > Decimal('0') else 'short'
                        entry_price = Decimal(p.get('entryPrice', '0'))
                        # If we are here, it means an existing position was found via CCXT.
                        # We should ideally load the strategy's state for this position from DB.
                        # For now, if self.position_side is None, we adopt this position.
                        if self.position_side is None: # Adopt if strategy state is clean
                            self.position_side = side
                            self.position_qty = abs(qty)
                            self.position_entry_price = entry_price
                            if side == 'long': self.high_water_mark = entry_price
                            else: self.low_water_mark = entry_price
                            self.logger.info(f"Adopted existing CCXT position: {side} {abs(qty)} {symbol} @ {entry_price}")

                        # Return details consistent with strategy's state if available and matches symbol
                        if self.position_side and self.position_qty > Decimal("0") and symbol == self.trading_pair:
                             return {'side': self.position_side, 'qty': self.position_qty, 'entry_price': self.position_entry_price}
            return None # No open position found via CCXT for this symbol
        except Exception as e:
            self.logger.error(f"Error fetching/parsing position for {symbol}: {e}")
            return None

    def _place_order(self, symbol, order_type, side, quantity, price=None, params=None, current_simulated_time_utc=None):
        if current_simulated_time_utc: # Backtesting
            self.logger.info(f"BACKTEST_SIM: Place {side} {order_type} for {quantity} {symbol} at {price if price else 'Market'}")
            # Backtester should handle order fill simulation
            return {"id": f"sim_{datetime.utcnow().timestamp()}", "status": "open", "simulated": True}
        
        try:
            formatted_qty = self._format_quantity(quantity)
            formatted_price = self._format_price(price) if price else None

            order = self.exchange_ccxt.create_order(symbol, order_type, side, float(formatted_qty), float(formatted_price) if formatted_price else None, params)
            self.logger.info(f"Order placed: {side} {order_type} {formatted_qty} {symbol} at {formatted_price if formatted_price else 'Market'}. OrderID: {order.get('id')}")
            # TODO: Record order in DB linked to self.user_sub_obj.id
            return order
        except Exception as e:
            self.logger.error(f"Failed to place {side} {order_type} for {symbol}: {e}", exc_info=True)
            return None

    def _cancel_order_by_id(self, symbol, order_id, current_simulated_time_utc=None):
        if not order_id: return
        if current_simulated_time_utc:
            self.logger.info(f"BACKTEST_SIM: Cancel order {order_id} for {symbol}")
            return True
        try:
            self.exchange_ccxt.cancel_order(order_id, symbol)
            self.logger.info(f"Cancelled order {order_id} for {symbol}")
            return True
        except ccxt.OrderNotFound:
            self.logger.warning(f"Order {order_id} for {symbol} not found to cancel (already filled/cancelled).")
            return True # Treat as success if not found
        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id} for {symbol}: {e}")
            return False

    def _reset_position_state(self):
        self.position_entry_price = None
        self.position_side = None
        self.position_qty = Decimal("0")
        self.high_water_mark = None
        self.low_water_mark = None
        self.trailing_stop_active = False
        self.current_trailing_stop_price = None
        self.active_stop_loss_order_id = None
        self.active_take_profit_order_id = None
        self.logger.info("Position state has been reset.")
        # TODO: If position is closed, update corresponding Position record in DB to closed.

    def _process_trading_logic(self, ohlcv_df, is_backtest=False, current_simulated_time_utc=None, historical_data_feed=None):
        symbol = self.trading_pair
        latest_candle_idx = -1 # Use the last fully closed candle
        
        if len(ohlcv_df) < max(self.bb_length, self.trend_ema_period, self.vol_filter_stdev_length + self.vol_filter_sma_length):
            self.logger.warning(f"Not enough data for indicator calculation on {symbol}. Need more than {max(self.bb_length, self.trend_ema_period)} candles, got {len(ohlcv_df)}")
            return {"action": "HOLD", "reason": "Insufficient data for indicators"} if is_backtest else None

        indicators = self._calculate_indicators(ohlcv_df)
        
        # Use values from the latest closed candle (index -1)
        close = Decimal(str(ohlcv_df['close'].iloc[latest_candle_idx]))
        upper_bb = Decimal(str(indicators['upper_bb'][latest_candle_idx]))
        lower_bb = Decimal(str(indicators['lower_bb'][latest_candle_idx]))
        trend_ema = Decimal(str(indicators['trend_ema'][latest_candle_idx]))
        vol_cond = indicators['vol_cond'][latest_candle_idx]

        # Crossover conditions (comparing current close with previous candle's BB relation)
        # Need -2 for previous candle's values if checking crossover logic strictly.
        # PineScript crossover(source, target) is true if source just crossed target.
        # close > upper_bb AND ref(close,1) < ref(upper_bb,1)
        prev_close = Decimal(str(ohlcv_df['close'].iloc[-2])) if len(ohlcv_df) >=2 else close
        prev_upper_bb = Decimal(str(indicators['upper_bb'][-2])) if len(indicators['upper_bb']) >=2 else upper_bb
        prev_lower_bb = Decimal(str(indicators['lower_bb'][-2])) if len(indicators['lower_bb']) >=2 else lower_bb

        buy_cond_bb_crossover = prev_close < prev_upper_bb and close > upper_bb
        sell_cond_bb_crossover = prev_close > prev_lower_bb and close < lower_bb
        
        buy_trend_cond = close > trend_ema
        sell_trend_cond = close < trend_ema

        final_buy_entry = buy_cond_bb_crossover and buy_trend_cond and vol_cond
        final_sell_entry = sell_cond_bb_crossover and sell_trend_cond and vol_cond

        # This method calls _get_current_position_details which might update self.position_side etc.
        # We need to ensure the state is consistent for this one processing cycle.
        # So, call it once and use its return, or rely on the fact that self.position_side is the source of truth
        # after _load_state_from_db_if_needed or initial adoption.
        # For now, assume self.position_side is the current state for this execution cycle.

        # --- Primary Exit Logic (BB Crossover Reversal) ---
        if self.position_side == 'long' and sell_cond_bb_crossover:
            self.logger.info(f"Primary Exit Long for {symbol} due to BB crossover at {close}.")
            self._cancel_order_by_id(symbol, self.active_stop_loss_order_id, current_simulated_time_utc)
            self._cancel_order_by_id(symbol, self.active_take_profit_order_id, current_simulated_time_utc)
            self._place_order(symbol, 'MARKET', 'sell', self.position_qty, params={'reduceOnly': True}, current_simulated_time_utc=current_simulated_time_utc)
            # TODO: DB Update for position close
            self._reset_position_state()
            return {"action": "EXIT_LONG", "price": float(close), "reason": "BB Crossover"} if is_backtest else None

        if self.position_side == 'short' and buy_cond_bb_crossover:
            self.logger.info(f"Primary Exit Short for {symbol} due to BB crossover at {close}.")
            self._cancel_order_by_id(symbol, self.active_stop_loss_order_id, current_simulated_time_utc)
            self._cancel_order_by_id(symbol, self.active_take_profit_order_id, current_simulated_time_utc)
            self._place_order(symbol, 'MARKET', 'buy', self.position_qty, params={'reduceOnly': True}, current_simulated_time_utc=current_simulated_time_utc)
            # TODO: DB Update for position close
            self._reset_position_state()
            return {"action": "EXIT_SHORT", "price": float(close), "reason": "BB Crossover"} if is_backtest else None

        # --- Entry Logic ---
        if not self.position_side: # No current position for this strategy instance
            entry_price = close # Use current close for market order entry
            qty_to_trade = self.order_quantity_usd / entry_price # Notional USD / price

            sl_price = Decimal('0')
            tp_price = Decimal('0')

            order_placed = False
            if final_buy_entry:
                self.logger.info(f"Entry Long signal for {symbol} at {entry_price}")
                entry_order = self._place_order(symbol, 'MARKET', 'buy', qty_to_trade, current_simulated_time_utc=current_simulated_time_utc)
                if entry_order or is_backtest: # If order placed or simulating
                    self.position_entry_price = entry_price
                    self.position_side = 'long'
                    self.position_qty = qty_to_trade
                    self.high_water_mark = entry_price
                    self.trailing_stop_active = False
                    self.current_trailing_stop_price = None
                    sl_price = entry_price * (Decimal('1') - self.stop_loss_pct)
                    tp_price = entry_price * (Decimal('1') + self.take_profit_pct)
                    order_placed = True
                    # TODO: DB: Create Position record. Store entry_price, side, qty, strategy_state (HWM etc.)
                    self.logger.info(f"DB_TODO: Create Position: LONG {qty_to_trade} {symbol} @ {entry_price}")


            elif final_sell_entry:
                self.logger.info(f"Entry Short signal for {symbol} at {entry_price}")
                entry_order = self._place_order(symbol, 'MARKET', 'sell', qty_to_trade, current_simulated_time_utc=current_simulated_time_utc)
                if entry_order or is_backtest:
                    self.position_entry_price = entry_price
                    self.position_side = 'short'
                    self.position_qty = qty_to_trade
                    self.low_water_mark = entry_price
                    self.trailing_stop_active = False
                    self.current_trailing_stop_price = None
                    sl_price = entry_price * (Decimal('1') + self.stop_loss_pct)
                    tp_price = entry_price * (Decimal('1') - self.take_profit_pct)
                    order_placed = True
                    # TODO: DB: Create Position record
                    self.logger.info(f"DB_TODO: Create Position: SHORT {qty_to_trade} {symbol} @ {entry_price}")
            
            if order_placed:
                # Place SL and TP orders
                exit_side = 'sell' if self.position_side == 'long' else 'buy'
                sl_order = self._place_order(symbol, 'STOP_MARKET', exit_side, self.position_qty, params={'stopPrice': self._format_price(sl_price), 'reduceOnly': True}, current_simulated_time_utc=current_simulated_time_utc)
                if sl_order: self.active_stop_loss_order_id = sl_order.get('id')
                
                tp_order = self._place_order(symbol, 'LIMIT', exit_side, self.position_qty, price=tp_price, params={'reduceOnly': True, 'timeInForce':'GTC'}, current_simulated_time_utc=current_simulated_time_utc) # Using LIMIT for TP
                if tp_order: self.active_take_profit_order_id = tp_order.get('id')

                if is_backtest: return {"action": f"ENTER_{self.position_side.upper()}", "price": float(entry_price), "qty": float(qty_to_trade), "sl": float(sl_price), "tp": float(tp_price)}


        # --- Simulated Trailing Stop Logic (if position exists) ---
        if self.position_side and self.position_entry_price:
            # Check if TP or SL would have been hit by this candle's H/L (for backtesting primarily, or if orders failed)
            # This is complex for live if relying on exchange for SL/TP execution.
            # For simplicity, live mode relies on exchange executing SL/TP.
            # Backtest mode would check historical_data_feed.get_candle_high_low() against SL/TP.

            activation_trigger_price = Decimal('0')
            new_potential_sl = Decimal('0')

            if self.position_side == 'long':
                if close > self.high_water_mark: self.high_water_mark = close
                activation_trigger_price = self.position_entry_price * (Decimal('1') + self.trailing_stop_activation_pct)
                if not self.trailing_stop_active and close >= activation_trigger_price:
                    self.trailing_stop_active = True
                    self.logger.info(f"Trailing stop ACTIVATE for LONG {symbol} at {close}. Activation Price: {activation_trigger_price}")

                if self.trailing_stop_active:
                    new_potential_sl = self.high_water_mark * (Decimal('1') - self.trailing_stop_offset_pct)
                    if self.current_trailing_stop_price is None or new_potential_sl > self.current_trailing_stop_price:
                        # Modify existing SL order
                        self.logger.info(f"Trailing stop for LONG {symbol} new SL: {new_potential_sl}. Old SL: {self.current_trailing_stop_price}")
                        if not is_backtest:
                            self._cancel_order_by_id(symbol, self.active_stop_loss_order_id) # Cancel old SL
                            # Place new SL. TP remains fixed.
                            new_sl_order = self._place_order(symbol, 'STOP_MARKET', 'sell', self.position_qty, params={'stopPrice': self._format_price(new_potential_sl), 'reduceOnly': True})
                            if new_sl_order: self.active_stop_loss_order_id = new_sl_order.get('id')
                        self.current_trailing_stop_price = new_potential_sl
                        # TODO: DB Update: Store new self.current_trailing_stop_price, self.active_stop_loss_order_id, self.trailing_stop_active, self.high_water_mark
                        self.logger.info(f"DB_TODO: Update Position state for Trailing Stop (Long)")


            elif self.position_side == 'short':
                if close < self.low_water_mark: self.low_water_mark = close
                activation_trigger_price = self.position_entry_price * (Decimal('1') - self.trailing_stop_activation_pct)
                if not self.trailing_stop_active and close <= activation_trigger_price:
                    self.trailing_stop_active = True
                    self.logger.info(f"Trailing stop ACTIVATE for SHORT {symbol} at {close}. Activation Price: {activation_trigger_price}")

                if self.trailing_stop_active:
                    new_potential_sl = self.low_water_mark * (Decimal('1') + self.trailing_stop_offset_pct)
                    if self.current_trailing_stop_price is None or new_potential_sl < self.current_trailing_stop_price:
                        self.logger.info(f"Trailing stop for SHORT {symbol} new SL: {new_potential_sl}. Old SL: {self.current_trailing_stop_price}")
                        if not is_backtest:
                            self._cancel_order_by_id(symbol, self.active_stop_loss_order_id)
                            new_sl_order = self._place_order(symbol, 'STOP_MARKET', 'buy', self.position_qty, params={'stopPrice': self._format_price(new_potential_sl), 'reduceOnly': True})
                            if new_sl_order: self.active_stop_loss_order_id = new_sl_order.get('id')
                        self.current_trailing_stop_price = new_potential_sl
                        # TODO: DB Update: Store new self.current_trailing_stop_price, self.active_stop_loss_order_id, self.trailing_stop_active, self.low_water_mark
                        self.logger.info(f"DB_TODO: Update Position state for Trailing Stop (Short)")

        if is_backtest: return {"action": "HOLD", "reason": "No conditions met"}
        return None


    def execute_live_signal(self, market_data_df=None): # market_data_df is for the self.trading_pair
        self.logger.info(f"Executing {self.name} for {self.trading_pair} UserSubID {self.user_sub_obj.id}")
        symbol = self.trading_pair

        # TODO: Add check for self.trading_pair being set. If not, log error and return.
        if not self.trading_pair or self.trading_pair == "UNSPECIFIED/USDT":
             self.logger.error("Trading pair not specified for this strategy instance.")
             return

        try:
            # Fetch enough data for all indicators
            # Longest period is trend_ema_period, vol_filter_stdev_length + vol_filter_sma_length
            limit_needed = max(self.bb_length, self.trend_ema_period, self.vol_filter_stdev_length + self.vol_filter_sma_length) + 50 # Buffer

            ohlcv_data = self.exchange_ccxt.fetch_ohlcv(symbol, timeframe=self.kline_interval, limit=limit_needed)
            if not ohlcv_data or len(ohlcv_data) < limit_needed - 45: # Allow some buffer miss
                self.logger.warning(f"Not enough OHLCV data for {symbol} on {self.kline_interval}. Got {len(ohlcv_data)}, need approx {limit_needed-45}")
                return

            ohlcv_df = pandas.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            ohlcv_df['timestamp'] = pandas.to_datetime(ohlcv_df['timestamp'], unit='ms')

            # Crucial: Load current position state from DB before processing logic
            # This ensures that if the strategy worker restarted, it picks up where it left off.
            self._load_state_from_db_if_needed(symbol) # This method needs to be implemented

            self._process_trading_logic(ohlcv_df, is_backtest=False)

        except ccxt.NetworkError as e:
            self.logger.error(f"CCXT NetworkError in {self.name} for {symbol}: {e}")
        except ccxt.ExchangeError as e:
            self.logger.error(f"CCXT ExchangeError in {self.name} for {symbol}: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error in {self.name} for {symbol}: {e}", exc_info=True)
            
    def _load_state_from_db_if_needed(self, symbol):
        # This method is called at the start of execute_live_signal
        # If self.position_side is None (meaning strategy instance just started or reset state),
        # then try to load an active position for this user_sub_obj.id and symbol from the DB.
        if self.position_side is None:
            self.logger.info(f"DB_TODO: Attempting to load persistent state for active position on {symbol} for sub_id {self.user_sub_obj.id}")
            # Example query:
            # active_db_pos = self.db_session.query(Position).filter_by(
            #    subscription_id=self.user_sub_obj.id,
            #    symbol=symbol,
            #    is_open=True
            # ).first()
            # if active_db_pos:
            #    self.position_side = active_db_pos.side
            #    self.position_entry_price = Decimal(str(active_db_pos.entry_price))
            #    self.position_qty = Decimal(str(active_db_pos.amount))
            #    # Load strategy_state_data (JSON field on Position model)
            #    if active_db_pos.strategy_state_data:
            #        state_data = json.loads(active_db_pos.strategy_state_data) # Assuming strategy_state_data is a JSON string
            #        self.high_water_mark = Decimal(str(state_data.get("high_water_mark"))) if state_data.get("high_water_mark") else None
            #        self.low_water_mark = Decimal(str(state_data.get("low_water_mark"))) if state_data.get("low_water_mark") else None
            #        self.trailing_stop_active = state_data.get("trailing_stop_active", False)
            #        self.current_trailing_stop_price = Decimal(str(state_data.get("current_trailing_stop_price"))) if state_data.get("current_trailing_stop_price") else None
            #        self.active_stop_loss_order_id = state_data.get("active_stop_loss_order_id")
            #        self.active_take_profit_order_id = state_data.get("active_take_profit_order_id")
            #        self.logger.info(f"Loaded persistent state for {self.position_side} position on {symbol}.")
            #    else: # If no strategy_state_data, try to initialize HWM/LWM if possible
            #        if self.position_side == 'long': self.high_water_mark = self.position_entry_price
            #        else: self.low_water_mark = self.position_entry_price
            # else:
            #    self.logger.info(f"No active persistent position found in DB for {symbol} and this subscription.")
            pass # End of DB_TODO block


    def run_backtest(self, historical_data_feed, current_simulated_time_utc):
        symbol = self.trading_pair # Strategy is single-pair
        # TODO: Add check for self.trading_pair being set.
        if not self.trading_pair or self.trading_pair == "UNSPECIFIED/USDT":
             self.logger.error("BACKTEST: Trading pair not specified.")
             return {"action": "ERROR", "reason": "Trading pair not set"}

        limit_needed = max(self.bb_length, self.trend_ema_period, self.vol_filter_stdev_length + self.vol_filter_sma_length) + 50

        # Get OHLCV data up to the current simulated time for indicator calculation
        # The historical_data_feed.get_ohlcv method should handle providing data ending at current_simulated_time_utc
        ohlcv_df = historical_data_feed.get_ohlcv(
            symbol=symbol,
            timeframe=self.kline_interval,
            limit=limit_needed,
            end_time_utc=current_simulated_time_utc # Ensure feed can provide data ending at this point
        )

        if ohlcv_df is None or ohlcv_df.empty or len(ohlcv_df) < limit_needed - 45:
            self.logger.warning(f"BACKTEST: Not enough OHLCV data for {symbol} at {current_simulated_time_utc}. Got {len(ohlcv_df) if ohlcv_df is not None else 0}")
            return {"action": "HOLD", "reason": "Insufficient data for backtest"}

        # The _process_trading_logic will use historical_data_feed to get current_price, equity, position
        return self._process_trading_logic(
            ohlcv_df,
            is_backtest=True,
            current_simulated_time_utc=current_simulated_time_utc,
            historical_data_feed=historical_data_feed
        )
