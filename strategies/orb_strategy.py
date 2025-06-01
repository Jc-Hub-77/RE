import logging
import numpy as np
# import talib # Not strictly needed for this strategy's core logic
import ccxt
from decimal import Decimal, ROUND_DOWN, ROUND_UP, getcontext
from datetime import datetime, time, date, timedelta
import pytz
import pandas # Required for DataFrame manipulation

# Ensure high precision for Decimal calculations
getcontext().prec = 18 # Set precision for Decimal

class OpeningRangeBreakoutStrategy:

    @staticmethod
    def get_parameters_definition():
        return {
            "trading_pair": {"type": "str", "label": "Trading Pair (e.g., BTC/USDT)", "default": "BTC/USDT"},
            "kline_interval": {"type": "str", "label": "Kline Interval for Breakout Logic", "default": "5m", "choices": ["1m", "3m", "5m", "15m", "1h"]},
            "orb_candle_hour_local": {"type": "int", "label": "ORB Candle Hour (Local Timezone)", "default": 9, "min": 0, "max": 23},
            "orb_candle_minute_local": {"type": "int", "label": "ORB Candle Minute (Local Timezone)", "default": 15, "min": 0, "max": 59, "step": 1},
            "orb_candle_timezone": {"type": "str", "label": "Timezone for ORB Candle Time", "default": "America/New_York",
                                    "choices": ["America/New_York", "America/Los_Angeles", "Europe/London", "Europe/Berlin", "Asia/Kolkata", "Asia/Tokyo", "Asia/Shanghai", "Australia/Sydney", "UTC"]},
            "orb_kline_interval": {"type": "str", "label": "Kline Interval for ORB Candle", "default": "15m", "choices": ["1m", "5m", "15m", "30m", "1h"]}, # Interval of the ORB candle itself
            "stop_loss_pct_from_orb": {"type": "float", "label": "Stop Loss % from ORB Level", "default": 0.5, "min": 0.01, "max": 5.0, "step": 0.01},
            "take_profit_pct_from_orb": {"type": "float", "label": "Take Profit % from ORB Level", "default": 1.5, "min": 0.01, "max": 10.0, "step": 0.01},
            "order_quantity_usd": {"type": "float", "label": "Order Quantity (USD Notional)", "default": 100.0, "min": 1.0},
            "leverage": {"type": "int", "label": "Leverage", "default": 10, "min": 1, "max": 100},
        }

    def __init__(self, db_session, user_sub_obj, strategy_params: dict, exchange_ccxt, logger=None):
        self.db_session = db_session
        self.user_sub_obj = user_sub_obj
        self.params = strategy_params
        self.exchange_ccxt = exchange_ccxt
        self.logger = logger if logger else logging.getLogger(__name__)
        self.name = "OpeningRangeBreakoutStrategy"

        self.trading_pair = self.params.get("trading_pair", "BTC/USDT")
        self.kline_interval = self.params.get("kline_interval", "5m") # Interval for breakout evaluation
        self.orb_kline_interval = self.params.get("orb_kline_interval", "15m") # Interval of the ORB candle

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

        self.orb_high = None
        self.orb_low = None
        self.orb_levels_set_for_utc_date = None
        self.active_position_side = None # 'long' or 'short'
        self.current_pos_qty = Decimal("0") # Store active trade info
        self.current_pos_entry_price = None # Store active trade info
        self.active_sl_tp_orders = {} # Store { 'sl_id': id, 'tp_id': id }

        self._fetch_market_precision()
        self._set_leverage()
        self.logger.info(f"{self.name} initialized for {self.trading_pair}, UserSubID {self.user_sub_obj.id}")

    def _fetch_market_precision(self):
        try:
            self.exchange_ccxt.load_markets() # Load markets if not already loaded
            market = self.exchange_ccxt.markets[self.trading_pair]
            self.quantity_precision_str = market['precision']['amount']
            self.price_precision_str = market['precision']['price']
        except Exception as e:
            self.logger.error(f"Error fetching precision for {self.trading_pair}: {e}. Using defaults.")
            self.quantity_precision_str = "0.00001"
            self.price_precision_str = "0.01"

    def _get_decimal_places(self, precision_str):
        if precision_str is None: return 8 # Default if None
        try:
            d_prec = Decimal(str(precision_str)) # Ensure it's a string for Decimal
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
                # Ensure trading_pair format is suitable for set_leverage (e.g., symbol without market type)
                symbol_for_leverage = self.trading_pair.split(':')[0] if ':' in self.trading_pair else self.trading_pair
                self.exchange_ccxt.set_leverage(self.leverage, symbol_for_leverage)
                self.logger.info(f"Leverage set to {self.leverage}x for {symbol_for_leverage}")
        except Exception as e:
            self.logger.warning(f"Could not set leverage for {self.trading_pair}: {e}")

    def _reset_daily_orb_state(self):
        self.orb_high = None
        self.orb_low = None
        self.orb_levels_set_for_utc_date = None
        self.logger.info("Daily ORB state reset.")

    def _determine_and_set_orb_levels(self, current_utc_datetime, historical_data_feed=None):
        target_local_time = time(self.orb_hour_local, self.orb_minute_local)
        current_est_datetime = current_utc_datetime.astimezone(self.orb_candle_tz)
        est_date_for_orb_candle = current_est_datetime.date()

        orb_candle_dt_local = self.orb_candle_tz.localize(datetime.combine(est_date_for_orb_candle, target_local_time))
        orb_candle_open_utc = orb_candle_dt_local.astimezone(pytz.utc)

        if orb_candle_open_utc > current_utc_datetime:
            self.logger.info(f"ORB candle time {orb_candle_open_utc.strftime('%Y-%m-%d %H:%M:%S %Z')} for {est_date_for_orb_candle.strftime('%Y-%m-%d %Z')} has not occurred yet (current UTC: {current_utc_datetime.strftime('%Y-%m-%d %H:%M:%S %Z')}).")
            if self.orb_levels_set_for_utc_date and self.orb_levels_set_for_utc_date == (current_utc_datetime.date() - timedelta(days=1)):
                 self.logger.info(f"Using previous day's ({self.orb_levels_set_for_utc_date}) ORB levels as today's ORB candle time has not passed.")
                 return True
            return False

        orb_candle_timestamp_ms = int(orb_candle_open_utc.timestamp() * 1000)
        self.logger.info(f"Attempting to fetch ORB candle for {self.trading_pair} at {orb_candle_open_utc.strftime('%Y-%m-%d %H:%M:%S %Z')} (Interval: {self.orb_kline_interval})")
        
        try:
            ohlcv = []
            if historical_data_feed:
                ohlcv = historical_data_feed.get_ohlcv(self.trading_pair, self.orb_kline_interval, since_utc_ms=orb_candle_timestamp_ms, limit=1)
            else:
                ohlcv = self.exchange_ccxt.fetch_ohlcv(self.trading_pair, self.orb_kline_interval, since=orb_candle_timestamp_ms, limit=1)

            if ohlcv and len(ohlcv) > 0:
                fetched_candle_ts_ms = ohlcv[0][0]
                if fetched_candle_ts_ms == orb_candle_timestamp_ms:
                    self.orb_high = Decimal(str(ohlcv[0][2])) # High
                    self.orb_low = Decimal(str(ohlcv[0][3]))  # Low
                    self.orb_levels_set_for_utc_date = orb_candle_open_utc.date()
                    self.logger.info(f"ORB levels set for {self.orb_levels_set_for_utc_date}: High={self.orb_high}, Low={self.orb_low}")
                    return True
                else:
                    self.logger.warning(f"Fetched candle timestamp {datetime.fromtimestamp(fetched_candle_ts_ms/1000, tz=pytz.utc)} does not match target ORB candle time {orb_candle_open_utc}.")
            else:
                self.logger.warning(f"Could not fetch ORB candle for {self.trading_pair} at {orb_candle_open_utc}.")
        except Exception as e:
            self.logger.error(f"Error fetching ORB candle: {e}", exc_info=True)
        
        self._reset_daily_orb_state()
        return False

    def _get_current_position_details(self, symbol, historical_data_feed=None):
        if self.active_position_side:
            return {'side': self.active_position_side, 'qty': self.current_pos_qty, 'entry_price': self.current_pos_entry_price}

        if historical_data_feed: return historical_data_feed.get_current_position(symbol)
        try:
            positions = self.exchange_ccxt.fetch_positions([symbol])
            for p in positions:
                if p['symbol'] == symbol: # Ensure it's the correct symbol format if needed
                    qty_str = p.get('contracts', '0')
                    if qty_str == '0' and 'info' in p and 'positionAmt' in p['info']: # Binance specific
                        qty_str = str(p['info']['positionAmt'])

                    qty = Decimal(qty_str)
                    if qty != Decimal('0'):
                        entry_price_str = p.get('entryPrice', '0')
                        if entry_price_str == '0' and 'info' in p and 'entryPrice' in p['info']: # Binance
                            entry_price_str = str(p['info']['entryPrice'])

                        return {'side': 'long' if qty > Decimal('0') else 'short',
                                'qty': abs(qty),
                                'entry_price': Decimal(entry_price_str)}
            return None
        except Exception as e:
            self.logger.error(f"Error fetching live position for {symbol}: {e}")
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
        # Simplified cancellation logic
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


    def _process_trading_logic(self, ohlcv_df, is_backtest=False, current_simulated_time_utc=None, historical_data_feed=None):
        symbol = self.trading_pair
        if len(ohlcv_df) < 4:
            self.logger.warning("Not enough data for ORB breakout check (need 4 candles for Pine logic).")
            return {"action": "HOLD", "reason": "Insufficient data"} if is_backtest else None

        current_high = Decimal(str(ohlcv_df['high'].iloc[-1]))
        current_low = Decimal(str(ohlcv_df['low'].iloc[-1]))
        prev_high = Decimal(str(ohlcv_df['high'].iloc[-2]))
        prev_low = Decimal(str(ohlcv_df['low'].iloc[-2]))
        two_ago_close = Decimal(str(ohlcv_df['close'].iloc[-3]))
        two_ago_high = Decimal(str(ohlcv_df['high'].iloc[-3]))
        two_ago_low = Decimal(str(ohlcv_df['low'].iloc[-3]))
        three_ago_close = Decimal(str(ohlcv_df['close'].iloc[-4]))

        if not self.orb_high or not self.orb_low:
            self.logger.info("ORB levels not set. Holding.")
            return {"action": "HOLD", "reason": "ORB levels not set"} if is_backtest else None

        if self.active_position_side:
            self.logger.info(f"Already in an active {self.active_position_side} position. Monitoring SL/TP.")
            return {"action": "HOLD", "reason": "Position already open"} if is_backtest else None
        
        crossed_over_s_high = (three_ago_close <= self.orb_high and two_ago_close > self.orb_high)
        buy_cond_final = (crossed_over_s_high and
                          prev_high > two_ago_high and
                          current_high > prev_high)

        crossed_under_s_low = (three_ago_close >= self.orb_low and two_ago_close < self.orb_low)
        sell_cond_final = (crossed_under_s_low and
                           prev_low < two_ago_low and
                           current_low < prev_low)

        entry_price = Decimal(str(ohlcv_df['close'].iloc[-1]))
        if entry_price == Decimal("0"): # Avoid division by zero if price somehow is zero
            self.logger.warning("Entry price is zero, cannot calculate quantity.")
            return {"action": "HOLD", "reason": "Entry price is zero"} if is_backtest else None
        qty_to_trade = self.order_quantity_usd / entry_price
        
        action_taken_type = None # To store "BUY" or "SELL" for backtest return

        if buy_cond_final:
            self.logger.info(f"ORB Buy Signal for {symbol} at {entry_price}")
            sl_price = self.orb_low * (Decimal('1') - self.stop_loss_pct)
            tp_price = self.orb_high * (Decimal('1') + self.take_profit_pct)
            
            if not is_backtest:
                entry_order = self._place_order(symbol, 'MARKET', 'buy', qty_to_trade)
                if entry_order:
                    self.active_position_side = 'long'
                    self.current_pos_qty = qty_to_trade
                    self.current_pos_entry_price = entry_price
                    self.logger.info(f"DB_TODO: Create Position: LONG {qty_to_trade} {symbol} @ {entry_price}")
                    sl_ord = self._place_order(symbol, 'STOP_MARKET', 'sell', qty_to_trade, params={'stopPrice': self._format_price(sl_price), 'reduceOnly':True})
                    tp_ord = self._place_order(symbol, 'TAKE_PROFIT_MARKET', 'sell', qty_to_trade, params={'stopPrice': self._format_price(tp_price), 'reduceOnly':True})
                    if sl_ord: self.active_sl_tp_orders['sl_id'] = sl_ord.get('id')
                    if tp_ord: self.active_sl_tp_orders['tp_id'] = tp_ord.get('id')
            else:
                self.active_position_side = 'long'
                self.current_pos_qty = qty_to_trade
                self.current_pos_entry_price = entry_price
                action_taken_type = "BUY"

        elif sell_cond_final:
            self.logger.info(f"ORB Sell Signal for {symbol} at {entry_price}")
            sl_price = self.orb_high * (Decimal('1') + self.stop_loss_pct)
            tp_price = self.orb_low * (Decimal('1') - self.take_profit_pct)
            if not is_backtest:
                entry_order = self._place_order(symbol, 'MARKET', 'sell', qty_to_trade)
                if entry_order:
                    self.active_position_side = 'short'
                    self.current_pos_qty = qty_to_trade
                    self.current_pos_entry_price = entry_price
                    self.logger.info(f"DB_TODO: Create Position: SHORT {qty_to_trade} {symbol} @ {entry_price}")
                    sl_ord = self._place_order(symbol, 'STOP_MARKET', 'buy', qty_to_trade, params={'stopPrice': self._format_price(sl_price), 'reduceOnly':True})
                    tp_ord = self._place_order(symbol, 'TAKE_PROFIT_MARKET', 'buy', qty_to_trade, params={'stopPrice': self._format_price(tp_price), 'reduceOnly':True})
                    if sl_ord: self.active_sl_tp_orders['sl_id'] = sl_ord.get('id')
                    if tp_ord: self.active_sl_tp_orders['tp_id'] = tp_ord.get('id')
            else:
                self.active_position_side = 'short'
                self.current_pos_qty = qty_to_trade
                self.current_pos_entry_price = entry_price
                action_taken_type = "SELL"

        if is_backtest:
            if action_taken_type:
                return {"action": action_taken_type, "price": float(entry_price), "qty": float(qty_to_trade), "sl": float(sl_price), "tp": float(tp_price)}
            else:
                return {"action": "HOLD", "reason": "No ORB signal"}
        return None


    def execute_live_signal(self, market_data_df=None):
        current_utc_dt = datetime.utcnow().replace(tzinfo=pytz.utc)

        # TODO: Load self.active_position_side from DB if it's None (strategy just started)
        # Example: self._load_active_position_from_db()

        if self.orb_levels_set_for_utc_date != current_utc_dt.date():
            self.logger.info(f"Date changed ({current_utc_dt.date()}) or ORB levels not set. Resetting/determining levels.")
            self._reset_daily_orb_state()
            if self.active_position_side: # If position carried overnight, close it
                 self.logger.info(f"Closing overnight position {self.active_position_side} for {self.trading_pair} due to date change.")
                 # Need current price to close
                 try:
                     ticker = self.exchange_ccxt.fetch_ticker(self.trading_pair)
                     close_price = Decimal(str(ticker['last']))
                     self._place_order(self.trading_pair, 'MARKET', 'sell' if self.active_position_side == 'long' else 'buy', self.current_pos_qty, params={'reduceOnly': True})
                     self.active_position_side = None; self.current_pos_qty = Decimal("0"); self.current_pos_entry_price = None
                     self._cancel_active_sl_tp_orders(self.trading_pair)
                 except Exception as e:
                     self.logger.error(f"Failed to close overnight position: {e}")

            self._determine_and_set_orb_levels(current_utc_dt)

        if not self.orb_high or not self.orb_low:
            self.logger.info("ORB levels are not available. Waiting.")
            return

        live_pos_check = self._get_current_position_details(self.trading_pair)
        if not live_pos_check and self.active_position_side is not None:
            self.logger.info(f"Position for {self.trading_pair} seems to have closed on exchange (SL/TP hit?). Resetting strategy state.")
            self.active_position_side = None; self.current_pos_qty = Decimal("0"); self.current_pos_entry_price = None
            self.active_sl_tp_orders = {}
            # TODO: DB: Ensure position is marked closed in DB.

        try:
            limit_needed = 4
            ohlcv_data = self.exchange_ccxt.fetch_ohlcv(self.trading_pair, timeframe=self.kline_interval, limit=limit_needed)
            if not ohlcv_data or len(ohlcv_data) < limit_needed:
                self.logger.warning(f"Not enough OHLCV data for {self.trading_pair} on {self.kline_interval}. Got {len(ohlcv_data)}")
                return

            ohlcv_df = pandas.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            ohlcv_df['timestamp'] = pandas.to_datetime(ohlcv_df['timestamp'], unit='ms')

            self._process_trading_logic(ohlcv_df, is_backtest=False)

        except Exception as e:
            self.logger.error(f"Error in ORB execute_live_signal: {e}", exc_info=True)


    def run_backtest(self, historical_data_feed, current_simulated_time_utc):
        current_utc_dt = current_simulated_time_utc

        if self.orb_levels_set_for_utc_date != current_utc_dt.date():
            self._reset_daily_orb_state()
            if self.active_position_side: # Position carried overnight
                 self.logger.info(f"[BACKTEST] Closing overnight {self.active_position_side} position for {self.trading_pair} at date change.")
                 self.active_position_side = None; self.current_pos_qty = Decimal("0"); self.current_pos_entry_price = None
                 self.active_sl_tp_orders = {} # Simulate SL/TP orders cancelled
            self._determine_and_set_orb_levels(current_utc_dt, historical_data_feed)

        if not self.orb_high or not self.orb_low:
            return {"action": "HOLD", "reason": "ORB levels not set for simulated day"}

        limit_needed = 4
        ohlcv_df = historical_data_feed.get_ohlcv(
            symbol=self.trading_pair,
            timeframe=self.kline_interval,
            limit=limit_needed,
            end_time_utc=current_simulated_time_utc
        )

        if ohlcv_df is None or ohlcv_df.empty or len(ohlcv_df) < limit_needed:
            return {"action": "HOLD", "reason": f"Insufficient OHLCV data for breakout check in backtest at {current_simulated_time_utc}"}

        return self._process_trading_logic(ohlcv_df, is_backtest=True, current_simulated_time_utc=current_simulated_time_utc, historical_data_feed=historical_data_feed)
