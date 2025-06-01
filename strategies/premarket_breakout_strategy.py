import logging
from datetime import datetime, time, date, timedelta
import pytz
import numpy as np # May not be strictly needed if all TA is via price action
import ccxt
from decimal import Decimal, ROUND_DOWN, ROUND_UP

# Assuming platform models are accessible for type hinting if needed,
# but the strategy itself won't import them directly for creation/update.
# from backend.models import Order, Position

class PreMarketBreakout:

    @staticmethod
    def get_parameters_definition():
        return {
            "trading_pair": {"type": "str", "label": "Trading Pair (e.g., BTC/USDT:USDT)", "default": "BTC/USDT:USDT"},
            "leverage": {"type": "int", "label": "Leverage", "default": 5, "min": 1, "max": 100},
            "stop_loss_percent": {"type": "float", "label": "Stop Loss % (from entry)", "default": 0.005, "min": 0.001, "max": 0.1, "step": 0.0001}, # 0.5%
            "take_profit_percent": {"type": "float", "label": "Take Profit % (from entry)", "default": 0.01, "min": 0.001, "max": 0.2, "step": 0.0001}, # 1%
            "risk_allocation_percent": {"type": "float", "label": "Risk Allocation % (of equity per trade)", "default": 0.02, "min": 0.001, "max": 0.1, "step": 0.001}, # 2%
            "kline_interval_for_levels": {"type": "str", "label": "Kline Interval for Pre-Market Levels", "default": "5m", "choices": ["1m", "3m", "5m", "15m"]},
            "kline_interval_for_breakout": {"type": "str", "label": "Kline Interval for Breakout Signal", "default": "1m", "choices": ["1m", "3m", "5m"]},
            "max_entry_deviation_percent": {"type": "float", "label": "Max Entry Price Deviation % (from breakout level)", "default": 0.001, "min": 0.0, "max": 0.01, "step": 0.0001}, # 0.1%
            "pre_market_start_time_est": {"type": "str", "label": "Pre-Market Start Time (EST HH:MM)", "default": "07:30"},
            "pre_market_end_time_est": {"type": "str", "label": "Pre-Market End Time (EST HH:MM)", "default": "09:29"},
            "market_open_time_est": {"type": "str", "label": "Market Open Time (EST HH:MM)", "default": "09:30"},
            "trading_session_end_time_est": {"type": "str", "label": "Trading Session End Time (EST HH:MM)", "default": "15:55"}, # EOD close a bit before actual market close
            "est_timezone": {"type": "str", "label": "EST Timezone Name", "default": "US/Eastern", "choices": ["US/Eastern", "America/New_York"]},
        }

    def __init__(self, db_session, user_sub_obj, strategy_params: dict, exchange_ccxt, logger=None):
        self.db_session = db_session
        self.user_sub_obj = user_sub_obj
        self.params = strategy_params
        self.exchange_ccxt = exchange_ccxt
        self.logger = logger if logger else logging.getLogger(__name__)
        self.name = "PreMarketBreakout"

        # Parse parameters
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
            # Fallback to defaults if parsing fails
            self.pre_market_start_time = time(7, 30)
            self.pre_market_end_time = time(9, 29)
            self.market_open_time = time(9, 30)
            self.trading_session_end_time = time(15, 55)

        # State variables
        self.premarket_high = None
        self.premarket_low = None
        self.max_deviation_high_entry = None # Calculated when levels are set
        self.max_deviation_low_entry = None  # Calculated when levels are set
        self.initialized_for_session_date = None # Stores the EST date for which levels are initialized
        self.last_trade_candle_timestamp = None # Timestamp of the last candle that resulted in a trade action
        self.target_notional_usdt_for_trade = Decimal("0.0") # Calculated based on equity and risk allocation

        self.quantity_precision = None
        self.price_precision = None
        self._fetch_market_precision()

        try:
            if hasattr(self.exchange_ccxt, 'set_leverage') and self.trading_pair.endswith(':USDT'): # Common suffix for futures
                self.exchange_ccxt.set_leverage(self.leverage, self.trading_pair)
                self.logger.info(f"Leverage set to {self.leverage}x for {self.trading_pair}")
        except Exception as e:
            self.logger.error(f"Failed to set leverage for {self.trading_pair}: {e}")

        self.logger.info(f"{self.name} strategy initialized for UserSubID {self.user_sub_obj.id if self.user_sub_obj else 'N/A'} with params (subset): pair={self.trading_pair}, leverage={self.leverage}")

    def _get_current_est_datetime(self, current_utc_dt=None):
        if current_utc_dt is None:
            current_utc_dt = datetime.utcnow()
        return current_utc_dt.replace(tzinfo=pytz.utc).astimezone(self.est_tz)

    def _fetch_market_precision(self):
        try:
            self.exchange_ccxt.load_markets()
            market_info = self.exchange_ccxt.markets[self.trading_pair]
            self.quantity_precision = int(market_info['precision']['amount'] if market_info['precision']['amount'] is not None else 8) # Default if None
            self.price_precision = int(market_info['precision']['price'] if market_info['precision']['price'] is not None else 8) # Default if None
            self.logger.info(f"Precision for {self.trading_pair}: Qty DP={self.quantity_precision}, Price DP={self.price_precision}")
        except Exception as e:
            self.logger.error(f"Error fetching market precision for {self.trading_pair}: {e}. Using defaults (8,8).")
            self.quantity_precision = 8 # Default decimal places
            self.price_precision = 8  # Default decimal places

    def _format_quantity(self, quantity: Decimal):
        return quantity.quantize(Decimal('1e-' + str(self.quantity_precision)), rounding=ROUND_DOWN)

    def _format_price(self, price: Decimal):
        return price.quantize(Decimal('1e-' + str(self.price_precision)), rounding=ROUND_DOWN) # Or ROUND_NEAREST depending on exchange

    def _reset_daily_state(self):
        self.premarket_high = None
        self.premarket_low = None
        self.max_deviation_high_entry = None
        self.max_deviation_low_entry = None
        self.initialized_for_session_date = None
        # self.last_trade_candle_timestamp = None # Don't reset this, it's per candle, not per day
        self.logger.info("Daily state reset.")

    def _initialize_session_levels(self, effective_est_datetime, historical_data_feed=None):
        est_date_to_fetch_for = effective_est_datetime.date()
        if self.initialized_for_session_date == est_date_to_fetch_for:
            self.logger.debug(f"Levels already initialized for {est_date_to_fetch_for}.")
            return True

        self.logger.info(f"Initializing pre-market levels for EST date: {est_date_to_fetch_for}")
        self._reset_daily_state() # Reset before initializing

        try:
            high, low = self._get_premarket_levels(est_date_to_fetch_for, historical_data_feed)
            if high is not None and low is not None:
                self.premarket_high = Decimal(str(high))
                self.premarket_low = Decimal(str(low))
                self.max_deviation_high_entry = self.premarket_high * (Decimal("1") + self.max_entry_deviation_percent)
                self.max_deviation_low_entry = self.premarket_low * (Decimal("1") - self.max_entry_deviation_percent)
                self.initialized_for_session_date = est_date_to_fetch_for
                self.logger.info(f"Pre-market levels for {est_date_to_fetch_for}: High={self.premarket_high}, Low={self.premarket_low}")
                self._calculate_target_notional_usdt(historical_data_feed)
                return True
            else:
                self.logger.warning(f"Failed to get pre-market levels for {est_date_to_fetch_for}.")
                return False
        except Exception as e:
            self.logger.error(f"Error initializing session levels: {e}", exc_info=True)
            return False

    def _get_premarket_levels(self, est_date_for_levels: date, historical_data_feed=None):
        start_dt_est = self.est_tz.localize(datetime.combine(est_date_for_levels, self.pre_market_start_time))
        end_dt_est = self.est_tz.localize(datetime.combine(est_date_for_levels, self.pre_market_end_time))

        start_timestamp_ms = int(start_dt_est.timestamp() * 1000)
        # Fetch up to, but not including, the end time candle. So, limit calculation needs care.
        # Or fetch slightly after and filter.
        # For simplicity, fetch_ohlcv usually includes the candle starting at 'since'
        # and limit determines how many. We need candles *within* the range.

        self.logger.info(f"Fetching pre-market OHLCV for {self.trading_pair} between {start_dt_est} and {end_dt_est} (EST)")
        
        # Calculate number of candles needed (approx)
        duration_seconds = (end_dt_est - start_dt_est).total_seconds()
        interval_seconds = self.exchange_ccxt.parse_timeframe(self.kline_interval_for_levels)
        limit_approx = int(duration_seconds / interval_seconds) + 5 # Add buffer
        
        try:
            if historical_data_feed:
                ohlcv_data = historical_data_feed.get_ohlcv(
                    symbol=self.trading_pair,
                    interval=self.kline_interval_for_levels,
                    since_utc_ms=start_timestamp_ms,
                    limit=limit_approx, # Adjust limit as needed
                    end_time_utc_ms=int(end_dt_est.timestamp() * 1000) # Pass end time for filtering in feed
                )
            else:
                # Fetch slightly more and filter, as 'since' behavior can vary.
                # Fetching for 1 day prior to ensure data availability if needed
                ohlcv_data = self.exchange_ccxt.fetch_ohlcv(
                    self.trading_pair,
                    timeframe=self.kline_interval_for_levels,
                    since=start_timestamp_ms - (24*60*60*1000), # Fetch more to be safe
                    limit=limit_approx + 288 # Add more buffer for live
                )

            if not ohlcv_data:
                self.logger.warning("No OHLCV data returned for pre-market period.")
                return None, None

            # Filter candles strictly within the pre-market window
            # OHLCV format: [timestamp, open, high, low, close, volume]
            pre_market_candles = [
                candle for candle in ohlcv_data
                if candle[0] >= start_timestamp_ms and candle[0] < int(end_dt_est.timestamp() * 1000)
            ]

            if not pre_market_candles:
                self.logger.warning(f"No candles found within the pre-market range {start_dt_est} - {end_dt_est}")
                return None, None

            highs = [candle[2] for candle in pre_market_candles]
            lows = [candle[3] for candle in pre_market_candles]

            return max(highs) if highs else None, min(lows) if lows else None
        except Exception as e:
            self.logger.error(f"Error fetching pre-market OHLCV: {e}", exc_info=True)
            return None, None

    def _calculate_target_notional_usdt(self, historical_data_feed=None):
        try:
            equity = Decimal("0.0")
            if historical_data_feed:
                equity = Decimal(str(historical_data_feed.get_current_equity()))
            else:
                balance = self.exchange_ccxt.fetch_balance(params={'type': 'future'}) # Ensure this is for futures
                if 'USDT' in balance['total']:
                    equity = Decimal(str(balance['total']['USDT']))
                else:
                    self.logger.warning("USDT balance not found in fetch_balance response.")
                    self.target_notional_usdt_for_trade = Decimal("0.0")
                    return

            if equity <= Decimal("0.0"):
                self.logger.warning("Equity is zero or negative. Cannot calculate trade notional.")
                self.target_notional_usdt_for_trade = Decimal("0.0")
                return

            self.target_notional_usdt_for_trade = equity * self.risk_allocation_percent * Decimal(str(self.leverage))
            # This notional is what the position value should be. Qty will be derived from this.
            self.logger.info(f"Calculated target notional USDT for trade: {self.target_notional_usdt_for_trade} (Equity: {equity})")

        except Exception as e:
            self.logger.error(f"Error calculating target notional USDT: {e}", exc_info=True)
            self.target_notional_usdt_for_trade = Decimal("0.0")


    def _get_current_position_details(self, historical_data_feed=None):
        # TODO: Robustly parse CCXT position data or use DB. This is a simplified placeholder.
        if historical_data_feed:
            return historical_data_feed.get_current_position(self.trading_pair) # Assumes feed provides this
        try:
            positions = self.exchange_ccxt.fetch_positions([self.trading_pair])
            if positions:
                # Find non-zero position for the symbol. CCXT fetch_positions can return multiple.
                for pos in positions:
                    if pos['symbol'] == self.trading_pair:
                        contracts = Decimal(str(pos.get('contracts', '0'))) # Amount of base currency
                        if 'info' in pos and 'positionAmt' in pos['info'] and contracts == Decimal('0'): # Binance specific
                             contracts = Decimal(str(pos['info']['positionAmt']))

                        if contracts != Decimal('0'):
                            side = 'long' if contracts > Decimal('0') else 'short'
                            entry_price = Decimal(str(pos.get('entryPrice', '0')))
                            if entry_price == Decimal('0') and 'info' in pos and 'entryPrice' in pos['info']: # Binance
                                entry_price = Decimal(str(pos['info']['entryPrice']))

                            return {
                                'side': side,
                                'qty': abs(contracts),
                                'entry_price': entry_price
                            }
            return None
        except Exception as e:
            self.logger.error(f"Error fetching current position for {self.trading_pair}: {e}")
            return None

    def _place_order_with_sl_tp(self, side, quantity_decimal: Decimal, entry_price_decimal: Decimal, sl_price_decimal: Decimal, tp_price_decimal: Decimal, current_simulated_time_utc=None):
        formatted_qty = self._format_quantity(quantity_decimal)
        # entry_price_str = self._format_price(entry_price_decimal) # Market order, so no price
        sl_price_str = self._format_price(sl_price_decimal)
        tp_price_str = self._format_price(tp_price_decimal)

        action_details = {
            "action": "OPEN_ORDER", "side": side, "symbol": self.trading_pair,
            "qty": float(formatted_qty), "entry_approx": float(entry_price_decimal),
            "sl": float(sl_price_str), "tp": float(tp_price_str),
            "timestamp_utc": current_simulated_time_utc or datetime.utcnow()
        }
        self.logger.info(f"Placing order: {action_details}")

        if current_simulated_time_utc: # Backtesting
            self.logger.info(f"[BACKTEST] Simulating {side} order: Qty {formatted_qty} for {self.trading_pair} at market (entry ~{entry_price_decimal}), SL {sl_price_str}, TP {tp_price_str}")
            # In a full backtest, this would interact with the backtest engine to simulate fill
            return {"status": "simulated_open", "order_id": f"sim_{datetime.utcnow().timestamp()}", **action_details}

        try:
            # TODO: Record Order and Position in DB via self.db_session for live trades.
            self._cancel_all_open_orders() # Clear any previous SL/TP before new trade

            order_type = 'MARKET'
            ccxt_side = 'buy' if side == 'long' else 'sell'

            self.logger.info(f"Submitting MARKET {ccxt_side} order: {self.trading_pair}, Qty: {str(formatted_qty)}")
            market_order = self.exchange_ccxt.create_order(self.trading_pair, order_type, ccxt_side, float(formatted_qty))
            self.logger.info(f"Market order response: {market_order.get('id', 'N/A')}")
            # Actual entry price might differ, could fetch last trade or use order details if available quickly

            sl_ccxt_side = 'sell' if side == 'long' else 'buy'
            tp_ccxt_side = 'sell' if side == 'long' else 'buy' # Same as SL for closing

            sl_params = {'stopPrice': str(sl_price_str), 'reduceOnly': True} # Ensure reduceOnly if exchange supports
            self.logger.info(f"Submitting STOP_MARKET {sl_ccxt_side} (SL) order: {self.trading_pair}, Qty: {str(formatted_qty)}, Stop: {sl_price_str}")
            sl_order = self.exchange_ccxt.create_order(self.trading_pair, 'STOP_MARKET', sl_ccxt_side, float(formatted_qty), params=sl_params)
            self.logger.info(f"SL order response: {sl_order.get('id', 'N/A')}")

            # TP order: Some exchanges use 'TAKE_PROFIT_MARKET' or require limit price for TAKE_PROFIT
            tp_params = {'stopPrice': str(tp_price_str), 'reduceOnly': True} # Price is trigger for TAKE_PROFIT_MARKET
            self.logger.info(f"Submitting TAKE_PROFIT_MARKET {tp_ccxt_side} (TP) order: {self.trading_pair}, Qty: {str(formatted_qty)}, Trigger: {tp_price_str}")
            tp_order = self.exchange_ccxt.create_order(self.trading_pair, 'TAKE_PROFIT_MARKET', tp_ccxt_side, float(formatted_qty), params=tp_params)
            self.logger.info(f"TP order response: {tp_order.get('id', 'N/A')}")

            return {"status": "live_orders_placed", "market_order_id": market_order.get('id'), **action_details}

        except Exception as e:
            self.logger.error(f"Error placing live order for {self.trading_pair}: {e}", exc_info=True)
            return {"status": "error", "message": str(e), **action_details}

    def _close_position(self, current_side, current_qty_decimal: Decimal, current_price_for_closing_decimal: Decimal, reason="signal", current_simulated_time_utc=None):
        formatted_qty = self._format_quantity(current_qty_decimal)
        action_details = {
            "action": "CLOSE_POSITION", "side_closed": current_side, "symbol": self.trading_pair,
            "qty": float(formatted_qty), "price_approx": float(current_price_for_closing_decimal), "reason": reason,
            "timestamp_utc": current_simulated_time_utc or datetime.utcnow()
        }
        self.logger.info(f"Closing position: {action_details}")

        if current_simulated_time_utc: # Backtesting
            self.logger.info(f"[BACKTEST] Simulating CLOSE {current_side} position: Qty {formatted_qty} for {self.trading_pair} at market (price ~{current_price_for_closing_decimal})")
            return {"status": "simulated_close", **action_details}

        try:
            # TODO: Update Position in DB to closed via self.db_session for live trades.
            self._cancel_all_open_orders() # Cancel associated SL/TP

            ccxt_side = 'sell' if current_side == 'long' else 'buy'
            self.logger.info(f"Submitting MARKET {ccxt_side} (close) order: {self.trading_pair}, Qty: {str(formatted_qty)}")
            close_order = self.exchange_ccxt.create_order(self.trading_pair, 'MARKET', ccxt_side, float(formatted_qty), params={'reduceOnly': True})
            self.logger.info(f"Close order response: {close_order.get('id', 'N/A')}")
            return {"status": "live_close_order_placed", "close_order_id": close_order.get('id'), **action_details}

        except Exception as e:
            self.logger.error(f"Error closing live position for {self.trading_pair}: {e}", exc_info=True)
            return {"status": "error", "message": str(e), **action_details}

    def _cancel_all_open_orders(self, current_simulated_time_utc=None):
        if current_simulated_time_utc: # Backtesting
            self.logger.info(f"[BACKTEST] Simulating cancel all open orders for {self.trading_pair}")
            return {"status": "simulated_cancel_all"}
        try:
            self.exchange_ccxt.cancel_all_orders(self.trading_pair)
            self.logger.info(f"Cancelled all open orders for {self.trading_pair}")
            return {"status": "live_orders_cancelled"}
        except Exception as e:
            self.logger.error(f"Error cancelling orders for {self.trading_pair}: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def _eod_close_logic(self, current_est_datetime, current_price_for_closing_decimal: Decimal, current_simulated_time_utc=None):
        self.logger.info(f"EOD check at {current_est_datetime.strftime('%Y-%m-%d %H:%M:%S')} EST")
        position = self._get_current_position_details(historical_data_feed=current_simulated_time_utc is not None) # Pass feed if backtesting
        if position:
            self.logger.info(f"EOD: Open position found for {self.trading_pair}. Closing it.")
            self._close_position(position['side'], position['qty'], current_price_for_closing_decimal, reason="EOD", current_simulated_time_utc=current_simulated_time_utc)
        
        self._cancel_all_open_orders(current_simulated_time_utc=current_simulated_time_utc) # Also cancel any pending entry orders
        self._reset_daily_state() # Reset for next day
        return {"action": "EOD_CLOSE_RESET"}


    def execute_live_signal(self, market_data_df=None): # market_data_df is not used here as we fetch specific klines
        now_est = self._get_current_est_datetime()
        now_est_time = now_est.time()
        
        # Initialize daily levels after market open or if not yet initialized for the day
        if now_est_time >= self.market_open_time and \
           (self.initialized_for_session_date is None or self.initialized_for_session_date != now_est.date()):
            if not self._initialize_session_levels(now_est):
                self.logger.warning("Failed to initialize session levels. Cannot proceed with trading logic.")
                return {"action": "HOLD", "reason": "Failed level initialization"}

        if not self.premarket_high or not self.premarket_low: # Levels not set
            if now_est_time < self.market_open_time and now_est_time >= self.pre_market_start_time:
                 self.logger.info(f"Pre-market period. Levels not yet finalized. Current EST: {now_est_time}")
            elif now_est_time >= self.market_open_time:
                 self.logger.warning(f"Market open but pre-market levels are not set. Attempting re-init.")
                 if not self._initialize_session_levels(now_est): # Try again
                      return {"action": "HOLD", "reason": "Levels not set post market open"}
            else: # Before pre-market start
                self.logger.info(f"Outside trading session hours. Current EST: {now_est_time}")
            return {"action": "HOLD", "reason": "Levels not set or outside trading hours"}

        # EOD Close Logic
        if now_est_time >= self.trading_session_end_time:
            # Fetch current price for EOD close
            try:
                ticker = self.exchange_ccxt.fetch_ticker(self.trading_pair)
                eod_close_price = Decimal(str(ticker['last']))
                return self._eod_close_logic(now_est, eod_close_price)
            except Exception as e:
                self.logger.error(f"Failed to fetch price for EOD close: {e}")
                return {"action": "ERROR", "reason": "Failed EOD price fetch"}

        # Fetch latest klines for breakout check
        try:
            # Fetch 2 candles to ensure we have the last fully closed one
            ohlcv_breakout = self.exchange_ccxt.fetch_ohlcv(self.trading_pair, timeframe=self.kline_interval_for_breakout, limit=2)
            if not ohlcv_breakout or len(ohlcv_breakout) < 2:
                self.logger.warning("Not enough data for breakout signal.")
                return {"action": "HOLD", "reason": "Insufficient breakout kline data"}

            # Last fully closed candle is the second to last in the list (index -2)
            # Current forming candle is the last (index -1)
            last_closed_candle = ohlcv_breakout[-2]
            last_closed_candle_high = Decimal(str(last_closed_candle[2]))
            last_closed_candle_low = Decimal(str(last_closed_candle[3]))
            last_closed_candle_close = Decimal(str(last_closed_candle[4]))
            last_closed_candle_timestamp_ms = last_closed_candle[0]

            # Avoid trading on the same candle multiple times if signal persists
            if self.last_trade_candle_timestamp == last_closed_candle_timestamp_ms:
                self.logger.info(f"Already traded or evaluated candle ending {datetime.fromtimestamp(last_closed_candle_timestamp_ms/1000)}. Waiting for next.")
                return {"action": "HOLD", "reason": "Already processed this candle"}

        except Exception as e:
            self.logger.error(f"Error fetching breakout klines: {e}", exc_info=True)
            return {"action": "ERROR", "reason": "Kline fetch error for breakout"}

        current_position = self._get_current_position_details()
        action_taken_this_cycle = False
        trade_result = None

        # Breakout Logic
        if not current_position:
            if last_closed_candle_close > self.premarket_high and last_closed_candle_close <= self.max_deviation_high_entry:
                self.logger.info(f"LONG Breakout detected: Price {last_closed_candle_close} > PM_High {self.premarket_high}")
                entry_price = last_closed_candle_close # Or current price from ticker if preferred
                sl_price = entry_price * (Decimal("1") - self.stop_loss_percent)
                tp_price = entry_price * (Decimal("1") + self.take_profit_percent)

                # Calculate quantity based on notional value allowed for the trade
                # Qty = Notional / Entry Price
                if self.target_notional_usdt_for_trade > Decimal("0.0") and entry_price > Decimal("0.0"):
                    quantity = self.target_notional_usdt_for_trade / entry_price
                    trade_result = self._place_order_with_sl_tp('long', quantity, entry_price, sl_price, tp_price)
                    action_taken_this_cycle = True
                else:
                    self.logger.warning("Target notional or entry price is zero. Cannot calculate quantity.")

            elif last_closed_candle_close < self.premarket_low and last_closed_candle_close >= self.max_deviation_low_entry:
                self.logger.info(f"SHORT Breakout detected: Price {last_closed_candle_close} < PM_Low {self.premarket_low}")
                entry_price = last_closed_candle_close
                sl_price = entry_price * (Decimal("1") + self.stop_loss_percent)
                tp_price = entry_price * (Decimal("1") - self.take_profit_percent)

                if self.target_notional_usdt_for_trade > Decimal("0.0") and entry_price > Decimal("0.0"):
                    quantity = self.target_notional_usdt_for_trade / entry_price
                    trade_result = self._place_order_with_sl_tp('short', quantity, entry_price, sl_price, tp_price)
                    action_taken_this_cycle = True
                else:
                    self.logger.warning("Target notional or entry price is zero. Cannot calculate quantity.")
        else: # Position exists, check for SL/TP (handled by exchange) or other exit conditions (e.g. time-based, not implemented here)
            self.logger.info(f"Position already open: {current_position}. Monitoring SL/TP via exchange.")
            # Could add logic here to check if SL/TP was hit if exchange doesn't notify, or if a manual close is needed based on new signals.
            # For this strategy, we typically let SL/TP orders on exchange handle exits unless EOD.

        if action_taken_this_cycle:
            self.last_trade_candle_timestamp = last_closed_candle_timestamp_ms
            return trade_result if trade_result else {"action": "ERROR", "reason": "Trade placement failed post-signal"}
        
        return {"action": "HOLD", "reason": "No breakout signal or position already exists"}


    def run_backtest(self, historical_data_feed, current_simulated_time_utc: datetime):
        now_est = self._get_current_est_datetime(current_simulated_time_utc)
        now_est_time = now_est.time()
        
        current_price_for_eval_decimal = Decimal(str(historical_data_feed.get_current_price(self.trading_pair, current_simulated_time_utc)))

        # Initialize daily levels
        if now_est_time >= self.market_open_time and \
           (self.initialized_for_session_date is None or self.initialized_for_session_date != now_est.date()):
            if not self._initialize_session_levels(now_est, historical_data_feed):
                return {"action": "HOLD", "reason": "Backtest: Failed level initialization"}
        
        if not self.premarket_high or not self.premarket_low:
            if now_est_time < self.market_open_time and now_est_time >= self.pre_market_start_time:
                 self.logger.info(f"[BACKTEST] Pre-market period. Levels not yet finalized. Current EST: {now_est_time}")
            elif now_est_time >= self.market_open_time:
                 self.logger.warning(f"[BACKTEST] Market open but pre-market levels are not set. Attempting re-init.")
                 if not self._initialize_session_levels(now_est, historical_data_feed):
                      return {"action": "HOLD", "reason": "Backtest: Levels not set post market open"}
            else:
                self.logger.info(f"[BACKTEST] Outside trading session hours. Current EST: {now_est_time}")
            return {"action": "HOLD", "reason": "Backtest: Levels not set or outside trading hours"}

        # EOD Close Logic
        if now_est_time >= self.trading_session_end_time:
            return self._eod_close_logic(now_est, current_price_for_eval_decimal, current_simulated_time_utc)

        # Fetch kline data for the breakout interval ending at current_simulated_time_utc
        # For backtesting, the 'current_simulated_time_utc' represents the END of the most recent kline interval.
        # We need the close of the candle that *just closed* at current_simulated_time_utc.
        # So, fetch_ohlcv with end_time = current_simulated_time_utc and limit=1 (or more for context)
        try:
            # To get the candle that closed *at or before* current_simulated_time_utc
            # If kline_interval is 1m, and current_simulated_time_utc is 09:31:00,
            # we want the candle for 09:30:00 - 09:31:00.
            # `historical_data_feed.get_ohlcv` should be designed to handle this.
            # Let's assume it returns candles where timestamp is the *start* of the candle.
            # We need the candle whose start time is `current_simulated_time_utc - interval_duration`.
            
            interval_duration_seconds = self.exchange_ccxt.parse_timeframe(self.kline_interval_for_breakout)
            # Fetch a few candles to be safe, then select the correct one.
            # Target end time for fetching is current_simulated_time_utc
            # Target start time for fetching is current_simulated_time_utc - some buffer
            buffer_duration = timedelta(minutes=self.exchange_ccxt.parse_timeframe(self.kline_interval_for_breakout) * 5 / 60) # 5 intervals
            since_utc_ms = int((current_simulated_time_utc - buffer_duration).timestamp() * 1000)
            
            ohlcv_breakout = historical_data_feed.get_ohlcv(
                self.trading_pair,
                self.kline_interval_for_breakout,
                since_utc_ms=since_utc_ms,
                limit=10, # Fetch a few, filter below
                end_time_utc_ms=int(current_simulated_time_utc.timestamp() * 1000)
            )

            if not ohlcv_breakout:
                self.logger.warning("[BACKTEST] No OHLCV data for breakout signal.")
                return {"action": "HOLD", "reason": "Backtest: Insufficient breakout kline data"}

            # Find the candle that *just closed* relative to current_simulated_time_utc
            # Candle timestamp is start of period. Interval in seconds.
            # A candle [t,o,h,l,c,v] is for period t to t + interval_seconds
            # It closes at t + interval_seconds.
            # We want the candle where t_start + interval_seconds == current_simulated_time_utc.timestamp()
            # Or, t_start == current_simulated_time_utc.timestamp() - interval_seconds
            target_candle_start_timestamp_ms = int((current_simulated_time_utc - timedelta(seconds=interval_duration_seconds)).timestamp() * 1000)

            last_closed_candle = None
            for candle_data in reversed(ohlcv_breakout): # Check most recent first
                if candle_data[0] == target_candle_start_timestamp_ms:
                    last_closed_candle = candle_data
                    break

            if not last_closed_candle:
                self.logger.warning(f"[BACKTEST] Could not find the specific kline ending at {current_simulated_time_utc}. Last available: {datetime.fromtimestamp(ohlcv_breakout[-1][0]/1000) if ohlcv_breakout else 'None'}")
                return {"action": "HOLD", "reason": "Backtest: Specific breakout kline not found"}

            last_closed_candle_high = Decimal(str(last_closed_candle[2]))
            last_closed_candle_low = Decimal(str(last_closed_candle[3]))
            last_closed_candle_close = Decimal(str(last_closed_candle[4]))
            last_closed_candle_timestamp_ms = last_closed_candle[0] # This is start of candle

            # Avoid trading on the same candle (using its start time as identifier)
            if self.last_trade_candle_timestamp == last_closed_candle_timestamp_ms:
                return {"action": "HOLD", "reason": "Backtest: Already processed this candle"}

        except Exception as e:
            self.logger.error(f"[BACKTEST] Error processing breakout klines: {e}", exc_info=True)
            return {"action": "ERROR", "reason": "Backtest: Kline processing error for breakout"}

        current_position = self._get_current_position_details(historical_data_feed)
        action_taken_this_cycle = False
        trade_action = {"action": "HOLD"} # Default

        # Breakout Logic (using last_closed_candle_close as the breakout price)
        if not current_position:
            entry_price_for_trade = last_closed_candle_close # Breakout confirmed by this candle's close

            if entry_price_for_trade > self.premarket_high and entry_price_for_trade <= self.max_deviation_high_entry:
                self.logger.info(f"[BACKTEST] LONG Breakout: Price {entry_price_for_trade} > PM_High {self.premarket_high}")
                sl_price = entry_price_for_trade * (Decimal("1") - self.stop_loss_percent)
                tp_price = entry_price_for_trade * (Decimal("1") + self.take_profit_percent)
                if self.target_notional_usdt_for_trade > Decimal("0.0") and entry_price_for_trade > Decimal("0.0"):
                    quantity = self.target_notional_usdt_for_trade / entry_price_for_trade
                    trade_action = self._place_order_with_sl_tp('long', quantity, entry_price_for_trade, sl_price, tp_price, current_simulated_time_utc)
                    action_taken_this_cycle = True
                else: self.logger.warning("[BACKTEST] Target notional or entry price zero.")

            elif entry_price_for_trade < self.premarket_low and entry_price_for_trade >= self.max_deviation_low_entry:
                self.logger.info(f"[BACKTEST] SHORT Breakout: Price {entry_price_for_trade} < PM_Low {self.premarket_low}")
                sl_price = entry_price_for_trade * (Decimal("1") + self.stop_loss_percent)
                tp_price = entry_price_for_trade * (Decimal("1") - self.take_profit_percent)
                if self.target_notional_usdt_for_trade > Decimal("0.0") and entry_price_for_trade > Decimal("0.0"):
                    quantity = self.target_notional_usdt_for_trade / entry_price_for_trade
                    trade_action = self._place_order_with_sl_tp('short', quantity, entry_price_for_trade, sl_price, tp_price, current_simulated_time_utc)
                    action_taken_this_cycle = True
                else: self.logger.warning("[BACKTEST] Target notional or entry price zero.")
        else:
            self.logger.info(f"[BACKTEST] Position already open: {current_position}. SL/TP handled by backtest engine based on subsequent price moves.")
            # Backtest engine should simulate SL/TP hits based on future klines for `current_price_for_eval_decimal`

        if action_taken_this_cycle:
            self.last_trade_candle_timestamp = last_closed_candle_timestamp_ms # Mark this candle's start time

        return trade_action
