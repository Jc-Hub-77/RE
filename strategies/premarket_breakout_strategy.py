# trading_platform/strategies/premarket_breakout_strategy.py
import datetime
import pytz 
import pandas as pd
import logging
import time
import json # For UserStrategySubscription parameters
from sqlalchemy.orm import Session
from backend.models import Position, Order, UserStrategySubscription

logger = logging.getLogger(__name__)

class PremarketBreakoutStrategy:
    def __init__(self, symbol: str, timeframe: str, capital: float = 10000, **custom_parameters):
        self.name = "PremarketBreakoutStrategy"
        self.symbol = symbol
        self.timeframe_str = timeframe 
        self.capital_param = capital # Store initial capital from params

        defaults = {
            "orb_hour": 9, "orb_minute": 15, "orb_timezone": "America/New_York", # Using ORB naming for consistency
            "tp_percent": 1.0, "sl_percent": 0.5,
            "position_size_percent_capital": 10.0,
            "lookback_bars_for_orb": 1, # Renamed from premarket_... to orb_...
            "premarket_max_deviation_percent": 2.0 # Kept this specific param name
        }
        self_params = {**defaults, **custom_parameters}
        # Ensure correct types for parameters
        self.orb_hour = int(self_params["orb_hour"])
        self.orb_minute = int(self_params["orb_minute"])
        self.orb_timezone = str(self_params["orb_timezone"])
        self.tp_percent = float(self_params["tp_percent"])
        self.sl_percent = float(self_params["sl_percent"])
        self.position_size_percent_capital = float(self_params["position_size_percent_capital"])
        self.lookback_bars_for_orb = int(self_params["lookback_bars_for_orb"])
        self.premarket_max_deviation_percent = float(self_params["premarket_max_deviation_percent"])
        
        # These are specific to the strategy's original naming, map them from ORB for internal use if needed
        # Or, adjust internal logic to use orb_hour/minute directly. For now, map for less internal change:
        self.premarket_start_hour_est = custom_parameters.get("premarket_start_hour_est", 7) # Example: if specific PM times are still needed
        self.premarket_start_minute_est = custom_parameters.get("premarket_start_minute_est", 30)
        self.premarket_end_hour_est = self.orb_hour # Align with ORB definition
        self.premarket_end_minute_est = self.orb_minute 
        self.market_open_hour_est = self.orb_hour # Market open is when ORB period ends
        self.market_open_minute_est = self.orb_minute

        try:
            self.pytz_orb_timezone = pytz.timezone(self.orb_timezone)
        except pytz.UnknownTimeZoneError:
            logger.warning(f"[{self.name}-{self.symbol}] Unknown ORB timezone '{self.orb_timezone}', defaulting to UTC.")
            self.pytz_orb_timezone = pytz.utc; self.orb_timezone = "UTC"

        self.tp_decimal = self.tp_percent / 100.0
        self.sl_decimal = self.sl_percent / 100.0
        self.position_size_percent_capital_decimal = self.position_size_percent_capital / 100.0
        self.premarket_max_deviation_decimal = self.premarket_max_deviation_percent / 100.0

        # In-memory state for ORB range (refreshed daily)
        self.opening_range_high = None
        self.opening_range_low = None
        self.opening_range_set_for_date = None 
        self.max_deviation_high = None # Calculated after ORB set
        self.max_deviation_low = None  # Calculated after ORB set
        
        self.price_precision = 8; self.quantity_precision = 8
        self._precisions_fetched_ = False
        
        logger.info(f"[{self.name}-{self.symbol}] Initialized with effective params: {self_params}")

    @classmethod
    def get_parameters_definition(cls):
        common_timezones = [ "UTC", "US/Eastern", "US/Central", "US/Pacific", "Europe/London", "America/New_York" ]
        return {
            "orb_hour": {"type": "int", "default": 9, "min": 0, "max": 23, "label": "Premarket End Hour (Timezone Specific)"}, # Renamed label for clarity
            "orb_minute": {"type": "int", "default": 30, "min": 0, "max": 59, "label": "Premarket End Minute (Timezone Specific)"}, # Assuming this is end of premarket / start of ORB
            "orb_timezone": {"type": "select", "default": "America/New_York", "options": common_timezones, "label": "Premarket/ORB Timezone"},
            "lookback_bars_for_orb": {"type": "int", "default": 15, "min":1, "max":120, "label": "Premarket Duration (bars for ORB definition)", "description":"Number of bars of strategy's timeframe before Premarket End Time to define the range."},
            "premarket_max_deviation_percent": {"type": "float", "default": 0.5, "min": 0.01, "max": 5.0, "step": 0.01, "label": "Max Price Deviation from Premarket (%)"},
            "tp_percent": {"type": "float", "default": 1.0, "label": "Take Profit (%)"},
            "sl_percent": {"type": "float", "default": 0.5, "label": "Stop Loss (%)"},
            "position_size_percent_capital": {"type": "float", "default": 10.0, "label": "Position Size (% of Effective Capital)"}
        }

    def _get_precisions_live(self, exchange_ccxt):
        if not self._precisions_fetched_:
            try:
                exchange_ccxt.load_markets(True)
                market = exchange_ccxt.market(self.symbol)
                self.price_precision = market['precision']['price']
                self.quantity_precision = market['precision']['amount']
                self._precisions_fetched_ = True
                logger.info(f"[{self.name}-{self.symbol}] Precisions: Price={self.price_precision}, Qty={self.quantity_precision}")
            except Exception as e: logger.error(f"[{self.name}-{self.symbol}] Error fetching live precisions: {e}", exc_info=True)

    def _format_price(self, price, exchange_ccxt): self._get_precisions_live(exchange_ccxt); return float(exchange_ccxt.price_to_precision(self.symbol, price))
    def _format_quantity(self, quantity, exchange_ccxt): self._get_precisions_live(exchange_ccxt); return float(exchange_ccxt.amount_to_precision(self.symbol, quantity))

    def _await_order_fill(self, exchange_ccxt, order_id: str, symbol: str, timeout_seconds: int = 60, check_interval_seconds: int = 3):
        start_time = time.time()
        logger.info(f"[{self.name}-{self.symbol}] Awaiting fill for order {order_id} (timeout: {timeout_seconds}s)")
        while time.time() - start_time < timeout_seconds:
            try:
                order = exchange_ccxt.fetch_order(order_id, symbol)
                logger.debug(f"[{self.name}-{self.symbol}] Order {order_id} status: {order['status']}")
                if order['status'] == 'closed': logger.info(f"[{self.name}-{self.symbol}] Order {order_id} filled. AvgPrice: {order.get('average')}, Qty: {order.get('filled')}"); return order
                if order['status'] in ['canceled', 'rejected', 'expired']: logger.warning(f"[{self.name}-{self.symbol}] Order {order_id} is {order['status']}."); return order
            except ccxt.OrderNotFound: logger.warning(f"[{self.name}-{self.symbol}] Order {order_id} not found. Retrying.")
            except Exception as e: logger.error(f"[{self.name}-{self.symbol}] Error fetching order {order_id}: {e}. Retrying.", exc_info=True)
            time.sleep(check_interval_seconds)
        logger.warning(f"[{self.name}-{self.symbol}] Timeout for order {order_id}. Final check.")
        try: final_status = exchange_ccxt.fetch_order(order_id, symbol); logger.info(f"[{self.name}-{self.symbol}] Final status for order {order_id}: {final_status['status']}"); return final_status
        except Exception as e: logger.error(f"[{self.name}-{self.symbol}] Final check for order {order_id} failed: {e}", exc_info=True); return None

    def _create_db_order(self, db_session: Session, subscription_id: int, **kwargs):
        db_order = Order(subscription_id=subscription_id, **kwargs); db_session.add(db_order); db_session.commit(); return db_order

    def _update_orb_range(self, df_in_orb_tz: pd.DataFrame, current_bar_dt_orb_tz: datetime.datetime):
        current_date_in_orb_tz = current_bar_dt_orb_tz.date()
        if self.opening_range_set_for_date == current_date_in_orb_tz: return

        self.opening_range_high = None; self.opening_range_low = None
        self.opening_range_set_for_date = current_date_in_orb_tz 

        # Premarket End Time (also the ORB reference time)
        pm_end_target_dt = datetime.datetime.combine(current_date_in_orb_tz, datetime.time(self.orb_hour, self.orb_minute), tzinfo=self.pytz_orb_timezone)
        
        # Filter data up to and including the ORB reference time
        df_for_orb_calc = df_in_orb_tz[df_in_orb_tz.index <= pm_end_target_dt]
        if df_for_orb_calc.empty: logger.debug(f"[{self.name}-{self.symbol}] No data up to ORB target time {pm_end_target_dt}."); return

        # Get the last `lookback_bars_for_orb` from this filtered data
        orb_slice_df = df_for_orb_calc.tail(self.lookback_bars_for_orb)
        
        if not orb_slice_df.empty and len(orb_slice_df) == self.lookback_bars_for_orb:
            # Ensure all bars in slice are on the same day as the target ORB date
            if all(b_idx.date() == current_date_in_orb_tz for b_idx in orb_slice_df.index):
                self.opening_range_high = orb_slice_df['High'].max()
                self.opening_range_low = orb_slice_df['Low'].min()
                self.max_deviation_high = self.opening_range_high * (1 + self.premarket_max_deviation_decimal)
                self.max_deviation_low = self.opening_range_low * (1 - self.premarket_max_deviation_decimal)
                logger.info(f"[{self.name}-{self.symbol}] ORB Set for {current_date_in_orb_tz}: H={self.opening_range_high:.2f}, L={self.opening_range_low:.2f}. MaxDev H/L: {self.max_deviation_high:.2f}/{self.max_deviation_low:.2f}")
            else: logger.debug(f"[{self.name}-{self.symbol}] ORB slice for {current_date_in_orb_tz} spanned multiple days. ORB not set.")
        else: logger.debug(f"[{self.name}-{self.symbol}] Not enough bars ({len(orb_slice_df)}/{self.lookback_bars_for_orb}) for ORB on {current_date_in_orb_tz} ending {pm_end_target_dt}")

    def run_backtest(self, historical_df: pd.DataFrame, htf_historical_df: pd.DataFrame = None):
        logger.info(f"Running backtest for {self.name} on {self.symbol}...")
        return {"pnl": 0, "trades": [], "message": "Backtest logic for PremarketBreakout needs to be reviewed and aligned with live logic if used for performance metrics."}

    def execute_live_signal(self, db_session: Session, subscription_id: int, market_data_df: pd.DataFrame, exchange_ccxt, user_sub_obj: UserStrategySubscription):
        logger.debug(f"[{self.name}-{self.symbol}] Executing live signal for sub {subscription_id}...")
        if market_data_df.empty or len(market_data_df) < 4 : logger.warning(f"[{self.name}-{self.symbol}] Insufficient market data."); return
        self._get_precisions_live(exchange_ccxt)

        df_utc = market_data_df.copy()
        if not isinstance(df_utc.index, pd.DatetimeIndex): df_utc.index = pd.to_datetime(df_utc.index)
        if df_utc.index.tzinfo is None: df_utc = df_utc.tz_localize('UTC')
        df_orb_tz = df_utc.tz_convert(self.pytz_orb_timezone)

        current_bar_dt_orb_tz = df_orb_tz.index[-1]
        self._update_orb_range(df_orb_tz, current_bar_dt_orb_tz)

        if self.opening_range_high is None or self.opening_range_low is None or self.opening_range_set_for_date != current_bar_dt_orb_tz.date():
            logger.debug(f"[{self.name}-{self.symbol}] ORB not set for current bar's date ({current_bar_dt_orb_tz.date()}). ORB set for: {self.opening_range_set_for_date}"); return
        
        # Market Open time in ORB timezone for the current bar's date
        market_open_dt_orb_tz = datetime.datetime.combine(current_bar_dt_orb_tz.date(), datetime.time(self.orb_hour, self.orb_minute), tzinfo=self.pytz_orb_timezone)
        # Note: The original script uses `market_open_hour_est` which might be different from `orb_hour`.
        # For this strategy, "market open" is effectively when the ORB period ends and trading can begin.
        
        if current_bar_dt_orb_tz < market_open_dt_orb_tz: # Don't trade before market "open" (i.e., after ORB period defined)
            logger.debug(f"[{self.name}-{self.symbol}] Market not yet open for breakout trading. Current: {current_bar_dt_orb_tz}, Open: {market_open_dt_orb_tz}"); return

        latest_bar_orb_tz = df_orb_tz.iloc[-1]
        price = latest_bar_orb_tz['Close'] # Use close of the last completed bar for decisions
        
        position_db = db_session.query(Position).filter(Position.subscription_id == subscription_id, Position.symbol == self.symbol, Position.is_open == True).first()

        # Exit Logic
        if position_db:
            exit_reason = None; side_to_close = None; filled_exit_order = None
            entry_price = position_db.entry_price
            if position_db.side == "long":
                sl_price = entry_price * (1 - self.sl_decimal); tp_price = entry_price * (1 + self.tp_decimal)
                if price <= sl_price: exit_reason = "SL"
                elif price >= tp_price: exit_reason = "TP"
                elif price < self.opening_range_low: exit_reason = "Price re-entered ORB (Low)" # Exit if price falls back below ORB low
                if exit_reason: side_to_close = 'sell'
            elif position_db.side == "short":
                sl_price = entry_price * (1 + self.sl_decimal); tp_price = entry_price * (1 - self.tp_decimal)
                if price >= sl_price: exit_reason = "SL"
                elif price <= tp_price: exit_reason = "TP"
                elif price > self.opening_range_high: exit_reason = "Price re-entered ORB (High)" # Exit if price rises back above ORB high
                if exit_reason: side_to_close = 'buy'

            if exit_reason and side_to_close:
                logger.info(f"[{self.name}-{self.symbol}] Closing {position_db.side} Pos ID {position_db.id} at {price}. Reason: {exit_reason}")
                close_qty = self._format_quantity(position_db.amount, exchange_ccxt)
                db_exit_order = self._create_db_order(db_session, subscription_id, symbol=self.symbol, order_type='market', side=side_to_close, amount=close_qty, status='pending_creation')
                try:
                    exit_receipt = exchange_ccxt.create_market_order(self.symbol, side_to_close, close_qty, params={'reduceOnly': True})
                    db_exit_order.order_id = exit_receipt['id']; db_exit_order.status = 'open'; db_session.commit()
                    filled_exit_order = self._await_order_fill(exchange_ccxt, exit_receipt['id'], self.symbol)
                    if filled_exit_order and filled_exit_order['status'] == 'closed':
                        db_exit_order.status='closed'; db_exit_order.price=filled_exit_order['average']; db_exit_order.filled=filled_exit_order['filled']; db_exit_order.cost=filled_exit_order['cost']; db_exit_order.updated_at=datetime.datetime.utcnow()
                        position_db.is_open=False; position_db.closed_at=datetime.datetime.utcnow()
                        pnl = (filled_exit_order['average'] - entry_price) * filled_exit_order['filled'] if position_db.side == 'long' else (entry_price - filled_exit_order['average']) * filled_exit_order['filled']
                        position_db.pnl=pnl; position_db.updated_at = datetime.datetime.utcnow()
                        logger.info(f"[{self.name}-{self.symbol}] {position_db.side} Pos ID {position_db.id} closed. PnL: {pnl:.2f}")
                    else: logger.error(f"[{self.name}-{self.symbol}] Exit order {exit_receipt['id']} failed. Pos ID {position_db.id} open."); db_exit_order.status = filled_exit_order.get('status', 'fill_check_failed') if filled_exit_order else 'fill_check_failed'
                    db_session.commit()
                except Exception as e: logger.error(f"[{self.name}-{self.symbol}] Error closing Pos ID {position_db.id}: {e}", exc_info=True); db_exit_order.status='error'; db_session.commit()
                return

        # Entry Logic
        if not position_db:
            # Pine conditions: crossover(close, s_high) -> close > s_high AND close[1] <= s_high
            # For live, we use the latest completed bar's close (`price`) and its previous bar's close (`prev_close`)
            prev_close = df_orb_tz['Close'].iloc[-2] if len(df_orb_tz) >= 2 else price # Fallback if only one bar
            
            buy_cond = price > self.opening_range_high and prev_close <= self.opening_range_high and price <= self.max_deviation_high
            sell_cond = price < self.opening_range_low and prev_close >= self.opening_range_low and price >= self.max_deviation_low
            
            entry_side = None
            if buy_cond: entry_side = "long"
            elif sell_cond: entry_side = "short"

            if entry_side:
                allocated_capital = json.loads(user_sub_obj.custom_parameters).get("capital", self.capital_param)
                position_size_usdt = allocated_capital * self.position_size_percent_capital_decimal
                asset_qty_to_trade = self._format_quantity(position_size_usdt / price, exchange_ccxt)
                if asset_qty_to_trade <= 0: logger.warning(f"[{self.name}-{self.symbol}] Asset quantity zero. Skipping."); return

                logger.info(f"[{self.name}-{self.symbol}] {entry_side.upper()} entry signal at {price}. Size: {asset_qty_to_trade}. ORB H:{self.opening_range_high:.2f}, L:{self.opening_range_low:.2f}")
                db_entry_order = self._create_db_order(db_session, subscription_id, symbol=self.symbol, order_type='market', side=entry_side, amount=asset_qty_to_trade, status='pending_creation')
                try:
                    entry_receipt = exchange_ccxt.create_market_order(self.symbol, entry_side, asset_qty_to_trade)
                    db_entry_order.order_id = entry_receipt['id']; db_entry_order.status = 'open'; db_session.commit()
                    filled_entry_order = self._await_order_fill(exchange_ccxt, entry_receipt['id'], self.symbol)
                    if filled_entry_order and filled_entry_order['status'] == 'closed':
                        db_entry_order.status='closed'; db_entry_order.price=filled_entry_order['average']; db_entry_order.filled=filled_entry_order['filled']; db_entry_order.cost=filled_entry_order['cost']; db_entry_order.updated_at = datetime.datetime.utcnow()
                        
                        new_pos = Position(subscription_id=subscription_id, symbol=self.symbol, exchange_name=str(exchange_ccxt.id), side=entry_side, amount=filled_entry_order['filled'], entry_price=filled_entry_order['average'], current_price=filled_entry_order['average'], is_open=True, created_at=datetime.datetime.utcnow(), updated_at=datetime.datetime.utcnow())
                        db_session.add(new_pos); db_session.commit()
                        logger.info(f"[{self.name}-{self.symbol}] {entry_side.upper()} Pos ID {new_pos.id} created. Entry: {new_pos.entry_price}, Size: {new_pos.amount}")
                        # Note: This strategy version doesn't place explicit SL/TP orders on exchange. It monitors price levels.
                    else: logger.error(f"[{self.name}-{self.symbol}] Entry order {entry_receipt['id']} failed. Pos not opened."); db_entry_order.status = filled_entry_order.get('status', 'fill_check_failed') if filled_entry_order else 'fill_check_failed'
                    db_session.commit()
                except Exception as e: logger.error(f"[{self.name}-{self.symbol}] Error during {entry_side} entry: {e}", exc_info=True); db_entry_order.status='error'; db_session.commit()
        logger.debug(f"[{self.name}-{self.symbol}] Live signal check complete.")
