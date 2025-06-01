import pandas as pd
import numpy as np
import logging
import time
import datetime
import pytz
import json # For UserStrategySubscription parameters
from sqlalchemy.orm import Session
from backend.models import Position, Order, UserStrategySubscription

logger = logging.getLogger(__name__)

class ORBStrategy:
    def __init__(self, symbol: str, timeframe: str, capital: float = 10000, **custom_parameters):
        self.name = "ORBStrategy" # Class name for clarity
        self.symbol = symbol
        self.timeframe_str = timeframe 
        self.capital_param = capital # Store the capital from parameters, might be overridden by subscription

        defaults = {
            "orb_hour": 9, "orb_minute": 15, "orb_timezone": "America/New_York",
            "tp_percent": 1.0, "sl_percent": 0.5,
            "position_size_percent_capital": 10.0,
            "lookback_bars_for_orb": 1 
        }
        self_params = {**defaults, **custom_parameters}
        for key, value in self_params.items():
            setattr(self, key, value)

        try:
            self.pytz_orb_timezone = pytz.timezone(self.orb_timezone)
        except pytz.UnknownTimeZoneError:
            logger.warning(f"[{self.name}-{self.symbol}] Unknown ORB timezone '{self.orb_timezone}', defaulting to UTC.")
            self.pytz_orb_timezone = pytz.utc; self.orb_timezone = "UTC"

        self.tp_decimal = self.tp_percent / 100.0
        self.sl_decimal = self.sl_percent / 100.0
        self.position_size_percent_capital_decimal = self.position_size_percent_capital / 100.0
        
        # In-memory state for ORB range (refreshed daily based on market time)
        self.opening_range_high = None
        self.opening_range_low = None
        self.opening_range_set_for_date = None # Tracks for which ORB-timezoned date range is set

        self.price_precision = 8; self.quantity_precision = 8
        self._precisions_fetched_ = False
        
        logger.info(f"[{self.name}-{self.symbol}] Initialized with effective params: {self_params}")

    @classmethod
    def get_parameters_definition(cls):
        common_timezones = [ "UTC", "US/Eastern", "US/Central", "US/Pacific", "Europe/London", "Europe/Berlin", "Asia/Kolkata", "Asia/Tokyo", "Australia/Sydney", "America/New_York" ] # Simplified list
        return {
            "orb_hour": {"type": "int", "default": 9, "min": 0, "max": 23, "label": "ORB Hour (Timezone Specific)"},
            "orb_minute": {"type": "int", "default": 15, "min": 0, "max": 59, "label": "ORB Minute (Timezone Specific)"},
            "orb_timezone": {"type": "select", "default": "America/New_York", "options": common_timezones, "label": "ORB Timezone"},
            "lookback_bars_for_orb": {"type": "int", "default": 1, "min":1, "max":10, "label": "ORB Lookback Bars"},
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

        orb_target_datetime = datetime.datetime.combine(current_date_in_orb_tz, datetime.time(self.orb_hour, self.orb_minute), tzinfo=self.pytz_orb_timezone)
        
        day_data = df_in_orb_tz[df_in_orb_tz.index.date == current_date_in_orb_tz]
        if day_data.empty: logger.debug(f"[{self.name}-{self.symbol}] No data for {current_date_in_orb_tz} in ORB timezone."); return

        # Find the index of the bar that is AT or JUST AFTER the orb_target_datetime
        # This bar's data (or `lookback_bars_for_orb` ending with this bar) defines the ORB.
        orb_defining_bars = day_data[day_data.index >= orb_target_datetime]
        if not orb_defining_bars.empty:
            orb_bar_end_timestamp = orb_defining_bars.index[0]
            try:
                orb_bar_actual_idx_pos = df_in_orb_tz.index.get_loc(orb_bar_end_timestamp)
                start_slice_idx = max(0, orb_bar_actual_idx_pos - (self.lookback_bars_for_orb - 1))
                orb_slice_df = df_in_orb_tz.iloc[start_slice_idx : orb_bar_actual_idx_pos + 1]

                if all(b_idx.date() == current_date_in_orb_tz for b_idx in orb_slice_df.index) and not orb_slice_df.empty:
                    self.opening_range_high = orb_slice_df['High'].max()
                    self.opening_range_low = orb_slice_df['Low'].min()
                    logger.info(f"[{self.name}-{self.symbol}] ORB Set for {current_date_in_orb_tz}: H={self.opening_range_high}, L={self.opening_range_low} from {len(orb_slice_df)} bars ending {orb_slice_df.index[-1]}")
                else: logger.debug(f"[{self.name}-{self.symbol}] ORB slice invalid for {current_date_in_orb_tz}.")
            except KeyError: logger.error(f"[{self.name}-{self.symbol}] Timestamp {orb_bar_end_timestamp} not found in main df_in_orb_tz index. ORB not set.")
        else: logger.debug(f"[{self.name}-{self.symbol}] No bar found at or after ORB time {orb_target_datetime}.")

    def run_backtest(self, historical_df: pd.DataFrame, htf_historical_df: pd.DataFrame = None):
        # (Backtesting logic remains largely unchanged, for offline simulation)
        logger.info(f"Running backtest for {self.name} on {self.symbol}...")
        # ... (original backtest logic can remain, ensure it uses decimal multipliers for SL/TP) ...
        return {"pnl": 0, "trades": [], "message": "Backtest logic for ORB needs review if used for performance metrics."}

    def execute_live_signal(self, db_session: Session, subscription_id: int, market_data_df: pd.DataFrame, exchange_ccxt, user_sub_obj: UserStrategySubscription):
        logger.debug(f"[{self.name}-{self.symbol}] Executing live signal for sub {subscription_id}...")
        if market_data_df.empty or len(market_data_df) < 4: logger.warning(f"[{self.name}-{self.symbol}] Insufficient market data."); return
        self._get_precisions_live(exchange_ccxt)

        df_utc = market_data_df.copy()
        if not isinstance(df_utc.index, pd.DatetimeIndex): df_utc.index = pd.to_datetime(df_utc.index)
        if df_utc.index.tzinfo is None: df_utc = df_utc.tz_localize('UTC')
        df = df_utc.tz_convert(self.pytz_orb_timezone) # Convert entire DF to ORB timezone for consistent indexing

        # Shifted columns for signal logic (must be done on ORB timezone df)
        df['c1'] = df['Close'].shift(1); df['c2'] = df['Close'].shift(2); df['c3'] = df['Close'].shift(3)
        df['h_curr'] = df['High']; df['h1'] = df['High'].shift(1); df['h2'] = df['High'].shift(2)
        df['l_curr'] = df['Low']; df['l1'] = df['Low'].shift(1); df['l2'] = df['Low'].shift(2)
        df.dropna(subset=['c3','h2','l2'], inplace=True) # Drop rows where shifted values are NaN
        if len(df) < 1: logger.warning(f"[{self.name}-{self.symbol}] Not enough data after shifts."); return

        current_bar_dt_orb_tz = df.index[-1] # This is the start time of the last completed bar
        self._update_orb_range(df, current_bar_dt_orb_tz) 

        if self.opening_range_high is None or self.opening_range_low is None or self.opening_range_set_for_date != current_bar_dt_orb_tz.date():
            logger.debug(f"[{self.name}-{self.symbol}] ORB not set for current bar's date ({current_bar_dt_orb_tz.date()}). ORB set for: {self.opening_range_set_for_date}"); return

        latest_bar = df.iloc[-1] # Use the last row for signal generation
        price = latest_bar['Close']
        c2, c3 = latest_bar['c2'], latest_bar['c3']
        h_curr, h1, h2 = latest_bar['h_curr'], latest_bar['h1'], latest_bar['h2']
        l_curr, l1, l2 = latest_bar['l_curr'], latest_bar['l1'], latest_bar['l2']

        if any(pd.isna(x) for x in [price, c2, c3, h_curr, h1, h2, l_curr, l1, l2]):
            logger.warning(f"[{self.name}-{self.symbol}] NaN data for signal check on bar {current_bar_dt_orb_tz}."); return

        position_db = db_session.query(Position).filter(Position.subscription_id == subscription_id, Position.symbol == self.symbol, Position.is_open == True).first()
        
        # Exit Logic
        if position_db:
            exit_reason = None; side_to_close = None; filled_exit_order = None
            entry_price = position_db.entry_price
            if position_db.side == "long":
                sl_price = entry_price * (1 - self.sl_decimal); tp_price = entry_price * (1 + self.tp_decimal)
                if price <= sl_price: exit_reason = "SL"
                elif price >= tp_price: exit_reason = "TP"
                if exit_reason: side_to_close = 'sell'
            elif position_db.side == "short":
                sl_price = entry_price * (1 + self.sl_decimal); tp_price = entry_price * (1 - self.tp_decimal)
                if price >= sl_price: exit_reason = "SL"
                elif price <= tp_price: exit_reason = "TP"
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
                    else: logger.error(f"[{self.name}-{self.symbol}] Exit order {exit_receipt['id']} failed. Pos ID {position_db.id} might still be open."); db_exit_order.status = filled_exit_order.get('status', 'fill_check_failed') if filled_exit_order else 'fill_check_failed'
                    db_session.commit()
                except Exception as e: logger.error(f"[{self.name}-{self.symbol}] Error closing Pos ID {position_db.id}: {e}", exc_info=True); db_exit_order.status='error'; db_session.commit()
                return # Action taken

        # Entry Logic (Only if ORB is set for the current bar's date and current time is past ORB time)
        if not position_db and current_bar_dt_orb_tz.time() > datetime.time(self.orb_hour, self.orb_minute):
            # Pine conditions: ta.crossover(close[2], s.high[0]) means close[2] > s.high[0] AND close[3] <= s.high[0]
            buy_cond = (c2 > self.opening_range_high and c3 <= self.opening_range_high) and (h1 > h2) and (h_curr > h1)
            sell_cond = (c2 < self.opening_range_low and c3 >= self.opening_range_low) and (l1 < l2) and (l_curr < l1)
            
            entry_side = None
            if buy_cond: entry_side = "long"
            elif sell_cond: entry_side = "short"

            if entry_side:
                allocated_capital = json.loads(user_sub_obj.custom_parameters).get("capital", self.capital_param)
                position_size_usdt = allocated_capital * self.position_size_percent_capital_decimal
                asset_qty_to_trade = self._format_quantity(position_size_usdt / price, exchange_ccxt)
                if asset_qty_to_trade <= 0: logger.warning(f"[{self.name}-{self.symbol}] Asset quantity zero. Skipping."); return

                logger.info(f"[{self.name}-{self.symbol}] {entry_side.upper()} entry signal at {price}. Size: {asset_qty_to_trade}. ORB H:{self.opening_range_high}, L:{self.opening_range_low}")
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
                        # For ORB, SL/TP are managed by checking price against calculated levels, not by placing separate exchange orders initially.
                    else: logger.error(f"[{self.name}-{self.symbol}] Entry order {entry_receipt['id']} failed. Pos not opened."); db_entry_order.status = filled_entry_order.get('status', 'fill_check_failed') if filled_entry_order else 'fill_check_failed'
                    db_session.commit()
                except Exception as e: logger.error(f"[{self.name}-{self.symbol}] Error during {entry_side} entry: {e}", exc_info=True); db_entry_order.status='error'; db_session.commit()
        logger.debug(f"[{self.name}-{self.symbol}] Live signal check complete.")
