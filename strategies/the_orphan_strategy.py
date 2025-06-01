# trading_platform/strategies/the_orphan_strategy.py
import datetime
import pytz 
import pandas as pd
import logging
import time
import json # For UserStrategySubscription parameters
from sqlalchemy.orm import Session
from backend.models import Position, Order, UserStrategySubscription

logger = logging.getLogger(__name__)

class TheOrphanStrategy:
    def __init__(self, symbol: str, timeframe: str, capital: float = 2000, **custom_parameters):
        self.name = "TheOrphanStrategy"
        self.symbol = symbol
        self.timeframe_str = timeframe
        self.capital_param = capital 

        defaults = {
            "bb_length": 14, "bb_stdev": 2.1, "trend_period": 90,
            "vol_filter_length": 15, "vol_ma_length": 28,
            "sl_percent": 2.0, "tp_percent": 9.0,
            "trail_stop_activation_percent": 0.5, 
            "trail_offset_percent": 0.5,        
            "position_size_percent_equity": 10.0 
        }
        self_params = {**defaults, **custom_parameters}
        for key, value in self_params.items():
            setattr(self, key, value)

        self.sl_decimal = self.sl_percent / 100.0
        self.tp_decimal = self.tp_percent / 100.0
        self.trail_stop_activation_decimal = self.trail_stop_activation_percent / 100.0
        self.trail_offset_decimal = self.trail_offset_percent / 100.0
        self.position_size_percent_equity_decimal = self.position_size_percent_equity / 100.0
        
        self.price_precision = 8; self.quantity_precision = 8
        self._precisions_fetched_ = False
        
        # No in-memory live state variables here; will be fetched/managed via DB in execute_live_signal
        logger.info(f"[{self.name}-{self.symbol}] Initialized with effective params: {self_params}")

    @classmethod
    def get_parameters_definition(cls):
        return {
            "bb_length": {"type": "int", "default": 14, "min": 1, "label": "BB Length"},
            "bb_stdev": {"type": "float", "default": 2.1, "min": 0.1, "label": "BB StdDev"},
            "trend_period": {"type": "int", "default": 90, "min": 1, "label": "Trend Filter Period (EMA)"},
            "vol_filter_length": {"type": "int", "default": 15, "min": 1, "label": "Volatility Filter Length (StdDev)"},
            "vol_ma_length": {"type": "int", "default": 28, "min": 1, "label": "Volatility Filter MA Length"},
            "sl_percent": {"type": "float", "default": 2.0, "min": 0.1, "label": "Stop Loss (%)"},
            "tp_percent": {"type": "float", "default": 9.0, "min": 0.1, "label": "Take Profit (%)"},
            "trail_stop_activation_percent": {"type": "float", "default": 0.5, "min": 0.0, "label": "Trailing Stop Activation Gain (%)"},
            "trail_offset_percent": {"type": "float", "default": 0.5, "min": 0.1, "label": "Trailing Offset from High/Low (%)"},
            "position_size_percent_equity": {"type": "float", "default": 10.0, "min": 0.1, "max": 100.0, "label": "Position Size (% of Effective Capital)"}
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

    def _calculate_indicators(self, df: pd.DataFrame):
        if df.empty or 'Close' not in df.columns: return None, None, None, None
        if len(df) < max(self.bb_length, self.trend_period, self.vol_filter_length + self.vol_ma_length -1): return None, None, None, None
        
        bbands = ta.volatility.BollingerBands(close=df['Close'], window=self.bb_length, window_dev=self.bb_stdev)
        upper_band = bbands.bollinger_hband()
        lower_band = bbands.bollinger_lband()
        ema_trend = ta.trend.EMAIndicator(close=df['Close'], window=self.trend_period).ema_indicator()
        
        vol_stddev = df['Close'].rolling(window=self.vol_filter_length).std()
        volatility_filter = vol_stddev > vol_stddev.rolling(window=self.vol_ma_length).mean()
        return upper_band, lower_band, ema_trend, volatility_filter

    def run_backtest(self, historical_df: pd.DataFrame, htf_historical_df: pd.DataFrame = None):
        logger.info(f"Running backtest for {self.name} on {self.symbol}...")
        # (Backtesting logic remains simplified as per original)
        return {"pnl": 0, "trades": [], "message": "Backtest for TheOrphanStrategy needs detailed review if used for performance."}

    def execute_live_signal(self, db_session: Session, subscription_id: int, market_data_df: pd.DataFrame, exchange_ccxt, user_sub_obj: UserStrategySubscription):
        logger.debug(f"[{self.name}-{self.symbol}] Executing live signal for sub {subscription_id}...")
        required_bars = max(self.bb_length, self.trend_period, self.vol_filter_length + self.vol_ma_length - 1) + 2
        if market_data_df.empty or len(market_data_df) < required_bars: logger.warning(f"[{self.name}-{self.symbol}] Insufficient market data."); return
        self._get_precisions_live(exchange_ccxt)

        df = market_data_df.copy()
        upper_band, lower_band, ema_trend, volatility_filter = self._calculate_indicators(df)
        if upper_band is None or pd.isna(upper_band.iloc[-1]): logger.warning(f"[{self.name}-{self.symbol}] Indicator calculation failed."); return

        df['upper'] = upper_band; df['lower'] = lower_band; df['ema_trend'] = ema_trend; df['volatility_filter'] = volatility_filter
        df['Close_prev1'] = df['Close'].shift(1); df['Close_prev2'] = df['Close'].shift(2)
        df.dropna(inplace=True) # Drop rows with NaNs from shifts or initial indicator calculations
        if len(df) < 1: logger.warning(f"[{self.name}-{self.symbol}] Not enough data after indicator calculation and NaN drop."); return

        latest = df.iloc[-1]; price = latest['Close']; prev_close1 = latest['Close_prev1']; prev_close2 = latest['Close_prev2']
        
        position_db = db_session.query(Position).filter(Position.subscription_id == subscription_id, Position.symbol == self.symbol, Position.is_open == True).first()
        
        # Load trailing stop state from Position.custom_data if it exists
        live_trailing_stop_price = 0.0; live_trailing_stop_activated = False
        if position_db and position_db.custom_data:
            custom_data = json.loads(position_db.custom_data)
            live_trailing_stop_price = custom_data.get('trailing_stop_price', 0.0)
            live_trailing_stop_activated = custom_data.get('trailing_stop_activated', False)

        # Exit Logic
        if position_db:
            exit_reason = None; side_to_close = None; entry_price = position_db.entry_price
            
            if position_db.side == "long":
                sl = entry_price * (1 - self.sl_decimal); tp = entry_price * (1 + self.tp_decimal)
                if not live_trailing_stop_activated and price >= entry_price * (1 + self.trail_stop_activation_decimal): # Activate trail
                    live_trailing_stop_activated = True; live_trailing_stop_price = price * (1 - self.trail_offset_decimal)
                if live_trailing_stop_activated: live_trailing_stop_price = max(live_trailing_stop_price, price * (1 - self.trail_offset_decimal))
                
                if price >= tp: exit_reason = "TP"
                elif price <= sl: exit_reason = "SL"
                elif live_trailing_stop_activated and price <= live_trailing_stop_price: exit_reason = "Trail Stop"
                elif prev_close1 >= latest['upper'] and prev_close2 < latest['upper']: exit_reason = "BB Exit" # Close[1] crosses over upper
                if exit_reason: side_to_close = 'sell'
            elif position_db.side == "short":
                sl = entry_price * (1 + self.sl_decimal); tp = entry_price * (1 - self.tp_decimal)
                if not live_trailing_stop_activated and price <= entry_price * (1 - self.trail_stop_activation_decimal): # Activate trail
                    live_trailing_stop_activated = True; live_trailing_stop_price = price * (1 + self.trail_offset_decimal)
                if live_trailing_stop_activated: live_trailing_stop_price = min(live_trailing_stop_price, price * (1 + self.trail_offset_decimal))

                if price <= tp: exit_reason = "TP"
                elif price >= sl: exit_reason = "SL"
                elif live_trailing_stop_activated and price >= live_trailing_stop_price: exit_reason = "Trail Stop"
                elif prev_close1 <= latest['lower'] and prev_close2 > latest['lower']: exit_reason = "BB Exit" # Close[1] crosses under lower
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
                        position_db.pnl=pnl; position_db.updated_at = datetime.datetime.utcnow(); position_db.custom_data = None # Clear custom data on close
                        logger.info(f"[{self.name}-{self.symbol}] {position_db.side} Pos ID {position_db.id} closed. PnL: {pnl:.2f}")
                    else: logger.error(f"[{self.name}-{self.symbol}] Exit order {exit_receipt['id']} failed. Pos {position_db.id} open."); db_exit_order.status = filled_exit_order.get('status', 'fill_check_failed') if filled_exit_order else 'fill_check_failed'
                    db_session.commit()
                except Exception as e: logger.error(f"[{self.name}-{self.symbol}] Error closing Pos {position_db.id}: {e}", exc_info=True); db_exit_order.status='error'; db_session.commit()
                return

            # If still in position, update custom_data with trailing stop state
            if position_db.is_open:
                current_custom_data = json.loads(position_db.custom_data) if position_db.custom_data else {}
                new_custom_data = {'trailing_stop_price': live_trailing_stop_price, 'trailing_stop_activated': live_trailing_stop_activated}
                if current_custom_data != new_custom_data : # Only update if changed
                     position_db.custom_data = json.dumps(new_custom_data)
                     position_db.updated_at = datetime.datetime.utcnow()
                     db_session.commit()


        # Entry Logic
        if not position_db:
            buy_cond_bb = price >= latest['lower'] and prev_close1 < latest['lower']
            buy_cond_trend = price > latest['ema_trend']; buy_cond_final = buy_cond_bb and buy_cond_trend and latest['volatility_filter']
            sell_cond_bb = price >= latest['upper'] and prev_close1 < latest['upper'] # Pine: crossover(close[1], upper)
            sell_cond_trend = price < latest['ema_trend']; sell_condition_final = sell_cond_bb and sell_cond_trend and latest['volatility_filter']
            
            entry_side = None
            if buy_cond_final: entry_side = "long"
            elif sell_condition_final: entry_side = "short"

            if entry_side:
                allocated_capital = json.loads(user_sub_obj.custom_parameters).get("capital", self.capital_param)
                position_size_usdt = allocated_capital * self.position_size_percent_equity_decimal
                asset_qty = self._format_quantity(position_size_usdt / price, exchange_ccxt)
                if asset_qty <= 0: logger.warning(f"[{self.name}-{self.symbol}] Asset quantity zero. Skipping."); return

                logger.info(f"[{self.name}-{self.symbol}] {entry_side.upper()} entry signal at {price}. Size: {asset_qty}")
                db_entry_order = self._create_db_order(db_session, subscription_id, symbol=self.symbol, order_type='market', side=entry_side, amount=asset_qty, status='pending_creation')
                try:
                    entry_receipt = exchange_ccxt.create_market_order(self.symbol, entry_side, asset_qty)
                    db_entry_order.order_id = entry_receipt['id']; db_entry_order.status = 'open'; db_session.commit()
                    filled_entry = self._await_order_fill(exchange_ccxt, entry_receipt['id'], self.symbol)
                    if filled_entry and filled_entry['status'] == 'closed':
                        db_entry_order.status='closed'; db_entry_order.price=filled_entry['average']; db_entry_order.filled=filled_entry['filled']; db_entry_order.cost=filled_entry['cost']; db_entry_order.updated_at=datetime.datetime.utcnow()
                        
                        new_pos = Position(subscription_id=subscription_id, symbol=self.symbol, exchange_name=str(exchange_ccxt.id), side=entry_side, amount=filled_entry['filled'], entry_price=filled_entry['average'], current_price=filled_entry['average'], is_open=True, created_at=datetime.datetime.utcnow(), updated_at=datetime.datetime.utcnow(), custom_data=json.dumps({'trailing_stop_price': 0.0, 'trailing_stop_activated': False}))
                        db_session.add(new_pos); db_session.commit()
                        logger.info(f"[{self.name}-{self.symbol}] {entry_side.upper()} Pos ID {new_pos.id} created. Entry: {new_pos.entry_price}, Size: {new_pos.amount}")
                        # Note: This strategy does not place explicit SL/TP orders on exchange. It monitors price levels.
                    else: logger.error(f"[{self.name}-{self.symbol}] Entry order {entry_receipt['id']} failed. Pos not opened."); db_entry_order.status = filled_entry.get('status', 'fill_check_failed') if filled_entry else 'fill_check_failed'
                    db_session.commit()
                except Exception as e: logger.error(f"[{self.name}-{self.symbol}] Error during {entry_side} entry: {e}", exc_info=True); db_entry_order.status='error'; db_session.commit()
        logger.debug(f"[{self.name}-{self.symbol}] Live signal check complete.")
