# trading_platform/strategies/rsi_divergence_strategy.py
import pandas as pd
import pandas_ta as ta
import numpy as np
import logging
import time # For awaiting order fills
import datetime # For timestamps
import json # For UserStrategySubscription parameters
from sqlalchemy.orm import Session
from backend.models import Position, Order, UserStrategySubscription # Ensure UserStrategySubscription is imported
from scipy.signal import find_peaks 

logger = logging.getLogger(__name__)

class RSIDivergenceStrategy:
    def __init__(self, symbol: str, timeframe: str, capital: float = 10000, **custom_parameters):
        self.name = "RSIDivergenceStrategy"
        self.symbol = symbol
        self.timeframe = timeframe
        
        defaults = {
            "rsi_period": 14, 
            "lookback_period": 20, 
            "peak_prominence": 0.5, 
            "risk_per_trade_percent": 1.5, 
            "stop_loss_percent": 2.0, 
            "take_profit_percent": 4.0 
        }
        self_params = {**defaults, **custom_parameters}
        for key, value in self_params.items():
            setattr(self, key, value)

        self.risk_per_trade_decimal = self.risk_per_trade_percent / 100.0
        self.stop_loss_decimal = self.stop_loss_percent / 100.0
        self.take_profit_decimal = self.take_profit_percent / 100.0
        
        self.price_precision = 8
        self.quantity_precision = 8
        self._precisions_fetched_ = False

        init_params_log = {k:v for k,v in self_params.items()}
        init_params_log.update({"symbol": symbol, "timeframe": timeframe, "capital_param": capital})
        logger.info(f"[{self.name}-{self.symbol}] Initialized with effective params: {init_params_log}")

    @classmethod
    def get_parameters_definition(cls):
        return {
            "rsi_period": {"type": "int", "default": 14, "min": 5, "max": 50, "label": "RSI Period"},
            "lookback_period": {"type": "int", "default": 20, "min": 10, "max": 100, "label": "Divergence Lookback Period"},
            "peak_prominence": {"type": "float", "default": 0.5, "min": 0.1, "max": 10, "step":0.1, "label": "Peak Prominence (RSI)"},
            "risk_per_trade_percent": {"type": "float", "default": 1.5, "min": 0.1, "max": 10.0, "step":0.1, "label": "Risk per Trade (%)"},
            "stop_loss_percent": {"type": "float", "default": 2.0, "min": 0.1, "step": 0.1, "label": "Stop Loss % from Entry"},
            "take_profit_percent": {"type": "float", "default": 4.0, "min": 0.1, "step": 0.1, "label": "Take Profit % from Entry"}
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

    def _calculate_rsi(self, df: pd.DataFrame):
        if 'close' not in df.columns: logger.error(f"[{self.name}-{self.symbol}] 'close' column missing for RSI."); return df
        df['rsi'] = ta.rsi(df['close'], length=self.rsi_period)
        return df

    def _find_divergence(self, price_series: pd.Series, rsi_series: pd.Series):
        if not price_series.index.equals(rsi_series.index): logger.error(f"[{self.name}-{self.symbol}] Price/RSI index mismatch."); return None, None
        
        price_std = price_series.std()
        if price_std == 0: price_std = price_series.mean() * 0.01 # Avoid zero std for flat lines

        price_low_indices, _ = find_peaks(-price_series.values, prominence=price_std * 0.1)
        price_high_indices, _ = find_peaks(price_series.values, prominence=price_std * 0.1)
        rsi_low_indices, _ = find_peaks(-rsi_series.values, prominence=self.peak_prominence)
        rsi_high_indices, _ = find_peaks(rsi_series.values, prominence=self.peak_prominence)

        if len(price_low_indices) >= 2 and len(rsi_low_indices) >= 2:
            p_low1_idx, p_low2_idx = price_low_indices[-2], price_low_indices[-1]
            rsi_l1_idx, rsi_l2_idx = -1, -1
            for r_idx in reversed(rsi_low_indices): 
                if r_idx <= p_low2_idx: rsi_l2_idx = r_idx; break
            for r_idx in reversed(rsi_low_indices):
                if r_idx <= p_low1_idx and r_idx < rsi_l2_idx : rsi_l1_idx = r_idx; break
            if rsi_l1_idx != -1 and rsi_l2_idx != -1 and \
               price_series.iloc[p_low2_idx] < price_series.iloc[p_low1_idx] and \
               rsi_series.iloc[rsi_l2_idx] > rsi_series.iloc[rsi_l1_idx] and \
               (len(price_series) - 1 - p_low2_idx) <= 3 : # Signal on one of last 3 bars
                return "bullish", p_low2_idx

        if len(price_high_indices) >= 2 and len(rsi_high_indices) >= 2:
            p_high1_idx, p_high2_idx = price_high_indices[-2], price_high_indices[-1]
            rsi_h1_idx, rsi_h2_idx = -1,-1
            for r_idx in reversed(rsi_high_indices):
                if r_idx <= p_high2_idx: rsi_h2_idx = r_idx; break
            for r_idx in reversed(rsi_high_indices):
                if r_idx <= p_high1_idx and r_idx < rsi_h2_idx: rsi_h1_idx = r_idx; break
            if rsi_h1_idx != -1 and rsi_h2_idx != -1 and \
               price_series.iloc[p_high2_idx] > price_series.iloc[p_high1_idx] and \
               rsi_series.iloc[rsi_h2_idx] < rsi_series.iloc[rsi_h1_idx] and \
               (len(price_series) - 1 - p_high2_idx) <= 3:
                return "bearish", p_high2_idx
        return None, None

    def run_backtest(self, historical_df: pd.DataFrame, htf_historical_df: pd.DataFrame = None):
        logger.info(f"Running backtest for {self.name} on {self.symbol}...")
        # (Backtesting logic remains simplified as per original)
        return {"pnl": 0, "trades": [], "message": "Backtest for RSIDivergence needs review if used for performance."}

    def execute_live_signal(self, db_session: Session, subscription_id: int, market_data_df: pd.DataFrame, exchange_ccxt, user_sub_obj: UserStrategySubscription):
        logger.debug(f"[{self.name}-{self.symbol}] Executing live signal for sub {subscription_id}...")
        if market_data_df.empty or len(market_data_df) < self.lookback_period + self.rsi_period:
            logger.warning(f"[{self.name}-{self.symbol}] Insufficient market data."); return
        self._get_precisions_live(exchange_ccxt)

        df = self._calculate_rsi(market_data_df.copy()); df.dropna(inplace=True)
        if len(df) < self.lookback_period: logger.warning(f"[{self.name}-{self.symbol}] Not enough data post-RSI calc."); return

        analysis_window_df = df.iloc[-self.lookback_period:] # Use most recent 'lookback_period' bars
        current_price = analysis_window_df['close'].iloc[-1]
        
        position_db = db_session.query(Position).filter(Position.subscription_id == subscription_id, Position.symbol == self.symbol, Position.is_open == True).first()

        # Exit Logic
        if position_db:
            exit_reason = None; side_to_close = None; filled_exit_order = None
            entry_price = position_db.entry_price
            if position_db.side == "long":
                sl_price = entry_price * (1 - self.sl_decimal); tp_price = entry_price * (1 + self.tp_decimal)
                if current_price <= sl_price: exit_reason = "SL"
                elif current_price >= tp_price: exit_reason = "TP"
                if exit_reason: side_to_close = 'sell'
            elif position_db.side == "short":
                sl_price = entry_price * (1 + self.sl_decimal); tp_price = entry_price * (1 - self.tp_decimal)
                if current_price >= sl_price: exit_reason = "SL"
                elif current_price <= tp_price: exit_reason = "TP"
                if exit_reason: side_to_close = 'buy'

            if exit_reason and side_to_close:
                logger.info(f"[{self.name}-{self.symbol}] Closing {position_db.side} Pos ID {position_db.id} at {current_price}. Reason: {exit_reason}")
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
                return # Action taken

        # Entry Logic
        if not position_db:
            divergence_type, signal_bar_index = self._find_divergence(analysis_window_df['close'], analysis_window_df['rsi'])
            # Ensure divergence signal is on the latest completed bar
            if divergence_type and signal_bar_index == len(analysis_window_df) - 1:
                allocated_capital = json.loads(user_sub_obj.custom_parameters).get("capital", self.capital_param) # Capital from subscription
                amount_to_risk_usd = allocated_capital * self.risk_per_trade_decimal
                sl_distance_usd = current_price * self.sl_decimal
                if sl_distance_usd == 0: logger.warning(f"[{self.name}-{self.symbol}] SL distance zero. Cannot size."); return
                
                position_size_asset = self._format_quantity(amount_to_risk_usd / sl_distance_usd, exchange_ccxt)
                if position_size_asset <= 0: logger.warning(f"[{self.name}-{self.symbol}] Asset quantity zero. Skipping."); return

                entry_side = "long" if divergence_type == "bullish" else "short"
                logger.info(f"[{self.name}-{self.symbol}] {divergence_type.upper()} RSI Divergence. {entry_side.upper()} entry at {current_price}. Size: {position_size_asset}")
                
                db_entry_order = self._create_db_order(db_session, subscription_id, symbol=self.symbol, order_type='market', side=entry_side, amount=position_size_asset, status='pending_creation')
                try:
                    entry_receipt = exchange_ccxt.create_market_order(self.symbol, entry_side, position_size_asset)
                    db_entry_order.order_id = entry_receipt['id']; db_entry_order.status = 'open'; db_session.commit()
                    filled_entry_order = self._await_order_fill(exchange_ccxt, entry_receipt['id'], self.symbol)
                    if filled_entry_order and filled_entry_order['status'] == 'closed':
                        db_entry_order.status='closed'; db_entry_order.price=filled_entry_order['average']; db_entry_order.filled=filled_entry_order['filled']; db_entry_order.cost=filled_entry_order['cost']; db_entry_order.updated_at = datetime.datetime.utcnow()
                        
                        new_pos = Position(subscription_id=subscription_id, symbol=self.symbol, exchange_name=str(exchange_ccxt.id), side=entry_side, amount=filled_entry_order['filled'], entry_price=filled_entry_order['average'], current_price=filled_entry_order['average'], is_open=True, created_at=datetime.datetime.utcnow(), updated_at=datetime.datetime.utcnow())
                        db_session.add(new_pos); db_session.commit()
                        logger.info(f"[{self.name}-{self.symbol}] {entry_side.upper()} Pos ID {new_pos.id} created. Entry: {new_pos.entry_price}, Size: {new_pos.amount}")
                        # Note: This strategy version relies on monitoring for SL/TP, not placing separate exchange orders.
                    else: logger.error(f"[{self.name}-{self.symbol}] Entry order {entry_receipt['id']} failed. Pos not opened."); db_entry_order.status = filled_entry_order.get('status', 'fill_check_failed') if filled_entry_order else 'fill_check_failed'
                    db_session.commit()
                except Exception as e: logger.error(f"[{self.name}-{self.symbol}] Error during {entry_side} entry: {e}", exc_info=True); db_entry_order.status='error'; db_session.commit()
        logger.debug(f"[{self.name}-{self.symbol}] Live signal check complete.")
