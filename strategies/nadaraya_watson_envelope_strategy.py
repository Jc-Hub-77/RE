import pandas as pd
import numpy as np
import logging
import time
import ta 
import json # For custom_data if used
from sqlalchemy.orm import Session
from backend.models import Position, Order, UserStrategySubscription

logger = logging.getLogger(__name__)

class NadarayaWatsonEnvelopeStrategy:
    def __init__(self, symbol: str, timeframe: str, capital: float = 10000, **custom_parameters):
        self.name = "NadarayaWatsonEnvelopeStrategy" # Ensure class name matches file if used for loading
        self.symbol = symbol
        self.timeframe = timeframe
        
        defaults = {
            "h_bandwidth": 8.0,
            "multiplier": 3.0,
            "tp_percent": 1.0,
            "sl_percent": 0.5,
            "position_size_percent_capital": 10.0 
        }
        self_params = {**defaults, **custom_parameters}
        for key, value in self_params.items():
            setattr(self, key, value)

        self.tp_decimal = self.tp_percent / 100.0
        self.sl_decimal = self.sl_percent / 100.0
        self.position_size_percent_capital_decimal = self.position_size_percent_capital / 100.0
        
        self.price_precision = 8
        self.quantity_precision = 8
        self._precisions_fetched_ = False

        init_params_log = {k:v for k,v in self_params.items()}
        init_params_log.update({"symbol": symbol, "timeframe": timeframe, "capital_param": capital})
        logger.info(f"[{self.name}-{self.symbol}] Initialized with effective params: {init_params_log}")

    @classmethod
    def get_parameters_definition(cls):
        return {
            "h_bandwidth": {"type": "float", "default": 8.0, "label": "Kernel Bandwidth (h)"},
            "multiplier": {"type": "float", "default": 3.0, "label": "Envelope Multiplier"},
            "tp_percent": {"type": "float", "default": 1.0, "label": "Take Profit (%)"},
            "sl_percent": {"type": "float", "default": 0.5, "label": "Stop Loss (%)"},
            "position_size_percent_capital": {"type": "float", "default": 10.0, "label": "Position Size (% of Capital)"}
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
    
    def _gauss(self, x, h): return np.exp(-((x ** 2) / (2 * h ** 2)))

    def _calculate_nadaraya_watson_envelope(self, close_prices_series: pd.Series):
        data = close_prices_series.values; n = len(data)
        if n == 0: return pd.Series(dtype='float64'), pd.Series(dtype='float64'), pd.Series(dtype='float64')
        y_hat = np.zeros(n)
        for i in range(n):
            weighted_sum = 0; total_weight = 0
            for j in range(n):
                weight = self._gauss(i - j, self.h_bandwidth)
                weighted_sum += data[j] * weight; total_weight += weight
            y_hat[i] = data[i] if total_weight == 0 else weighted_sum / total_weight
        mae_value = np.abs(data - y_hat).mean() * self.multiplier
        return pd.Series(y_hat, index=close_prices_series.index), pd.Series(y_hat + mae_value, index=close_prices_series.index), pd.Series(y_hat - mae_value, index=close_prices_series.index)

    def run_backtest(self, historical_df: pd.DataFrame, htf_historical_df: pd.DataFrame = None):
        # (Backtesting logic remains largely unchanged from original, for offline simulation)
        logger.info(f"Running backtest for {self.name} on {self.symbol}...")
        # ... (original backtest logic can remain, ensuring it uses decimal multipliers for SL/TP) ...
        return {"pnl": 0, "trades": [], "message": "Backtest logic for NadarayaWatson needs review if used for performance metric generation."}

    def execute_live_signal(self, db_session: Session, subscription_id: int, market_data_df: pd.DataFrame, exchange_ccxt, user_sub_obj: UserStrategySubscription):
        logger.debug(f"[{self.name}-{self.symbol}] Executing live signal for sub {subscription_id}...")
        if market_data_df.empty or 'Close' not in market_data_df.columns or len(market_data_df) < int(self.h_bandwidth):
            logger.warning(f"[{self.name}-{self.symbol}] Insufficient market data for envelope calculation."); return
        self._get_precisions_live(exchange_ccxt)

        close_prices = market_data_df['Close']
        _, upper_band, lower_band = self._calculate_nadaraya_watson_envelope(close_prices)
        if upper_band.empty or lower_band.empty or pd.isna(upper_band.iloc[-1]) or pd.isna(lower_band.iloc[-1]):
            logger.warning(f"[{self.name}-{self.symbol}] Envelope calculation failed or resulted in NaN for latest bar."); return
        
        current_price = close_prices.iloc[-1]; current_upper = upper_band.iloc[-1]; current_lower = lower_band.iloc[-1]
        if pd.isna(current_price): logger.warning(f"[{self.name}-{self.symbol}] Current price is NaN."); return
        
        logger.debug(f"[{self.name}-{self.symbol}] Price: {current_price}, Lower: {current_lower}, Upper: {current_upper}")
        
        position_db = db_session.query(Position).filter(Position.subscription_id == subscription_id, Position.symbol == self.symbol, Position.is_open == True).first()

        # Exit Logic
        if position_db:
            exit_reason = None; side_to_close = None; filled_exit_order = None
            if position_db.side == "long":
                sl_price = position_db.entry_price * (1 - self.sl_decimal)
                tp_price = position_db.entry_price * (1 + self.tp_decimal)
                if current_price <= sl_price: exit_reason = "SL"
                elif current_price >= tp_price: exit_reason = "TP"
                if exit_reason: side_to_close = 'sell'
            elif position_db.side == "short":
                sl_price = position_db.entry_price * (1 + self.sl_decimal)
                tp_price = position_db.entry_price * (1 - self.tp_decimal)
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
                        pnl = (filled_exit_order['average'] - position_db.entry_price) * filled_exit_order['filled'] if position_db.side == 'long' else (position_db.entry_price - filled_exit_order['average']) * filled_exit_order['filled']
                        position_db.pnl=pnl; position_db.updated_at = datetime.datetime.utcnow()
                        logger.info(f"[{self.name}-{self.symbol}] {position_db.side} Pos ID {position_db.id} closed. PnL: {pnl:.2f}")
                    else: logger.error(f"[{self.name}-{self.symbol}] Exit order {exit_receipt['id']} failed. Pos ID {position_db.id} might still be open."); db_exit_order.status = filled_exit_order.get('status', 'fill_check_failed') if filled_exit_order else 'fill_check_failed'
                    db_session.commit()
                except Exception as e: logger.error(f"[{self.name}-{self.symbol}] Error closing Pos ID {position_db.id}: {e}", exc_info=True); db_exit_order.status='error'; db_session.commit()
                return # Action taken

        # Entry Logic
        if not position_db:
            allocated_capital = json.loads(user_sub_obj.custom_parameters).get("capital", self.capital) # Use capital from subscription
            position_size_usdt = allocated_capital * self.position_size_percent_capital_decimal
            asset_qty_to_trade = self._format_quantity(position_size_usdt / current_price, exchange_ccxt)
            if asset_qty_to_trade <= 0: logger.warning(f"[{self.name}-{self.symbol}] Asset quantity zero. Skipping."); return

            entry_side = None
            if current_price <= current_lower: entry_side = "long"
            elif current_price >= current_upper: entry_side = "short"

            if entry_side:
                logger.info(f"[{self.name}-{self.symbol}] {entry_side.upper()} entry signal at {current_price}. Size: {asset_qty_to_trade}")
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
                        
                        # SL/TP orders are implicitly managed by checking price against calculated levels in each tick.
                        # No separate SL/TP orders are placed on the exchange for this specific strategy version.
                    else: logger.error(f"[{self.name}-{self.symbol}] Entry order {entry_receipt['id']} failed. Pos not opened."); db_entry_order.status = filled_entry_order.get('status', 'fill_check_failed') if filled_entry_order else 'fill_check_failed'
                    db_session.commit()
                except Exception as e: logger.error(f"[{self.name}-{self.symbol}] Error during {entry_side} entry: {e}", exc_info=True); db_entry_order.status='error'; db_session.commit()
        logger.debug(f"[{self.name}-{self.symbol}] Live signal check complete.")
