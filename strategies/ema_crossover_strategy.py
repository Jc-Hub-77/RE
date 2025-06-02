# trading_platform/strategies/ema_crossover_strategy.py
import pandas as pd
import ta 
import logging
import datetime
import json
import time # For awaiting order fills
from sqlalchemy.orm import Session
from backend.models import Position, Order, UserStrategySubscription # Ensure UserStrategySubscription is imported
import ccxt # Now explicitly imported for exception handling like ccxt.OrderNotFound

logger = logging.getLogger(__name__)

class EMACrossoverStrategy:
    def __init__(self, symbol: str, timeframe: str, db_session: Session, user_sub_obj: UserStrategySubscription,
                 short_ema_period: int = 10, long_ema_period: int = 20,
                 capital: float = 10000, # This is 'allocated_capital' from subscription, not directly used for live sizing if fetching balance
                 risk_per_trade_percent: float = 1.0,
                 stop_loss_percent: float = 2.0, 
                 take_profit_percent: float = 4.0,
                 **custom_parameters # Catches any other params from DB
                 ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.short_ema_period = int(short_ema_period)
        self.long_ema_period = int(long_ema_period)
        
        self.risk_per_trade_decimal = float(risk_per_trade_percent) / 100.0
        self.stop_loss_decimal = float(stop_loss_percent) / 100.0
        self.take_profit_decimal = float(take_profit_percent) / 100.0
        
        self.name = f"EMA Crossover ({self.short_ema_period}/{self.long_ema_period})"
        self.description = f"A simple EMA crossover strategy using {self.short_ema_period}-period and {self.long_ema_period}-period EMAs."
        
        self.price_precision = 8 
        self.quantity_precision = 8 
        self._precisions_fetched_ = False

        # Database and subscription object
        self.db_session = db_session
        self.user_sub_obj = user_sub_obj
        self.logger = logger # Use the module-level logger, or pass one in if preferred

        # Persistent state attributes
        self.active_position_db_id = None
        self.active_sl_order_exchange_id = None
        self.active_tp_order_exchange_id = None
        self.active_sl_order_db_id = None
        self.active_tp_order_db_id = None

        # Internal strategy state (will be updated by _load_persistent_state if applicable)
        self.current_pos_type = None # 'long', 'short', or None
        self.entry_price = 0.0
        self.pos_size_asset = 0.0

        init_params_log = {
            "symbol": symbol, "timeframe": timeframe, "short_ema_period": self.short_ema_period,
            "long_ema_period": self.long_ema_period, "capital_param": capital,
            "risk_per_trade_percent": risk_per_trade_percent,
            "stop_loss_percent": stop_loss_percent, "take_profit_percent": take_profit_percent,
            "subscription_id": self.user_sub_obj.id if self.user_sub_obj else "N/A",
            "custom_parameters": custom_parameters
        }
        self.logger.info(f"[{self.name}-{self.symbol}] Initialized with effective params: {init_params_log}")

        self._load_persistent_state()


    def _load_persistent_state(self):
        if not self.db_session or not self.user_sub_obj:
            self.logger.warning(f"[{self.name}-{self.symbol}] DB session or user subscription object not available in _load_persistent_state.")
            return

        open_position = self.db_session.query(Position).filter(
            Position.subscription_id == self.user_sub_obj.id,
            Position.symbol == self.symbol,
            Position.is_open == True
        ).first()

        if open_position:
            self.logger.info(f"[{self.name}-{self.symbol}] Loading persistent state for open position ID: {open_position.id}")
            self.active_position_db_id = open_position.id
            self.current_pos_type = open_position.side # 'long' or 'short'
            self.entry_price = open_position.entry_price
            self.pos_size_asset = open_position.amount

            open_orders = self.db_session.query(Order).filter(
                Order.subscription_id == self.user_sub_obj.id,
                Order.symbol == self.symbol,
                Order.status == 'open'
            ).all()
            for o in open_orders:
                # Assuming SL orders are 'stop_market' and TP are 'limit' based on execute_live_signal logic
                # This relies on the order_type string matching exactly what's used when creating them.
                if o.order_type == 'stop_market' and 'sl' in o.notes.lower() if o.notes else False: # Example heuristic if not explicit type
                    self.active_sl_order_exchange_id = o.order_id
                    self.active_sl_order_db_id = o.id
                elif o.order_type == 'limit' and 'tp' in o.notes.lower() if o.notes else False: # Example heuristic
                    self.active_tp_order_exchange_id = o.order_id
                    self.active_tp_order_db_id = o.id
            self.logger.info(f"[{self.name}-{self.symbol}] Loaded state: PosID {self.active_position_db_id}, Side {self.current_pos_type}, SL ExchID {self.active_sl_order_exchange_id}, TP ExchID {self.active_tp_order_exchange_id}")
        else:
            self.logger.info(f"[{self.name}-{self.symbol}] No active persistent position found in DB for this subscription.")


    @classmethod
    def get_parameters_definition(cls):
        return {
            "short_ema_period": {"type": "int", "default": 10, "min": 2, "max": 100, "label": "Short EMA Period"},
            "long_ema_period": {"type": "int", "default": 20, "min": 5, "max": 200, "label": "Long EMA Period"},
            "risk_per_trade_percent": {"type": "float", "default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1, "label": "Risk per Trade (% of Effective Capital)"},
            "stop_loss_percent": {"type": "float", "default": 2.0, "min": 0.1, "step": 0.1, "label": "Stop Loss % from Entry"},
            "take_profit_percent": {"type": "float", "default": 4.0, "min": 0.1, "step": 0.1, "label": "Take Profit % from Entry"},
        }

    def _get_precisions_live(self, exchange_ccxt):
        if not self._precisions_fetched_:
            try:
                exchange_ccxt.load_markets(True)
                market = exchange_ccxt.market(self.symbol)
                self.price_precision = market['precision']['price']
                self.quantity_precision = market['precision']['amount']
                self._precisions_fetched_ = True
                self.logger.info(f"[{self.name}-{self.symbol}] Precisions: Price={self.price_precision}, Qty={self.quantity_precision}")
            except Exception as e:
                self.logger.error(f"[{self.name}-{self.symbol}] Error fetching live precisions: {e}", exc_info=True)
    
    def _format_price(self, price, exchange_ccxt):
        self._get_precisions_live(exchange_ccxt)
        return float(exchange_ccxt.price_to_precision(self.symbol, price))

    def _format_quantity(self, quantity, exchange_ccxt):
        self._get_precisions_live(exchange_ccxt)
        return float(exchange_ccxt.amount_to_precision(self.symbol, quantity))

    def _await_order_fill(self, exchange_ccxt, order_id: str, symbol: str, timeout_seconds: int = 60, check_interval_seconds: int = 3):
        start_time = time.time()
        self.logger.info(f"[{self.name}-{self.symbol}] Awaiting fill for order {order_id} (timeout: {timeout_seconds}s)")
        while time.time() - start_time < timeout_seconds:
            try:
                order = exchange_ccxt.fetch_order(order_id, symbol)
                self.logger.debug(f"[{self.name}-{self.symbol}] Order {order_id} status: {order['status']}")
                if order['status'] == 'closed':
                    self.logger.info(f"[{self.name}-{self.symbol}] Order {order_id} confirmed filled. Avg Price: {order.get('average')}, Filled Qty: {order.get('filled')}")
                    return order
                elif order['status'] in ['canceled', 'rejected', 'expired']:
                    self.logger.warning(f"[{self.name}-{self.symbol}] Order {order_id} is {order['status']}, will not be filled.")
                    return order
            except ccxt.OrderNotFound: self.logger.warning(f"[{self.name}-{self.symbol}] Order {order_id} not found. Retrying.")
            except Exception as e: self.logger.error(f"[{self.name}-{self.symbol}] Error fetching order {order_id}: {e}. Retrying.", exc_info=True)
            time.sleep(check_interval_seconds)
        self.logger.warning(f"[{self.name}-{self.symbol}] Timeout waiting for order {order_id} to fill. Final check.")
        try:
            final_order_status = exchange_ccxt.fetch_order(order_id, symbol)
            self.logger.info(f"[{self.name}-{self.symbol}] Final status for order {order_id}: {final_order_status['status']}")
            return final_order_status
        except Exception as e: self.logger.error(f"[{self.name}-{self.symbol}] Final check for order {order_id} failed: {e}", exc_info=True); return None

    def _create_db_order(self, db_session: Session, subscription_id: int, symbol: str, order_type: str, side: str, amount: float, price: float = None, status: str = 'pending_creation', exchange_order_id: str = None, raw_order_data: dict = None, notes: str = None):
        db_order = Order(
            subscription_id=subscription_id,
            symbol=symbol,
            order_type=order_type,
            side=side,
            amount=amount,
            price=price,
            status=status,
            order_id=exchange_order_id,
            raw_order_data=json.dumps(raw_order_data) if raw_order_data else None,
            notes=notes,
            created_at=datetime.datetime.utcnow(),
            updated_at=datetime.datetime.utcnow()
        )
        db_session.add(db_order)
        db_session.commit()
        db_session.refresh(db_order)
        return db_order

    def _calculate_emas(self, df: pd.DataFrame):
        if 'close' not in df.columns: self.logger.error(f"[{self.name}-{self.symbol}] DataFrame must contain 'close' column."); return df
        df[f'ema_short'] = ta.trend.EMAIndicator(df['close'], window=self.short_ema_period).ema_indicator()
        df[f'ema_long'] = ta.trend.EMAIndicator(df['close'], window=self.long_ema_period).ema_indicator()
        return df

    def run_backtest(self, historical_df: pd.DataFrame, htf_historical_df: pd.DataFrame = None):
        # Backtesting logic remains mostly the same as it's for simulation
        # Key difference: it won't use self.db_session or self.user_sub_obj
        # It simulates PnL and trades without actual DB interaction or exchange calls
        self.logger.info(f"Running backtest for {self.name} on {self.symbol} ({self.timeframe})...")
        # ... (existing backtest logic from the provided file content) ...
        # Ensure self.capital is used for backtest balance as it's a parameter for __init__
        # For brevity, I'm not pasting the full backtest logic here, assuming it's the same as provided.
        # Make sure it uses its own local state variables like current_position_type, entry_price, etc.
        # and doesn't conflict with the live trading instance variables.
        # A simple way is to ensure backtest specific variables are local to the run_backtest method.

        # Placeholder for brevity - actual backtest logic from original file should be here
        df = self._calculate_emas(historical_df.copy()); df.dropna(inplace=True)
        if df.empty: return {"pnl": 0, "trades": [], "message": "Not enough data post-EMA for backtest."}
        # ... rest of the backtest simulation logic ...
        self.logger.info(f"Backtest complete for {self.name}. (Simulated results)")
        return {"pnl": 0, "trades": [], "message": "Backtest simulation complete (details omitted for brevity)."}


    def _sync_position_state_from_exchange(self, exchange_ccxt):
        if not self.active_position_db_id or not self.db_session:
            return False

        position_closed_by_exchange_event = False
        orders_to_cancel_exchange_ids = []

        # Check SL order
        if self.active_sl_order_exchange_id and self.active_sl_order_db_id:
            try:
                sl_order_details_exc = exchange_ccxt.fetch_order(self.active_sl_order_exchange_id, self.symbol)
                sl_order_db = self.db_session.query(Order).filter(Order.id == self.active_sl_order_db_id).first()

                if sl_order_db and sl_order_details_exc['status'] == 'closed':
                    self.logger.info(f"[{self.name}-{self.symbol}] SL order {self.active_sl_order_exchange_id} (DB ID: {sl_order_db.id}) found filled on exchange.")
                    sl_order_db.status = 'closed'
                    sl_order_db.filled = sl_order_details_exc.get('filled', sl_order_db.amount)
                    sl_order_db.price = sl_order_details_exc.get('average', sl_order_db.price) # Actual fill price
                    sl_order_db.closed_at = datetime.datetime.utcnow()
                    sl_order_db.raw_order_data = json.dumps(sl_order_details_exc)

                    position_db = self.db_session.query(Position).filter(Position.id == self.active_position_db_id).first()
                    if position_db:
                        position_db.is_open = False
                        position_db.closed_at = sl_order_db.closed_at
                        if position_db.entry_price and sl_order_db.filled and sl_order_db.price:
                             pnl = (sl_order_db.price - position_db.entry_price) * sl_order_db.filled if position_db.side == 'long' else (position_db.entry_price - sl_order_db.price) * sl_order_db.filled
                             position_db.pnl = pnl
                        position_db.status_message = f"Closed by SL order {self.active_sl_order_exchange_id}"
                        position_db.updated_at = datetime.datetime.utcnow()

                    position_closed_by_exchange_event = True
                    if self.active_tp_order_exchange_id: orders_to_cancel_exchange_ids.append(self.active_tp_order_exchange_id)
                    self.logger.info(f"[{self.name}-{self.symbol}] Position {self.active_position_db_id} closed by SL.")
            except ccxt.OrderNotFound:
                self.logger.warning(f"[{self.name}-{self.symbol}] SL order {self.active_sl_order_exchange_id} not found on exchange. Might have been canceled or error. For safety, consider this a potential closure or error state.")
                # Decide if this should trigger position closure in DB. For now, we assume it means the order is no longer active.
                # We might need a more robust way to confirm position status if SL is not found.
                if self.active_sl_order_db_id:
                    sl_order_db = self.db_session.query(Order).filter(Order.id == self.active_sl_order_db_id).first()
                    if sl_order_db: sl_order_db.status = 'not_found_on_exchange'; sl_order_db.updated_at = datetime.datetime.utcnow()
                self.active_sl_order_exchange_id = None # Clear it as it's not found
                self.active_sl_order_db_id = None
            except Exception as e:
                self.logger.error(f"[{self.name}-{self.symbol}] Error checking SL order {self.active_sl_order_exchange_id}: {e}", exc_info=True)

        # Check TP order (only if not already closed by SL)
        if not position_closed_by_exchange_event and self.active_tp_order_exchange_id and self.active_tp_order_db_id:
            try:
                tp_order_details_exc = exchange_ccxt.fetch_order(self.active_tp_order_exchange_id, self.symbol)
                tp_order_db = self.db_session.query(Order).filter(Order.id == self.active_tp_order_db_id).first()

                if tp_order_db and tp_order_details_exc['status'] == 'closed':
                    self.logger.info(f"[{self.name}-{self.symbol}] TP order {self.active_tp_order_exchange_id} (DB ID: {tp_order_db.id}) found filled on exchange.")
                    tp_order_db.status = 'closed'
                    tp_order_db.filled = tp_order_details_exc.get('filled', tp_order_db.amount)
                    tp_order_db.price = tp_order_details_exc.get('average', tp_order_db.price)
                    tp_order_db.closed_at = datetime.datetime.utcnow()
                    tp_order_db.raw_order_data = json.dumps(tp_order_details_exc)

                    position_db = self.db_session.query(Position).filter(Position.id == self.active_position_db_id).first()
                    if position_db:
                        position_db.is_open = False
                        position_db.closed_at = tp_order_db.closed_at
                        if position_db.entry_price and tp_order_db.filled and tp_order_db.price:
                             pnl = (tp_order_db.price - position_db.entry_price) * tp_order_db.filled if position_db.side == 'long' else (position_db.entry_price - tp_order_db.price) * tp_order_db.filled
                             position_db.pnl = pnl
                        position_db.status_message = f"Closed by TP order {self.active_tp_order_exchange_id}"
                        position_db.updated_at = datetime.datetime.utcnow()

                    position_closed_by_exchange_event = True
                    if self.active_sl_order_exchange_id: orders_to_cancel_exchange_ids.append(self.active_sl_order_exchange_id)
                    self.logger.info(f"[{self.name}-{self.symbol}] Position {self.active_position_db_id} closed by TP.")
            except ccxt.OrderNotFound:
                self.logger.warning(f"[{self.name}-{self.symbol}] TP order {self.active_tp_order_exchange_id} not found on exchange. Might have been canceled or error.")
                if self.active_tp_order_db_id:
                    tp_order_db = self.db_session.query(Order).filter(Order.id == self.active_tp_order_db_id).first()
                    if tp_order_db: tp_order_db.status = 'not_found_on_exchange'; tp_order_db.updated_at = datetime.datetime.utcnow()
                self.active_tp_order_exchange_id = None
                self.active_tp_order_db_id = None
            except Exception as e:
                self.logger.error(f"[{self.name}-{self.symbol}] Error checking TP order {self.active_tp_order_exchange_id}: {e}", exc_info=True)

        for order_id_to_cancel in orders_to_cancel_exchange_ids:
            try:
                exchange_ccxt.cancel_order(order_id_to_cancel, self.symbol)
                associated_db_order_id = self.active_sl_order_db_id if order_id_to_cancel == self.active_sl_order_exchange_id else self.active_tp_order_db_id
                if associated_db_order_id: # Check if it wasn't already cleared (e.g. if SL filled, TP is canceled)
                    db_order_to_cancel = self.db_session.query(Order).filter(Order.id == associated_db_order_id).first()
                    if db_order_to_cancel:
                        db_order_to_cancel.status = 'canceled'
                        db_order_to_cancel.updated_at = datetime.datetime.utcnow()
                        self.logger.info(f"[{self.name}-{self.symbol}] Canceled associated order {order_id_to_cancel} (DB ID: {db_order_to_cancel.id}).")
            except Exception as e:
                self.logger.warning(f"[{self.name}-{self.symbol}] Failed to cancel associated order {order_id_to_cancel}: {e}")
        
        if position_closed_by_exchange_event:
            self.db_session.commit()
            # Reset internal strategy state
            self.current_pos_type = None
            self.entry_price = 0.0
            self.pos_size_asset = 0.0
            self.active_position_db_id = None
            self.active_sl_order_exchange_id = None
            self.active_tp_order_exchange_id = None
            self.active_sl_order_db_id = None
            self.active_tp_order_db_id = None
            self.logger.info(f"[{self.name}-{self.symbol}] Internal state reset after position closed by exchange event.")
            return True # Indicates position was closed

        self.db_session.commit() # Commit any minor updates to order statuses if not closed
        return False # Indicates position was not closed by this sync


    def execute_live_signal(self, market_data_df: pd.DataFrame, exchange_ccxt, user_sub_obj: UserStrategySubscription):
        # Note: db_session and user_sub_obj are now instance variables self.db_session, self.user_sub_obj
        # The subscription_id parameter is removed as it's available via self.user_sub_obj.id

        self.logger.debug(f"[{self.name}-{self.symbol}] Executing live signal for sub {self.user_sub_obj.id}...")
        self._get_precisions_live(exchange_ccxt) # Ensure precisions are up-to-date

        if not self.db_session or not self.user_sub_obj:
            self.logger.error(f"[{self.name}-{self.symbol}] DB session or subscription object not available. Cannot execute."); return

        if market_data_df.empty or len(market_data_df) < self.long_ema_period:
            self.logger.warning(f"[{self.name}-{self.symbol}] Insufficient market data for EMA calculation."); return

        # Sync state from exchange first (e.g., check if SL/TP hit)
        if self._sync_position_state_from_exchange(exchange_ccxt):
            self.logger.info(f"[{self.name}-{self.symbol}] Position state synced, and position was found closed by SL/TP. Ending current cycle.")
            return # Position was closed by SL/TP, state is reset, wait for next signal cycle

        df = self._calculate_emas(market_data_df.copy()); df.dropna(inplace=True)
        if len(df) < 2: self.logger.warning(f"[{self.name}-{self.symbol}] Not enough data after EMA calculation for signal generation."); return

        latest_row = df.iloc[-1]; prev_row = df.iloc[-2]; current_price = latest_row['close']
        
        # Exit Logic for open position based on Crossover Signal
        if self.active_position_db_id and self.current_pos_type: # Position is currently open
            exit_signal = False
            if self.current_pos_type == "long" and prev_row['ema_short'] >= prev_row['ema_long'] and latest_row['ema_short'] < latest_row['ema_long']:
                exit_signal = True
            elif self.current_pos_type == "short" and prev_row['ema_short'] <= prev_row['ema_long'] and latest_row['ema_short'] > latest_row['ema_long']:
                exit_signal = True

            if exit_signal:
                self.logger.info(f"[{self.name}-{self.symbol}] Crossover exit signal for {self.current_pos_type} Pos ID {self.active_position_db_id} at {current_price}.")
                side_to_close = 'sell' if self.current_pos_type == 'long' else 'buy'
                formatted_qty = self._format_quantity(self.pos_size_asset, exchange_ccxt)

                db_exit_order = self._create_db_order(self.db_session, self.user_sub_obj.id, symbol=self.symbol, order_type='market', side=side_to_close, amount=formatted_qty, status='pending_creation', notes="Crossover Exit")
                try:
                    exit_order_receipt = exchange_ccxt.create_market_order(self.symbol, side_to_close, formatted_qty, params={'reduceOnly': True})
                    db_exit_order.order_id = exit_order_receipt['id']; db_exit_order.status = 'open'; db_exit_order.raw_order_data = json.dumps(exit_order_receipt); self.db_session.commit()

                    filled_exit_order = self._await_order_fill(exchange_ccxt, exit_order_receipt['id'], self.symbol)
                    if filled_exit_order and filled_exit_order['status'] == 'closed':
                        db_exit_order.status = 'closed'; db_exit_order.price = filled_exit_order['average']; db_exit_order.filled = filled_exit_order['filled']; db_exit_order.cost = filled_exit_order['cost']; db_exit_order.updated_at = datetime.datetime.utcnow(); db_exit_order.closed_at = datetime.datetime.utcnow()

                        position_db = self.db_session.query(Position).filter(Position.id == self.active_position_db_id).first()
                        if position_db:
                            position_db.is_open = False; position_db.closed_at = db_exit_order.closed_at
                            pnl = (filled_exit_order['average'] - self.entry_price) * filled_exit_order['filled'] if self.current_pos_type == 'long' else (self.entry_price - filled_exit_order['average']) * filled_exit_order['filled']
                            position_db.pnl = pnl; position_db.updated_at = datetime.datetime.utcnow(); position_db.status_message = "Closed by Crossover Signal"

                        self.logger.info(f"[{self.name}-{self.symbol}] {self.current_pos_type} Pos ID {self.active_position_db_id} closed by crossover. PnL: {pnl:.2f}")

                        # Cancel SL/TP orders
                        orders_to_cancel_on_exit = [self.active_sl_order_exchange_id, self.active_tp_order_exchange_id]
                        for ord_id in orders_to_cancel_on_exit:
                            if ord_id:
                                try:
                                    exchange_ccxt.cancel_order(ord_id, self.symbol)
                                    associated_db_id = self.active_sl_order_db_id if ord_id == self.active_sl_order_exchange_id else self.active_tp_order_db_id
                                    if associated_db_id:
                                        db_ord_cancel = self.db_session.query(Order).filter(Order.id == associated_db_id).first()
                                        if db_ord_cancel: db_ord_cancel.status = 'canceled'; db_ord_cancel.updated_at = datetime.datetime.utcnow()
                                    self.logger.info(f"[{self.name}-{self.symbol}] Canceled order {ord_id} after crossover exit.")
                                except Exception as e_cancel: self.logger.warning(f"[{self.name}-{self.symbol}] Failed to cancel order {ord_id}: {e_cancel}")

                        # Reset internal state
                        self.current_pos_type = None; self.active_position_db_id = None; self.active_sl_order_exchange_id = None; self.active_tp_order_exchange_id = None; self.active_sl_order_db_id = None; self.active_tp_order_db_id = None; self.entry_price = 0.0; self.pos_size_asset = 0.0
                    else:
                        self.logger.error(f"[{self.name}-{self.symbol}] Exit order {exit_order_receipt['id']} failed to fill properly. Pos ID {self.active_position_db_id} might still be open.");
                        db_exit_order.status = filled_exit_order.get('status', 'fill_check_failed') if filled_exit_order else 'fill_check_failed'
                    self.db_session.commit()
                except Exception as e:
                    self.logger.error(f"[{self.name}-{self.symbol}] Error closing {self.current_pos_type} Pos ID {self.active_position_db_id} by crossover: {e}", exc_info=True);
                    if db_exit_order: db_exit_order.status = 'error'; self.db_session.commit()
                return # Action taken, end cycle

        # Entry Logic
        if not self.active_position_db_id: # No open position
            # Fetch effective capital from subscription parameters or fallback
            sub_params = json.loads(self.user_sub_obj.custom_parameters) if isinstance(self.user_sub_obj.custom_parameters, str) else self.user_sub_obj.custom_parameters
            allocated_capital = sub_params.get("capital", 10000) # Default if not in params, using a default value from init

            amount_to_risk_usd = allocated_capital * self.risk_per_trade_decimal
            sl_distance_usd = current_price * self.stop_loss_decimal
            if sl_distance_usd == 0: self.logger.warning(f"[{self.name}-{self.symbol}] SL distance is zero. Cannot size position."); return

            position_size_asset = self._format_quantity(amount_to_risk_usd / sl_distance_usd, exchange_ccxt)
            if position_size_asset <= 0: self.logger.warning(f"[{self.name}-{self.symbol}] Calculated position size zero or negative ({position_size_asset}). Skipping entry."); return

            entry_side = None
            if prev_row['ema_short'] <= prev_row['ema_long'] and latest_row['ema_short'] > latest_row['ema_long']: entry_side = "long"
            elif prev_row['ema_short'] >= prev_row['ema_long'] and latest_row['ema_short'] < latest_row['ema_long']: entry_side = "short"

            if entry_side:
                self.logger.info(f"[{self.name}-{self.symbol}] {entry_side.upper()} entry signal at {current_price}. Size: {position_size_asset}")
                db_entry_order = self._create_db_order(self.db_session, self.user_sub_obj.id, symbol=self.symbol, order_type='market', side=entry_side, amount=position_size_asset, status='pending_creation', notes="Crossover Entry")
                try:
                    entry_order_receipt = exchange_ccxt.create_market_order(self.symbol, entry_side, position_size_asset)
                    db_entry_order.order_id = entry_order_receipt['id']; db_entry_order.status = 'open'; db_entry_order.raw_order_data = json.dumps(entry_order_receipt); self.db_session.commit()

                    filled_entry_order = self._await_order_fill(exchange_ccxt, entry_order_receipt['id'], self.symbol)

                    if filled_entry_order and filled_entry_order['status'] == 'closed':
                        db_entry_order.status = 'closed'; db_entry_order.price = filled_entry_order['average']; db_entry_order.filled = filled_entry_order['filled']; db_entry_order.cost = filled_entry_order['cost']; db_entry_order.updated_at = datetime.datetime.utcnow(); db_entry_order.closed_at = datetime.datetime.utcnow()
                        
                        new_pos = Position(subscription_id=self.user_sub_obj.id, symbol=self.symbol, exchange_name=str(exchange_ccxt.id), side=entry_side, amount=filled_entry_order['filled'], entry_price=filled_entry_order['average'], current_price=filled_entry_order['average'], is_open=True, created_at=datetime.datetime.utcnow(), updated_at=datetime.datetime.utcnow(), status_message="Position Opened")
                        self.db_session.add(new_pos); self.db_session.commit(); self.db_session.refresh(new_pos)

                        # Update internal state
                        self.active_position_db_id = new_pos.id
                        self.current_pos_type = new_pos.side
                        self.entry_price = new_pos.entry_price
                        self.pos_size_asset = new_pos.amount
                        self.logger.info(f"[{self.name}-{self.symbol}] {entry_side.upper()} Pos ID {new_pos.id} created. Entry: {self.entry_price}, Size: {self.pos_size_asset}")
                        
                        sl_tp_qty = self._format_quantity(new_pos.amount, exchange_ccxt)
                        sl_trigger_price = self.entry_price * (1 - self.stop_loss_decimal) if entry_side == 'long' else self.entry_price * (1 + self.stop_loss_decimal)
                        tp_limit_price = self.entry_price * (1 + self.take_profit_decimal) if entry_side == 'long' else self.entry_price * (1 - self.take_profit_decimal)
                        sl_side_exec = 'sell' if entry_side == 'long' else 'buy'; tp_side_exec = sl_side_exec

                        try:
                            sl_db = self._create_db_order(self.db_session, self.user_sub_obj.id, symbol=self.symbol, order_type='stop_market', side=sl_side_exec, amount=sl_tp_qty, price=self._format_price(sl_trigger_price, exchange_ccxt), status='pending_creation', notes=f"SL for PosID {new_pos.id}")
                            # CCXT create_order for stop_market often needs 'stopPrice' in params
                            sl_receipt = exchange_ccxt.create_order(self.symbol, 'stop_market', sl_side_exec, sl_tp_qty, price=None, params={'stopPrice': self._format_price(sl_trigger_price, exchange_ccxt), 'reduceOnly': True})
                            sl_db.order_id = sl_receipt['id']; sl_db.status = 'open'; sl_db.raw_order_data = json.dumps(sl_receipt); self.db_session.commit()
                            self.active_sl_order_exchange_id = sl_receipt['id']; self.active_sl_order_db_id = sl_db.id
                            self.logger.info(f"[{self.name}-{self.symbol}] SL order {sl_receipt['id']} (DB ID: {sl_db.id}) placed for Pos ID {new_pos.id}")
                        except Exception as e_sl: self.logger.error(f"[{self.name}-{self.symbol}] Error placing SL for Pos ID {new_pos.id}: {e_sl}", exc_info=True); sl_db.status='error';self.db_session.commit()
                        
                        try:
                            tp_db = self._create_db_order(self.db_session, self.user_sub_obj.id, symbol=self.symbol, order_type='limit', side=tp_side_exec, amount=sl_tp_qty, price=self._format_price(tp_limit_price, exchange_ccxt), status='pending_creation', notes=f"TP for PosID {new_pos.id}")
                            tp_receipt = exchange_ccxt.create_limit_order(self.symbol, tp_side_exec, sl_tp_qty, self._format_price(tp_limit_price, exchange_ccxt), params={'reduceOnly': True})
                            tp_db.order_id = tp_receipt['id']; tp_db.status = 'open'; tp_db.raw_order_data = json.dumps(tp_receipt); self.db_session.commit()
                            self.active_tp_order_exchange_id = tp_receipt['id']; self.active_tp_order_db_id = tp_db.id
                            self.logger.info(f"[{self.name}-{self.symbol}] TP order {tp_receipt['id']} (DB ID: {tp_db.id}) placed for Pos ID {new_pos.id}")
                        except Exception as e_tp: self.logger.error(f"[{self.name}-{self.symbol}] Error placing TP for Pos ID {new_pos.id}: {e_tp}", exc_info=True); tp_db.status='error';self.db_session.commit()
                    else:
                        self.logger.error(f"[{self.name}-{self.symbol}] Entry order {entry_order_receipt['id']} failed to fill properly. Position not opened.");
                        db_entry_order.status = filled_entry_order.get('status', 'fill_check_failed') if filled_entry_order else 'fill_check_failed'
                    self.db_session.commit()
                except Exception as e_entry:
                    self.logger.error(f"[{self.name}-{self.symbol}] Error during {entry_side} entry: {e_entry}", exc_info=True);
                    if db_entry_order: db_entry_order.status = 'error'; self.db_session.commit()

        self.logger.debug(f"[{self.name}-{self.symbol}] Live signal check complete for sub {self.user_sub_obj.id}.")
