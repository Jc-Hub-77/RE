# trading_platform/strategies/ema_crossover_strategy.py
import pandas as pd
import ta 
import logging
import datetime
import json
import time # For awaiting order fills
from sqlalchemy.orm import Session
from backend.models import Position, Order, UserStrategySubscription # Ensure UserStrategySubscription is imported
# import ccxt # Handled by runner via exchange_ccxt

logger = logging.getLogger(__name__)

class EMACrossoverStrategy:
    def __init__(self, symbol: str, timeframe: str, short_ema_period: int = 10, long_ema_period: int = 20, 
                 capital: float = 10000, # This is 'allocated_capital' from subscription, not directly used for live sizing if fetching balance
                 risk_per_trade_percent: float = 1.0, # Note: parameter name in definition is risk_per_trade_percent
                 stop_loss_percent: float = 2.0, 
                 take_profit_percent: float = 4.0,
                 **custom_parameters # Catches any other params from DB
                 ):
        self.symbol = symbol
        self.timeframe = timeframe # Timeframe for market_data_df
        self.short_ema_period = int(short_ema_period)
        self.long_ema_period = int(long_ema_period)
        
        # Convert percentages from UI (e.g., 1 for 1%) to decimals for calculation
        self.risk_per_trade_decimal = float(risk_per_trade_percent) / 100.0
        self.stop_loss_decimal = float(stop_loss_percent) / 100.0
        self.take_profit_decimal = float(take_profit_percent) / 100.0
        
        self.name = f"EMA Crossover ({self.short_ema_period}/{self.long_ema_period})"
        self.description = f"A simple EMA crossover strategy using {self.short_ema_period}-period and {self.long_ema_period}-period EMAs."
        
        self.price_precision = 8 
        self.quantity_precision = 8 
        self._precisions_fetched_ = False

        init_params_log = {
            "symbol": symbol, "timeframe": timeframe, "short_ema_period": self.short_ema_period,
            "long_ema_period": self.long_ema_period, "capital_param": capital, # Log the param as passed
            "risk_per_trade_percent": risk_per_trade_percent, # Log original percent
            "stop_loss_percent": stop_loss_percent, "take_profit_percent": take_profit_percent,
            "custom_parameters": custom_parameters
        }
        logger.info(f"[{self.name}-{self.symbol}] Initialized with effective params: {init_params_log}")

    @classmethod
    def get_parameters_definition(cls):
        return {
            "short_ema_period": {"type": "int", "default": 10, "min": 2, "max": 100, "label": "Short EMA Period"},
            "long_ema_period": {"type": "int", "default": 20, "min": 5, "max": 200, "label": "Long EMA Period"},
            "risk_per_trade_percent": {"type": "float", "default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1, "label": "Risk per Trade (% of Effective Capital)"},
            "stop_loss_percent": {"type": "float", "default": 2.0, "min": 0.1, "step": 0.1, "label": "Stop Loss % from Entry"},
            "take_profit_percent": {"type": "float", "default": 4.0, "min": 0.1, "step": 0.1, "label": "Take Profit % from Entry"},
            # Capital is usually managed by subscription or account balance, not a direct strategy param for live.
            # "capital": {"type": "float", "default": 10000, "min": 100, "label": "Allocated Capital (for sizing)"},
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
            except Exception as e:
                logger.error(f"[{self.name}-{self.symbol}] Error fetching live precisions: {e}", exc_info=True)
    
    def _format_price(self, price, exchange_ccxt):
        self._get_precisions_live(exchange_ccxt)
        return float(exchange_ccxt.price_to_precision(self.symbol, price))

    def _format_quantity(self, quantity, exchange_ccxt):
        self._get_precisions_live(exchange_ccxt)
        return float(exchange_ccxt.amount_to_precision(self.symbol, quantity))

    def _await_order_fill(self, exchange_ccxt, order_id: str, symbol: str, timeout_seconds: int = 60, check_interval_seconds: int = 3):
        start_time = time.time()
        logger.info(f"[{self.name}-{self.symbol}] Awaiting fill for order {order_id} (timeout: {timeout_seconds}s)")
        while time.time() - start_time < timeout_seconds:
            try:
                order = exchange_ccxt.fetch_order(order_id, symbol)
                logger.debug(f"[{self.name}-{self.symbol}] Order {order_id} status: {order['status']}")
                if order['status'] == 'closed':
                    logger.info(f"[{self.name}-{self.symbol}] Order {order_id} confirmed filled. Avg Price: {order.get('average')}, Filled Qty: {order.get('filled')}")
                    return order
                elif order['status'] in ['canceled', 'rejected', 'expired']:
                    logger.warning(f"[{self.name}-{self.symbol}] Order {order_id} is {order['status']}, will not be filled.")
                    return order
            except ccxt.OrderNotFound: logger.warning(f"[{self.name}-{self.symbol}] Order {order_id} not found. Retrying.")
            except Exception as e: logger.error(f"[{self.name}-{self.symbol}] Error fetching order {order_id}: {e}. Retrying.", exc_info=True)
            time.sleep(check_interval_seconds)
        logger.warning(f"[{self.name}-{self.symbol}] Timeout waiting for order {order_id} to fill. Final check.")
        try:
            final_order_status = exchange_ccxt.fetch_order(order_id, symbol)
            logger.info(f"[{self.name}-{self.symbol}] Final status for order {order_id}: {final_order_status['status']}")
            return final_order_status
        except Exception as e: logger.error(f"[{self.name}-{self.symbol}] Final check for order {order_id} failed: {e}", exc_info=True); return None

    def _create_db_order(self, db_session: Session, subscription_id: int, position_id: int = None, **kwargs):
        # Note: 'position_id' is not in the current Order model schema.
        db_order = Order(subscription_id=subscription_id, **kwargs)
        db_session.add(db_order); db_session.commit(); return db_order

    def _calculate_emas(self, df: pd.DataFrame):
        if 'close' not in df.columns: logger.error(f"[{self.name}-{self.symbol}] DataFrame must contain 'close' column."); return df
        df[f'ema_short'] = ta.trend.EMAIndicator(df['close'], window=self.short_ema_period).ema_indicator()
        df[f'ema_long'] = ta.trend.EMAIndicator(df['close'], window=self.long_ema_period).ema_indicator()
        return df

    def run_backtest(self, historical_df: pd.DataFrame, htf_historical_df: pd.DataFrame = None):
        # (Backtesting logic remains largely unchanged from original, assuming it's for offline simulation)
        logger.info(f"Running backtest for {self.name} on {self.symbol} ({self.timeframe})...")
        if historical_df.empty or len(historical_df) < self.long_ema_period:
            logger.warning("Not enough historical data for backtest."); return {"pnl": 0, "trades": [], "message": "Not enough data."}

        df = self._calculate_emas(historical_df.copy()); df.dropna(inplace=True)
        if df.empty: logger.warning("DataFrame empty after EMA calc."); return {"pnl": 0, "trades": [], "message": "Not enough data post-EMA."}

        trades_log = []; current_position_type = None; entry_price = 0.0; position_size = 0.0; balance = self.capital # self.capital is fine for backtest
        entry_time_ts = None

        for i in range(1, len(df)):
            prev_row = df.iloc[i-1]; current_row = df.iloc[i]; current_price = current_row['close']; current_time_ts = current_row.name.timestamp()

            if current_position_type == "long":
                sl_price = entry_price * (1 - self.stop_loss_decimal); tp_price = entry_price * (1 + self.take_profit_decimal)
                exit_signal = prev_row['ema_short'] >= prev_row['ema_long'] and current_row['ema_short'] < current_row['ema_long']
                if current_price <= sl_price or current_price >= tp_price or exit_signal:
                    reason = "SL" if current_price <= sl_price else ("TP" if current_price >= tp_price else "Crossover Exit")
                    exit_p = sl_price if current_price <= sl_price else (tp_price if current_price >= tp_price else current_price)
                    pnl = (exit_p - entry_price) * position_size; balance += pnl
                    trades_log.append({"entry_time": entry_time_ts, "exit_time": current_time_ts, "type": "long", "entry_price": entry_price, "exit_price": exit_p, "size": position_size, "pnl": pnl, "reason": reason})
                    current_position_type = None
            elif current_position_type == "short":
                sl_price = entry_price * (1 + self.stop_loss_decimal); tp_price = entry_price * (1 - self.take_profit_decimal)
                exit_signal = prev_row['ema_short'] <= prev_row['ema_long'] and current_row['ema_short'] > current_row['ema_long']
                if current_price >= sl_price or current_price <= tp_price or exit_signal:
                    reason = "SL" if current_price >= sl_price else ("TP" if current_price <= tp_price else "Crossover Exit")
                    exit_p = sl_price if current_price >= sl_price else (tp_price if current_price <= tp_price else current_price)
                    pnl = (entry_price - exit_p) * position_size; balance += pnl
                    trades_log.append({"entry_time": entry_time_ts, "exit_time": current_time_ts, "type": "short", "entry_price": entry_price, "exit_price": exit_p, "size": position_size, "pnl": pnl, "reason": reason})
                    current_position_type = None
            
            if current_position_type is None:
                amount_to_risk = balance * self.risk_per_trade_decimal
                sl_distance = current_price * self.stop_loss_decimal
                if sl_distance == 0: continue
                position_size = amount_to_risk / sl_distance
                entry_time_ts = current_time_ts

                if prev_row['ema_short'] <= prev_row['ema_long'] and current_row['ema_short'] > current_row['ema_long']: # Bullish crossover
                    entry_price = current_price; current_position_type = "long"
                    trades_log.append({"entry_time": entry_time_ts, "type": "long", "entry_price": entry_price, "size": position_size, "pnl": None, "reason": "Crossover Entry"})
                elif prev_row['ema_short'] >= prev_row['ema_long'] and current_row['ema_short'] < current_row['ema_long']: # Bearish crossover
                    entry_price = current_price; current_position_type = "short"
                    trades_log.append({"entry_time": entry_time_ts, "type": "short", "entry_price": entry_price, "size": position_size, "pnl": None, "reason": "Crossover Entry"})
        
        final_pnl = balance - self.capital
        logger.info(f"Backtest complete for {self.name}. Final PnL: {final_pnl:.2f}, Trades: {len([t for t in trades_log if t.get('exit_time')])}")
        return {"pnl": final_pnl, "trades": [t for t in trades_log if t.get('exit_time') is not None]}


    def execute_live_signal(self, db_session: Session, subscription_id: int, market_data_df: pd.DataFrame, exchange_ccxt, user_sub_obj: UserStrategySubscription):
        logger.debug(f"[{self.name}-{self.symbol}] Executing live signal for sub {subscription_id}...")
        if market_data_df.empty or len(market_data_df) < self.long_ema_period: logger.warning(f"[{self.name}-{self.symbol}] Insufficient market data."); return
        self._get_precisions_live(exchange_ccxt)

        df = self._calculate_emas(market_data_df.copy()); df.dropna(inplace=True)
        if len(df) < 2: logger.warning(f"[{self.name}-{self.symbol}] Not enough data after EMA calculation."); return

        latest_row = df.iloc[-1]; prev_row = df.iloc[-2]; current_price = latest_row['close']
        
        position_db = db_session.query(Position).filter(Position.subscription_id == subscription_id, Position.symbol == self.symbol, Position.is_open == True).first()
        current_pos_type = position_db.side if position_db else None
        entry_price = position_db.entry_price if position_db else 0.0
        pos_size_asset = position_db.amount if position_db else 0.0

        # Exit Logic
        if position_db:
            exit_reason = None; exit_price_target = None
            if current_pos_type == "long":
                sl_price = entry_price * (1 - self.stop_loss_decimal); tp_price = entry_price * (1 + self.take_profit_decimal)
                if current_price <= sl_price: exit_reason = "SL"; exit_price_target = sl_price
                elif current_price >= tp_price: exit_reason = "TP"; exit_price_target = tp_price
                elif prev_row['ema_short'] >= prev_row['ema_long'] and latest_row['ema_short'] < latest_row['ema_long']: exit_reason = "Crossover Exit"; exit_price_target = current_price
            elif current_pos_type == "short":
                sl_price = entry_price * (1 + self.stop_loss_decimal); tp_price = entry_price * (1 - self.take_profit_decimal)
                if current_price >= sl_price: exit_reason = "SL"; exit_price_target = sl_price
                elif current_price <= tp_price: exit_reason = "TP"; exit_price_target = tp_price
                elif prev_row['ema_short'] <= prev_row['ema_long'] and latest_row['ema_short'] > latest_row['ema_long']: exit_reason = "Crossover Exit"; exit_price_target = current_price

            if exit_reason:
                logger.info(f"[{self.name}-{self.symbol}] Closing {current_pos_type} Pos ID {position_db.id} at {current_price}. Reason: {exit_reason}")
                side_to_close = 'sell' if current_pos_type == 'long' else 'buy'
                formatted_qty = self._format_quantity(pos_size_asset, exchange_ccxt)
                db_exit_order = self._create_db_order(db_session, subscription_id, position_id=position_db.id, symbol=self.symbol, order_type='market', side=side_to_close, amount=formatted_qty, status='pending_creation')
                try:
                    exit_order_receipt = exchange_ccxt.create_market_order(self.symbol, side_to_close, formatted_qty, params={'reduceOnly': True})
                    db_exit_order.order_id = exit_order_receipt['id']; db_exit_order.status = 'open'; db_session.commit()
                    filled_exit_order = self._await_order_fill(exchange_ccxt, exit_order_receipt['id'], self.symbol)
                    if filled_exit_order and filled_exit_order['status'] == 'closed':
                        db_exit_order.status = 'closed'; db_exit_order.price = filled_exit_order['average']; db_exit_order.filled = filled_exit_order['filled']; db_exit_order.cost = filled_exit_order['cost']; db_exit_order.updated_at = datetime.datetime.utcnow();
                        position_db.is_open = False; position_db.closed_at = datetime.datetime.utcnow()
                        pnl = (filled_exit_order['average'] - entry_price) * filled_exit_order['filled'] if current_pos_type == 'long' else (entry_price - filled_exit_order['average']) * filled_exit_order['filled']
                        position_db.pnl = pnl; position_db.updated_at = datetime.datetime.utcnow()
                        logger.info(f"[{self.name}-{self.symbol}] {current_pos_type} Pos ID {position_db.id} closed. PnL: {pnl:.2f}")
                    else: logger.error(f"[{self.name}-{self.symbol}] Exit order {exit_order_receipt['id']} failed to fill. Pos ID {position_db.id} might still be open."); db_exit_order.status = filled_exit_order.get('status', 'fill_check_failed') if filled_exit_order else 'fill_check_failed'
                    db_session.commit()
                except Exception as e: logger.error(f"[{self.name}-{self.symbol}] Error closing {current_pos_type} Pos ID {position_db.id}: {e}", exc_info=True); db_exit_order.status = 'error'; db_session.commit()
                return # Action taken, end cycle

        # Entry Logic
        if not position_db: # No open position
            allocated_capital = json.loads(user_sub_obj.custom_parameters).get("capital", self.capital) # Use capital from subscription params
            amount_to_risk_usd = allocated_capital * self.risk_per_trade_decimal
            sl_distance_usd = current_price * self.stop_loss_decimal
            if sl_distance_usd == 0: logger.warning(f"[{self.name}-{self.symbol}] SL distance is zero. Cannot size position."); return
            position_size_asset = self._format_quantity(amount_to_risk_usd / sl_distance_usd, exchange_ccxt)
            if position_size_asset <= 0: logger.warning(f"[{self.name}-{self.symbol}] Calculated position size zero or negative. Skipping."); return

            entry_side = None
            if prev_row['ema_short'] <= prev_row['ema_long'] and latest_row['ema_short'] > latest_row['ema_long']: entry_side = "long"
            elif prev_row['ema_short'] >= prev_row['ema_long'] and latest_row['ema_short'] < latest_row['ema_long']: entry_side = "short"

            if entry_side:
                logger.info(f"[{self.name}-{self.symbol}] {entry_side.upper()} entry signal at {current_price}. Size: {position_size_asset}")
                db_entry_order = self._create_db_order(db_session, subscription_id, symbol=self.symbol, order_type='market', side=entry_side, amount=position_size_asset, status='pending_creation')
                try:
                    entry_order_receipt = exchange_ccxt.create_market_order(self.symbol, entry_side, position_size_asset)
                    db_entry_order.order_id = entry_order_receipt['id']; db_entry_order.status = 'open'; db_session.commit()
                    filled_entry_order = self._await_order_fill(exchange_ccxt, entry_order_receipt['id'], self.symbol)

                    if filled_entry_order and filled_entry_order['status'] == 'closed':
                        db_entry_order.status = 'closed'; db_entry_order.price = filled_entry_order['average']; db_entry_order.filled = filled_entry_order['filled']; db_entry_order.cost = filled_entry_order['cost']; db_entry_order.updated_at = datetime.datetime.utcnow()
                        
                        new_pos = Position(subscription_id=subscription_id, symbol=self.symbol, exchange_name=str(exchange_ccxt.id), side=entry_side, amount=filled_entry_order['filled'], entry_price=filled_entry_order['average'], current_price=filled_entry_order['average'], is_open=True, created_at=datetime.datetime.utcnow(), updated_at=datetime.datetime.utcnow())
                        db_session.add(new_pos); db_session.commit()
                        logger.info(f"[{self.name}-{self.symbol}] {entry_side.upper()} Pos ID {new_pos.id} created. Entry: {new_pos.entry_price}, Size: {new_pos.amount}")
                        
                        # Place SL/TP Orders (OCO not standard in CCXT, place as separate orders)
                        sl_tp_qty = self._format_quantity(new_pos.amount, exchange_ccxt)
                        sl_trigger_price = new_pos.entry_price * (1 - self.stop_loss_decimal) if entry_side == 'long' else new_pos.entry_price * (1 + self.stop_loss_decimal)
                        tp_limit_price = new_pos.entry_price * (1 + self.take_profit_decimal) if entry_side == 'long' else new_pos.entry_price * (1 - self.take_profit_decimal)
                        sl_side = 'sell' if entry_side == 'long' else 'buy'; tp_side = sl_side

                        try:
                            sl_db = self._create_db_order(db_session, subscription_id, position_id=new_pos.id, symbol=self.symbol, order_type='stop_market', side=sl_side, amount=sl_tp_qty, price=self._format_price(sl_trigger_price, exchange_ccxt), status='pending_creation')
                            sl_receipt = exchange_ccxt.create_order(self.symbol, 'stop_market', sl_side, sl_tp_qty, params={'stopPrice': self._format_price(sl_trigger_price, exchange_ccxt), 'reduceOnly': True})
                            sl_db.order_id = sl_receipt['id']; sl_db.status = 'open'; db_session.commit(); logger.info(f"[{self.name}-{self.symbol}] SL order {sl_receipt['id']} placed for Pos ID {new_pos.id}")
                        except Exception as e_sl: logger.error(f"[{self.name}-{self.symbol}] Error placing SL for Pos ID {new_pos.id}: {e_sl}", exc_info=True); sl_db.status='error';db_session.commit()
                        
                        try:
                            tp_db = self._create_db_order(db_session, subscription_id, position_id=new_pos.id, symbol=self.symbol, order_type='limit', side=tp_side, amount=sl_tp_qty, price=self._format_price(tp_limit_price, exchange_ccxt), status='pending_creation')
                            tp_receipt = exchange_ccxt.create_limit_order(self.symbol, tp_side, sl_tp_qty, self._format_price(tp_limit_price, exchange_ccxt), params={'reduceOnly': True})
                            tp_db.order_id = tp_receipt['id']; tp_db.status = 'open'; db_session.commit(); logger.info(f"[{self.name}-{self.symbol}] TP order {tp_receipt['id']} placed for Pos ID {new_pos.id}")
                        except Exception as e_tp: logger.error(f"[{self.name}-{self.symbol}] Error placing TP for Pos ID {new_pos.id}: {e_tp}", exc_info=True); tp_db.status='error';db_session.commit()
                    else: logger.error(f"[{self.name}-{self.symbol}] Entry order {entry_order_receipt['id']} failed to fill. Pos not opened."); db_entry_order.status = filled_entry_order.get('status', 'fill_check_failed') if filled_entry_order else 'fill_check_failed'
                    db_session.commit()
                except Exception as e_entry: logger.error(f"[{self.name}-{self.symbol}] Error during {entry_side} entry: {e_entry}", exc_info=True); db_entry_order.status = 'error'; db_session.commit()
        logger.debug(f"[{self.name}-{self.symbol}] Live signal check complete.")
