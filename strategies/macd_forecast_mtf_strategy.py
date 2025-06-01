# trading_platform/strategies/macd_forecast_mtf_strategy.py
import datetime
import logging
import pandas as pd
import ta 
import numpy as np 
import json
import time
from sqlalchemy.orm import Session
from backend.models import Position, Order, UserStrategySubscription # Added UserStrategySubscription

logger = logging.getLogger(__name__)

def percentile_linear_interpolation(data_array, percentile):
    if not data_array: return 0.0 
    # Ensure data_array contains numbers, convert if necessary, handle potential errors
    numeric_array = []
    for x in data_array:
        try:
            numeric_array.append(float(x))
        except (ValueError, TypeError):
            # logger.warning(f"Could not convert {x} to float for percentile calculation. Skipping.")
            pass # Skip non-numeric values
    if not numeric_array: return 0.0
    
    # Older numpy versions might not have 'method' parameter.
    # 'linear' is equivalent to older 'fraction' parameter behavior if needed.
    try:
        return np.percentile(np.array(numeric_array), percentile, method='linear')
    except TypeError: # Fallback for older numpy
        return np.percentile(np.array(numeric_array), percentile)


class MACDForecastMTFStrategy:
    def __init__(self, symbol: str, timeframe: str, capital: float = 1000,
                 htf: str = "240", # Higher timeframe for trend
                 fast_len: int = 12, slow_len: int = 26, signal_len: int = 9,
                 trend_determination: str = 'MACD - Signal', 
                 lot_size: float = 1.0, 
                 use_stop_loss: bool = True, stop_loss_percent: float = 2.0,
                 use_take_profit: bool = True, take_profit_percent: float = 4.0,
                 max_memory: int = 50, forecast_length: int = 100, 
                 forecast_top_percentile: int = 80, forecast_mid_percentile: int = 50,
                 forecast_bottom_percentile: int = 20, **custom_parameters):
        self.name = "MTF MACD Strategy with Forecasting"
        self.symbol = symbol
        self.timeframe = timeframe 
        
        self.htf = htf # Ensure this is a valid timeframe string CCXT understands, e.g., '4h' if '240' means minutes
        self.fast_len = int(fast_len)
        self.slow_len = int(slow_len)
        self.signal_len = int(signal_len)
        self.trend_determination = trend_determination
        
        self.lot_size = float(lot_size) # Position size in base currency (e.g., BTC amount)
        self.use_stop_loss = bool(use_stop_loss)
        self.stop_loss_decimal = float(stop_loss_percent) / 100.0
        self.use_take_profit = bool(use_take_profit)
        self.take_profit_decimal = float(take_profit_percent) / 100.0

        self.max_memory = int(max_memory)
        self.forecast_length = int(forecast_length)
        self.forecast_top_percentile = int(forecast_top_percentile)
        self.forecast_mid_percentile = int(forecast_mid_percentile)
        self.forecast_bottom_percentile = int(forecast_bottom_percentile)
        
        self.price_precision = 8 
        self.quantity_precision = 8
        self._precisions_fetched_ = False

        # In-memory state for forecast vectors (not persisted across restarts for now)
        # TODO: For robust forecasting state, persist forecast_memory_up/down and counters in DB (e.g., UserStrategySubscription.custom_data)
        self.forecast_state = {
            "up_idx_counter": 0, "dn_idx_counter": 0,
            "uptrend_init_price": None, "downtrend_init_price": None,
            "forecast_memory_up": {}, "forecast_memory_down": {}
        }
        
        init_params_log = {
            "symbol": symbol, "timeframe": timeframe, "htf": htf, "fast_len": fast_len, "slow_len": slow_len,
            "signal_len": signal_len, "trend_determination": trend_determination, "lot_size": lot_size,
            "use_stop_loss": use_stop_loss, "stop_loss_percent": stop_loss_percent,
            "use_take_profit": use_take_profit, "take_profit_percent": take_profit_percent,
            "max_memory": max_memory, "forecast_length": forecast_length, 
            "forecast_percentiles": (forecast_bottom_percentile, forecast_mid_percentile, forecast_top_percentile),
            "custom_parameters": custom_parameters
        }
        logger.info(f"[{self.name}-{self.symbol}] Initialized with params: {init_params_log}")

    @classmethod
    def get_parameters_definition(cls):
        return {
            "htf": {"type": "timeframe", "default": "4h", "label": "Higher Timeframe (Trend)"}, # Changed default to '4h'
            "fast_len": {"type": "int", "default": 12, "min": 2, "label": "MACD Fast Length"},
            "slow_len": {"type": "int", "default": 26, "min": 2, "label": "MACD Slow Length"},
            "signal_len": {"type": "int", "default": 9, "min": 2, "label": "MACD Signal Length"},
            "trend_determination": {"type": "select", "default": "MACD - Signal", "options": ["MACD", "MACD - Signal"], "label": "Trend Determination (HTF)"},
            "lot_size": {"type": "float", "default": 0.01, "min": 0.000001, "label": "Position Size (Base Asset Qty)"}, # Clarified label
            "use_stop_loss": {"type": "bool", "default": True, "label": "Use Stop Loss"},
            "stop_loss_percent": {"type": "float", "default": 2.0, "min": 0.1, "step": 0.1, "label": "Stop Loss %"},
            "use_take_profit": {"type": "bool", "default": True, "label": "Use Take Profit"},
            "take_profit_percent": {"type": "float", "default": 4.0, "min": 0.1, "step": 0.1, "label": "Take Profit %"},
            "max_memory": {"type": "int", "default": 50, "min": 2, "label": "Forecast Max Memory (bars)"},
            "forecast_length": {"type": "int", "default": 100, "min": 1, "label": "Forecast Projection Length (bars)"},
            "forecast_top_percentile": {"type": "int", "default": 80, "min": 51, "max": 99, "label": "Forecast Top Percentile"},
            "forecast_mid_percentile": {"type": "int", "default": 50, "min": 1, "max": 99, "label": "Forecast Mid Percentile"},
            "forecast_bottom_percentile": {"type": "int", "default": 20, "min": 1, "max": 49, "label": "Forecast Bottom Percentile"}
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
                if order['status'] == 'closed': logger.info(f"[{self.name}-{self.symbol}] Order {order_id} filled. AvgPrice: {order.get('average')}, FilledQty: {order.get('filled')}"); return order
                if order['status'] in ['canceled', 'rejected', 'expired']: logger.warning(f"[{self.name}-{self.symbol}] Order {order_id} is {order['status']}."); return order
            except ccxt.OrderNotFound: logger.warning(f"[{self.name}-{self.symbol}] Order {order_id} not found. Retrying.")
            except Exception as e: logger.error(f"[{self.name}-{self.symbol}] Error fetching order {order_id}: {e}. Retrying.", exc_info=True)
            time.sleep(check_interval_seconds)
        logger.warning(f"[{self.name}-{self.symbol}] Timeout for order {order_id}. Final check.")
        try: final_status = exchange_ccxt.fetch_order(order_id, symbol); logger.info(f"[{self.name}-{self.symbol}] Final status for order {order_id}: {final_status['status']}"); return final_status
        except Exception as e: logger.error(f"[{self.name}-{self.symbol}] Final check for order {order_id} failed: {e}", exc_info=True); return None

    def _create_db_order(self, db_session: Session, subscription_id: int, position_id: int = None, **kwargs):
        db_order = Order(subscription_id=subscription_id, **kwargs); db_session.add(db_order); db_session.commit(); return db_order

    def _calculate_macd(self, series_df, fast, slow, signal):
        if 'close' not in series_df.columns: logger.error(f"[{self.name}-{self.symbol}] 'close' column missing for MACD."); return pd.DataFrame()
        macd_ta = ta.trend.MACD(series_df['close'], window_fast=fast, window_slow=slow, window_sign=signal)
        return pd.DataFrame({ 'macd': macd_ta.macd(), 'histogram': macd_ta.macd_diff(), 'signal': macd_ta.macd_signal() })

    # --- Forecasting Methods ---
    def _populate_forecast_memory(self, is_uptrend_primary, current_close_price):
        fs = self.forecast_state # Use alias for brevity
        if is_uptrend_primary:
            if fs['uptrend_init_price'] is None: logger.warning(f"[{self.name}-{self.symbol}] Uptrend init price not set for forecast memory."); return
            price_delta = current_close_price - fs['uptrend_init_price']
            if fs['up_idx_counter'] not in fs['forecast_memory_up']: fs['forecast_memory_up'][fs['up_idx_counter']] = []
            fs['forecast_memory_up'][fs['up_idx_counter']].insert(0, price_delta)
            if len(fs['forecast_memory_up'][fs['up_idx_counter']]) > self.max_memory: fs['forecast_memory_up'][fs['up_idx_counter']].pop()
        else: # Downtrend
            if fs['downtrend_init_price'] is None: logger.warning(f"[{self.name}-{self.symbol}] Downtrend init price not set for forecast memory."); return
            price_delta = current_close_price - fs['downtrend_init_price']
            if fs['dn_idx_counter'] not in fs['forecast_memory_down']: fs['forecast_memory_down'][fs['dn_idx_counter']] = []
            fs['forecast_memory_down'][fs['dn_idx_counter']].insert(0, price_delta)
            if len(fs['forecast_memory_down'][fs['dn_idx_counter']]) > self.max_memory: fs['forecast_memory_down'][fs['dn_idx_counter']].pop()
    
    def _generate_forecast_bands(self, is_uptrend_primary, current_close_price, current_bar_idx_for_time):
        # This is a placeholder for the complex PineScript forecasting logic.
        # A full translation requires managing historical trend "vectors" and their lengths.
        logger.debug(f"[{self.name}-{self.symbol}] Conceptual forecast generation. IsUptrend: {is_uptrend_primary}")
        forecast_bands = {"upper": [], "mid": [], "lower": []}
        init_price = self.forecast_state['uptrend_init_price'] if is_uptrend_primary else self.forecast_state['downtrend_init_price']
        if init_price is None: init_price = current_close_price # Fallback

        for x in range(self.forecast_length):
            offset = (x / self.forecast_length) * 0.01 * init_price 
            forecast_bands["upper"].append({"time": current_bar_idx_for_time + x, "value": init_price + offset * 2}) # Conceptual
            forecast_bands["mid"].append({"time": current_bar_idx_for_time + x, "value": init_price + offset})     # Conceptual
            forecast_bands["lower"].append({"time": current_bar_idx_for_time + x, "value": init_price - offset * 0.5}) # Conceptual
        return forecast_bands

    def run_backtest(self, historical_df: pd.DataFrame, htf_historical_df: pd.DataFrame = None):
        # (Backtesting logic remains simplified as per original, focusing on live execution for this task)
        logger.info(f"Running backtest for {self.name} on {self.symbol}...")
        return {"pnl": 0, "trades": [], "message": "Backtesting for this strategy is conceptual and not fully implemented for detailed PnL."}

    def execute_live_signal(self, db_session: Session, subscription_id: int, market_data_df: pd.DataFrame, exchange_ccxt, user_sub_obj: UserStrategySubscription):
        logger.debug(f"[{self.name}-{self.symbol}] Executing live signal for sub {subscription_id}...")
        if market_data_df.empty: logger.warning(f"[{self.name}-{self.symbol}] Market data is empty."); return
        self._get_precisions_live(exchange_ccxt)
        
        # Load persistent forecast state if available (e.g., from user_sub_obj.custom_data)
        # For now, forecast_state is in-memory and resets on strategy re-init.
        # sub_custom_data = json.loads(user_sub_obj.custom_parameters) if isinstance(user_sub_obj.custom_parameters, str) else user_sub_obj.custom_parameters
        # self.forecast_state = sub_custom_data.get("forecast_state", self.forecast_state) # Example load

        # Calculate Indicators
        primary_macd_df = self._calculate_macd(market_data_df, self.fast_len, self.slow_len, self.signal_len)
        df = market_data_df.join(primary_macd_df)
        
        try: # Fetch and join HTF data
            ohlcv_htf = exchange_ccxt.fetch_ohlcv(self.symbol, self.htf, limit=self.slow_len + self.signal_len + 50)
            if not ohlcv_htf: logger.warning(f"[{self.name}-{self.symbol}] Could not fetch HTF data."); return
            df_htf = pd.DataFrame(ohlcv_htf, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_htf['timestamp'] = pd.to_datetime(df_htf['timestamp'], unit='ms'); df_htf.set_index('timestamp', inplace=True)
            htf_macd_df = self._calculate_macd(df_htf, self.fast_len, self.slow_len, self.signal_len)
            df = pd.merge_asof(df.sort_index(), htf_macd_df.sort_index().add_prefix('htf_'), left_index=True, right_index=True, direction='forward')
        except Exception as e: logger.error(f"[{self.name}-{self.symbol}] Error fetching/processing HTF data: {e}", exc_info=True); return
        
        df.fillna(method='ffill', inplace=True); df.dropna(inplace=True)
        if len(df) < 2: logger.warning(f"[{self.name}-{self.symbol}] Not enough processed data for signal."); return

        latest = df.iloc[-1]; prev = df.iloc[-2]; current_price = latest['close']

        # Trend and Trigger Logic
        htf_trend_val = latest['htf_macd'] - latest['htf_signal'] if self.trend_determination == 'MACD - Signal' else latest['htf_macd']
        htf_uptrend = htf_trend_val > 0; htf_downtrend = htf_trend_val < 0
        
        primary_trend_val_now = latest['macd'] - latest['signal'] if self.trend_determination == 'MACD - Signal' else latest['macd']
        primary_trend_val_prev = prev['macd'] - prev['signal'] if self.trend_determination == 'MACD - Signal' else prev['macd']
        
        primary_uptrend_now = primary_trend_val_now > 0; primary_downtrend_now = primary_trend_val_now < 0
        primary_uptrend_prev = primary_trend_val_prev > 0; primary_downtrend_prev = primary_trend_val_prev < 0

        crossed_above = primary_trend_val_prev <= 0 and primary_trend_val_now > 0
        crossed_below = primary_trend_val_prev >= 0 and primary_trend_val_now < 0
        trigger = crossed_above or crossed_below

        # Forecast state updates (conceptual, as true Pine-like state is hard)
        if primary_uptrend_now and not primary_uptrend_prev: self.forecast_state['uptrend_init_price'] = current_price
        if primary_downtrend_now and not primary_downtrend_prev: self.forecast_state['downtrend_init_price'] = current_price
        if primary_uptrend_now: self._populate_forecast_memory(True, current_price)
        if primary_downtrend_now: self._populate_forecast_memory(False, current_price)
        self.forecast_state['up_idx_counter'] = 0 if not primary_uptrend_now else self.forecast_state['up_idx_counter'] + 1
        self.forecast_state['dn_idx_counter'] = 0 if not primary_downtrend_now else self.forecast_state['dn_idx_counter'] + 1
        # if trigger: forecast_bands = self._generate_forecast_bands(primary_uptrend_now, current_price, len(df)-1) # Example call

        position_db = db_session.query(Position).filter(Position.subscription_id == subscription_id, Position.symbol == self.symbol, Position.is_open == True).first()

        # Exit Logic
        if position_db:
            exit_reason = None; side_to_close = None
            if position_db.side == "long":
                sl_price = position_db.entry_price * (1 - self.stop_loss_decimal)
                tp_price = position_db.entry_price * (1 + self.take_profit_decimal)
                if self.use_stop_loss and current_price <= sl_price: exit_reason = "SL"
                elif self.use_take_profit and current_price >= tp_price: exit_reason = "TP"
                elif trigger and primary_downtrend_now: exit_reason = "Opposing Signal (Short)" # Exit long on short signal
                if exit_reason: side_to_close = 'sell'
            elif position_db.side == "short":
                sl_price = position_db.entry_price * (1 + self.stop_loss_decimal)
                tp_price = position_db.entry_price * (1 - self.take_profit_decimal)
                if self.use_stop_loss and current_price >= sl_price: exit_reason = "SL"
                elif self.use_take_profit and current_price <= tp_price: exit_reason = "TP"
                elif trigger and primary_uptrend_now: exit_reason = "Opposing Signal (Long)" # Exit short on long signal
                if exit_reason: side_to_close = 'buy'

            if exit_reason and side_to_close:
                logger.info(f"[{self.name}-{self.symbol}] Closing {position_db.side} Pos ID {position_db.id} at {current_price}. Reason: {exit_reason}")
                close_qty = self._format_quantity(position_db.amount, exchange_ccxt)
                db_exit_order = self._create_db_order(db_session, subscription_id, position_id=position_db.id, symbol=self.symbol, order_type='market', side=side_to_close, amount=close_qty, status='pending_creation')
                try:
                    exit_receipt = exchange_ccxt.create_market_order(self.symbol, side_to_close, close_qty, params={'reduceOnly': True})
                    db_exit_order.order_id = exit_receipt['id']; db_exit_order.status = 'open'; db_session.commit()
                    filled_exit = self._await_order_fill(exchange_ccxt, exit_receipt['id'], self.symbol)
                    if filled_exit and filled_exit['status'] == 'closed':
                        db_exit_order.status='closed'; db_exit_order.price=filled_exit['average']; db_exit_order.filled=filled_exit['filled']; db_exit_order.cost=filled_exit['cost']; db_exit_order.updated_at=datetime.datetime.utcnow()
                        position_db.is_open=False; position_db.closed_at=datetime.datetime.utcnow()
                        pnl = (filled_exit['average'] - position_db.entry_price) * filled_exit['filled'] if position_db.side == 'long' else (position_db.entry_price - filled_exit['average']) * filled_exit['filled']
                        position_db.pnl=pnl; position_db.updated_at = datetime.datetime.utcnow()
                        logger.info(f"[{self.name}-{self.symbol}] {position_db.side} Pos ID {position_db.id} closed. PnL: {pnl:.2f}")
                    else: logger.error(f"[{self.name}-{self.symbol}] Exit order {exit_receipt['id']} failed. Pos ID {position_db.id} might still be open."); db_exit_order.status = filled_exit.get('status', 'fill_check_failed') if filled_exit else 'fill_check_failed'
                    db_session.commit()
                except Exception as e: logger.error(f"[{self.name}-{self.symbol}] Error closing Pos ID {position_db.id}: {e}", exc_info=True); db_exit_order.status='error'; db_session.commit()
                return # Action taken

        # Entry Logic
        if not position_db:
            entry_side = None
            if crossed_above and primary_uptrend_now and htf_uptrend: entry_side = "long"
            elif crossed_below and primary_downtrend_now and htf_downtrend: entry_side = "short"

            if entry_side:
                entry_qty = self._format_quantity(self.lot_size, exchange_ccxt) # Using fixed lot_size
                if entry_qty <= 0: logger.warning(f"[{self.name}-{self.symbol}] Lot size zero or negative. Skipping."); return
                logger.info(f"[{self.name}-{self.symbol}] {entry_side.upper()} entry signal at {current_price}. Size: {entry_qty}")
                db_entry_order = self._create_db_order(db_session, subscription_id, symbol=self.symbol, order_type='market', side=entry_side, amount=entry_qty, status='pending_creation')
                try:
                    entry_receipt = exchange_ccxt.create_market_order(self.symbol, entry_side, entry_qty)
                    db_entry_order.order_id = entry_receipt['id']; db_entry_order.status = 'open'; db_session.commit()
                    filled_entry = self._await_order_fill(exchange_ccxt, entry_receipt['id'], self.symbol)
                    if filled_entry and filled_entry['status'] == 'closed':
                        db_entry_order.status='closed'; db_entry_order.price=filled_entry['average']; db_entry_order.filled=filled_entry['filled']; db_entry_order.cost=filled_entry['cost']; db_entry_order.updated_at=datetime.datetime.utcnow()
                        
                        new_pos = Position(subscription_id=subscription_id, symbol=self.symbol, exchange_name=str(exchange_ccxt.id), side=entry_side, amount=filled_entry['filled'], entry_price=filled_entry['average'], current_price=filled_entry['average'], is_open=True, created_at=datetime.datetime.utcnow(), updated_at=datetime.datetime.utcnow())
                        db_session.add(new_pos); db_session.commit()
                        logger.info(f"[{self.name}-{self.symbol}] {entry_side.upper()} Pos ID {new_pos.id} created. Entry: {new_pos.entry_price}, Size: {new_pos.amount}")
                        
                        # Place SL/TP (OCO not standard, place as separate)
                        sl_tp_qty = self._format_quantity(new_pos.amount, exchange_ccxt)
                        sl_trigger = new_pos.entry_price * (1 - self.stop_loss_decimal) if entry_side == 'long' else new_pos.entry_price * (1 + self.stop_loss_decimal)
                        tp_limit = new_pos.entry_price * (1 + self.take_profit_decimal) if entry_side == 'long' else new_pos.entry_price * (1 - self.take_profit_decimal)
                        sl_tp_side = 'sell' if entry_side == 'long' else 'buy'

                        if self.use_stop_loss:
                            db_sl = self._create_db_order(db_session, subscription_id, position_id=new_pos.id, symbol=self.symbol, order_type='stop_market', side=sl_tp_side, amount=sl_tp_qty, price=self._format_price(sl_trigger, exchange_ccxt), status='pending_creation')
                            try: sl_receipt = exchange_ccxt.create_order(self.symbol, 'stop_market', sl_tp_side, sl_tp_qty, params={'stopPrice': self._format_price(sl_trigger, exchange_ccxt), 'reduceOnly':True}); db_sl.order_id=sl_receipt['id']; db_sl.status='open'; logger.info(f"[{self.name}-{self.symbol}] SL order {sl_receipt['id']} for Pos {new_pos.id}")
                            except Exception as e_sl: logger.error(f"[{self.name}-{self.symbol}] Error SL for Pos {new_pos.id}: {e_sl}", exc_info=True); db_sl.status='error'
                            db_session.commit()
                        if self.use_take_profit:
                            db_tp = self._create_db_order(db_session, subscription_id, position_id=new_pos.id, symbol=self.symbol, order_type='limit', side=sl_tp_side, amount=sl_tp_qty, price=self._format_price(tp_limit, exchange_ccxt), status='pending_creation')
                            try: tp_receipt = exchange_ccxt.create_limit_order(self.symbol, sl_tp_side, sl_tp_qty, self._format_price(tp_limit, exchange_ccxt), params={'reduceOnly':True}); db_tp.order_id=tp_receipt['id']; db_tp.status='open'; logger.info(f"[{self.name}-{self.symbol}] TP order {tp_receipt['id']} for Pos {new_pos.id}")
                            except Exception as e_tp: logger.error(f"[{self.name}-{self.symbol}] Error TP for Pos {new_pos.id}: {e_tp}", exc_info=True); db_tp.status='error'
                            db_session.commit()
                    else: logger.error(f"[{self.name}-{self.symbol}] Entry order {entry_receipt['id']} failed. Pos not opened."); db_entry_order.status = filled_entry.get('status', 'fill_check_failed') if filled_entry else 'fill_check_failed'
                    db_session.commit()
                except Exception as e: logger.error(f"[{self.name}-{self.symbol}] Error during {entry_side} entry: {e}", exc_info=True); db_entry_order.status='error'; db_session.commit()
        logger.debug(f"[{self.name}-{self.symbol}] Live signal check complete.")
