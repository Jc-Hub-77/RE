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
from backend import strategy_utils # Import the new strategy_utils
import ccxt # For ccxt.OrderNotFound

logger = logging.getLogger(__name__)

class RSIDivergenceStrategy:
    def __init__(self, symbol: str, timeframe: str, db_session: Session, user_sub_obj: UserStrategySubscription, capital: float = 10000, **custom_parameters):
        self.name = "RSIDivergenceStrategy"
        self.symbol = symbol
        self.timeframe = timeframe
        
        # Store db_session and user_sub_obj
        self.db_session = db_session
        self.user_sub_obj = user_sub_obj
        self.logger = logger # Use module-level logger
        self.capital_param = capital # Store capital from init for risk calc

        defaults = {
            "rsi_period": 14, 
            "lookback_period": 20, 
            "peak_prominence": 0.5, 
            "risk_per_trade_percent": 1.5, 
            "stop_loss_percent": 2.0, 
            "take_profit_percent": 4.0,
            # New parameters
            "order_fill_timeout_seconds": 60,
            "order_fill_check_interval_seconds": 3,
            "divergence_signal_window": 3
        }
        self_params = {**defaults, **custom_parameters}
        for key, value in self_params.items():
            setattr(self, key, value)

        self.risk_per_trade_decimal = float(self.risk_per_trade_percent) / 100.0
        self.stop_loss_decimal = float(self.stop_loss_percent) / 100.0
        self.take_profit_decimal = float(self.take_profit_percent) / 100.0
        
        self.price_precision = 8
        self.quantity_precision = 8
        self._precisions_fetched_ = False

        # State attributes
        self.active_position_db_id = None
        self.entry_price_internal_state = 0.0
        self.side_internal_state = None # 'long' or 'short'
        self.qty_internal_state = 0.0

        init_params_log = {k:v for k,v in self_params.items()}
        init_params_log.update({"symbol": symbol, "timeframe": timeframe, "capital_param": self.capital_param, "subscription_id": self.user_sub_obj.id})
        self.logger.info(f"[{self.name}-{self.symbol}] Initialized with effective params: {init_params_log}")
        
        self._load_persistent_state()

    def _load_persistent_state(self):
        if not self.db_session or not self.user_sub_obj:
            self.logger.warning(f"[{self.name}-{self.symbol}] DB session or user_sub_obj not available for loading state.")
            self.active_position_db_id = None
            self.entry_price_internal_state = 0.0 
            self.side_internal_state = None
            self.qty_internal_state = 0.0
            return

        open_position = strategy_utils.get_open_strategy_position(
            self.db_session, self.user_sub_obj.id, self.symbol
        )
        if open_position:
            self.logger.info(f"[{self.name}-{self.symbol}] Loading persistent state for open position ID: {open_position.id}")
            self.active_position_db_id = open_position.id
            self.entry_price_internal_state = open_position.entry_price 
            self.side_internal_state = open_position.side
            self.qty_internal_state = open_position.amount
        else:
            self.logger.info(f"[{self.name}-{self.symbol}] No active persistent position found for SubID {self.user_sub_obj.id}.")
            self.active_position_db_id = None
            self.entry_price_internal_state = 0.0
            self.side_internal_state = None
            self.qty_internal_state = 0.0

    @classmethod
    def validate_parameters(cls, params: dict) -> dict:
        """Validates strategy-specific parameters."""
        definition = cls.get_parameters_definition()
        validated_params = {}
        _logger = logging.getLogger(__name__) # Use a logger instance

        for key, def_value in definition.items():
            val_type_str = def_value.get("type")
            choices = def_value.get("options") # Or "choices" if that's used
            min_val = def_value.get("min")
            max_val = def_value.get("max")
            default_val = def_value.get("default")

            user_val = params.get(key)

            if user_val is None: # Parameter not provided by user
                if default_val is not None:
                    user_val = default_val # Apply default
                else:
                    raise ValueError(f"Required parameter '{key}' is missing and has no default.")
            
            # Type checking and coercion
            if val_type_str == "int":
                try:
                    user_val = int(user_val)
                except (ValueError, TypeError):
                    raise ValueError(f"Parameter '{key}' must be an integer. Got: {user_val}")
            elif val_type_str == "float":
                try:
                    user_val = float(user_val)
                except (ValueError, TypeError):
                    raise ValueError(f"Parameter '{key}' must be a float. Got: {user_val}")
            elif val_type_str == "string": # Assuming "timeframe" type is string-like
                if not isinstance(user_val, str):
                    raise ValueError(f"Parameter '{key}' must be a string. Got: {user_val}")
            elif val_type_str == "bool": # Not in current definition but good to have
                if not isinstance(user_val, bool):
                    if str(user_val).lower() in ['true', 'yes', '1']: user_val = True
                    elif str(user_val).lower() in ['false', 'no', '0']: user_val = False
                    else: raise ValueError(f"Parameter '{key}' must be a boolean. Got: {user_val}")
            
            # Choice validation
            if choices and user_val not in choices:
                raise ValueError(f"Parameter '{key}' value '{user_val}' is not in valid choices: {choices}")
            
            # Min/Max validation
            if val_type_str in ["int", "float"]:
                if min_val is not None and user_val < min_val:
                    raise ValueError(f"Parameter '{key}' value {user_val} is less than min {min_val}.")
                if max_val is not None and user_val > max_val:
                    raise ValueError(f"Parameter '{key}' value {user_val} is greater than max {max_val}.")
            
            validated_params[key] = user_val

        # Check for unknown parameters
        for key_param in params:
            if key_param not in definition:
                # This strategy's __init__ takes symbol, timeframe, capital as named args, 
                # and **custom_parameters which are defined in get_parameters_definition.
                # So, any key in params not in definition is an unknown custom parameter.
                _logger.warning(f"Unknown parameter '{key_param}' provided for {cls.__name__}. It will be ignored.")
        
        return validated_params

    @classmethod
    def get_parameters_definition(cls):
        # Parameters as defined previously
        return {
            "rsi_period": {"type": "int", "default": 14, "min": 5, "max": 50, "label": "RSI Period"},
            "lookback_period": {"type": "int", "default": 20, "min": 10, "max": 100, "label": "Divergence Lookback Period"},
            "peak_prominence": {"type": "float", "default": 0.5, "min": 0.1, "max": 10, "step":0.1, "label": "Peak Prominence (RSI)"},
            "risk_per_trade_percent": {"type": "float", "default": 1.5, "min": 0.1, "max": 10.0, "step":0.1, "label": "Risk per Trade (%)"},
            "stop_loss_percent": {"type": "float", "default": 2.0, "min": 0.1, "step": 0.1, "label": "Stop Loss % from Entry"},
            "take_profit_percent": {"type": "float", "default": 4.0, "min": 0.1, "step": 0.1, "label": "Take Profit % from Entry"},
            # New parameter definitions
            "order_fill_timeout_seconds": {"type": "int", "default": 60, "min":10, "max":300, "label": "Order Fill Timeout (s)"},
            "order_fill_check_interval_seconds": {"type": "int", "default": 3, "min":1, "max":30, "label": "Order Fill Check Interval (s)"},
            "divergence_signal_window": {"type": "int", "default": 3, "min":1, "max":10, "label": "Divergence Signal Window (bars)"}
        }

    def _get_precisions_live(self, exchange_ccxt):
        # Same as provided
        if not self._precisions_fetched_:
            try:
                exchange_ccxt.load_markets(True)
                market = exchange_ccxt.market(self.symbol)
                self.price_precision = market['precision']['price']
                self.quantity_precision = market['precision']['amount']
                self._precisions_fetched_ = True
                self.logger.info(f"[{self.name}-{self.symbol}] Precisions: Price={self.price_precision}, Qty={self.quantity_precision}")
            except Exception as e: self.logger.error(f"[{self.name}-{self.symbol}] Error fetching live precisions: {e}", exc_info=True)

    def _format_price(self, price, exchange_ccxt): self._get_precisions_live(exchange_ccxt); return float(exchange_ccxt.price_to_precision(self.symbol, price))
    def _format_quantity(self, quantity, exchange_ccxt): self._get_precisions_live(exchange_ccxt); return float(exchange_ccxt.amount_to_precision(self.symbol, quantity))

    def _await_order_fill(self, exchange_ccxt, order_id: str, symbol: str): # Removed defaults
        # Same as provided, but uses instance attributes for timeout/interval
        start_time = time.time()
        self.logger.info(f"[{self.name}-{self.symbol}] Awaiting fill for order {order_id} (timeout: {self.order_fill_timeout_seconds}s)")
        while time.time() - start_time < self.order_fill_timeout_seconds:
            try:
                order = exchange_ccxt.fetch_order(order_id, symbol)
                self.logger.debug(f"[{self.name}-{self.symbol}] Order {order_id} status: {order['status']}")
                if order['status'] == 'closed': self.logger.info(f"[{self.name}-{self.symbol}] Order {order_id} filled. AvgPrice: {order.get('average')}, Qty: {order.get('filled')}"); return order
                if order['status'] in ['canceled', 'rejected', 'expired']: self.logger.warning(f"[{self.name}-{self.symbol}] Order {order_id} is {order['status']}."); return order
            except ccxt.OrderNotFound: self.logger.warning(f"[{self.name}-{self.symbol}] Order {order_id} not found. Retrying.")
            except Exception as e: self.logger.error(f"[{self.name}-{self.symbol}] Error fetching order {order_id}: {e}. Retrying.", exc_info=True)
            time.sleep(self.order_fill_check_interval_seconds)
        self.logger.warning(f"[{self.name}-{self.symbol}] Timeout for order {order_id}. Final check.")
        try: final_status = exchange_ccxt.fetch_order(order_id, symbol); self.logger.info(f"[{self.name}-{self.symbol}] Final status for order {order_id}: {final_status['status']}"); return final_status
        except Exception as e: self.logger.error(f"[{self.name}-{self.symbol}] Final check for order {order_id} failed: {e}", exc_info=True); return None

    def _calculate_rsi(self, df: pd.DataFrame):
        # Same as provided
        if 'close' not in df.columns: self.logger.error(f"[{self.name}-{self.symbol}] 'close' column missing for RSI."); return df
        df['rsi'] = ta.rsi(df['close'], length=self.rsi_period)
        return df

    def _find_divergence(self, price_series: pd.Series, rsi_series: pd.Series):
        # Same as provided
        if not price_series.index.equals(rsi_series.index): self.logger.error(f"[{self.name}-{self.symbol}] Price/RSI index mismatch."); return None, None
        price_std = price_series.std()
        if price_std == 0: price_std = price_series.mean() * 0.01 
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
               (len(price_series) - 1 - p_low2_idx) <= self.divergence_signal_window : 
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
               (len(price_series) - 1 - p_high2_idx) <= self.divergence_signal_window:
                return "bearish", p_high2_idx
        return None, None

    def run_backtest(self, ohlcv_data: pd.DataFrame, initial_capital: float = 10000.0, params: Optional[Dict[str,Any]] = None):
        self.logger.info(f"[{self.name}-{self.symbol}] Starting backtest...")

        # Override instance params if provided in backtest call
        original_instance_params = self.params
        if params:
            self.params = {**self.params, **params}
            self.rsi_period = int(self.params.get("rsi_period", 14))
            self.lookback_period = int(self.params.get("lookback_period", 20))
            self.peak_prominence = float(self.params.get("peak_prominence", 0.5))
            self.risk_per_trade_decimal = float(self.params.get("risk_per_trade_percent", 1.5)) / 100.0
            self.stop_loss_decimal = float(self.params.get("stop_loss_percent", 2.0)) / 100.0
            self.take_profit_decimal = float(self.params.get("take_profit_percent", 4.0)) / 100.0
            self.logger.info(f"Backtest using merged params: {self.params}")

        equity = Decimal(str(initial_capital))
        trades_log = []
        equity_curve = [{'timestamp': ohlcv_data.index[0].isoformat(), 'equity': float(initial_capital)}]

        bt_position_side = None
        bt_entry_price = Decimal("0")
        bt_position_qty = Decimal("0")
        bt_sl_level = Decimal("0")
        bt_tp_level = Decimal("0")

        # Calculate RSI once
        df_with_rsi = self._calculate_rsi(ohlcv_data.copy())
        df_with_rsi.dropna(inplace=True)

        min_data_len_for_divergence = self.lookback_period 
        if len(df_with_rsi) < min_data_len_for_divergence:
            self.logger.warning("Not enough data for the entire backtest period after RSI calculation and initial lookback.")
            return {"pnl": 0, "total_trades": 0, "trades_log": [], "equity_curve": equity_curve, "message": "Not enough data."}

        # Iterate from the point where we have enough data for the first divergence check
        for i in range(min_data_len_for_divergence -1, len(df_with_rsi)):
            current_analysis_window_df = df_with_rsi.iloc[i - self.lookback_period + 1 : i + 1]
            current_bar = current_analysis_window_df.iloc[-1]
            current_timestamp = current_bar.name
            current_price = Decimal(str(current_bar['close']))
            current_low = Decimal(str(current_bar['low']))
            current_high = Decimal(str(current_bar['high']))

            # Exit Logic
            if bt_position_side:
                exit_trade = False; exit_price = Decimal("0"); exit_reason = ""
                if bt_position_side == 'long':
                    if current_low <= bt_sl_level: exit_price = bt_sl_level; exit_reason = "SL"; exit_trade = True
                    elif current_high >= bt_tp_level: exit_price = bt_tp_level; exit_reason = "TP"; exit_trade = True
                elif bt_position_side == 'short':
                    if current_high >= bt_sl_level: exit_price = bt_sl_level; exit_reason = "SL"; exit_trade = True
                    elif current_low <= bt_tp_level: exit_price = bt_tp_level; exit_reason = "TP"; exit_trade = True
                
                if exit_trade:
                    pnl = (exit_price - bt_entry_price) * bt_position_qty if bt_position_side == 'long' else (bt_entry_price - exit_price) * bt_position_qty
                    pnl *= self.leverage # Assuming leverage is part of PnL calculation
                    equity += pnl
                    trades_log.append({
                        'timestamp': current_timestamp.isoformat(), 'type': 'sell' if bt_position_side == 'long' else 'buy',
                        'price': float(exit_price), 'quantity': float(bt_position_qty),
                        'pnl_realized': float(pnl), 'reason': exit_reason, 'equity': float(equity)
                    })
                    self.logger.info(f"[BT] Close {bt_position_side.upper()}: Qty {bt_position_qty} at {exit_price}. PnL: {pnl:.2f}. Equity: {equity:.2f}. Reason: {exit_reason}")
                    bt_position_side = None; bt_position_qty = Decimal("0")
            
            # Entry Logic
            if not bt_position_side:
                divergence_type, signal_bar_idx_in_window = self._find_divergence(current_analysis_window_df['close'], current_analysis_window_df['rsi'])
                # Ensure signal is on the latest bar of the current window
                if divergence_type and signal_bar_idx_in_window == len(current_analysis_window_df) - 1:
                    bt_entry_price = current_price # Entry at close of signal bar
                    entry_side = 'long' if divergence_type == "bullish" else 'short'
                    
                    sl_distance = bt_entry_price * self.stop_loss_decimal
                    if sl_distance == Decimal("0"): 
                        self.logger.warning(f"[BT] SL distance is zero for entry at {bt_entry_price}. Skipping trade."); continue

                    # Simplified position sizing for backtest: risk % of current equity
                    risk_amount_per_trade = equity * self.risk_per_trade_decimal
                    bt_position_qty_unleveraged = risk_amount_per_trade / sl_distance
                    bt_position_qty = bt_position_qty_unleveraged # Leverage is applied to PnL, not directly to qty here for risk calc

                    if bt_position_qty > Decimal("0"):
                        if entry_side == 'long':
                            bt_sl_level = bt_entry_price - sl_distance
                            bt_tp_level = bt_entry_price + (bt_entry_price * self.take_profit_decimal)
                        else: # short
                            bt_sl_level = bt_entry_price + sl_distance
                            bt_tp_level = bt_entry_price - (bt_entry_price * self.take_profit_decimal)
                        
                        bt_position_side = entry_side
                        trades_log.append({
                            'timestamp': current_timestamp.isoformat(), 'type': 'buy' if entry_side == 'long' else 'sell',
                            'price': float(bt_entry_price), 'quantity': float(bt_position_qty),
                            'pnl_realized': 0.0, 'reason': f'{divergence_type.capitalize()} Divergence', 'equity': float(equity)
                        })
                        self.logger.info(f"[BT] Open {entry_side.upper()}: Qty {bt_position_qty:.8f} at {bt_entry_price:.2f}. SL: {bt_sl_level:.2f}, TP: {bt_tp_level:.2f}")
                    else:
                        self.logger.warning(f"[BT] Calculated position qty is zero or negative for {self.symbol}. Skipping trade.")

            equity_curve.append({'timestamp': current_timestamp.isoformat(), 'equity': float(equity)})

        final_pnl = equity - Decimal(str(initial_capital))
        final_pnl_percent = (final_pnl / Decimal(str(initial_capital))) * Decimal("100") if initial_capital > 0 else Decimal("0")
        win_trades = sum(1 for t in trades_log if t['pnl_realized'] > 0)
        loss_trades = sum(1 for t in trades_log if t['pnl_realized'] < 0)

        self.logger.info(f"[{self.name}-{self.symbol}] Backtest finished. Final PnL: {final_pnl:.2f} ({final_pnl_percent:.2f}%)")
        
        if params: self.params = original_instance_params # Restore original params

        return {
            "pnl": float(final_pnl), "pnl_percentage": float(final_pnl_percent),
            "total_trades": len(trades_log), "winning_trades": win_trades, "losing_trades": loss_trades,
            "sharpe_ratio": 0.0, "max_drawdown": 0.0, # Placeholders
            "trades_log": trades_log, "equity_curve": equity_curve,
            "message": "Backtest completed successfully.",
            "initial_capital": initial_capital, "final_equity": float(equity)
        }

    def execute_live_signal(self, market_data_df: pd.DataFrame, exchange_ccxt):
        self.logger.debug(f"[{self.name}-{self.symbol}] Executing live signal for SubID {self.user_sub_obj.id}...")
        if not self.db_session or not self.user_sub_obj:
            self.logger.error(f"[{self.name}-{self.symbol}] DB session or user_sub_obj not available. Cannot execute."); return
        
        if market_data_df.empty or len(market_data_df) < self.lookback_period + self.rsi_period:
            self.logger.warning(f"[{self.name}-{self.symbol}] Insufficient market data."); return
        self._get_precisions_live(exchange_ccxt)

        df = self._calculate_rsi(market_data_df.copy()); df.dropna(inplace=True)
        if len(df) < self.lookback_period: self.logger.warning(f"[{self.name}-{self.symbol}] Not enough data post-RSI calc."); return

        analysis_window_df = df.iloc[-self.lookback_period:]
        current_price = analysis_window_df['close'].iloc[-1]
        
        # Exit Logic
        if self.active_position_db_id:
            exit_reason = None; side_to_close = None
            if self.side_internal_state == "long":
                sl_price = self.entry_price_internal_state * (1 - self.stop_loss_decimal)
                tp_price = self.entry_price_internal_state * (1 + self.take_profit_decimal)
                if current_price <= sl_price: exit_reason = "SL"
                elif current_price >= tp_price: exit_reason = "TP"
                if exit_reason: side_to_close = 'sell'
            elif self.side_internal_state == "short":
                sl_price = self.entry_price_internal_state * (1 + self.stop_loss_decimal)
                tp_price = self.entry_price_internal_state * (1 - self.take_profit_decimal)
                if current_price >= sl_price: exit_reason = "SL"
                elif current_price <= tp_price: exit_reason = "TP"
                if exit_reason: side_to_close = 'buy'

            if exit_reason and side_to_close:
                self.logger.info(f"[{self.name}-{self.symbol}] Closing {self.side_internal_state} PosID {self.active_position_db_id} at {current_price}. Reason: {exit_reason}")
                close_qty = self._format_quantity(self.qty_internal_state, exchange_ccxt)
                
                db_exit_order = strategy_utils.create_strategy_order_in_db(self.db_session, self.user_sub_obj.id, self.symbol, 'market', side_to_close, close_qty, notes=f"Exit due to {exit_reason}")
                if not db_exit_order: self.logger.error(f"[{self.name}-{self.symbol}] Failed to create DB record for exit order."); return

                try:
                    exit_receipt = exchange_ccxt.create_market_order(self.symbol, side_to_close, close_qty, params={'reduceOnly': True})
                    strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=db_exit_order.id, updates={'order_id': exit_receipt['id'], 'status': 'open', 'raw_order_data': json.dumps(exit_receipt)})
                    
                    filled_exit_order = self._await_order_fill(exchange_ccxt, exit_receipt['id'], self.symbol)
                    if filled_exit_order and filled_exit_order['status'] == 'closed':
                        strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=db_exit_order.id, updates={'status':'closed', 'price':filled_exit_order['average'], 'filled':filled_exit_order['filled'], 'cost':filled_exit_order['cost'], 'closed_at': datetime.datetime.utcnow(), 'raw_order_data': json.dumps(filled_exit_order)})
                        
                        closed_pos = strategy_utils.close_strategy_position_in_db(self.db_session, self.active_position_db_id, filled_exit_order['average'], filled_exit_order['filled'], f"Closed by {exit_reason}")
                        if closed_pos: self.logger.info(f"[{self.name}-{self.symbol}] {self.side_internal_state} PosID {self.active_position_db_id} closed. PnL: {closed_pos.pnl:.2f}")
                        
                        self.active_position_db_id = None; self.entry_price_internal_state = 0.0; self.side_internal_state = None; self.qty_internal_state = 0.0
                    else: 
                        self.logger.error(f"[{self.name}-{self.symbol}] Exit order {exit_receipt['id']} failed. PosID {self.active_position_db_id} may still be open."); 
                        strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=db_exit_order.id, updates={'status': filled_exit_order.get('status', 'fill_check_failed') if filled_exit_order else 'fill_check_failed'})
                except Exception as e: 
                    self.logger.error(f"[{self.name}-{self.symbol}] Error closing PosID {self.active_position_db_id}: {e}", exc_info=True); 
                    strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=db_exit_order.id, updates={'status':'error'})
                return 

        # Entry Logic
        if not self.active_position_db_id:
            divergence_type, signal_bar_index = self._find_divergence(analysis_window_df['close'], analysis_window_df['rsi'])
            if divergence_type and signal_bar_index == len(analysis_window_df) - 1: # Ensure signal is on the latest completed bar
                # Use capital from subscription parameters if available, else from init
                sub_params = json.loads(self.user_sub_obj.custom_parameters) if isinstance(self.user_sub_obj.custom_parameters, str) else self.user_sub_obj.custom_parameters
                allocated_capital = sub_params.get("capital", self.capital_param)

                amount_to_risk_usd = allocated_capital * self.risk_per_trade_decimal
                sl_distance_usd = current_price * self.stop_loss_decimal # SL based on current price for sizing
                if sl_distance_usd == 0: self.logger.warning(f"[{self.name}-{self.symbol}] SL distance zero. Cannot size."); return
                
                position_size_asset = self._format_quantity(amount_to_risk_usd / sl_distance_usd, exchange_ccxt)
                if position_size_asset <= 0: self.logger.warning(f"[{self.name}-{self.symbol}] Asset quantity zero. Skipping."); return

                entry_side = "long" if divergence_type == "bullish" else "short"
                self.logger.info(f"[{self.name}-{self.symbol}] {divergence_type.upper()} RSI Divergence. {entry_side.upper()} entry at {current_price}. Size: {position_size_asset}")
                
                db_entry_order = strategy_utils.create_strategy_order_in_db(self.db_session, self.user_sub_obj.id, self.symbol, 'market', entry_side, position_size_asset, notes=f"{divergence_type.capitalize()} Divergence Entry")
                if not db_entry_order: self.logger.error(f"[{self.name}-{self.symbol}] Failed to create DB record for entry order."); return
                
                try:
                    entry_receipt = exchange_ccxt.create_market_order(self.symbol, entry_side, position_size_asset)
                    strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=db_entry_order.id, updates={'order_id': entry_receipt['id'], 'status': 'open', 'raw_order_data': json.dumps(entry_receipt)})
                    
                    filled_entry_order = self._await_order_fill(exchange_ccxt, entry_receipt['id'], self.symbol)
                    if filled_entry_order and filled_entry_order['status'] == 'closed':
                        strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=db_entry_order.id, updates={'status':'closed', 'price':filled_entry_order['average'], 'filled':filled_entry_order['filled'], 'cost':filled_entry_order['cost'], 'closed_at': datetime.datetime.utcnow(), 'raw_order_data': json.dumps(filled_entry_order)})
                        
                        new_pos = strategy_utils.create_strategy_position_in_db(self.db_session, self.user_sub_obj.id, self.symbol, str(exchange_ccxt.id), entry_side, filled_entry_order['filled'], filled_entry_order['average'], status_message=f"{divergence_type.capitalize()} Divergence Entry")
                        if new_pos:
                            self.active_position_db_id = new_pos.id
                            self.entry_price_internal_state = new_pos.entry_price
                            self.side_internal_state = new_pos.side
                            self.qty_internal_state = new_pos.amount
                            self.logger.info(f"[{self.name}-{self.symbol}] {entry_side.upper()} PosID {new_pos.id} created. Entry: {new_pos.entry_price}, Size: {new_pos.amount}")
                        else:
                            self.logger.error(f"[{self.name}-{self.symbol}] Failed to create Position DB record after entry.")
                    else: 
                        self.logger.error(f"[{self.name}-{self.symbol}] Entry order {entry_receipt['id']} failed. Pos not opened."); 
                        strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=db_entry_order.id, updates={'status': filled_entry_order.get('status', 'fill_check_failed') if filled_entry_order else 'fill_check_failed'})
                except Exception as e: 
                    self.logger.error(f"[{self.name}-{self.symbol}] Error during {entry_side} entry: {e}", exc_info=True); 
                    if db_entry_order: strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=db_entry_order.id, updates={'status':'error'})
        
        self.logger.debug(f"[{self.name}-{self.symbol}] Live signal check complete for SubID {self.user_sub_obj.id}.")
