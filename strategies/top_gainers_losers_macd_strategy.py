import logging
import numpy as np
import talib
import ccxt # For type hints if needed
import json
from datetime import datetime
import time # For helpers if used
from decimal import Decimal, ROUND_DOWN, ROUND_UP, ROUND_NEAREST, getcontext # Added ROUND_NEAREST
from typing import Optional, Dict, Any, List

import pandas as pd # Standard alias
from sqlalchemy.orm import Session # Optional for backtest
from backend.models import Position, Order, UserStrategySubscription # Optional for backtest
from backend import strategy_utils # Optional for backtest

getcontext().prec = 18

class TopGainersLosersMACD:
    
    @staticmethod
    def get_parameters_definition():
        return {
            "trading_pair": {"type": "str", "label": "Trading Pair (for backtest)", "default": "BTC/USDT"}, # Added for backtest context
            "kline_interval": {"type": "str", "label": "Kline Interval for MACD", "default": "15m"},
            "leverage": {"type": "int", "label": "Leverage", "default": 3, "min": 1, "max": 25},
            "stop_loss_pct": {"type": "float", "label": "Stop Loss % (e.g., 0.05 for 5%)", "default": 0.05}, # Renamed from stop_loss_percent
            "risk_per_trade_pct": {"type": "float", "label": "Risk Per Trade % (of balance)", "default": 0.05}, # Renamed
            "order_quantity_usd": {"type": "float", "label": "Order Quantity USD (fixed if risk_per_trade_pct=0)", "default": 0}, # Added for fixed qty option

            "macd_fast_period": {"type": "int", "label": "MACD Fast Period", "default": 34},
            "macd_slow_period": {"type": "int", "label": "MACD Slow Period", "default": 144},
            "macd_signal_period": {"type": "int", "label": "MACD Signal Period", "default": 9},
            
            # These are mostly for live multi-symbol scanning, will be ignored or adapted in single-symbol backtest
            "top_n_symbols_to_scan": {"type": "int", "label": "Top N Symbols to Scan (Live)", "default": 10},
            "max_concurrent_trades": {"type": "int", "label": "Max Concurrent Trades (Live)", "default": 2},
            "min_volume_usdt_24h": {"type": "float", "label": "Min. 24h QuoteVolume (Live)", "default": 1000000.0},
            "min_price_change_percent_filter": {"type": "float", "label": "Min. Price Change % (Live)", "default": 3.0},
            "min_candles_for_macd": {"type": "int", "label": "Min. Candles for MACD", "default": 144}, # Used for data check
            # New parameters
            "order_fill_max_retries": {"type": "int", "default": 10, "min": 1, "max": 30, "label": "Order Fill Max Retries"},
            "order_fill_delay_seconds": {"type": "int", "default": 2, "min": 1, "max": 10, "label": "Order Fill Delay (s)"},
            "data_fetch_buffer": {"type": "int", "default": 50, "min": 0, "max": 200, "label": "Data Fetch Buffer (candles)"},
        }

    def __init__(self, db_session: Optional[Session], user_sub_obj: Optional[UserStrategySubscription], 
                 strategy_params: dict, exchange_ccxt: Optional[Any], logger_obj=None):
        self.db_session = db_session
        self.user_sub_obj = user_sub_obj
        # self.params = strategy_params # Raw params stored if needed
        self.exchange_ccxt = exchange_ccxt
        self.logger = logger_obj if logger_obj else logging.getLogger(__name__)
        self.name = "TopGainersLosersMACD"

        self._initialize_parameters(strategy_params)

        # Live trading specific initializations
        if self.db_session and self.user_sub_obj and self.exchange_ccxt:
            self._live_trading_setup()
        else:
            self.logger.info(f"{self.name} initialized for BACKTESTING or without DB/Exchange context.")

    def _initialize_parameters(self, params_override: dict):
        defaults = self.get_parameters_definition()
        current_params = {k: v['default'] for k, v in defaults.items()}
        current_params.update(params_override)

        self.trading_pair = current_params.get("trading_pair") # Used by backtest for logging
        self.kline_interval = current_params.get("kline_interval")
        self.leverage = int(current_params.get("leverage"))
        self.stop_loss_pct = Decimal(str(current_params.get("stop_loss_pct"))) # Renamed from stop_loss_percent
        self.risk_per_trade_pct = Decimal(str(current_params.get("risk_per_trade_pct"))) # Renamed
        self.order_quantity_usd = Decimal(str(current_params.get("order_quantity_usd")))


        self.macd_fast_period = int(current_params.get("macd_fast_period"))
        self.macd_slow_period = int(current_params.get("macd_slow_period"))
        self.macd_signal_period = int(current_params.get("macd_signal_period"))
        self.min_candles_for_macd = int(current_params.get("min_candles_for_macd"))

        # Live-specific params (stored but may not be used in single-symbol backtest)
        self.top_n_symbols_to_scan = int(current_params.get("top_n_symbols_to_scan"))
        self.max_concurrent_trades = int(current_params.get("max_concurrent_trades"))
        self.min_volume_usdt_24h = float(current_params.get("min_volume_usdt_24h"))
        self.min_price_change_percent_filter = float(current_params.get("min_price_change_percent_filter"))
        
        # Initialize new parameters
        self.order_fill_max_retries = int(current_params.get("order_fill_max_retries", 10))
        self.order_fill_delay_seconds = int(current_params.get("order_fill_delay_seconds", 2))
        self.data_fetch_buffer = int(current_params.get("data_fetch_buffer", 50))

        self.params = current_params # Store fully resolved params

    def _live_trading_setup(self):
        self.active_trades: Dict[str, Dict[str, Any]] = {} 
        self.logger.info(f"{self.name} live setup for UserSubID {self.user_sub_obj.id if self.user_sub_obj else 'N/A'}")
        # self._load_all_persistent_positions() # Part of live execution cycle

    @classmethod
    def validate_parameters(cls, params: dict) -> dict:
        """Validates strategy-specific parameters."""
        definition = cls.get_parameters_definition()
        validated_params = {}
        _logger = logging.getLogger(__name__) # Use a logger instance

        for key, def_value in definition.items():
            val_type_str = def_value.get("type")
            choices = def_value.get("choices") 
            min_val = def_value.get("min")
            max_val = def_value.get("max")
            default_val = def_value.get("default")

            user_val = params.get(key)

            if user_val is None: # Parameter not provided by user
                if default_val is not None:
                    user_val = default_val # Apply default
                else:
                    # This implies a required parameter (no default) is missing.
                    # For this strategy, all defined params have defaults.
                    raise ValueError(f"Required parameter '{key}' is missing (and unexpectedly has no default in definition).")
            
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
            elif val_type_str == "str": 
                if not isinstance(user_val, str):
                    raise ValueError(f"Parameter '{key}' must be a string. Got: {user_val}")
            elif val_type_str == "bool": # Not in current definition
                if not isinstance(user_val, bool):
                    if str(user_val).lower() in ['true', 'yes', '1']: user_val = True
                    elif str(user_val).lower() in ['false', 'no', '0']: user_val = False
                    else: raise ValueError(f"Parameter '{key}' must be a boolean. Got: {user_val}")
            
            # Choice validation (Not in current definition)
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
                _logger.warning(f"Unknown parameter '{key_param}' provided for {cls.__name__}. It will be ignored.")
        
        return validated_params

    # Live trading methods like _load_all_persistent_positions, _get_top_gainer_loser_candidates,
    # _calculate_trade_qty (live version), _place_order_with_sl, _close_open_position, _sync_individual_position_state,
    # For the backtest, we'll use simplified position sizing and trade simulation.

    def _await_order_fill(self, order_id: str, symbol: str) -> Optional[Dict]: # Added method
        """ Awaits order fill by polling the exchange. """
        # Using instance parameters for timeout and interval
        start_time = time.time() 
        self.logger.info(f"[{self.name}-{symbol}] Awaiting fill for order {order_id} (timeout: {self.order_fill_timeout_seconds}s)")
        for attempt in range(self.order_fill_max_retries):
            try:
                order = self.exchange_ccxt.fetch_order(order_id, symbol)
                if order['status'] == 'closed':
                    self.logger.info(f"Order {order_id} filled.")
                    return order
                elif order['status'] in ['canceled', 'rejected', 'expired']:
                    self.logger.warning(f"Order {order_id} is {order['status']}. Not waiting further.")
                    return order
                self.logger.info(f"Order {order_id} status is {order['status']}. Attempt {attempt + 1}/{self.order_fill_max_retries}. Waiting...")
                time.sleep(self.order_fill_delay_seconds)
            except ccxt.OrderNotFound:
                self.logger.warning(f"Order {order_id} not found. Retrying.")
            except Exception as e:
                self.logger.error(f"Error fetching order {order_id}: {e}. Retrying.", exc_info=True)
        self.logger.warning(f"Timeout for order {order_id}. Final check.")
        try:
            final_status = self.exchange_ccxt.fetch_order(order_id, symbol)
            self.logger.info(f"Final status for order {order_id}: {final_status['status']}")
            return final_status
        except Exception as e:
            self.logger.error(f"Final check for order {order_id} failed: {e}", exc_info=True)
        return None

    def run_backtest(self, ohlcv_data: pd.DataFrame, initial_capital: float, params: dict) -> Dict[str, Any]:
        self.logger.info(f"[{self.name}-{self.trading_pair}] Starting backtest...")
        self._initialize_parameters(params) # Apply backtest-specific parameters
        
        self.logger.warning("This backtest runs TopGainersLosersMACD strategy on a SINGLE symbol. "
                            "The 'top gainers/losers' scanning aspect is NOT simulated. "
                            "Results will reflect MACD strategy performance on the provided symbol's data.")

        if not isinstance(ohlcv_data.index, pd.DatetimeIndex):
            ohlcv_data['timestamp'] = pd.to_datetime(ohlcv_data['timestamp'])
            ohlcv_data.set_index('timestamp', inplace=True)
        
        if len(ohlcv_data) < self.min_candles_for_macd:
            return {"pnl": 0, "trades_log": [], "equity_curve": [], "message": "Not enough data for MACD calculation."}

        closes_np = ohlcv_data['close'].to_numpy(dtype=float)
        opens_np = ohlcv_data['open'].to_numpy(dtype=float)
        
        macd, signal, hist = talib.MACD(closes_np, 
                                        fastperiod=self.macd_fast_period, 
                                        slowperiod=self.macd_slow_period, 
                                        signalperiod=self.macd_signal_period)

        df = ohlcv_data.copy()
        df['macd_hist'] = hist
        df['is_green_candle'] = df['close'] > df['open']
        df['is_red_candle'] = df['close'] < df['open']

        equity = Decimal(str(initial_capital))
        trades_log = []
        equity_curve = [{'timestamp': df.index[0].isoformat(), 'equity': float(equity)}]

        bt_position_side: Optional[str] = None
        bt_entry_price = Decimal("0")
        bt_position_qty = Decimal("0")
        bt_sl_price = Decimal("0")
        
        # Determine valid start index
        first_valid_idx = df.dropna(subset=['macd_hist']).index.min()
        if pd.isna(first_valid_idx):
            return {"pnl": 0, "trades_log": [], "equity_curve": [], "message": "Not enough data for MACD hist."}
        
        start_iloc = df.index.get_loc(first_valid_idx)
        start_iloc = max(start_iloc, 1) # Need previous bar for hist comparison

        for i in range(start_iloc, len(df)):
            current_bar = df.iloc[i]
            prev_bar = df.iloc[i-1]

            current_timestamp = current_bar.name.isoformat()
            current_price = Decimal(str(current_bar['close']))
            current_low = Decimal(str(current_bar['low']))
            current_high = Decimal(str(current_bar['high']))

            current_macd_hist = current_bar['macd_hist']
            prev_macd_hist = prev_bar['macd_hist']
            is_green_candle = current_bar['is_green_candle']
            is_red_candle = current_bar['is_red_candle']
            
            if pd.isna(current_macd_hist) or pd.isna(prev_macd_hist):
                equity_curve.append({'timestamp': current_timestamp, 'equity': float(equity)})
                continue

            # Exit Logic (SL only for this simplified backtest, no TP defined in original logic directly)
            if bt_position_side:
                exit_trade = False; exit_price = Decimal("0"); exit_reason = ""
                if bt_position_side == 'long' and current_low <= bt_sl_price:
                    exit_price = bt_sl_price; exit_reason = "SL"; exit_trade = True
                elif bt_position_side == 'short' and current_high >= bt_sl_price:
                    exit_price = bt_sl_price; exit_reason = "SL"; exit_trade = True
                
                # Reversal Signal as Exit (if MACD flips against position)
                if not exit_trade:
                    if bt_position_side == 'long' and current_macd_hist < prev_macd_hist and is_red_candle:
                        exit_price = current_price; exit_reason = "MACD Reversal"; exit_trade = True
                    elif bt_position_side == 'short' and current_macd_hist > prev_macd_hist and is_green_candle:
                        exit_price = current_price; exit_reason = "MACD Reversal"; exit_trade = True

                if exit_trade:
                    pnl = (exit_price - bt_entry_price) * bt_position_qty if bt_position_side == 'long' else (bt_entry_price - exit_price) * bt_position_qty
                    pnl_leveraged = pnl * Decimal(str(self.leverage))
                    equity += pnl_leveraged
                    trades_log.append({
                        'timestamp': current_timestamp, 'type': 'exit ' + bt_position_side,
                        'price': float(exit_price), 'quantity': float(bt_position_qty),
                        'pnl_realized': float(pnl_leveraged), 'reason': exit_reason, 'equity': float(equity)
                    })
                    bt_position_side = None; bt_position_qty = Decimal("0")

            # Entry Logic
            if not bt_position_side:
                entry_signal: Optional[str] = None
                # Buy Signal: MACD hist increasing, green candle
                if current_macd_hist > prev_macd_hist and is_green_candle:
                    entry_signal = 'long'
                # Sell Signal: MACD hist decreasing, red candle
                elif current_macd_hist < prev_macd_hist and is_red_candle:
                    entry_signal = 'short'
                
                if entry_signal:
                    bt_entry_price = current_price
                    
                    # Position Sizing
                    if self.order_quantity_usd > Decimal("0"): # Fixed USD notional
                        bt_position_qty = self.order_quantity_usd / bt_entry_price
                    elif self.risk_per_trade_pct > Decimal("0") and self.stop_loss_pct > Decimal("0"): # Risk % based
                        sl_price_calc = bt_entry_price * (Decimal('1') - self.stop_loss_pct) if entry_signal == 'long' else bt_entry_price * (Decimal('1') + self.stop_loss_pct)
                        price_diff_abs = abs(bt_entry_price - sl_price_calc)
                        if price_diff_abs > Decimal("0"):
                            risk_amount_equity = equity * self.risk_per_trade_pct
                            # Position Notional Value = Risk Amount / (SL % of price)
                            # SL % of price = abs(entry - SL_price) / entry
                            position_notional_value = risk_amount_equity / (price_diff_abs / bt_entry_price)
                            bt_position_qty = position_notional_value / bt_entry_price
                        else:
                            bt_position_qty = Decimal("0") # Avoid division by zero if SL is too close or zero
                    else: # Fallback if no valid sizing param
                        bt_position_qty = Decimal("0")
                        self.logger.warning("No valid position sizing parameters (order_quantity_usd or risk_per_trade_pct with stop_loss_pct). Qty is 0.")


                    if bt_position_qty > Decimal("0"):
                        bt_position_side = entry_signal
                        bt_sl_price = bt_entry_price * (Decimal('1') - self.stop_loss_pct) if entry_signal == 'long' else bt_entry_price * (Decimal('1') + self.stop_loss_pct)
                        
                        trades_log.append({
                            'timestamp': current_timestamp, 'type': 'entry ' + entry_signal,
                            'price': float(bt_entry_price), 'quantity': float(bt_position_qty),
                            'pnl_realized': 0.0, 'reason': 'MACD Signal', 'equity': float(equity)
                        })
                    else:
                        self.logger.warning(f"[BT] Calculated position qty is zero or negative for {self.trading_pair}. Skipping trade.")
            
            equity_curve.append({'timestamp': current_timestamp, 'equity': float(equity)})

        final_pnl = equity - Decimal(str(initial_capital))
        final_pnl_percent = (final_pnl / Decimal(str(initial_capital))) * Decimal("100") if initial_capital > 0 else Decimal("0")
        
        total_trades = len([t for t in trades_log if 'entry' in t['type']])
        winning_trades = sum(1 for t in trades_log if t['pnl_realized'] > 0)
        losing_trades = sum(1 for t in trades_log if t['pnl_realized'] < 0)

        self.logger.info(f"[{self.name}-{self.trading_pair}] Backtest finished. Final PnL: {final_pnl:.2f} ({final_pnl_percent:.2f}%)")
        
        return {
            "pnl": float(final_pnl), "pnl_percentage": float(final_pnl_percent),
            "total_trades": total_trades, "winning_trades": winning_trades, "losing_trades": losing_trades,
            "sharpe_ratio": 0.0, "max_drawdown": 0.0, # Placeholder
            "trades_log": trades_log, "equity_curve": equity_curve,
            "message": "Backtest completed for TopGainersLosersMACD (single symbol MACD logic).",
            "initial_capital": float(initial_capital), "final_equity": float(equity)
        }

    def execute_live_signal(self):
        # Live trading logic (kept for completeness but not focus of this task)
        self.logger.info(f"Executing live signal for {self.name} (Not Implemented in detail for this backtest task)")
        pass

```
