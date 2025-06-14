import logging
import numpy as np
import talib
import ccxt # Keep for potential type hints if exchange_ccxt is used in live
from decimal import Decimal, ROUND_DOWN, ROUND_UP, ROUND_NEAREST, getcontext
from datetime import datetime # Ensure time is imported if used by helpers
import time # For _await_order_fill if used by helpers
import json 
from typing import Optional, Dict, Any, List

import pandas as pd # Standard alias
from sqlalchemy.orm import Session # Optional for backtest
from backend.models import Position, Order, UserStrategySubscription # Optional for backtest
from backend import strategy_utils # Optional for backtest

getcontext().prec = 18

class TheOrphanStrategy:

    @staticmethod
    def get_parameters_definition():
        return {
            "trading_pair": {"type": "str", "label": "Trading Pair", "default": "BTC/USDT"},
            "kline_interval": {"type": "str", "label": "Kline Interval", "default": "1h"}, # Data input timeframe
            "leverage": {"type": "int", "label": "Leverage", "default": 10},
            "order_quantity_usd": {"type": "float", "label": "Order Quantity (USD)", "default": 100.0},

            "bb_length": {"type": "int", "label": "BB Length", "default": 24},
            "bb_stdev": {"type": "float", "label": "BB StdDev", "default": 2.1},
            "trend_ema_period": {"type": "int", "label": "Trend EMA Period", "default": 200}, # Common default
            "vol_filter_stdev_length": {"type": "int", "label": "Vol Filter STDEV Length", "default": 15},
            "vol_filter_sma_length": {"type": "int", "label": "Vol Filter SMA Length (of STDEV)", "default": 28},
            
            "use_stop_loss": {"type": "bool", "label": "Use Stop Loss (Initial)", "default": True},
            "stop_loss_pct": {"type": "float", "label": "Stop Loss %", "default": 2.0},
            "use_take_profit": {"type": "bool", "label": "Use Take Profit", "default": True},
            "take_profit_pct": {"type": "float", "label": "Take Profit %", "default": 9.0}, # As per original
            
            "use_trailing_stop": {"type": "bool", "label": "Use Trailing Stop", "default": True},
            "trailing_stop_activation_pct": {"type": "float", "label": "TSL Activation % Profit", "default": 1.0}, # Adjusted
            "trailing_stop_offset_pct": {"type": "float", "label": "TSL Offset %", "default": 1.0}, # Adjusted
            # New parameters
            "order_fill_timeout_seconds": {"type": "int", "default": 60, "min": 10, "max": 300, "label": "Order Fill Timeout (s)"},
            "order_fill_check_interval_seconds": {"type": "int", "default": 3, "min": 1, "max": 30, "label": "Order Fill Check Interval (s)"},
        }

    def __init__(self, db_session: Optional[Session], user_sub_obj: Optional[UserStrategySubscription], 
                 strategy_params: dict, exchange_ccxt: Optional[Any], logger_obj=None):
        self.db_session = db_session
        self.user_sub_obj = user_sub_obj
        # self.params = strategy_params # Raw params stored if needed, _initialize_parameters sets attributes
        self.exchange_ccxt = exchange_ccxt
        self.logger = logger_obj if logger_obj else logging.getLogger(__name__)
        self.name = "TheOrphanStrategy"
        
        self._initialize_parameters(strategy_params) # Set attributes from params

        # Live trading specific initializations
        if self.db_session and self.user_sub_obj and self.exchange_ccxt:
            self._live_trading_setup()
        else:
            self.logger.info(f"{self.name} initialized for BACKTESTING or without DB/Exchange context.")


    def _initialize_parameters(self, params_override: dict):
        defaults = self.get_parameters_definition()
        current_params = {k: v['default'] for k, v in defaults.items()}
        current_params.update(params_override)

        self.trading_pair = current_params.get("trading_pair")
        self.kline_interval = current_params.get("kline_interval")
        self.leverage = int(current_params.get("leverage"))
        self.order_quantity_usd = Decimal(str(current_params.get("order_quantity_usd")))

        self.bb_length = int(current_params.get("bb_length"))
        self.bb_stdev = float(current_params.get("bb_stdev"))
        self.trend_ema_period = int(current_params.get("trend_ema_period"))
        self.vol_filter_stdev_length = int(current_params.get("vol_filter_stdev_length"))
        self.vol_filter_sma_length = int(current_params.get("vol_filter_sma_length"))
        
        self.use_stop_loss = current_params.get("use_stop_loss")
        self.stop_loss_pct = Decimal(str(current_params.get("stop_loss_pct"))) / Decimal("100")
        self.use_take_profit = current_params.get("use_take_profit")
        self.take_profit_pct = Decimal(str(current_params.get("take_profit_pct"))) / Decimal("100")
        
        self.use_trailing_stop = current_params.get("use_trailing_stop")
        self.trailing_stop_activation_pct = Decimal(str(current_params.get("trailing_stop_activation_pct"))) / Decimal("100")
        self.trailing_stop_offset_pct = Decimal(str(current_params.get("trailing_stop_offset_pct"))) / Decimal("100")
        
        # Initialize new parameters
        self.order_fill_timeout_seconds = int(current_params.get("order_fill_timeout_seconds", 60))
        self.order_fill_check_interval_seconds = int(current_params.get("order_fill_check_interval_seconds", 3))
        
        self.params = current_params # Store the fully resolved params

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
            elif val_type_str == "str": 
                if not isinstance(user_val, str):
                    raise ValueError(f"Parameter '{key}' must be a string. Got: {user_val}")
            elif val_type_str == "bool":
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
                _logger.warning(f"Unknown parameter '{key_param}' provided for {cls.__name__}. It will be ignored.")
        
        return validated_params

    def _live_trading_setup(self):
        """Initializes attributes specific to live trading."""
        self.active_position_db_id: Optional[int] = None
        self.position_entry_price: Optional[Decimal] = None
        self.position_side: Optional[str] = None 
        self.position_qty: Decimal = Decimal("0")
        self.high_water_mark: Optional[Decimal] = None
        self.low_water_mark: Optional[Decimal] = None
        self.trailing_stop_active: bool = False
        self.current_trailing_stop_price: Optional[Decimal] = None
        self.active_sl_exchange_id: Optional[str] = None
        self.active_tp_exchange_id: Optional[str] = None
        self.sl_order_db_id: Optional[int] = None
        self.tp_order_db_id: Optional[int] = None
        
        self.price_precision_str: Optional[str] = None
        self.quantity_precision_str: Optional[str] = None
        self._precisions_fetched_ = False

        self._fetch_market_precision()
        # self._load_persistent_state() # Called in live execution cycle
        self._set_leverage()
        self.logger.info(f"{self.name} live setup for {self.trading_pair}, UserSubID {self.user_sub_obj.id if self.user_sub_obj else 'N/A'}")

    # All helper methods like _fetch_market_precision, _load_persistent_state, _save_persistent_state,
    # _format_quantity, _format_price, _set_leverage, _await_order_fill, _place_order, _cancel_order_by_id,
    # _sync_exchange_position_state, _reset_internal_trade_state etc. are assumed to be here for live trading.
    # For run_backtest, they are not directly called, but _calculate_indicators is used.

    def _await_order_fill(self, order_id: str, symbol: str) -> Optional[Dict]: # Added symbol argument
        """ Awaits order fill by polling the exchange, using instance parameters for timeout/interval. """
        # Ensure 'time' module is imported at the top of the file: import time
        start_time = time.time()
        self.logger.info(f"[{self.name}-{symbol}] Awaiting fill for order {order_id} (timeout: {self.order_fill_timeout_seconds}s)")
        for attempt in range(self.order_fill_max_retries if hasattr(self, 'order_fill_max_retries') else 5): # Fallback for max_retries if not set
            try:
                order = self.exchange_ccxt.fetch_order(order_id, symbol)
                if order['status'] == 'closed':
                    self.logger.info(f"Order {order_id} filled.")
                    return order
                elif order['status'] in ['canceled', 'rejected', 'expired']:
                    self.logger.warning(f"Order {order_id} is {order['status']}. Not waiting further.")
                    return order
                self.logger.info(f"Order {order_id} status is {order['status']}. Attempt {attempt + 1}/"
                                 f"{self.order_fill_max_retries if hasattr(self, 'order_fill_max_retries') else 5}. Waiting...")
                time.sleep(self.order_fill_check_interval_seconds if hasattr(self, 'order_fill_check_interval_seconds') else 3) # Fallback for interval
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

    def _calculate_indicators(self, ohlcv_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        indicators = {}
        close_prices = ohlcv_df['close'].to_numpy(dtype=float)
        
        upper_bb, middle_bb, lower_bb = talib.BBANDS(close_prices, timeperiod=self.bb_length, nbdevup=self.bb_stdev, nbdevdn=self.bb_stdev, matype=0)
        indicators['upper_bb'] = upper_bb
        indicators['middle_bb'] = middle_bb
        indicators['lower_bb'] = lower_bb
        
        indicators['trend_ema'] = talib.EMA(close_prices, timeperiod=self.trend_ema_period)
        
        vol_std = talib.STDDEV(close_prices, timeperiod=self.vol_filter_stdev_length, nbdev=1) # nbdev=1 for stddev
        vol_sma_of_std = talib.SMA(vol_std, timeperiod=self.vol_filter_sma_length)
        indicators['vol_cond'] = vol_std > vol_sma_of_std # True if current vol > its SMA
        
        return indicators

    def run_backtest(self, ohlcv_data: pd.DataFrame, initial_capital: float, params: dict) -> Dict[str, Any]:
        self.logger.info(f"[{self.name}-{self.trading_pair}] Starting backtest...")
        self._initialize_parameters(params)

        if not isinstance(ohlcv_data.index, pd.DatetimeIndex):
            ohlcv_data['timestamp'] = pd.to_datetime(ohlcv_data['timestamp'])
            ohlcv_data.set_index('timestamp', inplace=True)

        indicators_dict = self._calculate_indicators(ohlcv_data)
        
        df = ohlcv_data.copy()
        df['upper_bb'] = indicators_dict['upper_bb']
        df['middle_bb'] = indicators_dict['middle_bb']
        df['lower_bb'] = indicators_dict['lower_bb']
        df['trend_ema'] = indicators_dict['trend_ema']
        df['vol_cond'] = indicators_dict['vol_cond']

        equity = Decimal(str(initial_capital))
        trades_log = []
        equity_curve = [{'timestamp': df.index[0].isoformat(), 'equity': float(equity)}]

        bt_position_side: Optional[str] = None
        bt_entry_price = Decimal("0")
        bt_position_qty = Decimal("0")
        bt_initial_sl_price = Decimal("0") # Fixed SL based on entry
        bt_tp_price = Decimal("0")
        
        # Trailing Stop State for Backtest
        bt_high_water_mark: Optional[Decimal] = None
        bt_low_water_mark: Optional[Decimal] = None
        bt_trailing_stop_active: bool = False
        bt_current_trailing_sl_price: Optional[Decimal] = None

        # Determine valid start index
        first_valid_idx = df.dropna(subset=['upper_bb', 'lower_bb', 'trend_ema', 'vol_cond']).index.min()
        if pd.isna(first_valid_idx):
            return {"pnl": 0, "trades_log": [], "equity_curve": [], "message": "Not enough data for indicators."}
        start_iloc = df.index.get_loc(first_valid_idx)
        start_iloc = max(start_iloc, 1) # Need previous bar for crossover

        for i in range(start_iloc, len(df)):
            current_bar = df.iloc[i]
            prev_bar = df.iloc[i-1]

            current_timestamp = current_bar.name.isoformat()
            current_price = Decimal(str(current_bar['close']))
            current_low = Decimal(str(current_bar['low']))
            current_high = Decimal(str(current_bar['high']))

            val_upper_bb = Decimal(str(current_bar['upper_bb']))
            val_lower_bb = Decimal(str(current_bar['lower_bb']))
            val_trend_ema = Decimal(str(current_bar['trend_ema']))
            val_vol_cond = current_bar['vol_cond']

            prev_price = Decimal(str(prev_bar['close']))
            prev_upper_bb = Decimal(str(prev_bar['upper_bb']))
            prev_lower_bb = Decimal(str(prev_bar['lower_bb']))

            # Conditions
            buy_cond_bb_crossover = prev_price < prev_upper_bb and current_price > val_upper_bb
            sell_cond_bb_crossover = prev_price > prev_lower_bb and current_price < val_lower_bb
            buy_trend_cond = current_price > val_trend_ema
            sell_trend_cond = current_price < val_trend_ema
            
            final_buy_entry = buy_cond_bb_crossover and buy_trend_cond and val_vol_cond
            final_sell_entry = sell_cond_bb_crossover and sell_trend_cond and val_vol_cond

            # Exit Logic
            if bt_position_side:
                exit_trade = False; exit_price = Decimal("0"); exit_reason = ""
                effective_sl_price = bt_current_trailing_sl_price if bt_trailing_stop_active and bt_current_trailing_sl_price else bt_initial_sl_price

                if bt_position_side == 'long':
                    if self.use_stop_loss and current_low <= effective_sl_price: exit_price = effective_sl_price; exit_reason = "SL/TSL"; exit_trade = True
                    elif self.use_take_profit and current_high >= bt_tp_price: exit_price = bt_tp_price; exit_reason = "TP"; exit_trade = True
                    elif sell_cond_bb_crossover: exit_price = current_price; exit_reason = "BB Crossover Exit"; exit_trade = True
                
                elif bt_position_side == 'short':
                    if self.use_stop_loss and current_high >= effective_sl_price: exit_price = effective_sl_price; exit_reason = "SL/TSL"; exit_trade = True
                    elif self.use_take_profit and current_low <= bt_tp_price: exit_price = bt_tp_price; exit_reason = "TP"; exit_trade = True
                    elif buy_cond_bb_crossover: exit_price = current_price; exit_reason = "BB Crossover Exit"; exit_trade = True

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
                    # Reset TSL state on exit
                    bt_high_water_mark = None; bt_low_water_mark = None
                    bt_trailing_stop_active = False; bt_current_trailing_sl_price = None
            
            # Entry Logic
            if not bt_position_side:
                entry_signal: Optional[str] = None
                if final_buy_entry: entry_signal = 'long'
                elif final_sell_entry: entry_signal = 'short'
                
                if entry_signal:
                    bt_entry_price = current_price
                    bt_position_qty = self.order_quantity_usd / bt_entry_price
                    
                    if bt_position_qty > Decimal("0"):
                        bt_position_side = entry_signal
                        if self.use_stop_loss:
                            bt_initial_sl_price = bt_entry_price * (Decimal('1') - self.stop_loss_pct) if entry_signal == 'long' else bt_entry_price * (Decimal('1') + self.stop_loss_pct)
                        if self.use_take_profit:
                            bt_tp_price = bt_entry_price * (Decimal('1') + self.take_profit_pct) if entry_signal == 'long' else bt_entry_price * (Decimal('1') - self.take_profit_pct)
                        
                        # Initialize TSL state on entry
                        if entry_signal == 'long': bt_high_water_mark = bt_entry_price
                        else: bt_low_water_mark = bt_entry_price
                        bt_trailing_stop_active = False
                        bt_current_trailing_sl_price = None # Initial SL is fixed

                        trades_log.append({
                            'timestamp': current_timestamp, 'type': 'entry ' + entry_signal,
                            'price': float(bt_entry_price), 'quantity': float(bt_position_qty),
                            'pnl_realized': 0.0, 'reason': 'Strategy Signal', 'equity': float(equity)
                        })
                    else:
                        self.logger.warning(f"[BT] Calculated position qty is zero or negative. Skipping trade.")

            # Trailing Stop Update Logic (if in position and TSL enabled)
            if bt_position_side and self.use_trailing_stop and bt_entry_price > 0:
                new_tsl_candidate = None
                if bt_position_side == 'long':
                    if current_high > (bt_high_water_mark or bt_entry_price): bt_high_water_mark = current_high
                    if not bt_trailing_stop_active and current_price >= bt_entry_price * (Decimal('1') + self.trailing_stop_activation_pct):
                        bt_trailing_stop_active = True
                    if bt_trailing_stop_active and bt_high_water_mark: # Ensure HWM is set
                        new_tsl_candidate = bt_high_water_mark * (Decimal('1') - self.trailing_stop_offset_pct)
                        if new_tsl_candidate > (bt_current_trailing_sl_price or bt_initial_sl_price or Decimal("-Infinity")): # Ensure TSL only moves up
                            bt_current_trailing_sl_price = new_tsl_candidate
                
                elif bt_position_side == 'short':
                    if current_low < (bt_low_water_mark or bt_entry_price): bt_low_water_mark = current_low
                    if not bt_trailing_stop_active and current_price <= bt_entry_price * (Decimal('1') - self.trailing_stop_activation_pct):
                        bt_trailing_stop_active = True
                    if bt_trailing_stop_active and bt_low_water_mark: # Ensure LWM is set
                        new_tsl_candidate = bt_low_water_mark * (Decimal('1') + self.trailing_stop_offset_pct)
                        if new_tsl_candidate < (bt_current_trailing_sl_price or bt_initial_sl_price or Decimal("Infinity")): # Ensure TSL only moves down
                            bt_current_trailing_sl_price = new_tsl_candidate
            
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
            "message": "Backtest completed for TheOrphanStrategy.",
            "initial_capital": float(initial_capital), "final_equity": float(equity)
        }

    def execute_live_signal(self):
        # Live trading logic (kept for completeness of class structure, but not focus of this task)
        self.logger.info(f"Executing live signal for {self.name} (Not Implemented in detail for this backtest task)")
        # This would typically involve:
        # self._load_persistent_state()
        # self._sync_exchange_position_state()
        # Fetching live OHLCV data
        # self._process_trading_logic(live_ohlcv_df)
        pass

```
