import logging
import numpy as np
import talib
import pandas as pd
from decimal import Decimal, ROUND_DOWN, ROUND_UP, ROUND_NEAREST, getcontext
import json
from datetime import datetime
import time # For _await_order_fill, if used in live mode
from typing import Optional, Dict, Any, List

from sqlalchemy.orm import Session
from backend.models import Position, Order, UserStrategySubscription # Optional for backtest
from backend import strategy_utils # Optional for backtest

getcontext().prec = 18

class NadarayaWatsonEnvelopeStrategy: # Renamed class

    @staticmethod
    def get_parameters_definition():
        return {
            "trading_pair": {"type": "str", "label": "Trading Pair", "default": "BTC/USDT"},
            "leverage": {"type": "int", "label": "Leverage", "default": 10, "min": 1, "max": 100},
            "order_quantity_usd": {"type": "float", "label": "Order Quantity (USD Notional)", "default": 100.0, "min": 1.0},
            
            "nw_timeframe": {"type": "str", "label": "Nadaraya-Watson Timeframe", "default": "15m"}, # Data input timeframe
            "nw_h_bandwidth": {"type": "float", "label": "NW: h (Bandwidth)", "default": 8.0, "min": 1.0},
            "nw_env_multiplier": {"type": "float", "label": "NW: Envelope Multiplier (ATR/StdDev)", "default": 2.0, "min": 0.1},
            "nw_yhat_lookback": {"type": "int", "label": "NW: y_hat Lookback (for estimator)", "default": 50, "min": 5}, # Increased default based on common usage
            "nw_residual_lookback": {"type": "int", "label": "NW: Residual Lookback (for ATR/StdDev)", "default": 20, "min": 5},
            "nw_env_type": {"type": "str", "label": "NW: Envelope Type", "default": "ATR", "choices": ["ATR", "StdDev"]},

            "trend_linreg_lookback": {"type": "int", "label": "Trend: LinReg Channel Lookback", "default": 50, "min": 5},
            "trend_ema_slope_period": {"type": "int", "label": "Trend: EMA Slope Period (on y_hat or close)", "default": 20, "min": 2},
            
            "use_stop_loss": {"type": "bool", "label": "Use Stop Loss", "default": True},
            "stop_loss_pct": {"type": "float", "label": "Stop Loss % (from entry)", "default": 1.5, "min": 0.1, "step": 0.1},
            "use_take_profit": {"type": "bool", "label": "Use Take Profit", "default": True},
            "take_profit_pct": {"type": "float", "label": "Take Profit % (from entry)", "default": 3.0, "min": 0.1, "step": 0.1},
            "exit_on_midline_cross": {"type": "bool", "label": "Exit on NW Midline Cross", "default": True},
        }

    def __init__(self, db_session: Optional[Session], user_sub_obj: Optional[UserStrategySubscription], 
                 strategy_params: dict, exchange_ccxt: Optional[Any], logger_obj=None):
        self.db_session = db_session
        self.user_sub_obj = user_sub_obj
        self.params = strategy_params # Store raw params
        self.exchange_ccxt = exchange_ccxt # Optional, for live trading aspects
        self.logger = logger_obj if logger_obj else logging.getLogger(__name__)
        self.name = "NadarayaWatsonEnvelopeStrategy"

        self._initialize_parameters(strategy_params)

        # Live trading specific initializations (not strictly needed for backtest method alone)
        if self.db_session and self.user_sub_obj and self.exchange_ccxt:
            self._live_trading_setup()
        else:
            self.logger.info(f"{self.name} initialized for BACKTESTING or without DB/Exchange context.")

    def _initialize_parameters(self, params_override: dict):
        """Helper to set parameters from defaults or overrides."""
        defaults = self.get_parameters_definition()
        current_params = {k: v['default'] for k, v in defaults.items()}
        current_params.update(params_override) # Apply overrides from constructor

        self.trading_pair = current_params.get("trading_pair")
        self.leverage = int(current_params.get("leverage"))
        self.order_quantity_usd = Decimal(str(current_params.get("order_quantity_usd")))
        
        self.nw_timeframe = current_params.get("nw_timeframe")
        self.nw_h_bandwidth = float(current_params.get("nw_h_bandwidth"))
        self.nw_env_multiplier = float(current_params.get("nw_env_multiplier"))
        self.nw_yhat_lookback = int(current_params.get("nw_yhat_lookback"))
        self.nw_residual_lookback = int(current_params.get("nw_residual_lookback"))
        self.nw_env_type = current_params.get("nw_env_type")

        self.trend_linreg_lookback = int(current_params.get("trend_linreg_lookback"))
        self.trend_ema_slope_period = int(current_params.get("trend_ema_slope_period"))

        self.use_stop_loss = current_params.get("use_stop_loss")
        self.stop_loss_pct = Decimal(str(current_params.get("stop_loss_pct"))) / Decimal("100")
        self.use_take_profit = current_params.get("use_take_profit")
        self.take_profit_pct = Decimal(str(current_params.get("take_profit_pct"))) / Decimal("100")
        self.exit_on_midline_cross = current_params.get("exit_on_midline_cross")
        
        # Store merged params if needed for other parts of the class
        self.params = current_params # Update self.params to the merged version

    # validate_parameters method was already added in the previous turn.
    # This is unexpected. I will ensure the content is correct.
    # The method from the previous turn seems fine.
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
        self.sl_order_db_id: Optional[int] = None
        self.tp_order_db_id: Optional[int] = None
        self.active_sl_tp_orders: Dict[str, Optional[str]] = {}
        self.active_position_side: Optional[str] = None
        self.position_entry_price: Optional[Decimal] = None
        self.position_qty: Decimal = Decimal("0")
        
        self.price_precision_str: Optional[str] = None
        self.quantity_precision_str: Optional[str] = None
        self._precisions_fetched_ = False

        self._fetch_market_precision()
        # self._load_persistent_state() # If managing state for live trading
        self._set_leverage()
        self.logger.info(f"{self.name} initialized for LIVE trading {self.trading_pair}, UserSubID {self.user_sub_obj.id if self.user_sub_obj else 'N/A'}")
        # Placeholder for other live-specific methods like _fetch_market_precision, _load_persistent_state etc.
        # These are not directly used by run_backtest but would be part of a full live strategy class.

    # _live_trading_setup method was here.

    def _fetch_market_precision(self):
        if not self._precisions_fetched_ and self.exchange_ccxt:
            try:
                self.exchange_ccxt.load_markets(True)
                market = self.exchange_ccxt.markets[self.trading_pair]
                self.quantity_precision_str = str(market['precision']['amount'])
                self.price_precision_str = str(market['precision']['price'])
                self._precisions_fetched_ = True
            except Exception: # Simplified error handling
                self.quantity_precision_str = "0.00001"; self.price_precision_str = "0.01"

    def _set_leverage(self):
        # Simplified version for context
        if self.exchange_ccxt and hasattr(self.exchange_ccxt, 'set_leverage'):
            try:
                self.exchange_ccxt.set_leverage(self.leverage, self.trading_pair.split(':')[0])
            except Exception: pass # Simplified

    def _gauss(self, x: float, h: float) -> float:
        if h == 0: return 1.0 if x == 0 else 0.0 # Avoid division by zero if h is exactly 0
        return np.exp(-((x ** 2) / (2 * h ** 2)))

    def _causal_nadaraya_watson_estimator(self, data_series_np: np.ndarray, h_bandwidth: float, y_hat_lookback: int) -> np.ndarray:
        n = len(data_series_np)
        y_hat_arr = np.full(n, np.nan)
        for i in range(n):
            start_idx = max(0, i - y_hat_lookback + 1)
            current_window = data_series_np[start_idx : i + 1]
            
            weighted_sum = 0.0
            total_weight = 0.0
            
            # Kernel regression for the current point `i` using `current_window`
            # The "distance" for the kernel is how far back in the window the point is
            for j_local, val_j in enumerate(current_window):
                # distance: (length of window - 1) is current point, 0 is oldest point in window
                gauss_dist = float((len(current_window) - 1) - j_local) 
                weight = self._gauss(gauss_dist, h_bandwidth)
                weighted_sum += val_j * weight
                total_weight += weight
            
            if total_weight > 1e-9: # Prevent division by zero or near-zero
                y_hat_arr[i] = weighted_sum / total_weight
            elif len(current_window) > 0: # Fallback for edge cases
                y_hat_arr[i] = current_window[-1] 
        return y_hat_arr

    def _calculate_residuals_metric(self, actual_prices_np: np.ndarray, y_hat_np: np.ndarray, lookback: int, metric_type: str = "ATR") -> np.ndarray:
        residuals = actual_prices_np - y_hat_np
        abs_residuals = np.abs(residuals)
        
        metric_values = np.full(len(actual_prices_np), np.nan)
        
        if metric_type == "ATR":
            # ATR calculation needs high, low, close. We only have close and y_hat.
            # A common proxy for ATR of residuals is to use MA of absolute residuals.
            # Alternatively, if full OHLC is available, calculate True Range of residuals (more complex)
            # For simplicity, using Moving Average of Absolute Residuals as a proxy for ATR-like volatility of residuals
            if lookback > 0 and len(abs_residuals) >= lookback:
                 metric_values = talib.SMA(abs_residuals, timeperiod=lookback)
        elif metric_type == "StdDev":
            if lookback > 0 and len(residuals) >= lookback:
                metric_values = talib.STDDEV(residuals, timeperiod=lookback, nbdev=1) # nbdev=1 for 1 std dev
        
        return metric_values

    def run_backtest(self, ohlcv_data: pd.DataFrame, initial_capital: float, params: dict) -> Dict[str, Any]:
        self.logger.info(f"[{self.name}-{self.trading_pair}] Starting backtest...")
        self._initialize_parameters(params) # Apply backtest-specific parameters

        if not isinstance(ohlcv_data.index, pd.DatetimeIndex):
            ohlcv_data['timestamp'] = pd.to_datetime(ohlcv_data['timestamp'])
            ohlcv_data.set_index('timestamp', inplace=True)

        close_prices_np = ohlcv_data['close'].to_numpy(dtype=float)

        # 1. Nadaraya-Watson Estimator (y_hat - the midline)
        y_hat = self._causal_nadaraya_watson_estimator(close_prices_np, self.nw_h_bandwidth, self.nw_yhat_lookback)
        
        # 2. Envelope Bands
        residuals_metric = self._calculate_residuals_metric(close_prices_np, y_hat, self.nw_residual_lookback, self.nw_env_type)
        upper_band = y_hat + (residuals_metric * self.nw_env_multiplier)
        lower_band = y_hat - (residuals_metric * self.nw_env_multiplier)

        # 3. Trend: Linear Regression Channel (on y_hat or close)
        # Using talib.LINEARREG_SLOPE on y_hat to determine trend direction might be simpler than full channel
        # For a channel, you'd typically use LINEARREG on price, then get angle or check if price is above/below.
        # Let's use LINEARREG_SLOPE on y_hat to indicate trend of the NW estimator itself.
        linreg_source = y_hat # or close_prices_np
        valid_linreg_source_for_slope = linreg_source[~np.isnan(linreg_source)] # Remove NaNs for TA-Lib
        
        trend_slope_values = np.full(len(linreg_source), np.nan)
        if len(valid_linreg_source_for_slope) >= self.trend_linreg_lookback:
            slope_calc = talib.LINEARREG_SLOPE(valid_linreg_source_for_slope, timeperiod=self.trend_linreg_lookback)
            # Pad slope_calc to match original length of linreg_source
            nan_padding_count = len(linreg_source) - len(slope_calc)
            trend_slope_values = np.pad(slope_calc, (nan_padding_count, 0), 'constant', constant_values=np.nan)


        # 4. Trend: EMA Slope (on y_hat or close)
        ema_source = y_hat # or close_prices_np
        valid_ema_source = ema_source[~np.isnan(ema_source)]
        
        ema_values = np.full(len(ema_source), np.nan)
        ema_slope_values = np.full(len(ema_source), np.nan)

        if len(valid_ema_source) >= self.trend_ema_slope_period:
            ema_calc = talib.EMA(valid_ema_source, timeperiod=self.trend_ema_slope_period)
            nan_padding_ema = len(ema_source) - len(ema_calc)
            ema_values = np.pad(ema_calc, (nan_padding_ema, 0), 'constant', constant_values=np.nan)
            
            # Calculate slope of EMA: (EMA_today - EMA_yesterday)
            ema_slope_values[1:] = ema_values[1:] - ema_values[:-1] # Difference calculation, first will be NaN
            ema_slope_values[0] = np.nan # Explicitly set the first slope to NaN


        # Combine into DataFrame for iteration
        df = ohlcv_data.copy()
        df['y_hat'] = y_hat
        df['upper_band'] = upper_band
        df['lower_band'] = lower_band
        df['trend_slope'] = trend_slope_values # LinReg slope of y_hat
        df['ema_slope'] = ema_slope_values     # EMA slope of y_hat

        # Initialize portfolio
        equity = Decimal(str(initial_capital))
        trades_log = []
        equity_curve = [{'timestamp': df.index[0].isoformat(), 'equity': float(equity)}]

        bt_position_side: Optional[str] = None
        bt_entry_price = Decimal("0")
        bt_position_qty = Decimal("0")
        bt_sl_price = Decimal("0")
        bt_tp_price = Decimal("0")

        # Determine valid start index after all indicator calculations
        # Find first row where all necessary indicators are non-NaN
        first_valid_idx = df.dropna(subset=['y_hat', 'upper_band', 'lower_band', 'trend_slope', 'ema_slope']).index.min()
        if pd.isna(first_valid_idx):
             self.logger.warning("Not enough data to start backtest after indicator calculation.")
             return {"pnl": 0, "trades_log": [], "equity_curve": [], "message": "Not enough data."}
        
        start_iloc = df.index.get_loc(first_valid_idx)
        start_iloc = max(start_iloc, 1) # Ensure we have a previous bar for midline cross checks

        for i in range(start_iloc, len(df)):
            current_bar = df.iloc[i]
            prev_bar = df.iloc[i-1] # For midline cross check

            current_timestamp = current_bar.name.isoformat()
            current_price = Decimal(str(current_bar['close']))
            current_low = Decimal(str(current_bar['low']))
            current_high = Decimal(str(current_bar['high']))

            # Indicator values
            val_y_hat = Decimal(str(current_bar['y_hat']))
            val_upper_band = Decimal(str(current_bar['upper_band']))
            val_lower_band = Decimal(str(current_bar['lower_band']))
            val_trend_slope = Decimal(str(current_bar['trend_slope'])) # LinReg slope
            val_ema_slope = Decimal(str(current_bar['ema_slope']))

            prev_price = Decimal(str(prev_bar['close']))
            prev_y_hat = Decimal(str(prev_bar['y_hat']))

            # Exit Logic
            if bt_position_side:
                exit_trade = False; exit_price = Decimal("0"); exit_reason = ""
                if bt_position_side == 'long':
                    if self.use_stop_loss and current_low <= bt_sl_price: exit_price = bt_sl_price; exit_reason = "SL"; exit_trade = True
                    elif self.use_take_profit and current_high >= bt_tp_price: exit_price = bt_tp_price; exit_reason = "TP"; exit_trade = True
                    elif self.exit_on_midline_cross and prev_price > prev_y_hat and current_price <= val_y_hat: # Cross below midline
                        exit_price = current_price; exit_reason = "Midline Cross"; exit_trade = True
                elif bt_position_side == 'short':
                    if self.use_stop_loss and current_high >= bt_sl_price: exit_price = bt_sl_price; exit_reason = "SL"; exit_trade = True
                    elif self.use_take_profit and current_low <= bt_tp_price: exit_price = bt_tp_price; exit_reason = "TP"; exit_trade = True
                    elif self.exit_on_midline_cross and prev_price < prev_y_hat and current_price >= val_y_hat: # Cross above midline
                        exit_price = current_price; exit_reason = "Midline Cross"; exit_trade = True
                
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
                # Long condition: Price touches lower band, LinReg trend is up, EMA slope is up
                if current_price <= val_lower_band and val_trend_slope > Decimal("0") and val_ema_slope > Decimal("0"):
                    entry_signal = 'long'
                # Short condition: Price touches upper band, LinReg trend is down, EMA slope is down
                elif current_price >= val_upper_band and val_trend_slope < Decimal("0") and val_ema_slope < Decimal("0"):
                    entry_signal = 'short'
                
                if entry_signal:
                    bt_entry_price = current_price 
                    bt_position_qty = self.order_quantity_usd / bt_entry_price
                    
                    if bt_position_qty > Decimal("0"):
                        bt_position_side = entry_signal
                        if self.use_stop_loss:
                            bt_sl_price = bt_entry_price * (Decimal('1') - self.stop_loss_pct) if entry_signal == 'long' else bt_entry_price * (Decimal('1') + self.stop_loss_pct)
                        if self.use_take_profit:
                            bt_tp_price = bt_entry_price * (Decimal('1') + self.take_profit_pct) if entry_signal == 'long' else bt_entry_price * (Decimal('1') - self.take_profit_pct)
                        
                        trades_log.append({
                            'timestamp': current_timestamp, 'type': 'entry ' + entry_signal,
                            'price': float(bt_entry_price), 'quantity': float(bt_position_qty),
                            'pnl_realized': 0.0, 'reason': 'NW Envelope Signal', 'equity': float(equity)
                        })
                    else:
                        self.logger.warning(f"[BT] Calculated position qty is zero or negative. Skipping trade.")
            
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
            "message": "Backtest completed for NadarayaWatsonEnvelopeStrategy.",
            "initial_capital": float(initial_capital), "final_equity": float(equity)
        }

    def execute_live_signal(self):
        # Placeholder for live execution, not the focus of this task
        self.logger.info(f"Executing live signal for {self.name} (Not Implemented for this task scope)")
        pass

# Example Usage (for testing, not part of the class itself)
# if __name__ == '__main__':
#     logging.basicConfig(level=logging.INFO)
#     logger = logging.getLogger("BacktestRunner_NWEnvelope")

#     num_bars = 1000
#     base_time = datetime(2023, 1, 1)
#     time_deltas = [pd.Timedelta(minutes=15*i) for i in range(num_bars)] # Assuming 15m timeframe data
#     timestamps = [base_time + delta for delta in time_deltas]
    
#     data_large = {
#         'open': np.random.normal(loc=30000, scale=100, size=num_bars),
#         'high': np.random.normal(loc=30050, scale=100, size=num_bars),
#         'low': np.random.normal(loc=29950, scale=100, size=num_bars),
#         'close': np.random.normal(loc=30000, scale=100, size=num_bars),
#         'volume': np.random.uniform(10, 200, size=num_bars)
#     }
#     sample_ohlcv_df = pd.DataFrame(data_large, index=pd.DatetimeIndex(timestamps))
#     sample_ohlcv_df['high'] = sample_ohlcv_df[['open', 'close']].max(axis=1) + np.random.uniform(0, 20, size=num_bars)
#     sample_ohlcv_df['low'] = sample_ohlcv_df[['open', 'close']].min(axis=1) - np.random.uniform(0, 20, size=num_bars)

#     bt_params = {
#         "trading_pair": "ETH/USDT",
#         "nw_timeframe": "15m", # Informational, data is this timeframe
#         "nw_h_bandwidth": 8.0,
#         "nw_env_multiplier": 2.5,
#         "nw_yhat_lookback": 50,
#         "nw_residual_lookback": 30,
#         "nw_env_type": "ATR", # or "StdDev"
#         "trend_linreg_lookback": 60,
#         "trend_ema_slope_period": 25,
#         "order_quantity_usd": 1000.0,
#         "leverage": 5,
#         "use_stop_loss": True, "stop_loss_pct": 1.0,
#         "use_take_profit": True, "take_profit_pct": 2.5,
#         "exit_on_midline_cross": True
#     }

#     strategy = NadarayaWatsonEnvelopeStrategy(db_session=None, user_sub_obj=None, 
#                                             strategy_params=bt_params, 
#                                             exchange_ccxt=None, logger_obj=logger)
    
#     results = strategy.run_backtest(ohlcv_data=sample_ohlcv_df.copy(), 
#                                     initial_capital=10000.0, 
#                                     params=bt_params) # Pass params again to ensure _initialize_parameters uses them

#     logger.info(f"NW Envelope Backtest Results: {results['message']}")
#     logger.info(f"Initial Capital: {results['initial_capital']:.2f}, Final Equity: {results['final_equity']:.2f}")
#     logger.info(f"PNL: {results['pnl']:.2f} ({results['pnl_percentage']:.2f}%)")
#     logger.info(f"Trades: {results['total_trades']} (W: {results['winning_trades']}, L: {results['losing_trades']})")

    # import matplotlib.pyplot as plt
    # eq_df = pd.DataFrame(results['equity_curve']).set_index('timestamp')
    # eq_df.index = pd.to_datetime(eq_df.index)
    # eq_df['equity'].plot(title='NW Envelope Equity Curve')
    # plt.show()

    # trades_df = pd.DataFrame(results['trades_log']).set_index('timestamp')
    # trades_df.index = pd.to_datetime(trades_df.index)
    # print("\nTrades Log:")
    # print(trades_df)

```
