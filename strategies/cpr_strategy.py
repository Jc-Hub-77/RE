# trading_platform/strategies/cpr_strategy.py
import datetime
import time # Keep for _await_order_fill if used locally
import pytz
import numpy as np
import pandas as pd
import ta 
import logging
import json
from sqlalchemy.orm import Session
from backend.models import Position, UserStrategySubscription, Order 
from backend import strategy_utils # Import new strategy_utils
from decimal import Decimal, ROUND_DOWN, ROUND_UP, getcontext
import ccxt # For ccxt.OrderNotFound

logger = logging.getLogger(__name__)
getcontext().prec = 18 # Set precision for Decimal

class CPRStrategy:
    def __init__(self, db_session: Session, user_sub_obj: UserStrategySubscription, strategy_params: dict, exchange_ccxt, logger_obj=None): # Renamed logger
        self.db_session = db_session
        self.user_sub_obj = user_sub_obj
        self.params = strategy_params
        self.exchange_ccxt = exchange_ccxt
        self.logger = logger_obj if logger_obj else logging.getLogger(__name__) # Use passed logger
        self.name = "CPR Strategy"
        
        # Load parameters
        self.symbol = self.params.get("symbol", "BTC/USDT") # Ensure symbol is set from params
        self.timeframe = self.params.get("timeframe", "1h") 
        self.cpr_timeframe = '1d' # Fixed for CPR calculation
        
        # Using self.capital for risk calculation, assuming it's allocated capital for this strategy instance
        self.capital = Decimal(str(self.params.get("capital", "10000"))) # Ensure capital is Decimal
        self.risk_percent = Decimal(str(self.params.get("risk_percent", "1.0"))) / Decimal("100")
        self.leverage = int(self.params.get("leverage", 3))
        self.take_profit_percent = Decimal(str(self.params.get("take_profit_percent", "0.8"))) / Decimal("100")
        self.distance_threshold_percent = Decimal(str(self.params.get("distance_threshold_percent", "0.24"))) / Decimal("100")
        self.max_volatility_threshold_percent = Decimal(str(self.params.get("max_volatility_threshold_percent", "3.48"))) / Decimal("100")
        self.distance_condition_type = self.params.get("distance_condition_type", "Above")
        self.sl_percent_from_entry = Decimal(str(self.params.get("sl_percent_from_entry", "3.5"))) / Decimal("100")
        self.pullback_percent_for_entry = Decimal(str(self.params.get("pullback_percent_for_entry", "0.2"))) / Decimal("100")
        self.s1_bc_dist_thresh_low_percent = Decimal(str(self.params.get("s1_bc_dist_thresh_low_percent", "2.2"))) / Decimal("100")
        self.s1_bc_dist_thresh_high_percent = Decimal(str(self.params.get("s1_bc_dist_thresh_high_percent", "2.85"))) / Decimal("100")
        self.rsi_threshold_entry = float(self.params.get("rsi_threshold_entry", 25.0))
        self.use_prev_day_cpr_tp_filter = self.params.get("use_prev_day_cpr_tp_filter", True)
        self.reduced_tp_percent_if_filter = Decimal(str(self.params.get("reduced_tp_percent_if_filter", "0.2"))) / Decimal("100")
        self.use_monthly_cpr_filter_entry = self.params.get("use_monthly_cpr_filter_entry", True)

        # In-memory state for daily calculated data
        self.daily_cpr: Optional[tuple] = None 
        self.weekly_cpr: Optional[tuple] = None
        self.monthly_cpr: Optional[tuple] = None
        self.daily_indicators: Optional[pd.Series] = None 
        self.today_daily_open_utc: Optional[float] = None # Store as float
        self.data_prepared_for_utc_date: Optional[datetime.date] = None 
        self.monthly_cpr_filter_active: bool = False
        self.last_entry_attempt_utc_time: Optional[datetime.datetime] = None

        # Position and Order State
        self.active_position_db_id: Optional[int] = None
        self.sl_order_db_id: Optional[int] = None
        self.tp_order_db_id: Optional[int] = None
        self.active_sl_exchange_id: Optional[str] = None
        self.active_tp_exchange_id: Optional[str] = None
        # These might be redundant if _load_persistent_position_state directly sets them from Position obj
        # self.active_position_side: Optional[str] = None 
        # self.current_pos_entry_price: Optional[Decimal] = None
        # self.current_pos_qty: Decimal = Decimal("0")


        self.price_precision: Optional[float] = None # Store as float from CCXT
        self.quantity_precision: Optional[float] = None # Store as float from CCXT
        self._precisions_fetched_ = False
 
        self.logger.info(f"Initializing {self.name} for {self.symbol} with UserSubID {self.user_sub_obj.id}...")
        self._fetch_precisions() # Fetch precisions first
        self._load_persistent_position_state() # Load any existing state
        self._set_leverage()
        self.logger.info(f"{self.name} initialized. Active Pos DB ID: {self.active_position_db_id}")


    def _get_init_params_log(self): # Same as original
        return {k:v for k,v in self.__dict__.items() if not k.startswith('_') and k not in ['daily_cpr', 'weekly_cpr', 'monthly_cpr', 'daily_indicators', 'db_session', 'user_sub_obj', 'exchange_ccxt', 'params']}

    @classmethod
    def get_parameters_definition(cls): # Same as original
        return {
            "symbol": {"type": "string", "default": "BTC/USDT", "label": "Trading Symbol"},
            "timeframe": {"type": "timeframe", "default": "1h", "label": "Execution Timeframe"}, # This is execution timeframe
            # cpr_timeframe is fixed to '1d' internally for CPR calcs
            "risk_percent": {"type": "float", "default": 1.0, "label": "Risk % per Trade"},
            "leverage": {"type": "int", "default": 3, "label": "Leverage"},
            "take_profit_percent": {"type": "float", "default": 0.8, "label": "Take Profit %"},
            "distance_threshold_percent": {"type": "float", "default": 0.24, "label": "DailyOpen to BC Distance Threshold %"},
            "max_volatility_threshold_percent": {"type": "float", "default": 3.48, "label": "Max S1-BC Volatility Threshold %"},
            "distance_condition_type": {"type": "select", "default": "Above", "options": ["Above", "Below"], "label": "DailyOpen vs BC Distance Condition"},
            "sl_percent_from_entry": {"type": "float", "default": 3.5, "label": "Stop Loss % from Entry"},
            "pullback_percent_for_entry": {"type": "float", "default": 0.2, "label": "Pullback % for Entry Target"},
            "s1_bc_dist_thresh_low_percent": {"type": "float", "default": 2.2, "label": "S1-BC Dist. Bypass Threshold Low %"},
            "s1_bc_dist_thresh_high_percent": {"type": "float", "default": 2.85, "label": "S1-BC Dist. Bypass Threshold High %"},
            "rsi_threshold_entry": {"type": "float", "default": 25.0, "label": "RSI Entry Threshold (Daily)"},
            "use_prev_day_cpr_tp_filter": {"type": "bool", "default": True, "label": "Use Prev. Day CPR for Reduced TP"},
            "reduced_tp_percent_if_filter": {"type": "float", "default": 0.2, "label": "Reduced TP % if Filter Active"},
            "use_monthly_cpr_filter_entry": {"type": "bool", "default": True, "label": "Use Monthly CPR Entry Filter"}
        }

    def _fetch_precisions(self): # Renamed from _get_precisions
        if not self._precisions_fetched_:
            try:
                self.exchange_ccxt.load_markets(True)
                market = self.exchange_ccxt.market(self.symbol)
                self.price_precision = market['precision']['price'] # Store as float
                self.quantity_precision = market['precision']['amount'] # Store as float
                self._precisions_fetched_ = True
                self.logger.info(f"[{self.name}-{self.symbol}] Precisions: Price={self.price_precision}, Qty={self.quantity_precision}")
            except Exception as e:
                self.logger.error(f"[{self.name}-{self.symbol}] Error fetching precision: {e}", exc_info=True)
                # Keep them as None if fetch fails, formatting functions will handle None
                self.price_precision = None; self.quantity_precision = None


    def _format_price(self, price: Decimal) -> str: # Takes Decimal, returns str
        if self.price_precision is None: self._fetch_precisions() # Ensure precisions are fetched
        if self.price_precision is None: return str(price) # Fallback if still None
        return self.exchange_ccxt.price_to_precision(self.symbol, float(price))

    def _format_quantity(self, quantity: Decimal) -> str: # Takes Decimal, returns str
        if self.quantity_precision is None: self._fetch_precisions()
        if self.quantity_precision is None: return str(quantity) # Fallback
        return self.exchange_ccxt.amount_to_precision(self.symbol, float(quantity))

    def _reset_trade_state(self):
        self.active_position_db_id = None
        # self.active_position_side = None # These are implicitly reset by active_position_db_id being None
        # self.current_pos_entry_price = None
        # self.current_pos_qty = Decimal("0")
        self.sl_order_db_id = None
        self.tp_order_db_id = None
        self.active_sl_exchange_id = None
        self.active_tp_exchange_id = None
        self.logger.info(f"[{self.name}-{self.symbol}] Internal trade state reset.")

    def _load_persistent_position_state(self):
        if not self.db_session or not self.user_sub_obj:
            self.logger.warning(f"[{self.name}-{self.symbol}] DB session/user_sub_obj not available for loading state.")
            self._reset_trade_state()
            return

        open_pos = strategy_utils.get_open_strategy_position(self.db_session, self.user_sub_obj.id, self.symbol)
        if open_pos:
            self.logger.info(f"[{self.name}-{self.symbol}] Loading state for open position ID: {open_pos.id}")
            self.active_position_db_id = open_pos.id
            # The strategy's direct position attributes (side, entry, qty) will be set when needed by fetching Position by ID
            
            if open_pos.custom_data:
                try:
                    state_data = json.loads(open_pos.custom_data)
                    self.active_sl_exchange_id = state_data.get("active_sl_exchange_id")
                    self.active_tp_exchange_id = state_data.get("active_tp_exchange_id")
                    self.sl_order_db_id = state_data.get("sl_order_db_id")
                    self.tp_order_db_id = state_data.get("tp_order_db_id")
                    self.logger.info(f"[{self.name}-{self.symbol}] Loaded custom state: SL ExchID {self.active_sl_exchange_id}, TP ExchID {self.active_tp_exchange_id}")
                except json.JSONDecodeError:
                    self.logger.error(f"[{self.name}-{self.symbol}] Error decoding custom_data for pos {open_pos.id}. Querying open orders.")
                    self._query_and_set_open_sl_tp_orders_from_db()
            else:
                self._query_and_set_open_sl_tp_orders_from_db()
        else:
            self.logger.info(f"[{self.name}-{self.symbol}] No active persistent position found for SubID {self.user_sub_obj.id}.")
            self._reset_trade_state()
            
    def _query_and_set_open_sl_tp_orders_from_db(self): # Helper for loading state
        self.active_sl_exchange_id = None; self.sl_order_db_id = None
        self.active_tp_exchange_id = None; self.tp_order_db_id = None
        # Assuming SL is STOP_MARKET and TP is TAKE_PROFIT_MARKET or LIMIT. Adjust types if different.
        sl_db_orders = strategy_utils.get_open_orders_for_subscription(self.db_session, self.user_sub_obj.id, self.symbol, order_type='stop_market')
        if sl_db_orders: self.active_sl_exchange_id = sl_db_orders[0].order_id; self.sl_order_db_id = sl_db_orders[0].id
        tp_db_orders = strategy_utils.get_open_orders_for_subscription(self.db_session, self.user_sub_obj.id, self.symbol, order_type='take_profit_market') # Or 'limit'
        if not tp_db_orders: tp_db_orders = strategy_utils.get_open_orders_for_subscription(self.db_session, self.user_sub_obj.id, self.symbol, order_type='limit')
        if tp_db_orders: self.active_tp_exchange_id = tp_db_orders[0].order_id; self.tp_order_db_id = tp_db_orders[0].id
        self.logger.info(f"[{self.name}-{self.symbol}] Queried open SL/TP orders. SL ExchID: {self.active_sl_exchange_id}, TP ExchID: {self.active_tp_exchange_id}")


    def _save_position_custom_state(self):
        if not self.active_position_db_id or not self.db_session:
            self.logger.debug(f"[{self.name}-{self.symbol}] No active pos DB ID or session to save custom state.")
            return
        
        pos_to_update = self.db_session.query(Position).filter(Position.id == self.active_position_db_id).first()
        if pos_to_update:
            state_data = {
                "active_sl_exchange_id": self.active_sl_exchange_id,
                "active_tp_exchange_id": self.active_tp_exchange_id,
                "sl_order_db_id": self.sl_order_db_id,
                "tp_order_db_id": self.tp_order_db_id
            }
            pos_to_update.custom_data = json.dumps(state_data)
            pos_to_update.updated_at = datetime.utcnow()
            try:
                self.db_session.commit()
                self.logger.info(f"[{self.name}-{self.symbol}] Saved custom state for PosID {self.active_position_db_id}: {state_data}")
            except Exception as e:
                self.logger.error(f"[{self.name}-{self.symbol}] Error saving custom state for PosID {self.active_position_db_id}: {e}", exc_info=True)
                self.db_session.rollback()
        else:
             self.logger.warning(f"[{self.name}-{self.symbol}] Position {self.active_position_db_id} not found to save custom state.")


    def _set_leverage(self): # Same as original
        try:
            if hasattr(self.exchange_ccxt, 'set_leverage'):
                self.exchange_ccxt.set_leverage(self.leverage, self.symbol)
                self.logger.info(f"Leverage set to {self.leverage}x for {self.symbol}")
        except Exception as e: self.logger.warning(f"Could not set leverage for {self.symbol}: {e}")

    def _calculate_cpr(self, prev_day_high, prev_day_low, prev_day_close): # Same as original
        P = (prev_day_high + prev_day_low + prev_day_close) / 3; TC = (prev_day_high + prev_day_low) / 2 ; BC = (P - TC) + P
        if TC < BC: TC, BC = BC, TC 
        R1 = (P*2)-prev_day_low; S1 = (P*2)-prev_day_high; R2=P+(prev_day_high-prev_day_low); S2=P-(prev_day_high-prev_day_low)
        R3=P+2*(prev_day_high-prev_day_low); S3=P-2*(prev_day_high-prev_day_low); R4=R3+(R2-R1); S4=S3-(S1-S2)
        return P, TC, BC, R1, S1, R2, S2, R3, S3, R4, S4

    def _calculate_indicators(self, df_daily: pd.DataFrame): # Same as original
        if df_daily.empty or len(df_daily) < 50: logger.warning(f"Not enough daily data for indicators."); return None
        indicators = pd.Series(dtype='float64'); price_data = df_daily['close'] 
        indicators['EMA_21'] = ta.trend.EMAIndicator(price_data, window=21).ema_indicator().iloc[-1]
        indicators['EMA_50'] = ta.trend.EMAIndicator(price_data, window=50).ema_indicator().iloc[-1]
        indicators['RSI'] = ta.momentum.RSIIndicator(price_data, window=14).rsi().iloc[-1]
        macd_obj = ta.trend.MACD(price_data, window_fast=12, window_slow=26, window_sign=9)
        if macd_obj is not None: indicators['MACD_Histo']=macd_obj.macd_diff().iloc[-1]; indicators['MACD']=macd_obj.macd().iloc[-1]; indicators['MACD_Signal']=macd_obj.macd_signal().iloc[-1]
        else: indicators['MACD_Histo']=indicators['MACD']=indicators['MACD_Signal']=np.nan
        return indicators.fillna(0)

    def _prepare_daily_data_live(self, exchange_ccxt): # Same as original, ensure self.symbol
        logger.info(f"Preparing daily data for {self.symbol} on {datetime.now(pytz.utc).date()}")
        now_utc = datetime.now(pytz.utc); today_utc_date = now_utc.date()
        try:
            ohlcv_daily = exchange_ccxt.fetch_ohlcv(self.symbol, '1d', limit=60)
            if not ohlcv_daily or len(ohlcv_daily) < 2: logger.warning(f"Not enough daily OHLCV for {self.symbol}."); self.data_prepared_for_utc_date = None; return
            df_daily = pd.DataFrame(ohlcv_daily, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']); df_daily['timestamp'] = pd.to_datetime(df_daily['timestamp'], unit='ms'); df_daily = df_daily.sort_values('timestamp').set_index('timestamp')
            if not df_daily.empty and df_daily.index[-1].date() == today_utc_date: self.today_daily_open_utc = df_daily['open'].iloc[-1]
            else: 
                try: 
                    since_ts = int(datetime(today_utc_date.year,today_utc_date.month,today_utc_date.day,0,0,0,tzinfo=pytz.utc).timestamp()*1000)
                    recent_ohlcv = exchange_ccxt.fetch_ohlcv(self.symbol, '1h', since=since_ts, limit=1)
                    if recent_ohlcv: self.today_daily_open_utc = recent_ohlcv[0][1]
                    elif not df_daily.empty: self.today_daily_open_utc = df_daily['close'].iloc[-1] 
                    else: self.today_daily_open_utc = None; logger.warning(f"Cannot determine today's open for {self.symbol}."); return
                except Exception as e_open: logger.warning(f"Error fetching today's open for {self.symbol}: {e_open}."); self.today_daily_open_utc = df_daily['close'].iloc[-1] if not df_daily.empty else None; if self.today_daily_open_utc is None: return

            prev_day_data = df_daily[df_daily.index.date == (today_utc_date - timedelta(days=1))]
            if prev_day_data.empty: self.daily_cpr = None; logger.warning(f"No prev day data for Daily CPR {self.symbol}.")
            else: self.daily_cpr = self._calculate_cpr(prev_day_data.iloc[-1]['high'], prev_day_data.iloc[-1]['low'], prev_day_data.iloc[-1]['close'])
            self.daily_indicators = self._calculate_indicators(df_daily[df_daily.index.date < today_utc_date])
            ohlcv_w = exchange_ccxt.fetch_ohlcv(self.symbol, '1w', limit=2); self.weekly_cpr = self._calculate_cpr(ohlcv_w[-2][2], ohlcv_w[-2][3], ohlcv_w[-2][4]) if ohlcv_w and len(ohlcv_w)>1 else None
            ohlcv_m = exchange_ccxt.fetch_ohlcv(self.symbol, '1M', limit=2); self.monthly_cpr = self._calculate_cpr(ohlcv_m[-2][2], ohlcv_m[-2][3], ohlcv_m[-2][4]) if ohlcv_m and len(ohlcv_m)>1 else None
            self.monthly_cpr_filter_active = False
            if self.use_monthly_cpr_filter_entry and self.monthly_cpr and self.today_daily_open_utc is not None:
                 mP, mTC, mBC, *_ = self.monthly_cpr
                 if mBC <= self.today_daily_open_utc <= mTC: self.monthly_cpr_filter_active = True; logger.info(f"Monthly CPR filter ACTIVE for {self.symbol}.")
            self.data_prepared_for_utc_date = today_utc_date; logger.info(f"Daily data prepared for {self.symbol} on {self.data_prepared_for_utc_date}.")
        except Exception as e: logger.error(f"Error preparing daily data for {self.symbol}: {e}", exc_info=True); self.data_prepared_for_utc_date = None

    def _await_order_fill(self, exchange_order_id: str): # Uses self.exchange_ccxt, self.symbol
        # Same as TheOrphanStrategy's version
        start_time = time.time()
        self.logger.info(f"[{self.name}-{self.symbol}] Awaiting fill for order {exchange_order_id} (timeout: 60s)")
        while time.time() - start_time < 60: # 60s timeout
            try:
                order = self.exchange_ccxt.fetch_order(exchange_order_id, self.symbol)
                if order['status'] == 'closed': self.logger.info(f"Order {exchange_order_id} filled."); return order
                if order['status'] in ['canceled', 'rejected', 'expired']: self.logger.warning(f"Order {exchange_order_id} is {order['status']}."); return order
            except ccxt.OrderNotFound: self.logger.warning(f"Order {exchange_order_id} not found. Retrying.")
            except Exception as e: self.logger.error(f"Error fetching order {exchange_order_id}: {e}. Retrying.", exc_info=True)
            time.sleep(3) # Check every 3s
        self.logger.warning(f"Timeout for order {exchange_order_id}. Final check.")
        try: return self.exchange_ccxt.fetch_order(exchange_order_id, self.symbol)
        except Exception as e: self.logger.error(f"Final check for order {exchange_order_id} failed: {e}", exc_info=True); return None
    
    # --- Refactored Trade Execution using strategy_utils ---
    def _open_position_live(self, side: str, current_market_price: float): # Renamed and simplified
        self.logger.info(f"[{self.name}-{self.symbol}] Attempting to open {side.upper()} for SubID {self.user_sub_obj.id} near {current_market_price}")
        entry_price_decimal = Decimal(str(current_market_price))
        
        risk_amount_usd = self.capital * self.risk_percent # self.capital is the strategy instance's allocated capital
        sl_distance_price_decimal = entry_price_decimal * self.sl_percent_from_entry
        stop_loss_price_decimal = entry_price_decimal - sl_distance_price_decimal if side == 'long' else entry_price_decimal + sl_distance_price_decimal
        
        if sl_distance_price_decimal == Decimal("0"): self.logger.warning(f"SL distance is zero."); return
        # Position size based on risk and SL distance (notional value)
        position_notional_usd = risk_amount_usd / self.sl_percent_from_entry 
        position_size_asset = position_notional_usd / entry_price_decimal
        
        if position_size_asset <= Decimal("0"): self.logger.warning(f"Calculated position size zero or negative. Skipping."); return

        db_entry_order = strategy_utils.create_strategy_order_in_db(self.db_session, self.user_sub_obj.id, self.symbol, 'market', side, float(position_size_asset), price=float(entry_price_decimal), notes="CPR Entry")
        if not db_entry_order: return

        try:
            order_receipt = self.exchange_ccxt.create_market_order(self.symbol, side, float(self._format_quantity(position_size_asset))) # Use formatted qty
            strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=db_entry_order.id, updates={'order_id': order_receipt['id'], 'status': 'open', 'raw_order_data': json.dumps(order_receipt)})
            
            filled_order = self._await_order_fill(order_receipt['id'])
            if not (filled_order and filled_order['status'] == 'closed'):
                strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=db_entry_order.id, updates={'status': filled_order.get('status', 'fill_check_failed') if filled_order else 'fill_check_failed'})
                self.logger.error(f"Market entry order {order_receipt['id']} failed/timed out."); return

            actual_filled_price = Decimal(str(filled_order['average'])); actual_filled_qty = Decimal(str(filled_order['filled']))
            strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=db_entry_order.id, updates={'status': 'closed', 'price': float(actual_filled_price), 'filled': float(actual_filled_qty), 'cost': float(filled_order.get('cost', actual_filled_price * actual_filled_qty)), 'closed_at': datetime.utcnow()})

            if actual_filled_qty <= Decimal("0"): self.logger.warning("Filled zero quantity. Skipping position."); return

            new_pos_db = strategy_utils.create_strategy_position_in_db(self.db_session, self.user_sub_obj.id, self.symbol, str(self.exchange_ccxt.id), side, float(actual_filled_qty), float(actual_filled_price))
            if not new_pos_db: self.logger.error("Failed to create Position DB record."); return
            
            self.active_position_db_id = new_pos_db.id
            # self.active_position_side = side # These are set by _load_persistent_position_state by reading Position table
            # self.current_pos_entry_price = actual_filled_price
            # self.current_pos_qty = actual_filled_qty
            self._load_persistent_position_state() # Reload to get all attributes set from new_pos_db

            sl_tp_qty_str = self._format_quantity(actual_filled_qty)
            tp_price_decimal = actual_filled_price * (Decimal('1') + self.take_profit_percent) if side == 'long' else actual_filled_price * (Decimal('1') - self.take_profit_percent)
            
            # Adjust TP if prev day CPR filter is active
            if self.use_prev_day_cpr_tp_filter and self.daily_cpr:
                prev_P, prev_TC, prev_BC, *_ = self.daily_cpr
                if side == 'long' and actual_filled_price < prev_P and tp_price_decimal > prev_P : # Long below Pivot, TP above Pivot
                    tp_price_decimal = actual_filled_price * (Decimal('1') + self.reduced_tp_percent_if_filter)
                    self.logger.info(f"Reduced TP for LONG due to prev day CPR filter. New TP: {tp_price_decimal}")
                elif side == 'short' and actual_filled_price > prev_P and tp_price_decimal < prev_P: # Short above Pivot, TP below Pivot
                    tp_price_decimal = actual_filled_price * (Decimal('1') - self.reduced_tp_percent_if_filter)
                    self.logger.info(f"Reduced TP for SHORT due to prev day CPR filter. New TP: {tp_price_decimal}")


            sl_order_type = 'stop_market'; tp_order_type = 'take_profit_market' # Or 'limit' for TP
            sl_exch_side = 'sell' if side == 'long' else 'buy'; tp_exch_side = sl_exch_side
            
            sl_db = strategy_utils.create_strategy_order_in_db(self.db_session, self.user_sub_obj.id, self.symbol, sl_order_type, sl_exch_side, float(sl_tp_qty_str), float(self._format_price(stop_loss_price_decimal)), notes="CPR SL")
            if sl_db:
                try:
                    sl_receipt = self.exchange_ccxt.create_order(self.symbol, sl_order_type, sl_exch_side, float(sl_tp_qty_str), params={'stopPrice': self._format_price(stop_loss_price_decimal), 'reduceOnly': True})
                    strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=sl_db.id, updates={'order_id': sl_receipt['id'], 'status': 'open', 'raw_order_data': json.dumps(sl_receipt)})
                    self.sl_order_db_id = sl_db.id; self.active_sl_exchange_id = sl_receipt['id']
                except Exception as e_sl: self.logger.error(f"Failed to place SL: {e_sl}"); strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=sl_db.id, updates={'status':'error'})
            
            tp_db = strategy_utils.create_strategy_order_in_db(self.db_session, self.user_sub_obj.id, self.symbol, tp_order_type, tp_exch_side, float(sl_tp_qty_str), float(self._format_price(tp_price_decimal)), notes="CPR TP")
            if tp_db:
                try: # TAKE_PROFIT_MARKET often needs stopPrice, LIMIT needs price
                    tp_params = {'stopPrice': self._format_price(tp_price_decimal), 'reduceOnly': True} if tp_order_type == 'take_profit_market' else {'reduceOnly': True}
                    tp_price_for_order = None if tp_order_type == 'take_profit_market' else float(self._format_price(tp_price_decimal))
                    tp_receipt = self.exchange_ccxt.create_order(self.symbol, tp_order_type, tp_exch_side, float(sl_tp_qty_str), tp_price_for_order, tp_params)
                    strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=tp_db.id, updates={'order_id': tp_receipt['id'], 'status': 'open', 'raw_order_data': json.dumps(tp_receipt)})
                    self.tp_order_db_id = tp_db.id; self.active_tp_exchange_id = tp_receipt['id']
                except Exception as e_tp: self.logger.error(f"Failed to place TP: {e_tp}"); strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=tp_db.id, updates={'status':'error'})
            
            self._save_position_custom_state() # Save SL/TP IDs to Position.custom_data
        except Exception as e: self.logger.error(f"Unexpected error opening {side.upper()} position: {e}", exc_info=True)


    def _close_position_live(self, reason: str, exchange_ccxt, closing_trigger_order: Optional[dict] = None):
        if not self.active_position_db_id: self.logger.warning("Close called but no active_position_db_id."); return
        
        pos_to_close = self.db_session.query(Position).filter(Position.id == self.active_position_db_id).first()
        if not pos_to_close or not pos_to_close.is_open: self.logger.warning(f"Position {self.active_position_db_id} not found or already closed."); self._reset_trade_state(); return
        
        self.logger.info(f"Attempting to close PosID {pos_to_close.id} due to: {reason}")

        # Cancel open SL/TP orders
        if self.active_sl_exchange_id and (not closing_trigger_order or closing_trigger_order['id'] != self.active_sl_exchange_id):
            try: exchange_ccxt.cancel_order(self.active_sl_exchange_id, self.symbol); strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=self.sl_order_db_id, updates={'status':'canceled', 'status_message':f'Canceled, pos closed by {reason}'})
            except Exception as e: self.logger.warning(f"Could not cancel SL {self.active_sl_exchange_id}: {e}")
        if self.active_tp_exchange_id and (not closing_trigger_order or closing_trigger_order['id'] != self.active_tp_exchange_id):
            try: exchange_ccxt.cancel_order(self.active_tp_exchange_id, self.symbol); strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=self.tp_order_db_id, updates={'status':'canceled', 'status_message':f'Canceled, pos closed by {reason}'})
            except Exception as e: self.logger.warning(f"Could not cancel TP {self.active_tp_exchange_id}: {e}")

        actual_close_price = None; actual_closed_quantity = pos_to_close.amount
        if closing_trigger_order: # Closed by SL/TP fill
            actual_close_price = Decimal(str(closing_trigger_order.get('average', pos_to_close.current_price)))
            actual_closed_quantity = Decimal(str(closing_trigger_order.get('filled', pos_to_close.amount)))
        else: # Closing with a new market order
            side_to_close = 'sell' if pos_to_close.side == 'long' else 'buy'
            qty_str = self._format_quantity(pos_to_close.amount)
            db_market_close_order = strategy_utils.create_strategy_order_in_db(self.db_session, self.user_sub_obj.id, self.symbol, 'market', side_to_close, float(qty_str), notes=f"CPR Close: {reason}")
            if db_market_close_order:
                try:
                    close_receipt = exchange_ccxt.create_market_order(self.symbol, side_to_close, float(qty_str), params={'reduceOnly':True})
                    strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=db_market_close_order.id, updates={'order_id':close_receipt['id'], 'status':'open'})
                    filled_details = self._await_order_fill(close_receipt['id'])
                    if filled_details and filled_details['status'] == 'closed':
                        actual_close_price = Decimal(str(filled_details['average'])); actual_closed_quantity = Decimal(str(filled_details['filled']))
                        strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=db_market_close_order.id, updates={'status':'closed', 'price':float(actual_close_price), 'filled':float(actual_closed_quantity), 'closed_at':datetime.utcnow()})
                    else: self.logger.error(f"Market close order {close_receipt['id']} failed to fill for Pos {pos_to_close.id}.")
                except Exception as e: self.logger.error(f"Error placing market close for Pos {pos_to_close.id}: {e}"); strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=db_market_close_order.id, updates={'status':'error'})
        
        if actual_close_price is not None:
            strategy_utils.close_strategy_position_in_db(self.db_session, pos_to_close.id, actual_close_price, actual_closed_quantity, reason)
        else: # Fallback if close price couldn't be determined (e.g. market order failed critically)
            pos_to_close.status_message = f"Close attempt failed: {reason}. Manual review needed."; pos_to_close.updated_at = datetime.utcnow(); self.db_session.commit()
            self.logger.error(f"Position {pos_to_close.id} close attempt failed to confirm fill price. Marked with error.")

        self._reset_trade_state()
        self._save_position_custom_state() # Save cleared state to custom_data of closed position

    def _sync_exchange_position_state(self, exchange_ccxt):
        if not self.active_position_db_id: return False
        
        pos_closed_by_sync = False
        if self.active_sl_exchange_id and self.sl_order_db_id:
            try:
                sl_details = exchange_ccxt.fetch_order(self.active_sl_exchange_id, self.symbol)
                if sl_details['status'] == 'closed':
                    self.logger.info(f"SL order {self.active_sl_exchange_id} found filled during sync.")
                    strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=self.sl_order_db_id, updates={'status':'closed', 'price':sl_details.get('average'), 'filled':sl_details.get('filled'), 'closed_at':datetime.utcnow(), 'raw_order_data':json.dumps(sl_details)})
                    self._close_position_live("SL Hit (synced)", exchange_ccxt, closing_trigger_order=sl_details)
                    pos_closed_by_sync = True
            except ccxt.OrderNotFound: strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=self.sl_order_db_id, updates={'status':'not_found_on_sync'}); self.active_sl_exchange_id=None; self.sl_order_db_id=None; self._save_position_custom_state()
            except Exception as e: self.logger.error(f"Error syncing SL {self.active_sl_exchange_id}: {e}")

        if not pos_closed_by_sync and self.active_tp_exchange_id and self.tp_order_db_id:
            try:
                tp_details = exchange_ccxt.fetch_order(self.active_tp_exchange_id, self.symbol)
                if tp_details['status'] == 'closed':
                    self.logger.info(f"TP order {self.active_tp_exchange_id} found filled during sync.")
                    strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=self.tp_order_db_id, updates={'status':'closed', 'price':tp_details.get('average'), 'filled':tp_details.get('filled'), 'closed_at':datetime.utcnow(), 'raw_order_data':json.dumps(tp_details)})
                    self._close_position_live("TP Hit (synced)", exchange_ccxt, closing_trigger_order=tp_details)
                    pos_closed_by_sync = True
            except ccxt.OrderNotFound: strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=self.tp_order_db_id, updates={'status':'not_found_on_sync'}); self.active_tp_exchange_id=None; self.tp_order_db_id=None; self._save_position_custom_state()
            except Exception as e: self.logger.error(f"Error syncing TP {self.active_tp_exchange_id}: {e}")
        
        return pos_closed_by_sync

    def _check_entry_conditions_live(self, exchange_ccxt): # Removed db_session, subscription_id
        # Uses self.daily_cpr, self.daily_indicators, self.today_daily_open_utc, self.symbol
        # Calls self._open_position_live
        # ... (Original condition logic) ...
        # If conditions met: self._open_position_live(side, current_price, exchange_ccxt)
        # For brevity, not fully reproducing the complex condition logic here, assuming it's adapted to use instance vars.
        if self.daily_cpr is None or self.daily_indicators is None or self.today_daily_open_utc is None: return
        P, TC, BC, R1, S1, *_ = self.daily_cpr # Extract relevant CPR levels
        daily_open = self.today_daily_open_utc
        rsi_daily = self.daily_indicators.get('RSI', np.nan)
        if np.isnan(rsi_daily): return
        bc_dist_pct = abs(daily_open - BC) / BC if BC != 0 else float('inf')
        dist_cond_met = (self.distance_condition_type == "Above" and daily_open > BC and bc_dist_pct >= self.distance_threshold_percent) or \
                        (self.distance_condition_type == "Below" and daily_open < BC and bc_dist_pct >= self.distance_threshold_percent)
        if not dist_cond_met: return
        s1_bc_dist_pct = abs(S1 - BC) / BC if BC != 0 else float('inf')
        if not (self.s1_bc_dist_thresh_low_percent <= s1_bc_dist_pct <= self.s1_bc_dist_thresh_high_percent): return
        if rsi_daily > self.rsi_threshold_entry: return
        if self.use_monthly_cpr_filter_entry and self.monthly_cpr_filter_active: return
        
        try: current_price = float(exchange_ccxt.fetch_ticker(self.symbol)['last'])
        except Exception as e: logger.error(f"Ticker fetch error for entry: {e}"); return

        target_entry = 0.0
        if self.distance_condition_type == "Above": target_entry = daily_open * (1 - float(self.pullback_percent_for_entry))
        else: target_entry = daily_open * (1 + float(self.pullback_percent_for_entry))
        
        if (self.distance_condition_type == "Above" and current_price <= target_entry) or \
           (self.distance_condition_type == "Below" and current_price >= target_entry):
            self.logger.info(f"Entry conditions met. Price ({current_price}) vs target ({target_entry}). Opening LONG.") # CPR is typically long only
            self._open_position_live('long', current_price) # exchange_ccxt passed via self


    def _check_exit_conditions_live(self, exchange_ccxt): # Removed db_session, subscription_id, current_position_db
        # Uses self.active_position_db_id, self.daily_cpr
        # Calls self._close_position_live
        if not self.active_position_db_id: return # No active position to check exit for
        # SL/TP hits are handled by _sync_exchange_position_state
        # This method handles discretionary exits based on CPR levels or EOD
        # ... (Original condition logic) ...
        # If conditions met: self._close_position_live("Reason", exchange_ccxt)
        if self.daily_cpr is None: return
        P, TC, BC, *_ = self.daily_cpr
        try: current_price = float(exchange_ccxt.fetch_ticker(self.symbol)['last'])
        except Exception as e: logger.error(f"Ticker fetch error for exit: {e}"); return
        
        # Example: Close long if price drops below BC (central pivot)
        # Note: self.active_position_side is loaded by _load_persistent_position_state
        current_pos = strategy_utils.get_open_strategy_position(self.db_session, self.user_sub_obj.id, self.symbol) # Get fresh side
        if current_pos and current_pos.side == "long" and current_price <= BC:
            self.logger.info(f"Price ({current_price}) hit BC ({BC}). Closing LONG PosID {self.active_position_db_id}.")
            self._close_position_live("BC Hit", exchange_ccxt); return

        now_utc = datetime.now(pytz.utc)
        if now_utc.hour == 23 and now_utc.minute >= 55: # EOD
             self.logger.info(f"End of day. Closing PosID {self.active_position_db_id}.")
             self._close_position_live("End of Day", exchange_ccxt); return


    def execute_live_signal(self, exchange_ccxt_in=None): # Optional exchange_ccxt_in for flexibility
        current_exchange_ccxt = exchange_ccxt_in if exchange_ccxt_in else self.exchange_ccxt
        if not current_exchange_ccxt:
            self.logger.error(f"[{self.name}-{self.symbol}] Exchange instance not available for SubID {self.user_sub_obj.id}."); return
        
        self.logger.debug(f"[{self.name}-{self.symbol}] Executing live signal for SubID {self.user_sub_obj.id}...")
        # self._load_persistent_position_state() # Called in __init__, consider if refresh needed per cycle

        if self._sync_exchange_position_state(current_exchange_ccxt):
            self.logger.info(f"[{self.name}-{self.symbol}] Position closed by SL/TP sync. Cycle ended.")
            return

        now_utc = datetime.now(pytz.utc)
        if self.data_prepared_for_utc_date != now_utc.date():
            if now_utc.hour == 0 and now_utc.minute < 15: self._prepare_daily_data_live(current_exchange_ccxt)
            elif self.data_prepared_for_utc_date is None : self._prepare_daily_data_live(current_exchange_ccxt)
        
        if self.data_prepared_for_utc_date == now_utc.date():
            if not self.active_position_db_id: # No active position
                if now_utc.hour == 0 and now_utc.minute < 10: # Entry window
                     if self.last_entry_attempt_utc_time is None or (now_utc - self.last_entry_attempt_utc_time).total_seconds() > 300: 
                         self._check_entry_conditions_live(current_exchange_ccxt)
                         self.last_entry_attempt_utc_time = now_utc
                     else: self.logger.debug(f"In entry cooldown.")
                else: self.logger.debug(f"Not within entry window (00:00-00:10 UTC).")
            else: # Active position exists
                self._check_exit_conditions_live(current_exchange_ccxt)
        else:
            self.logger.debug(f"Daily data for {now_utc.date()} not yet prepared. Current: {self.data_prepared_for_utc_date}")
        self.logger.debug(f"Live signal execution cycle finished for SubID {self.user_sub_obj.id}.")

    def run_backtest(self, historical_data_feed, current_simulated_time_utc):
        self.logger.warning(f"[{self.name}-{self.symbol}] Backtesting for CPR strategy is complex. This is a simplified conceptual outline.")
        return {"pnl": 0, "trades": [], "message": "CPR backtesting not fully implemented in this refactor."}

```
