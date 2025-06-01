# trading_platform/strategies/cpr_strategy.py
import datetime
import time
import pytz
import numpy as np
import pandas as pd
import ta # For indicators
import logging
import json
from sqlalchemy.orm import Session
from backend.models import Position, UserStrategySubscription, Order 
# import ccxt # Handled by runner, passed as exchange_ccxt

logger = logging.getLogger(__name__)

class CPRStrategy:
    def __init__(self, symbol: str, timeframe: str, capital: float = 10000, 
                 cpr_timeframe: str = '1d', 
                 risk_percent: float = 1.0,
                 leverage: int = 3,
                 take_profit_percent: float = 0.8,
                 distance_threshold_percent: float = 0.24, 
                 max_volatility_threshold_percent: float = 3.48, 
                 distance_condition_type: str = "Above", 
                 sl_percent_from_entry: float = 3.5, 
                 pullback_percent_for_entry: float = 0.2, 
                 s1_bc_dist_thresh_low_percent: float = 2.2, 
                 s1_bc_dist_thresh_high_percent: float = 2.85, 
                 rsi_threshold_entry: float = 25.0, 
                 use_prev_day_cpr_tp_filter: bool = True,
                 reduced_tp_percent_if_filter: float = 0.2, 
                 use_monthly_cpr_filter_entry: bool = True 
                 ):
        self.name = "CPR Strategy"
        self.symbol = symbol
        self.timeframe = timeframe 
        self.cpr_timeframe = '1d' 
        
        self.capital = capital
        self.risk_percent = risk_percent / 100.0 
        self.leverage = leverage
        self.take_profit_percent = take_profit_percent / 100.0
        self.distance_threshold_percent = distance_threshold_percent / 100.0
        self.max_volatility_threshold_percent = max_volatility_threshold_percent / 100.0
        self.distance_condition_type = distance_condition_type
        self.sl_percent_from_entry = sl_percent_from_entry / 100.0
        self.pullback_percent_for_entry = pullback_percent_for_entry / 100.0
        self.s1_bc_dist_thresh_low_percent = s1_bc_dist_thresh_low_percent / 100.0
        self.s1_bc_dist_thresh_high_percent = s1_bc_dist_thresh_high_percent / 100.0
        self.rsi_threshold_entry = rsi_threshold_entry
        self.use_prev_day_cpr_tp_filter = use_prev_day_cpr_tp_filter
        self.reduced_tp_percent_if_filter = reduced_tp_percent_if_filter / 100.0
        self.use_monthly_cpr_filter_entry = use_monthly_cpr_filter_entry

        # In-memory state for daily calculated data (refreshed daily)
        self.daily_cpr = None 
        self.weekly_cpr = None
        self.monthly_cpr = None
        self.daily_indicators = None 
        self.today_daily_open_utc = None 
        self.data_prepared_for_utc_date = None 
        self.monthly_cpr_filter_active = False
        self.last_entry_attempt_utc_time = None # Cooldown for entry attempts

        # Exchange specific, fetched once
        self.price_precision = 8 
        self.quantity_precision = 8 
        self._precisions_fetched_ = False
 
        logger.info(f"Initialized {self.name} for {self.symbol} with parameters: {self._get_init_params_log()}")

    def _get_init_params_log(self):
        # Helper to log constructor parameters without sensitive ones
        return {k:v for k,v in self.__dict__.items() if not k.startswith('_') and k not in ['daily_cpr', 'weekly_cpr', 'monthly_cpr', 'daily_indicators']}

    @classmethod
    def get_parameters_definition(cls):
        return {
            "symbol": {"type": "string", "default": "BTC/USDT", "label": "Trading Symbol"},
            "timeframe": {"type": "timeframe", "default": "1h", "label": "Execution Timeframe"},
            "risk_percent": {"type": "float", "default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1, "label": "Risk % per Trade"},
            "leverage": {"type": "int", "default": 3, "min": 1, "max": 20, "label": "Leverage"},
            "take_profit_percent": {"type": "float", "default": 0.8, "min": 0.1, "step": 0.1, "label": "Take Profit %"},
            "distance_threshold_percent": {"type": "float", "default": 0.24, "min": 0.01, "step": 0.01, "label": "DailyOpen to BC Distance Threshold %"},
            "max_volatility_threshold_percent": {"type": "float", "default": 3.48, "min": 0.1, "step": 0.1, "label": "Max S1-BC Volatility Threshold %"},
            "distance_condition_type": {"type": "select", "default": "Above", "options": ["Above", "Below"], "label": "DailyOpen vs BC Distance Condition"},
            "sl_percent_from_entry": {"type": "float", "default": 3.5, "min": 0.1, "step": 0.1, "label": "Stop Loss % from Entry"},
            "pullback_percent_for_entry": {"type": "float", "default": 0.2, "min": 0.0, "step": 0.01, "label": "Pullback % for Entry Target"},
            "s1_bc_dist_thresh_low_percent": {"type": "float", "default": 2.2, "min": 0.1, "step": 0.1, "label": "S1-BC Dist. Bypass Threshold Low %"},
            "s1_bc_dist_thresh_high_percent": {"type": "float", "default": 2.85, "min": 0.1, "step": 0.1, "label": "S1-BC Dist. Bypass Threshold High %"},
            "rsi_threshold_entry": {"type": "float", "default": 25.0, "min": 0, "max": 100, "label": "RSI Entry Threshold (Daily)"},
            "use_prev_day_cpr_tp_filter": {"type": "bool", "default": True, "label": "Use Prev. Day CPR for Reduced TP"},
            "reduced_tp_percent_if_filter": {"type": "float", "default": 0.2, "min": 0.05, "step": 0.01, "label": "Reduced TP % if Filter Active"},
            "use_monthly_cpr_filter_entry": {"type": "bool", "default": True, "label": "Use Monthly CPR Entry Filter"}
        }

    def _get_precisions(self, exchange_ccxt):
        if not self._precisions_fetched_: # Use the flag correctly
            try:
                exchange_ccxt.load_markets(True) # Force reload if needed
                market = exchange_ccxt.market(self.symbol)
                self.price_precision = market['precision']['price']
                self.quantity_precision = market['precision']['amount']
                self._precisions_fetched_ = True
                logger.info(f"[{self.name}-{self.symbol}] Precisions: Price={self.price_precision}, Qty={self.quantity_precision}")
            except Exception as e:
                logger.error(f"[{self.name}-{self.symbol}] Error fetching precision: {e}", exc_info=True)

    def _format_price(self, price, exchange_ccxt):
        self._get_precisions(exchange_ccxt)
        return float(exchange_ccxt.price_to_precision(self.symbol, price))

    def _format_quantity(self, quantity, exchange_ccxt):
        self._get_precisions(exchange_ccxt)
        return float(exchange_ccxt.amount_to_precision(self.symbol, quantity))

    def _calculate_cpr(self, prev_day_high, prev_day_low, prev_day_close):
        P = (prev_day_high + prev_day_low + prev_day_close) / 3
        TC = (prev_day_high + prev_day_low) / 2 
        BC = (P - TC) + P
        if TC < BC: TC, BC = BC, TC 
        R1 = (P * 2) - prev_day_low
        S1 = (P * 2) - prev_day_high
        R2 = P + (prev_day_high - prev_day_low)
        S2 = P - (prev_day_high - prev_day_low)
        R3 = P + 2 * (prev_day_high - prev_day_low) 
        S3 = P - 2 * (prev_day_high - prev_day_low) 
        R4 = R3 + (R2 - R1)
        S4 = S3 - (S1 - S2)
        return P, TC, BC, R1, S1, R2, S2, R3, S3, R4, S4

    def _calculate_indicators(self, df_daily: pd.DataFrame):
        if df_daily.empty or len(df_daily) < 50: 
            logger.warning(f"[{self.name}-{self.symbol}] Not enough daily data to calculate all indicators (need 50, got {len(df_daily)}).")
            return None
        indicators = pd.Series(dtype='float64')
        price_data = df_daily['close'] 
        indicators['EMA_21'] = ta.trend.EMAIndicator(price_data, window=21).ema_indicator().iloc[-1]
        indicators['EMA_50'] = ta.trend.EMAIndicator(price_data, window=50).ema_indicator().iloc[-1]
        indicators['RSI'] = ta.momentum.RSIIndicator(price_data, window=14).rsi().iloc[-1]
        macd_obj = ta.trend.MACD(price_data, window_fast=12, window_slow=26, window_sign=9)
        if macd_obj is not None:
             indicators['MACD_Histo'] = macd_obj.macd_diff().iloc[-1]
             indicators['MACD'] = macd_obj.macd().iloc[-1]      
             indicators['MACD_Signal'] = macd_obj.macd_signal().iloc[-1]
        else:
            indicators['MACD_Histo'] = indicators['MACD'] = indicators['MACD_Signal'] = np.nan
        return indicators.fillna(0) # Consider if 0 is appropriate for NaN EMAs/MACD

    def _prepare_daily_data_live(self, exchange_ccxt):
        logger.info(f"[{self.name}-{self.symbol}] Preparing daily data (CPR, indicators) for {datetime.datetime.now(pytz.utc).date()}")
        now_utc = datetime.datetime.now(pytz.utc)
        today_utc_date = now_utc.date()
        
        try:
            ohlcv_daily = exchange_ccxt.fetch_ohlcv(self.symbol, '1d', limit=60) # Fetch 60 days for indicators
            if not ohlcv_daily or len(ohlcv_daily) < 2:
                logger.warning(f"[{self.name}-{self.symbol}] Not enough daily OHLCV data fetched ({len(ohlcv_daily) if ohlcv_daily else 0} candles).")
                self.data_prepared_for_utc_date = None
                return

            df_daily = pd.DataFrame(ohlcv_daily, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_daily['timestamp'] = pd.to_datetime(df_daily['timestamp'], unit='ms')
            df_daily = df_daily.sort_values('timestamp').set_index('timestamp')
            
            if not df_daily.empty and df_daily.index[-1].date() == today_utc_date:
                self.today_daily_open_utc = df_daily['open'].iloc[-1]
            else: 
                try: 
                    since_timestamp = int(datetime.datetime(today_utc_date.year, today_utc_date.month, today_utc_date.day, 0, 0, 0, tzinfo=pytz.utc).timestamp()*1000)
                    recent_ohlcv = exchange_ccxt.fetch_ohlcv(self.symbol, '1h', since=since_timestamp, limit=1)
                    if recent_ohlcv: self.today_daily_open_utc = recent_ohlcv[0][1]
                    elif not df_daily.empty: self.today_daily_open_utc = df_daily['close'].iloc[-1] 
                    else: self.today_daily_open_utc = None; logger.warning(f"[{self.name}-{self.symbol}] Could not determine today's open price."); return
                except Exception as e_open: 
                    logger.warning(f"[{self.name}-{self.symbol}] Error fetching today's open price: {e_open}. Using last known close.")
                    self.today_daily_open_utc = df_daily['close'].iloc[-1] if not df_daily.empty else None
                    if self.today_daily_open_utc is None: logger.error(f"[{self.name}-{self.symbol}] Critical: Cannot determine today's open price."); return


            prev_day_data_for_cpr = df_daily[df_daily.index.date == (today_utc_date - datetime.timedelta(days=1))]
            if prev_day_data_for_cpr.empty:
                logger.warning(f"[{self.name}-{self.symbol}] Previous day's data not found for Daily CPR calculation.")
                self.daily_cpr = None
            else:
                prev_day_candle = prev_day_data_for_cpr.iloc[-1]
                self.daily_cpr = self._calculate_cpr(prev_day_candle['high'], prev_day_candle['low'], prev_day_candle['close'])

            self.daily_indicators = self._calculate_indicators(df_daily[df_daily.index.date < today_utc_date]) 

            ohlcv_weekly = exchange_ccxt.fetch_ohlcv(self.symbol, '1w', limit=2) 
            if ohlcv_weekly and len(ohlcv_weekly) > 1:
                 self.weekly_cpr = self._calculate_cpr(ohlcv_weekly[-2][2], ohlcv_weekly[-2][3], ohlcv_weekly[-2][4]) 
            else: self.weekly_cpr = None; logger.warning(f"[{self.name}-{self.symbol}] Not enough weekly data for CPR.")

            ohlcv_monthly = exchange_ccxt.fetch_ohlcv(self.symbol, '1M', limit=2) 
            if ohlcv_monthly and len(ohlcv_monthly) > 1:
                 self.monthly_cpr = self._calculate_cpr(ohlcv_monthly[-2][2], ohlcv_monthly[-2][3], ohlcv_monthly[-2][4])
            else: self.monthly_cpr = None; logger.warning(f"[{self.name}-{self.symbol}] Not enough monthly data for CPR.")
            
            self.monthly_cpr_filter_active = False
            if self.use_monthly_cpr_filter_entry and self.monthly_cpr and self.today_daily_open_utc is not None:
                 monthly_P, monthly_TC, monthly_BC, *_ = self.monthly_cpr
                 if monthly_BC <= self.today_daily_open_utc <= monthly_TC:
                      self.monthly_cpr_filter_active = True
                      logger.info(f"[{self.name}-{self.symbol}] Monthly CPR filter is ACTIVE.")

            self.data_prepared_for_utc_date = today_utc_date
            logger.info(f"[{self.name}-{self.symbol}] Daily data prepared for {self.data_prepared_for_utc_date}. Today's Open: {self.today_daily_open_utc}")
            logger.debug(f"[{self.name}-{self.symbol}] Daily CPR: {self.daily_cpr}")
            logger.debug(f"[{self.name}-{self.symbol}] Daily Indicators: {self.daily_indicators.to_dict() if self.daily_indicators is not None else 'None'}")

        except Exception as e:
            logger.error(f"[{self.name}-{self.symbol}] Error preparing daily data: {e}", exc_info=True)
            self.data_prepared_for_utc_date = None


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
            except ccxt.OrderNotFound:
                logger.warning(f"[{self.name}-{self.symbol}] Order {order_id} not found via fetch_order. May have filled quickly or error. Retrying.")
            except Exception as e:
                logger.error(f"[{self.name}-{self.symbol}] Error fetching order {order_id}: {e}. Retrying.", exc_info=True)
            
            time.sleep(check_interval_seconds)
        
        logger.warning(f"[{self.name}-{self.symbol}] Timeout waiting for order {order_id} to fill. Performing final check.")
        try: 
            final_order_status = exchange_ccxt.fetch_order(order_id, symbol)
            logger.info(f"[{self.name}-{self.symbol}] Final status for order {order_id}: {final_order_status['status']}")
            return final_order_status
        except Exception as e:
            logger.error(f"[{self.name}-{self.symbol}] Final check for order {order_id} also failed: {e}", exc_info=True)
            return None

    def _check_entry_conditions_live(self, db_session: Session, subscription_id: int, exchange_ccxt):
        logger.info(f"[{self.name}-{self.symbol}] Checking entry conditions for sub ID {subscription_id}.")
        if self.daily_cpr is None or self.daily_indicators is None or self.today_daily_open_utc is None:
            logger.warning(f"[{self.name}-{self.symbol}] Daily data not prepared. Skipping entry check."); return

        P, TC, BC, R1, S1, R2, S2, R3, S3, R4, S4 = self.daily_cpr
        daily_open = self.today_daily_open_utc
        rsi_daily = self.daily_indicators.get('RSI', np.nan) if self.daily_indicators is not None else np.nan

        if np.isnan(rsi_daily): logger.warning(f"[{self.name}-{self.symbol}] Daily RSI not available. Skipping."); return

        bc_distance_percent = abs(daily_open - BC) / BC * 100.0 if BC != 0 else float('inf')
        distance_condition_met = False
        if self.distance_condition_type == "Above" and daily_open > BC and bc_distance_percent >= self.distance_threshold_percent * 100: distance_condition_met = True
        elif self.distance_condition_type == "Below" and daily_open < BC and bc_distance_percent >= self.distance_threshold_percent * 100: distance_condition_met = True

        if not distance_condition_met: return logger.debug(f"[{self.name}-{self.symbol}] Entry Fail: DailyOpen ({daily_open}) vs BC ({BC}) dist ({bc_distance_percent:.2f}%) invalid.")
        
        s1_bc_distance_percent = abs(S1 - BC) / BC * 100.0 if BC != 0 else float('inf')
        if not (self.s1_bc_dist_thresh_low_percent * 100 <= s1_bc_distance_percent <= self.s1_bc_dist_thresh_high_percent * 100):
            return logger.debug(f"[{self.name}-{self.symbol}] Entry Fail: S1-BC dist ({s1_bc_distance_percent:.2f}%) out of range.")

        if rsi_daily > self.rsi_threshold_entry: return logger.debug(f"[{self.name}-{self.symbol}] Entry Fail: Daily RSI ({rsi_daily:.2f}) > threshold ({self.rsi_threshold_entry:.2f}).")
        if self.use_monthly_cpr_filter_entry and self.monthly_cpr_filter_active: return logger.debug(f"[{self.name}-{self.symbol}] Entry Fail: Monthly CPR filter active.")

        try:
            ticker = exchange_ccxt.fetch_ticker(self.symbol)
            current_price = ticker['last']
        except Exception as e: logger.error(f"[{self.name}-{self.symbol}] Error fetching ticker: {e}", exc_info=True); return

        if self.distance_condition_type == "Above":
             target_entry_price = daily_open * (1 - self.pullback_percent_for_entry)
             if current_price <= target_entry_price:
                  logger.info(f"[{self.name}-{self.symbol}] Entry conditions met. Price ({current_price}) <= target ({target_entry_price}). Opening LONG.")
                  self._open_long_position_live(db_session, subscription_id, current_price, exchange_ccxt)
             else: logger.debug(f"[{self.name}-{self.symbol}] Entry conditions met, waiting for pullback. Price ({current_price}) > target ({target_entry_price}).")
        elif self.distance_condition_type == "Below":
             target_entry_price = daily_open * (1 + self.pullback_percent_for_entry)
             if current_price >= target_entry_price:
                  logger.info(f"[{self.name}-{self.symbol}] Entry conditions met. Price ({current_price}) >= target ({target_entry_price}). Opening LONG.")
                  self._open_long_position_live(db_session, subscription_id, current_price, exchange_ccxt)
             else: logger.debug(f"[{self.name}-{self.symbol}] Entry conditions met, waiting for pullback. Price ({current_price}) < target ({target_entry_price}).")


    def _open_long_position_live(self, db_session: Session, subscription_id: int, current_market_price: float, exchange_ccxt):
        logger.info(f"[{self.name}-{self.symbol}] Attempting to open LONG for sub {subscription_id} near {current_market_price}")
        intended_entry_price = current_market_price 

        try:
            risk_amount = self.capital * self.risk_percent
            sl_distance_price = intended_entry_price * self.sl_percent_from_entry
            stop_loss_price = intended_entry_price - sl_distance_price
            
            position_size_quote = risk_amount / sl_distance_price if sl_distance_price != 0 else 0
            position_size_asset = (position_size_quote / intended_entry_price) * self.leverage if intended_entry_price != 0 else 0

            if position_size_asset <= 0: logger.warning(f"[{self.name}-{self.symbol}] Calc position size zero or negative ({position_size_asset:.8f}). Skipping."); return

            formatted_quantity = self._format_quantity(position_size_asset, exchange_ccxt)
            if float(formatted_quantity) <= 0: logger.warning(f"[{self.name}-{self.symbol}] Formatted quantity is zero ({formatted_quantity}). Skipping order."); return
            formatted_stop_loss_price = self._format_price(stop_loss_price, exchange_ccxt)
            
            logger.info(f"[{self.name}-{self.symbol}] Calculated: Risk ${risk_amount:.2f}, SL Price {formatted_stop_loss_price}, Qty {formatted_quantity}")

            entry_order_db = Order(subscription_id=subscription_id, symbol=self.symbol, order_type='market', side='buy', amount=float(formatted_quantity), status='pending_creation', created_at=datetime.datetime.utcnow(), updated_at=datetime.datetime.utcnow())
            db_session.add(entry_order_db); db_session.commit()

            order_receipt = exchange_ccxt.create_market_buy_order(self.symbol, float(formatted_quantity))
            entry_order_db.order_id = order_receipt['id']; entry_order_db.status = order_receipt.get('status', 'open'); db_session.commit()
            logger.info(f"[{self.name}-{self.symbol}] Market BUY order {order_receipt['id']} placed.")

            filled_order_details = self._await_order_fill(exchange_ccxt, order_receipt['id'], self.symbol)
            if not filled_order_details or filled_order_details['status'] != 'closed':
                logger.error(f"[{self.name}-{self.symbol}] Market BUY order {order_receipt['id']} failed/timed out. Status: {filled_order_details.get('status') if filled_order_details else 'Unknown'}")
                entry_order_db.status = filled_order_details.get('status', 'fill_check_failed') if filled_order_details else 'fill_check_failed'; db_session.commit()
                return

            actual_filled_price = float(filled_order_details['average']); actual_filled_quantity = float(filled_order_details['filled'])
            entry_order_db.status = 'closed'; entry_order_db.price = actual_filled_price; entry_order_db.filled = actual_filled_quantity; entry_order_db.cost = filled_order_details.get('cost'); entry_order_db.updated_at = datetime.datetime.utcnow(); db_session.commit()
            logger.info(f"[{self.name}-{self.symbol}] Market BUY order {order_receipt['id']} filled. Avg Price: {actual_filled_price}, Qty: {actual_filled_quantity}")

            if actual_filled_quantity <= 0: logger.warning(f"[{self.name}-{self.symbol}] Filled zero quantity. Skipping position."); return

            new_position = Position(subscription_id=subscription_id, symbol=self.symbol, exchange_name=str(exchange_ccxt.id), side="long", amount=actual_filled_quantity, entry_price=actual_filled_price, current_price=actual_filled_price, is_open=True, created_at=datetime.datetime.now(pytz.utc), updated_at=datetime.datetime.now(pytz.utc))
            db_session.add(new_position); db_session.commit(); logger.info(f"[{self.name}-{self.symbol}] Position ID {new_position.id} created.")

            sl_tp_quantity = self._format_quantity(actual_filled_quantity, exchange_ccxt)
            take_profit_price = actual_filled_price * (1 + self.take_profit_percent)
            formatted_take_profit_price = self._format_price(take_profit_price, exchange_ccxt)

            try:
                sl_params = {'stopPrice': formatted_stop_loss_price, 'reduceOnly': True}
                sl_order_receipt = exchange_ccxt.create_order(self.symbol, 'stop_market', 'sell', float(sl_tp_quantity), None, sl_params)
                new_sl_db = Order(subscription_id=subscription_id, order_id=sl_order_receipt['id'], symbol=self.symbol, order_type='stop_market', side='sell', amount=float(sl_tp_quantity), price=formatted_stop_loss_price, status='open', created_at=datetime.datetime.utcnow(),updated_at=datetime.datetime.utcnow())
                db_session.add(new_sl_db); logger.info(f"[{self.name}-{self.symbol}] SL order {sl_order_receipt['id']} placed for Pos ID {new_position.id}.")
            except Exception as e_sl: logger.error(f"[{self.name}-{self.symbol}] Failed to place SL for Pos ID {new_position.id}: {e_sl}", exc_info=True)
            
            try:
                tp_params = {'reduceOnly': True}
                tp_order_receipt = exchange_ccxt.create_limit_sell_order(self.symbol, float(sl_tp_quantity), formatted_take_profit_price, tp_params)
                new_tp_db = Order(subscription_id=subscription_id, order_id=tp_order_receipt['id'], symbol=self.symbol, order_type='limit', side='sell', amount=float(sl_tp_quantity), price=formatted_take_profit_price, status='open', created_at=datetime.datetime.utcnow(),updated_at=datetime.datetime.utcnow())
                db_session.add(new_tp_db); logger.info(f"[{self.name}-{self.symbol}] TP order {tp_order_receipt['id']} placed for Pos ID {new_position.id}.")
            except Exception as e_tp: logger.error(f"[{self.name}-{self.symbol}] Failed to place TP for Pos ID {new_position.id}: {e_tp}", exc_info=True)
            db_session.commit()
        except ccxt.InsufficientFunds as e: logger.error(f"[{self.name}-{self.symbol}] Insufficient funds: {e}", exc_info=True)
        except ccxt.NetworkError as e: logger.error(f"[{self.name}-{self.symbol}] Network error on entry: {e}", exc_info=True)
        except ccxt.ExchangeError as e: logger.error(f"[{self.name}-{self.symbol}] Exchange error on entry: {e}", exc_info=True)
        except Exception as e: logger.error(f"[{self.name}-{self.symbol}] Unexpected error opening LONG: {e}", exc_info=True)


    def _check_exit_conditions_live(self, db_session: Session, subscription_id: int, current_position_db: Position, exchange_ccxt):
        logger.info(f"[{self.name}-{self.symbol}] Checking exit for position ID {current_position_db.id} (Sub {subscription_id}).")
        open_orders_db = db_session.query(Order).filter(Order.subscription_id == subscription_id, Order.symbol == self.symbol, Order.status == 'open').all()
        sl_order_db = next((o for o in open_orders_db if o.order_type in ['stop_market', 'stop_limit', 'stop']), None)
        tp_order_db = next((o for o in open_orders_db if o.order_type in ['limit', 'take_profit_limit', 'take_profit']), None)

        try:
            if sl_order_db:
                sl_order_exchange = exchange_ccxt.fetch_order(sl_order_db.order_id, self.symbol)
                if sl_order_exchange['status'] == 'closed':
                    logger.info(f"[{self.name}-{self.symbol}] SL order {sl_order_db.order_id} filled. Closing position.")
                    sl_order_db.status = 'closed'; sl_order_db.filled = sl_order_exchange.get('filled', sl_order_db.amount); sl_order_db.updated_at = datetime.datetime.utcnow(); db_session.commit()
                    self._close_position_live(db_session, subscription_id, current_position_db, "Stop Loss Hit", exchange_ccxt, sl_order_exchange); return
            if tp_order_db:
                tp_order_exchange = exchange_ccxt.fetch_order(tp_order_db.order_id, self.symbol)
                if tp_order_exchange['status'] == 'closed':
                    logger.info(f"[{self.name}-{self.symbol}] TP order {tp_order_db.order_id} filled. Closing position.")
                    tp_order_db.status = 'closed'; tp_order_db.filled = tp_order_exchange.get('filled', tp_order_db.amount); tp_order_db.updated_at = datetime.datetime.utcnow(); db_session.commit()
                    self._close_position_live(db_session, subscription_id, current_position_db, "Take Profit Hit", exchange_ccxt, tp_order_exchange); return
        except Exception as e: logger.error(f"[{self.name}-{self.symbol}] Error checking SL/TP order status: {e}", exc_info=True)
        
        try:
            ticker = exchange_ccxt.fetch_ticker(self.symbol); current_price = ticker['last']
        except Exception as e: logger.error(f"[{self.name}-{self.symbol}] Error fetching ticker for exit check: {e}", exc_info=True); return

        if self.daily_cpr is None: logger.warning(f"[{self.name}-{self.symbol}] Daily CPR data not available for exit check."); return
        P, TC, BC, *_ = self.daily_cpr

        if current_position_db.side == "long" and current_price <= BC:
            logger.info(f"[{self.name}-{self.symbol}] Price ({current_price}) hit BC ({BC}). Closing LONG.")
            self._close_position_live(db_session, subscription_id, current_position_db, "BC Hit", exchange_ccxt); return

        now_utc = datetime.datetime.now(pytz.utc)
        if now_utc.hour == 23 and now_utc.minute >= 55: # End of Day EOD
             logger.info(f"[{self.name}-{self.symbol}] End of day approaching. Closing position.")
             self._close_position_live(db_session, subscription_id, current_position_db, "End of Day", exchange_ccxt); return
        
        logger.debug(f"[{self.name}-{self.symbol}] No manual exit conditions met for position ID {current_position_db.id}.")


    def _close_position_live(self, db_session: Session, subscription_id: int, current_position_db: Position, reason: str, exchange_ccxt, closing_trigger_order: dict = None):
        logger.info(f"[{self.name}-{self.symbol}] Attempting to close Pos ID {current_position_db.id} (Sub {subscription_id}) due to: {reason}")
        open_orders_for_pos = db_session.query(Order).filter(Order.subscription_id == subscription_id, Order.symbol == self.symbol, Order.status == 'open').all()

        for order_db in open_orders_for_pos:
            if closing_trigger_order and order_db.order_id == closing_trigger_order.get('id'): continue
            try:
                exchange_ccxt.cancel_order(order_db.order_id, self.symbol)
                logger.info(f"[{self.name}-{self.symbol}] Cancelled associated order {order_db.order_id} for closing position.")
                order_db.status = 'canceled'; order_db.updated_at = datetime.datetime.utcnow()
            except Exception as e: logger.warning(f"[{self.name}-{self.symbol}] Could not cancel associated order {order_db.order_id}: {e}")
        db_session.commit()

        actual_close_price = None; actual_closed_quantity = current_position_db.amount

        if closing_trigger_order: 
            actual_close_price = float(closing_trigger_order.get('average', current_position_db.current_price))
            actual_closed_quantity = float(closing_trigger_order.get('filled', current_position_db.amount))
            logger.info(f"[{self.name}-{self.symbol}] Position closed by pre-existing order {closing_trigger_order['id']}. Price: {actual_close_price}, Qty: {actual_closed_quantity}.")
        else: 
            try:
                side_to_close = 'sell' if current_position_db.side == 'long' else 'buy'
                formatted_qty_to_close = self._format_quantity(current_position_db.amount, exchange_ccxt)
                
                market_close_order_db = Order(subscription_id=subscription_id, symbol=self.symbol, order_type='market', side=side_to_close, amount=float(formatted_qty_to_close), status='pending_creation', created_at=datetime.datetime.utcnow(), updated_at=datetime.datetime.utcnow())
                db_session.add(market_close_order_db); db_session.commit()

                close_order_receipt = exchange_ccxt.create_market_order(self.symbol, side_to_close, float(formatted_qty_to_close))
                market_close_order_db.order_id = close_order_receipt['id']; market_close_order_db.status = 'open'; db_session.commit()
                logger.info(f"[{self.name}-{self.symbol}] Market {side_to_close.upper()} order {close_order_receipt['id']} placed to close position.")

                filled_details = self._await_order_fill(exchange_ccxt, close_order_receipt['id'], self.symbol)
                if not filled_details or filled_details['status'] != 'closed':
                    logger.error(f"[{self.name}-{self.symbol}] Market close order {close_order_receipt['id']} failed to fill. CRITICAL: Position might still be open.")
                    market_close_order_db.status = filled_details.get('status', 'fill_check_failed') if filled_details else 'fill_check_failed'; db_session.commit()
                    return 
                
                actual_close_price = float(filled_details['average']); actual_closed_quantity = float(filled_details['filled'])
                market_close_order_db.status = 'closed'; market_close_order_db.price = actual_close_price; market_close_order_db.filled = actual_closed_quantity; market_close_order_db.cost = filled_details.get('cost'); db_session.commit()
                logger.info(f"[{self.name}-{self.symbol}] Market close order {close_order_receipt['id']} filled. Price: {actual_close_price}, Qty: {actual_closed_quantity}.")
            except Exception as e:
                logger.error(f"[{self.name}-{self.symbol}] Error placing market order to close position: {e}", exc_info=True)
                db_session.commit(); return 

        current_position_db.is_open = False; current_position_db.closed_at = datetime.datetime.now(pytz.utc); current_position_db.updated_at = datetime.datetime.now(pytz.utc)
        if current_position_db.entry_price is not None and actual_closed_quantity > 0 and actual_close_price is not None:
            pnl = (actual_close_price - current_position_db.entry_price) * actual_closed_quantity if current_position_db.side == 'long' else (current_position_db.entry_price - actual_close_price) * actual_closed_quantity
            current_position_db.pnl = pnl
            logger.info(f"[{self.name}-{self.symbol}] Position ID {current_position_db.id} closed in DB. PnL: {pnl:.2f}. Reason: {reason}")
        else: logger.warning(f"[{self.name}-{self.symbol}] Could not calculate PnL for Pos ID {current_position_db.id} due to missing data.")
        db_session.commit()
        self.last_entry_attempt_utc_time = None 

    def run_backtest(self, historical_df: pd.DataFrame, htf_historical_df: pd.DataFrame = None):
        logger.warning(f"[{self.name}-{self.symbol}] Backtesting for CPR strategy is complex. This is a simplified conceptual outline.")
        return {"pnl": 0, "trades": [], "message": "CPR backtesting not fully implemented yet."}

    def execute_live_signal(self, db_session: Session, subscription_id: int, market_data_df: pd.DataFrame = None, exchange_ccxt=None):
        if not exchange_ccxt: logger.error(f"[{self.name}-{self.symbol}] Exchange instance not provided for sub {subscription_id}."); return
        logger.debug(f"[{self.name}-{self.symbol}] Executing live signal for sub {subscription_id}...")
        self._get_precisions(exchange_ccxt) 
        now_utc = datetime.datetime.now(pytz.utc)

        if self.data_prepared_for_utc_date != now_utc.date():
            if now_utc.hour == 0 and now_utc.minute < 15: 
                self._prepare_daily_data_live(exchange_ccxt)
            elif self.data_prepared_for_utc_date is None : 
                 self._prepare_daily_data_live(exchange_ccxt)
        
        current_position_db = db_session.query(Position).filter(
            Position.subscription_id == subscription_id,
            Position.symbol == self.symbol, 
            Position.is_open == True
        ).first()

        if self.data_prepared_for_utc_date == now_utc.date():
            if current_position_db is None: 
                if now_utc.hour == 0 and now_utc.minute < 10: 
                     if self.last_entry_attempt_utc_time is None or \
                        (now_utc - self.last_entry_attempt_utc_time).total_seconds() > 300: 
                         self._check_entry_conditions_live(db_session, subscription_id, exchange_ccxt)
                         self.last_entry_attempt_utc_time = now_utc
                     else: logger.debug(f"[{self.name}-{self.symbol}] In entry cooldown for sub {subscription_id}.")
                else: logger.debug(f"[{self.name}-{self.symbol}] Not within entry window (00:00-00:10 UTC) for sub {subscription_id}.")
            else: 
                self._check_exit_conditions_live(db_session, subscription_id, current_position_db, exchange_ccxt)
        else:
            logger.debug(f"[{self.name}-{self.symbol}] Daily data for {now_utc.date()} not yet prepared for sub {subscription_id}. Current prepared date: {self.data_prepared_for_utc_date}")

        logger.debug(f"[{self.name}-{self.symbol}] Live signal execution cycle finished for sub {subscription_id}.")
