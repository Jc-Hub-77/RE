import logging
import numpy as np
import talib
import ccxt
from decimal import Decimal, ROUND_DOWN, ROUND_UP, getcontext
from datetime import datetime, time # Ensure time is imported
import json # For custom_data
from typing import Optional, Dict, Any, List

import pandas
from sqlalchemy.orm import Session
from backend.models import Position, Order, UserStrategySubscription
from backend import strategy_utils

# Ensure high precision for Decimal calculations
getcontext().prec = 18

class TheOrphanStrategy:

    @staticmethod
    def get_parameters_definition():
        return {
            "trading_pair": {"type": "str", "label": "Trading Pair", "default": "BTC/USDT"},
            "bb_length": {"type": "int", "label": "BB Length", "default": 24},
            "bb_stdev": {"type": "float", "label": "BB StdDev", "default": 2.1},
            "trend_ema_period": {"type": "int", "label": "Trend EMA Period", "default": 365},
            "vol_filter_stdev_length": {"type": "int", "label": "Vol Filter STDEV Length", "default": 15},
            "vol_filter_sma_length": {"type": "int", "label": "Vol Filter SMA Length", "default": 28},
            "stop_loss_pct": {"type": "float", "label": "Stop Loss %", "default": 2.0},
            "take_profit_pct": {"type": "float", "label": "Take Profit %", "default": 9.0},
            "trailing_stop_activation_pct": {"type": "float", "label": "TSL Activation % Profit", "default": 0.5},
            "trailing_stop_offset_pct": {"type": "float", "label": "TSL Offset %", "default": 0.5},
            "kline_interval": {"type": "str", "label": "Kline Interval", "default": "1h"},
            "leverage": {"type": "int", "label": "Leverage", "default": 10},
            "order_quantity_usd": {"type": "float", "label": "Order Quantity (USD)", "default": 100.0}
        }

    def __init__(self, db_session: Session, user_sub_obj: UserStrategySubscription, strategy_params: dict, exchange_ccxt, logger_obj=None): # Renamed logger
        self.db_session = db_session
        self.user_sub_obj = user_sub_obj
        self.params = strategy_params
        self.exchange_ccxt = exchange_ccxt
        self.logger = logger_obj if logger_obj else logging.getLogger(__name__)
        self.name = "TheOrphanStrategy"
        
        self.trading_pair = self.params.get("trading_pair", "BTC/USDT")
        self.bb_length = int(self.params.get("bb_length", 24))
        self.bb_stdev = float(self.params.get("bb_stdev", 2.1))
        self.trend_ema_period = int(self.params.get("trend_ema_period", 365))
        self.vol_filter_stdev_length = int(self.params.get("vol_filter_stdev_length", 15))
        self.vol_filter_sma_length = int(self.params.get("vol_filter_sma_length", 28))
        self.stop_loss_pct = Decimal(str(self.params.get("stop_loss_pct", "2.0"))) / Decimal("100")
        self.take_profit_pct = Decimal(str(self.params.get("take_profit_pct", "9.0"))) / Decimal("100")
        self.trailing_stop_activation_pct = Decimal(str(self.params.get("trailing_stop_activation_pct", "0.5"))) / Decimal("100")
        self.trailing_stop_offset_pct = Decimal(str(self.params.get("trailing_stop_offset_pct", "0.5"))) / Decimal("100")
        self.kline_interval = self.params.get("kline_interval", "1h")
        self.leverage = int(self.params.get("leverage", 10))
        self.order_quantity_usd = Decimal(str(self.params.get("order_quantity_usd", "100.0")))

        # State attributes
        self.active_position_db_id: Optional[int] = None
        self.position_entry_price: Optional[Decimal] = None
        self.position_side: Optional[str] = None # 'long' or 'short'
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
        self._load_persistent_state()
        self._set_leverage()
        self.logger.info(f"{self.name} initialized for {self.trading_pair}, UserSubID {self.user_sub_obj.id}. PosID: {self.active_position_db_id}")

    def _reset_sl_tp_and_trailing_state_internals(self):
        self.high_water_mark = None
        self.low_water_mark = None
        self.trailing_stop_active = False
        self.current_trailing_stop_price = None
        self.active_sl_exchange_id = None
        self.active_tp_exchange_id = None
        self.sl_order_db_id = None
        self.tp_order_db_id = None

    def _reset_internal_trade_state(self):
        self.active_position_db_id = None
        self.position_entry_price = None
        self.position_side = None
        self.position_qty = Decimal("0")
        self._reset_sl_tp_and_trailing_state_internals()
        self.logger.info(f"[{self.name}-{self.trading_pair}] Internal trade state fully reset.")

    def _load_persistent_state(self):
        if not self.db_session or not self.user_sub_obj:
            self.logger.warning(f"[{self.name}-{self.trading_pair}] DB session/user_sub_obj not available. Cannot load state.")
            self._reset_internal_trade_state()
            return

        open_pos = strategy_utils.get_open_strategy_position(self.db_session, self.user_sub_obj.id, self.trading_pair)
        if open_pos:
            self.logger.info(f"[{self.name}-{self.trading_pair}] Loading state for open position ID: {open_pos.id}")
            self.active_position_db_id = open_pos.id
            self.position_side = open_pos.side
            self.position_entry_price = Decimal(str(open_pos.entry_price)) if open_pos.entry_price is not None else None
            self.position_qty = Decimal(str(open_pos.amount)) if open_pos.amount is not None else Decimal("0")
            
            if open_pos.custom_data:
                try:
                    state_data = json.loads(open_pos.custom_data)
                    self.high_water_mark = Decimal(str(state_data.get("high_water_mark"))) if state_data.get("high_water_mark") else (self.position_entry_price if self.position_side == 'long' else None)
                    self.low_water_mark = Decimal(str(state_data.get("low_water_mark"))) if state_data.get("low_water_mark") else (self.position_entry_price if self.position_side == 'short' else None)
                    self.trailing_stop_active = state_data.get("trailing_stop_active", False)
                    self.current_trailing_stop_price = Decimal(str(state_data.get("current_trailing_stop_price"))) if state_data.get("current_trailing_stop_price") else None
                    self.active_sl_exchange_id = state_data.get("active_sl_exchange_id")
                    self.active_tp_exchange_id = state_data.get("active_tp_exchange_id")
                    self.sl_order_db_id = state_data.get("sl_order_db_id")
                    self.tp_order_db_id = state_data.get("tp_order_db_id")
                    self.logger.info(f"[{self.name}-{self.trading_pair}] Loaded custom state: {state_data}")
                except json.JSONDecodeError:
                    self.logger.error(f"[{self.name}-{self.trading_pair}] Error decoding custom_data for pos {open_pos.id}. Initializing trailing state.")
                    self._initialize_trailing_state_for_loaded_position()
            else:
                self.logger.info(f"[{self.name}-{self.trading_pair}] No custom_data for pos {open_pos.id}. Initializing trailing state.")
                self._initialize_trailing_state_for_loaded_position()
        else:
            self.logger.info(f"[{self.name}-{self.trading_pair}] No active persistent position found.")
            self._reset_internal_trade_state()

    def _initialize_trailing_state_for_loaded_position(self):
        if self.position_side == 'long' and self.position_entry_price: self.high_water_mark = self.position_entry_price
        elif self.position_side == 'short' and self.position_entry_price: self.low_water_mark = self.position_entry_price
        else: self.high_water_mark = None; self.low_water_mark = None
        self.trailing_stop_active = False
        self.current_trailing_stop_price = None
        # SL/TP exchange IDs might need to be queried if not in custom_data
        # For now, assuming they are part of custom_data or re-established if needed by strategy logic
        self.active_sl_exchange_id = None 
        self.active_tp_exchange_id = None
        self.sl_order_db_id = None
        self.tp_order_db_id = None


    def _save_persistent_state(self):
        if not self.active_position_db_id or not self.db_session:
            self.logger.debug(f"[{self.name}-{self.trading_pair}] No active pos DB ID or DB session to save custom state.")
            return
        
        pos_to_update = self.db_session.query(Position).filter(Position.id == self.active_position_db_id).first()
        if pos_to_update:
            state_data = {
                "high_water_mark": str(self.high_water_mark) if self.high_water_mark else None,
                "low_water_mark": str(self.low_water_mark) if self.low_water_mark else None,
                "trailing_stop_active": self.trailing_stop_active,
                "current_trailing_stop_price": str(self.current_trailing_stop_price) if self.current_trailing_stop_price else None,
                "active_sl_exchange_id": self.active_sl_exchange_id,
                "active_tp_exchange_id": self.active_tp_exchange_id,
                "sl_order_db_id": self.sl_order_db_id,
                "tp_order_db_id": self.tp_order_db_id
            }
            pos_to_update.custom_data = json.dumps(state_data)
            pos_to_update.updated_at = datetime.utcnow()
            try:
                self.db_session.commit()
                self.logger.info(f"[{self.name}-{self.trading_pair}] Saved custom state for PosID {self.active_position_db_id}: {state_data}")
            except Exception as e:
                self.logger.error(f"[{self.name}-{self.trading_pair}] Error saving custom state for PosID {self.active_position_db_id}: {e}", exc_info=True)
                self.db_session.rollback()
        else:
             self.logger.warning(f"[{self.name}-{self.trading_pair}] Position {self.active_position_db_id} not found to save custom state.")


    def _fetch_market_precision(self): # Same as original
        if not self._precisions_fetched_:
            try:
                self.exchange_ccxt.load_markets(True)
                market = self.exchange_ccxt.markets[self.trading_pair]
                self.quantity_precision_str = str(market['precision']['amount'])
                self.price_precision_str = str(market['precision']['price'])
                self._precisions_fetched_ = True
                self.logger.info(f"Precision for {self.trading_pair}: Qty Prec Str={self.quantity_precision_str}, Price Prec Str={self.price_precision_str}")
            except Exception as e:
                self.logger.error(f"Error fetching precision for {self.trading_pair}: {e}. Using defaults.", exc_info=True)
                self.quantity_precision_str = "0.00001"; self.price_precision_str = "0.01"    

    def _get_decimal_places(self, precision_str: Optional[str]) -> int: # Same as original
        if precision_str is None: self.logger.warning("Precision string is None, using default 8."); return 8
        try:
            if 'e-' in precision_str.lower():
                num_val = float(precision_str); precision_str = format(num_val, f'.{abs(int(precision_str.split("e-")[1]))}f')
            d_prec = Decimal(precision_str)
            if d_prec.as_tuple().exponent < 0: return abs(d_prec.as_tuple().exponent)
            return 0 
        except Exception as e: self.logger.warning(f"Could not parse precision string '{precision_str}'. Error: {e}. Using default 8."); return 8

    def _format_quantity(self, quantity: Decimal) -> str: # Same as original
        places = self._get_decimal_places(self.quantity_precision_str)
        return str(quantity.quantize(Decimal('1e-' + str(places)), rounding=ROUND_DOWN))

    def _format_price(self, price: Decimal) -> str: # Same as original
        places = self._get_decimal_places(self.price_precision_str)
        return str(price.quantize(Decimal('1e-' + str(places)), rounding=ROUND_NEAREST))

    def _set_leverage(self): # Same as original
        try:
            if hasattr(self.exchange_ccxt, 'set_leverage'):
                self.exchange_ccxt.set_leverage(self.leverage, self.trading_pair)
                self.logger.info(f"Leverage set to {self.leverage}x for {self.trading_pair}")
        except Exception as e: self.logger.warning(f"Could not set leverage for {self.trading_pair}: {e}")

    def _calculate_indicators(self, ohlcv_df): # Same as original
        indicators = {}; close_prices = ohlcv_df['close'].to_numpy(dtype=float)
        upper_bb, middle_bb, lower_bb = talib.BBANDS(close_prices, timeperiod=self.bb_length, nbdevup=self.bb_stdev, nbdevdn=self.bb_stdev, matype=0)
        indicators['upper_bb'] = upper_bb; indicators['middle_bb'] = middle_bb; indicators['lower_bb'] = lower_bb
        indicators['trend_ema'] = talib.EMA(close_prices, timeperiod=self.trend_ema_period)
        vol_std = talib.STDDEV(close_prices, timeperiod=self.vol_filter_stdev_length)
        vol_sma_of_std = talib.SMA(vol_std, timeperiod=self.vol_filter_sma_length)
        indicators['vol_cond'] = vol_std > vol_sma_of_std
        return indicators

    def _await_order_fill(self, exchange_order_id: str): # Simplified
        # Same as original, but use self.exchange_ccxt and self.trading_pair
        start_time = time.time()
        self.logger.info(f"[{self.name}-{self.trading_pair}] Awaiting fill for order {exchange_order_id} (timeout: 60s)")
        while time.time() - start_time < 60:
            try:
                order = self.exchange_ccxt.fetch_order(exchange_order_id, self.trading_pair)
                if order['status'] == 'closed': self.logger.info(f"Order {exchange_order_id} filled."); return order
                if order['status'] in ['canceled', 'rejected', 'expired']: self.logger.warning(f"Order {exchange_order_id} is {order['status']}."); return order
            except ccxt.OrderNotFound: self.logger.warning(f"Order {exchange_order_id} not found. Retrying.")
            except Exception as e: self.logger.error(f"Error fetching order {exchange_order_id}: {e}. Retrying.", exc_info=True)
            time.sleep(3)
        self.logger.warning(f"Timeout for order {exchange_order_id}. Final check.")
        try: final_status = self.exchange_ccxt.fetch_order(exchange_order_id, self.trading_pair); self.logger.info(f"Final status for order {exchange_order_id}: {final_status['status']}"); return final_status
        except Exception as e: self.logger.error(f"Final check for order {exchange_order_id} failed: {e}", exc_info=True); return None

    def _place_order(self, order_type: str, side: str, quantity: Decimal, price: Optional[Decimal]=None, params: Optional[Dict]=None) -> Optional[Order]:
        db_order = None
        try:
            formatted_qty_str = self._format_quantity(quantity)
            formatted_price_str = self._format_price(price) if price else None
            
            db_order = strategy_utils.create_strategy_order_in_db(
                self.db_session, self.user_sub_obj.id, self.trading_pair, order_type.lower(), side, 
                float(formatted_qty_str), float(formatted_price_str) if formatted_price_str else None, 
                status='pending_exchange', notes=f"{self.name} {order_type} {side}"
            )
            if not db_order: self.logger.error(f"Failed to create DB order for {side} {quantity} {self.trading_pair}."); return None

            exchange_order = self.exchange_ccxt.create_order(self.trading_pair, order_type, side, float(formatted_qty_str), float(formatted_price_str) if formatted_price_str else None, params)
            
            updated_db_order = strategy_utils.update_strategy_order_in_db(
                self.db_session, order_db_id=db_order.id, 
                updates={'order_id': exchange_order.get('id'), 'status': 'open', 'raw_order_data': json.dumps(exchange_order)}
            )
            self.logger.info(f"Order placed: {side} {order_type} {formatted_qty_str} {self.trading_pair}. ExchID: {exchange_order.get('id')}, DB_ID: {db_order.id}")
            return updated_db_order # Return updated DB Order object
        except Exception as e:
            self.logger.error(f"Failed to place {side} {order_type} for {self.trading_pair}: {e}", exc_info=True)
            if db_order and db_order.id: strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=db_order.id, updates={'status': 'error_on_exchange', 'status_message': str(e)[:255]})
        return None

    def _cancel_order_by_id(self, exchange_order_id_to_cancel: str, associated_db_order_id: Optional[int]):
        if not exchange_order_id_to_cancel: return
        try:
            self.exchange_ccxt.cancel_order(exchange_order_id_to_cancel, self.trading_pair)
            self.logger.info(f"Cancelled order {exchange_order_id_to_cancel} for {self.trading_pair}")
            if associated_db_order_id:
                strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=associated_db_order_id, updates={'status': 'canceled'})
            else: # Fallback if DB ID not directly known, try to find by exchange ID
                 strategy_utils.update_strategy_order_in_db(self.db_session, exchange_order_id=exchange_order_id_to_cancel, subscription_id=self.user_sub_obj.id, symbol=self.trading_pair, updates={'status': 'canceled'})
        except ccxt.OrderNotFound:
            self.logger.warning(f"Order {exchange_order_id_to_cancel} not found to cancel (already filled/cancelled).")
            if associated_db_order_id: strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=associated_db_order_id, updates={'status': 'not_found_on_cancel'})
            else: strategy_utils.update_strategy_order_in_db(self.db_session, exchange_order_id=exchange_order_id_to_cancel, subscription_id=self.user_sub_obj.id, symbol=self.trading_pair, updates={'status': 'not_found_on_cancel'})
        except Exception as e:
            self.logger.error(f"Failed to cancel order {exchange_order_id_to_cancel} for {self.trading_pair}: {e}")

    def _sync_exchange_position_state(self): # No current_price needed if just checking orders
        if not self.active_position_db_id or not self.db_session: return False
        
        position_closed_event = False
        if self.active_sl_exchange_id and self.sl_order_db_id:
            try:
                sl_details = self.exchange_ccxt.fetch_order(self.active_sl_exchange_id, self.trading_pair)
                if sl_details['status'] == 'closed':
                    self.logger.info(f"SL order {self.active_sl_exchange_id} filled.")
                    strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=self.sl_order_db_id, updates={'status': 'closed', 'price': sl_details.get('average'), 'filled': sl_details.get('filled'), 'closed_at': datetime.utcnow(), 'raw_order_data': json.dumps(sl_details)})
                    strategy_utils.close_strategy_position_in_db(self.db_session, self.active_position_db_id, Decimal(str(sl_details['average'])), Decimal(str(sl_details['filled'])), f"Closed by SL {self.active_sl_exchange_id}")
                    position_closed_event = True
                    if self.active_tp_exchange_id: self._cancel_order_by_id(self.active_tp_exchange_id, self.tp_order_db_id)
            except ccxt.OrderNotFound: self.logger.warning(f"SL order {self.active_sl_exchange_id} not found in sync."); strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=self.sl_order_db_id, updates={'status': 'not_found_on_sync'}); self.active_sl_exchange_id = None; self.sl_order_db_id = None
            except Exception as e: self.logger.error(f"Error syncing SL order {self.active_sl_exchange_id}: {e}")

        if not position_closed_event and self.active_tp_exchange_id and self.tp_order_db_id:
            try:
                tp_details = self.exchange_ccxt.fetch_order(self.active_tp_exchange_id, self.trading_pair)
                if tp_details['status'] == 'closed':
                    self.logger.info(f"TP order {self.active_tp_exchange_id} filled.")
                    strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=self.tp_order_db_id, updates={'status': 'closed', 'price': tp_details.get('average'), 'filled': tp_details.get('filled'), 'closed_at': datetime.utcnow(), 'raw_order_data': json.dumps(tp_details)})
                    strategy_utils.close_strategy_position_in_db(self.db_session, self.active_position_db_id, Decimal(str(tp_details['average'])), Decimal(str(tp_details['filled'])), f"Closed by TP {self.active_tp_exchange_id}")
                    position_closed_event = True
                    if self.active_sl_exchange_id: self._cancel_order_by_id(self.active_sl_exchange_id, self.sl_order_db_id)
            except ccxt.OrderNotFound: self.logger.warning(f"TP order {self.active_tp_exchange_id} not found in sync."); strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=self.tp_order_db_id, updates={'status': 'not_found_on_sync'}); self.active_tp_exchange_id = None; self.tp_order_db_id = None
            except Exception as e: self.logger.error(f"Error syncing TP order {self.active_tp_exchange_id}: {e}")

        if position_closed_event:
            self._reset_internal_trade_state()
            self._save_persistent_state() # Save cleared state
            return True
        return False

    def _process_trading_logic(self, ohlcv_df: pandas.DataFrame):
        # Uses self.position_side, self.position_entry_price etc.
        # Calls self._place_order, self._cancel_order_by_id
        # Updates self.active_sl_exchange_id, self.active_tp_exchange_id, self.sl_order_db_id, self.tp_order_db_id
        # Calls self._save_persistent_state() after changes
        # ... (Existing logic for BB, EMA, Vol, signals) ...
        symbol = self.trading_pair
        latest_candle_idx = -1 
        
        if len(ohlcv_df) < max(self.bb_length, self.trend_ema_period, self.vol_filter_stdev_length + self.vol_filter_sma_length):
            self.logger.warning(f"Not enough data for indicator calculation on {symbol}."); return

        indicators = self._calculate_indicators(ohlcv_df)
        close = Decimal(str(ohlcv_df['close'].iloc[latest_candle_idx]))
        upper_bb = Decimal(str(indicators['upper_bb'][latest_candle_idx])); lower_bb = Decimal(str(indicators['lower_bb'][latest_candle_idx]))
        trend_ema = Decimal(str(indicators['trend_ema'][latest_candle_idx])); vol_cond = indicators['vol_cond'][latest_candle_idx]
        prev_close = Decimal(str(ohlcv_df['close'].iloc[-2])) if len(ohlcv_df) >=2 else close
        prev_upper_bb = Decimal(str(indicators['upper_bb'][-2])) if len(indicators['upper_bb']) >=2 else upper_bb
        prev_lower_bb = Decimal(str(indicators['lower_bb'][-2])) if len(indicators['lower_bb']) >=2 else lower_bb

        buy_cond_bb_crossover = prev_close < prev_upper_bb and close > upper_bb
        sell_cond_bb_crossover = prev_close > prev_lower_bb and close < lower_bb
        buy_trend_cond = close > trend_ema; sell_trend_cond = close < trend_ema
        final_buy_entry = buy_cond_bb_crossover and buy_trend_cond and vol_cond
        final_sell_entry = sell_cond_bb_crossover and sell_trend_cond and vol_cond

        # Primary Exit Logic (BB Crossover Reversal)
        if self.position_side == 'long' and sell_cond_bb_crossover:
            self.logger.info(f"Primary Exit Long for {symbol} due to BB crossover at {close}.")
            self._cancel_order_by_id(self.active_sl_exchange_id, self.sl_order_db_id)
            self._cancel_order_by_id(self.active_tp_exchange_id, self.tp_order_db_id)
            exit_order_db = self._place_order('market', 'sell', self.position_qty, params={'reduceOnly': True})
            if exit_order_db and exit_order_db.order_id:
                filled_exit = self._await_order_fill(exit_order_db.order_id)
                if filled_exit and filled_exit['status']=='closed':
                    strategy_utils.close_strategy_position_in_db(self.db_session, self.active_position_db_id, Decimal(str(filled_exit['average'])), Decimal(str(filled_exit['filled'])), "Exit Long (BB Crossover)")
            self._reset_internal_trade_state(); self._save_persistent_state(); return

        if self.position_side == 'short' and buy_cond_bb_crossover:
            self.logger.info(f"Primary Exit Short for {symbol} due to BB crossover at {close}.")
            self._cancel_order_by_id(self.active_sl_exchange_id, self.sl_order_db_id)
            self._cancel_order_by_id(self.active_tp_exchange_id, self.tp_order_db_id)
            exit_order_db = self._place_order('market', 'buy', self.position_qty, params={'reduceOnly': True})
            if exit_order_db and exit_order_db.order_id:
                filled_exit = self._await_order_fill(exit_order_db.order_id)
                if filled_exit and filled_exit['status']=='closed':
                     strategy_utils.close_strategy_position_in_db(self.db_session, self.active_position_db_id, Decimal(str(filled_exit['average'])), Decimal(str(filled_exit['filled'])), "Exit Short (BB Crossover)")
            self._reset_internal_trade_state(); self._save_persistent_state(); return
        
        # Entry Logic
        if not self.position_side:
            entry_price = close
            if entry_price == Decimal("0"): self.logger.warning("Entry price zero."); return
            qty_to_trade = self.order_quantity_usd / entry_price
            
            side_to_enter = None
            if final_buy_entry: side_to_enter = 'long'
            elif final_sell_entry: side_to_enter = 'short'

            if side_to_enter:
                self.logger.info(f"Entry {side_to_enter.upper()} signal for {symbol} at {entry_price}")
                entry_order_db = self._place_order('market', side_to_enter, qty_to_trade)
                if entry_order_db and entry_order_db.order_id:
                    filled_entry = self._await_order_fill(entry_order_db.order_id)
                    if filled_entry and filled_entry['status'] == 'closed':
                        actual_entry_price = Decimal(str(filled_entry['average']))
                        actual_filled_qty = Decimal(str(filled_entry['filled']))
                        strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=entry_order_db.id, updates={'status':'closed', 'price':float(actual_entry_price), 'filled':float(actual_filled_qty), 'cost': float(actual_entry_price*actual_filled_qty), 'closed_at':datetime.utcnow()})

                        new_pos_db = strategy_utils.create_strategy_position_in_db(self.db_session, self.user_sub_obj.id, symbol, str(self.exchange_ccxt.id), side_to_enter, float(actual_filled_qty), float(actual_entry_price), status_message=f"{self.name} Entry")
                        if new_pos_db:
                            self.active_position_db_id = new_pos_db.id; self.position_side = side_to_enter
                            self.position_entry_price = actual_entry_price; self.position_qty = actual_filled_qty
                            if side_to_enter == 'long': self.high_water_mark = actual_entry_price
                            else: self.low_water_mark = actual_entry_price
                            self.trailing_stop_active = False; self.current_trailing_stop_price = None
                            
                            sl_pct_val = self.stop_loss_pct; tp_pct_val = self.take_profit_pct
                            sl_price = actual_entry_price * (Decimal('1') - sl_pct_val) if side_to_enter == 'long' else actual_entry_price * (Decimal('1') + sl_pct_val)
                            tp_price = actual_entry_price * (Decimal('1') + tp_pct_val) if side_to_enter == 'long' else actual_entry_price * (Decimal('1') - tp_pct_val)
                            
                            exit_side = 'sell' if side_to_enter == 'long' else 'buy'
                            sl_order_db = self._place_order('STOP_MARKET', exit_side, actual_filled_qty, params={'stopPrice': self._format_price(sl_price), 'reduceOnly':True})
                            if sl_order_db: self.sl_order_db_id = sl_order_db.id; self.active_sl_exchange_id = sl_order_db.order_id
                            
                            tp_order_db = self._place_order('TAKE_PROFIT_MARKET', exit_side, actual_filled_qty, params={'stopPrice': self._format_price(tp_price), 'reduceOnly':True}) # Some exchanges use TAKE_PROFIT_MARKET, others use LIMIT with special flags
                            if tp_order_db: self.tp_order_db_id = tp_order_db.id; self.active_tp_exchange_id = tp_order_db.order_id
                            
                            self._save_persistent_state()
                        else: self.logger.error("Failed to create position in DB after entry.")
                    else: self.logger.error(f"Entry order {entry_order_db.order_id} did not fill."); strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=entry_order_db.id, updates={'status': filled_entry.get('status','fill_check_failed') if filled_entry else 'fill_check_failed'})
                else: self.logger.error(f"Failed to place entry order for {side_to_enter} or create DB record.")
            return # Action taken or attempted

        # Trailing Stop Logic (if position exists)
        if self.position_side and self.position_entry_price:
            new_tsl_price = None
            if self.position_side == 'long':
                if close > (self.high_water_mark or self.position_entry_price): self.high_water_mark = close
                if not self.trailing_stop_active and close >= self.position_entry_price * (Decimal('1') + self.trailing_stop_activation_pct):
                    self.trailing_stop_active = True; self.logger.info(f"TSL activated for LONG at {close}")
                if self.trailing_stop_active: new_tsl_price = self.high_water_mark * (Decimal('1') - self.trailing_stop_offset_pct)
                if new_tsl_price and (self.current_trailing_stop_price is None or new_tsl_price > self.current_trailing_stop_price):
                    self.logger.info(f"Updating TSL for LONG. New: {new_tsl_price}, Old: {self.current_trailing_stop_price}")
                    if self.active_sl_exchange_id: self._cancel_order_by_id(self.active_sl_exchange_id, self.sl_order_db_id)
                    new_sl_order_db = self._place_order('STOP_MARKET', 'sell', self.position_qty, params={'stopPrice': self._format_price(new_tsl_price), 'reduceOnly':True})
                    if new_sl_order_db: self.active_sl_exchange_id = new_sl_order_db.order_id; self.sl_order_db_id = new_sl_order_db.id; self.current_trailing_stop_price = new_tsl_price; self._save_persistent_state()

            elif self.position_side == 'short':
                if close < (self.low_water_mark or self.position_entry_price): self.low_water_mark = close
                if not self.trailing_stop_active and close <= self.position_entry_price * (Decimal('1') - self.trailing_stop_activation_pct):
                    self.trailing_stop_active = True; self.logger.info(f"TSL activated for SHORT at {close}")
                if self.trailing_stop_active: new_tsl_price = self.low_water_mark * (Decimal('1') + self.trailing_stop_offset_pct)
                if new_tsl_price and (self.current_trailing_stop_price is None or new_tsl_price < self.current_trailing_stop_price):
                    self.logger.info(f"Updating TSL for SHORT. New: {new_tsl_price}, Old: {self.current_trailing_stop_price}")
                    if self.active_sl_exchange_id: self._cancel_order_by_id(self.active_sl_exchange_id, self.sl_order_db_id)
                    new_sl_order_db = self._place_order('STOP_MARKET', 'buy', self.position_qty, params={'stopPrice': self._format_price(new_tsl_price), 'reduceOnly':True})
                    if new_sl_order_db: self.active_sl_exchange_id = new_sl_order_db.order_id; self.sl_order_db_id = new_sl_order_db.id; self.current_trailing_stop_price = new_tsl_price; self._save_persistent_state()
        
    def execute_live_signal(self):
        self.logger.info(f"Executing {self.name} for {self.trading_pair} UserSubID {self.user_sub_obj.id}")
        current_utc_dt = datetime.utcnow().replace(tzinfo=pytz.utc)
        
        try:
            current_price_for_sync = Decimal(str(self.exchange_ccxt.fetch_ticker(self.trading_pair)['last']))
            if self._sync_exchange_position_state(current_price_for_sync): # Pass current price
                self.logger.info(f"[{self.name}-{self.trading_pair}] Position closed by SL/TP sync. Cycle ended.")
                return
        except Exception as e:
            self.logger.error(f"[{self.name}-{self.trading_pair}] Error during pre-logic sync/price fetch: {e}", exc_info=True); return

        # EOD Close Logic (conceptual, needs specific EOD time)
        # For this strategy, EOD close might not be standard unless specified.
        # If needed, it would be similar to ORB's _perform_eod_close_if_needed.
        # For now, assuming no EOD close specific to this strategy beyond normal SL/TP/reversals.

        try:
            limit_needed = max(self.bb_length, self.trend_ema_period, self.vol_filter_stdev_length + self.vol_filter_sma_length) + 50
            ohlcv_data = self.exchange_ccxt.fetch_ohlcv(self.trading_pair, timeframe=self.kline_interval, limit=limit_needed)
            if not ohlcv_data or len(ohlcv_data) < limit_needed - 45:
                self.logger.warning(f"Not enough OHLCV data for {self.trading_pair}. Got {len(ohlcv_data)}"); return

            ohlcv_df = pandas.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            self._process_trading_logic(ohlcv_df)
        except Exception as e:
            self.logger.error(f"Error in {self.name} execute_live_signal: {e}", exc_info=True)

    def run_backtest(self, historical_data_feed, current_simulated_time_utc):
        # Backtest logic remains simulation-based
        self.logger.info(f"Running backtest for {self.name} on {self.trading_pair} at {current_simulated_time_utc}...")
        # ... (Full backtesting logic as previously provided, using local variables for state) ...
        self.logger.warning("Backtesting for TheOrphanStrategy needs a dedicated simulation loop. Returning placeholder.")
        return {"action": "HOLD", "reason": "Backtest simulation for this strategy is complex and not fully implemented in this refactor."}

```
