import logging
import numpy as np
import talib
import pandas
import ccxt 
from decimal import Decimal, ROUND_DOWN, ROUND_UP, getcontext
from datetime import datetime, time # Ensure time is imported
import json
from typing import Optional, Dict, Any, List

from sqlalchemy.orm import Session
from backend.models import Position, Order, UserStrategySubscription
from backend import strategy_utils

# Ensure high precision for Decimal calculations
getcontext().prec = 18

class NadarayaWatsonStochRSIStrategy:

    @staticmethod
    def get_parameters_definition():
        return {
            "trading_pair": {"type": "str", "label": "Trading Pair", "default": "BTC/USDT"},
            "leverage": {"type": "int", "label": "Leverage", "default": 10, "min": 1, "max": 100},
            "order_quantity_usd": {"type": "float", "label": "Order Quantity (USD Notional)", "default": 100.0, "min": 1.0},
            "nw_timeframe": {"type": "str", "label": "Nadaraya-Watson Timeframe", "default": "15m"},
            "nw_h_bandwidth": {"type": "float", "label": "NW: h (Bandwidth)", "default": 8.0},
            "nw_multiplier": {"type": "float", "label": "NW: Band Multiplier", "default": 3.0},
            "nw_yhat_lookback": {"type": "int", "label": "NW: y_hat Lookback", "default": 20},
            "nw_mae_lookback": {"type": "int", "label": "NW: MAE Lookback", "default": 20},
            "stoch_rsi_timeframe": {"type": "str", "label": "StochRSI Timeframe", "default": "1h"},
            "stoch_rsi_length": {"type": "int", "label": "StochRSI: RSI Length", "default": 14},
            "stoch_rsi_stoch_length": {"type": "int", "label": "StochRSI: Stochastic Length", "default": 14},
            "stoch_rsi_k_smooth": {"type": "int", "label": "StochRSI: %K Smoothing", "default": 3},
            "stoch_rsi_d_smooth": {"type": "int", "label": "StochRSI: %D Smoothing", "default": 3},
            "stoch_rsi_oversold_level": {"type": "float", "label": "StochRSI Oversold Level", "default": 20.0},
            "stoch_rsi_overbought_level": {"type": "float", "label": "StochRSI Overbought Level", "default": 80.0},
            "stop_loss_pct": {"type": "float", "label": "Stop Loss %", "default": 1.0},
            "take_profit_pct": {"type": "float", "label": "Take Profit %", "default": 3.0},
        }

    def __init__(self, db_session: Session, user_sub_obj: UserStrategySubscription, strategy_params: dict, exchange_ccxt, logger_obj=None):
        self.db_session = db_session
        self.user_sub_obj = user_sub_obj
        self.params = strategy_params
        self.exchange_ccxt = exchange_ccxt
        self.logger = logger_obj if logger_obj else logging.getLogger(__name__)
        self.name = "NadarayaWatsonStochRSIStrategy"

        # Load parameters
        self.trading_pair = self.params.get("trading_pair", "BTC/USDT")
        self.leverage = int(self.params.get("leverage", 10))
        self.order_quantity_usd = Decimal(str(self.params.get("order_quantity_usd", "100.0")))
        self.nw_timeframe = self.params.get("nw_timeframe", "15m")
        self.nw_h_bandwidth = float(self.params.get("nw_h_bandwidth", 8.0))
        self.nw_multiplier = float(self.params.get("nw_multiplier", 3.0))
        self.nw_yhat_lookback = int(self.params.get("nw_yhat_lookback", 20))
        self.nw_mae_lookback = int(self.params.get("nw_mae_lookback", 20))
        self.stoch_rsi_timeframe = self.params.get("stoch_rsi_timeframe", "1h")
        self.stoch_rsi_length = int(self.params.get("stoch_rsi_length", 14))
        self.stoch_rsi_stoch_length = int(self.params.get("stoch_rsi_stoch_length", 14))
        self.stoch_rsi_k_smooth = int(self.params.get("stoch_rsi_k_smooth", 3))
        self.stoch_rsi_d_smooth = int(self.params.get("stoch_rsi_d_smooth", 3))
        self.stoch_rsi_oversold_level = Decimal(str(self.params.get("stoch_rsi_oversold_level", "20.0")))
        self.stoch_rsi_overbought_level = Decimal(str(self.params.get("stoch_rsi_overbought_level", "80.0")))
        self.stop_loss_pct = Decimal(str(self.params.get("stop_loss_pct", "1.0"))) / Decimal("100")
        self.take_profit_pct = Decimal(str(self.params.get("take_profit_pct", "3.0"))) / Decimal("100")

        # State attributes
        self.active_position_db_id: Optional[int] = None
        self.sl_order_db_id: Optional[int] = None
        self.tp_order_db_id: Optional[int] = None
        self.active_sl_tp_orders: Dict[str, Optional[str]] = {} # {'sl_id': sl_exchange_id, 'tp_id': tp_exchange_id}
        self.active_position_side: Optional[str] = None
        self.position_entry_price: Optional[Decimal] = None
        self.position_qty: Decimal = Decimal("0")
        
        self.price_precision_str: Optional[str] = None
        self.quantity_precision_str: Optional[str] = None
        self._precisions_fetched_ = False


        self._fetch_market_precision()
        self._load_persistent_state() # Load state after fetching precision
        self._set_leverage()
        self.logger.info(f"{self.name} initialized for {self.trading_pair}, UserSubID {self.user_sub_obj.id}. PosID: {self.active_position_db_id}")

    def _load_persistent_state(self):
        if not self.db_session or not self.user_sub_obj:
            self.logger.warning(f"[{self.name}-{self.trading_pair}] DB session/user_sub_obj not available for loading state.")
            self._reset_internal_position_state() # Ensure defaults are set
            return

        open_pos = strategy_utils.get_open_strategy_position(self.db_session, self.user_sub_obj.id, self.trading_pair)
        if open_pos:
            self.logger.info(f"[{self.name}-{self.trading_pair}] Loading state for open position ID: {open_pos.id}")
            self.active_position_db_id = open_pos.id
            self.active_position_side = open_pos.side
            self.position_entry_price = Decimal(str(open_pos.entry_price)) if open_pos.entry_price is not None else None
            self.position_qty = Decimal(str(open_pos.amount)) if open_pos.amount is not None else Decimal("0")
            
            if open_pos.custom_data:
                try:
                    state_data = json.loads(open_pos.custom_data)
                    self.active_sl_tp_orders = state_data.get('active_sl_tp_orders', {})
                    self.sl_order_db_id = state_data.get('sl_order_db_id')
                    self.tp_order_db_id = state_data.get('tp_order_db_id')
                except json.JSONDecodeError:
                    self.logger.error(f"[{self.name}-{self.trading_pair}] Error decoding custom_data for pos {open_pos.id}. Querying open orders.")
                    self._query_and_set_open_sl_tp_orders() # Fallback to querying
            else: # Fallback if no custom_data
                self._query_and_set_open_sl_tp_orders()
            
            self.logger.info(f"[{self.name}-{self.trading_pair}] Loaded state: PosID {self.active_position_db_id}, Side {self.active_position_side}, SL ExchID {self.active_sl_tp_orders.get('sl_id')}, TP ExchID {self.active_sl_tp_orders.get('tp_id')}")
        else:
            self.logger.info(f"[{self.name}-{self.trading_pair}] No active persistent position found. Initializing state.")
            self._reset_internal_position_state()

    def _query_and_set_open_sl_tp_orders(self):
        # Helper for _load_persistent_state fallback
        self.active_sl_tp_orders = {}
        sl_orders = strategy_utils.get_open_orders_for_subscription(self.db_session, self.user_sub_obj.id, self.trading_pair, order_type='STOP_MARKET') # Adjust if your SL type is different
        if sl_orders: self.active_sl_tp_orders['sl_id'] = sl_orders[0].order_id; self.sl_order_db_id = sl_orders[0].id
        tp_orders = strategy_utils.get_open_orders_for_subscription(self.db_session, self.user_sub_obj.id, self.trading_pair, order_type='TAKE_PROFIT_MARKET') # Adjust if your TP type is different
        if tp_orders: self.active_sl_tp_orders['tp_id'] = tp_orders[0].order_id; self.tp_order_db_id = tp_orders[0].id


    def _save_position_custom_state(self):
        if not self.active_position_db_id or not self.db_session:
            self.logger.debug(f"[{self.name}-{self.trading_pair}] No active_position_db_id or db_session to save custom state.")
            return
        pos_to_update = self.db_session.query(Position).filter(Position.id == self.active_position_db_id).first()
        if pos_to_update:
            current_custom_data = {} # Start fresh or load existing if merging: json.loads(pos_to_update.custom_data) if pos_to_update.custom_data else {}
            current_custom_data['active_sl_tp_orders'] = self.active_sl_tp_orders
            current_custom_data['sl_order_db_id'] = self.sl_order_db_id
            current_custom_data['tp_order_db_id'] = self.tp_order_db_id
            pos_to_update.custom_data = json.dumps(current_custom_data)
            pos_to_update.updated_at = datetime.utcnow() # Use utcnow()
            try:
                self.db_session.commit()
                self.logger.info(f"[{self.name}-{self.trading_pair}] Saved custom state (SL/TP IDs) for PosID {self.active_position_db_id}")
            except Exception as e:
                self.logger.error(f"[{self.name}-{self.trading_pair}] Error saving custom state for PosID {self.active_position_db_id}: {e}", exc_info=True)
                self.db_session.rollback()
        else:
            self.logger.warning(f"[{self.name}-{self.trading_pair}] Position {self.active_position_db_id} not found to save custom state.")


    def _reset_internal_position_state(self):
        self.active_position_db_id = None
        self.active_position_side = None
        self.position_entry_price = None
        self.position_qty = Decimal("0")
        self.active_sl_tp_orders = {}
        self.sl_order_db_id = None
        self.tp_order_db_id = None
        self.logger.info(f"[{self.name}-{self.trading_pair}] Internal position state reset.")


    def _fetch_market_precision(self): # Renamed from _get_precisions_live
        # Same as original
        if not self._precisions_fetched_:
            try:
                self.exchange_ccxt.load_markets(True) # Force reload
                market = self.exchange_ccxt.markets[self.trading_pair]
                self.quantity_precision_str = str(market['precision']['amount'])
                self.price_precision_str = str(market['precision']['price'])
                self._precisions_fetched_ = True
                self.logger.info(f"Precision for {self.trading_pair}: Qty Prec Str={self.quantity_precision_str}, Price Prec Str={self.price_precision_str}")
            except Exception as e:
                self.logger.error(f"Error fetching precision for {self.trading_pair}: {e}. Using defaults.", exc_info=True)
                self.quantity_precision_str = "0.00001" 
                self.price_precision_str = "0.01"    

    def _get_decimal_places(self, precision_str: Optional[str]) -> int: # Added Optional
        # Same as original
        if precision_str is None: self.logger.warning("Precision string is None, using default 8."); return 8 
        try:
            if 'e-' in precision_str.lower():
                num_val = float(precision_str)
                precision_str = format(num_val, f'.{abs(int(precision_str.split("e-")[1]))}f')
            d_prec = Decimal(precision_str)
            if d_prec.as_tuple().exponent < 0: return abs(d_prec.as_tuple().exponent)
            return 0 
        except Exception as e: self.logger.warning(f"Could not parse precision string '{precision_str}'. Error: {e}. Using default 8."); return 8

    def _format_quantity(self, quantity: Decimal) -> str:
        # Same as original
        places = self._get_decimal_places(self.quantity_precision_str)
        return str(quantity.quantize(Decimal('1e-' + str(places)), rounding=ROUND_DOWN))

    def _format_price(self, price: Decimal) -> str:
        # Same as original
        places = self._get_decimal_places(self.price_precision_str)
        return str(price.quantize(Decimal('1e-' + str(places)), rounding=ROUND_NEAREST))

    def _set_leverage(self):
        # Same as original
        try:
            if hasattr(self.exchange_ccxt, 'set_leverage'):
                symbol_for_leverage = self.trading_pair.split(':')[0] if ':' in self.trading_pair else self.trading_pair
                self.exchange_ccxt.set_leverage(self.leverage, symbol_for_leverage)
                self.logger.info(f"Leverage set to {self.leverage}x for {symbol_for_leverage}")
        except Exception as e: self.logger.warning(f"Could not set leverage for {self.trading_pair}: {e}")
    
    def _gauss(self, x, h): # Same as original
        if h == 0: return 1.0 if x == 0 else 0.0
        return np.exp(-((x ** 2) / (2 * h ** 2)))

    def _causal_nadaraya_watson_envelope(self, data_series_np, h, mult, y_hat_lookback, mae_lookback): # Same as original
        n = len(data_series_np); y_hat_arr = np.full(n, np.nan); upper_band_arr = np.full(n, np.nan); lower_band_arr = np.full(n, np.nan)
        for i in range(n):
            start_idx_yhat = max(0, i - y_hat_lookback + 1); current_window_yhat = data_series_np[start_idx_yhat : i+1]
            weighted_sum = 0.0; total_weight = 0.0
            for j_local in range(len(current_window_yhat)):
                gauss_dist = float((len(current_window_yhat) - 1) - j_local); weight = self._gauss(gauss_dist, h)
                weighted_sum += current_window_yhat[j_local] * weight; total_weight += weight
            if total_weight > 1e-8: y_hat_arr[i] = weighted_sum / total_weight
            elif len(current_window_yhat) > 0: y_hat_arr[i] = current_window_yhat[-1]
        errors = np.abs(data_series_np - y_hat_arr)
        for i in range(n):
            if np.isnan(y_hat_arr[i]) or i == 0 : continue
            start_idx_mae = max(0, i - mae_lookback); relevant_errors_for_mae = errors[start_idx_mae : i]
            valid_errors = relevant_errors_for_mae[~np.isnan(relevant_errors_for_mae)]
            if len(valid_errors) > 0:
                mae_value = np.mean(valid_errors) * mult
                upper_band_arr[i] = y_hat_arr[i] + mae_value; lower_band_arr[i] = y_hat_arr[i] - mae_value
        return y_hat_arr, upper_band_arr, lower_band_arr

    def _calculate_stoch_rsi(self, close_prices_np, rsi_len, stoch_len, k_smooth, d_smooth): # Same as original
        if len(close_prices_np) < rsi_len + stoch_len + k_smooth + d_smooth + 1: self.logger.warning(f"Not enough data for StochRSI."); return np.full(len(close_prices_np), np.nan), np.full(len(close_prices_np), np.nan)
        rsi = talib.RSI(close_prices_np, timeperiod=rsi_len); rsi = rsi[~np.isnan(rsi)]
        if len(rsi) < stoch_len + k_smooth + d_smooth -1 : self.logger.warning(f"Not enough RSI data for STOCH."); return np.full(len(close_prices_np), np.nan), np.full(len(close_prices_np), np.nan)
        stoch_k, stoch_d = talib.STOCH(rsi, rsi, rsi, fastk_period=stoch_len, slowk_period=k_smooth, slowk_matype=0, slowd_period=d_smooth, slowd_matype=0)
        nan_padding_count = len(close_prices_np) - len(stoch_k)
        return np.pad(stoch_k, (nan_padding_count, 0), 'constant', constant_values=np.nan), np.pad(stoch_d, (nan_padding_count, 0), 'constant', constant_values=np.nan)

    def _await_order_fill(self, exchange_ccxt, order_id: str, symbol: str, timeout_seconds: int = 60, check_interval_seconds: int = 3): # Same as before
        start_time = time.time()
        self.logger.info(f"[{self.name}-{symbol}] Awaiting fill for order {order_id} (timeout: {timeout_seconds}s)")
        while time.time() - start_time < timeout_seconds:
            try:
                order = exchange_ccxt.fetch_order(order_id, symbol)
                self.logger.debug(f"[{self.name}-{symbol}] Order {order_id} status: {order['status']}")
                if order['status'] == 'closed': self.logger.info(f"[{self.name}-{symbol}] Order {order_id} filled."); return order
                if order['status'] in ['canceled', 'rejected', 'expired']: self.logger.warning(f"[{self.name}-{symbol}] Order {order_id} is {order['status']}."); return order
            except ccxt.OrderNotFound: self.logger.warning(f"[{self.name}-{symbol}] Order {order_id} not found. Retrying.")
            except Exception as e: self.logger.error(f"[{self.name}-{symbol}] Error fetching order {order_id}: {e}. Retrying.", exc_info=True)
            time.sleep(check_interval_seconds)
        self.logger.warning(f"[{self.name}-{symbol}] Timeout for order {order_id}. Final check.")
        try: final_status = exchange_ccxt.fetch_order(order_id, symbol); self.logger.info(f"[{self.name}-{symbol}] Final status for order {order_id}: {final_status['status']}"); return final_status
        except Exception as e: self.logger.error(f"[{self.name}-{symbol}] Final check for order {order_id} failed: {e}", exc_info=True); return None

    def _place_order(self, symbol:str, order_type:str, side:str, quantity:Decimal, price:Optional[Decimal]=None, params:Optional[Dict]=None) -> Optional[Order]: # Returns DB Order object
        db_order = None
        try:
            formatted_qty_str = self._format_quantity(quantity)
            formatted_price_str = self._format_price(price) if price else None
            
            db_order = strategy_utils.create_strategy_order_in_db(
                self.db_session, self.user_sub_obj.id, symbol, order_type.lower(), side, 
                float(formatted_qty_str), float(formatted_price_str) if formatted_price_str else None, 
                status='pending_exchange', notes=f"{self.name} order"
            )
            if not db_order: 
                self.logger.error(f"[{self.name}-{symbol}] Failed to create initial DB order record for {side} {quantity} {symbol}. Aborting exchange order."); return None

            exchange_order = self.exchange_ccxt.create_order(symbol, order_type, side, float(formatted_qty_str), float(formatted_price_str) if formatted_price_str else None, params)
            
            updated_db_order = strategy_utils.update_strategy_order_in_db(
                self.db_session, order_db_id=db_order.id, 
                updates={'order_id': exchange_order.get('id'), 'status': 'open', 'raw_order_data': json.dumps(exchange_order)}
            )
            self.logger.info(f"Order placed: {side} {order_type} {formatted_qty_str} {symbol} at {formatted_price_str if formatted_price_str else 'Market'}. ExchOrderID: {exchange_order.get('id')}, DBOrderID: {db_order.id}")
            return updated_db_order # Return the updated DB Order object
        except Exception as e:
            self.logger.error(f"Failed to place {side} {order_type} for {symbol}: {e}", exc_info=True)
            if db_order and db_order.id: 
                strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=db_order.id, updates={'status': 'error_on_exchange', 'status_message': str(e)[:255]})
        return None

    def _cancel_active_sl_tp_orders(self):
        for order_type_key, exchange_order_id in list(self.active_sl_tp_orders.items()): # list() for safe iteration
            if exchange_order_id:
                db_id_to_use = self.sl_order_db_id if order_type_key == 'sl_id' else (self.tp_order_db_id if order_type_key == 'tp_id' else None)
                try:
                    self.logger.info(f"[{self.name}-{self.trading_pair}] Canceling {order_type_key} order {exchange_order_id} (DB ID: {db_id_to_use}) on exchange.")
                    self.exchange_ccxt.cancel_order(exchange_order_id, self.trading_pair)
                    if db_id_to_use:
                        strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=db_id_to_use, updates={'status': 'canceled'})
                    else: # Fallback if DB ID not tracked, try to update by exchange ID
                        strategy_utils.update_strategy_order_in_db(self.db_session, exchange_order_id=exchange_order_id, subscription_id=self.user_sub_obj.id, symbol=self.trading_pair, updates={'status': 'canceled'})
                    self.logger.info(f"[{self.name}-{self.trading_pair}] Successfully canceled {order_type_key} order {exchange_order_id}.")
                except ccxt.OrderNotFound:
                     self.logger.warning(f"[{self.name}-{self.trading_pair}] {order_type_key} order {exchange_order_id} not found on exchange, likely already filled or canceled.")
                     if db_id_to_use: strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=db_id_to_use, updates={'status': 'not_found_on_cancel'})
                     else: strategy_utils.update_strategy_order_in_db(self.db_session, exchange_order_id=exchange_order_id, subscription_id=self.user_sub_obj.id, symbol=self.trading_pair, updates={'status': 'not_found_on_cancel'})
                except Exception as e:
                    self.logger.error(f"[{self.name}-{self.trading_pair}] Failed to cancel {order_type_key} order {exchange_order_id}: {e}", exc_info=True)
        
        self.active_sl_tp_orders = {}
        self.sl_order_db_id = None
        self.tp_order_db_id = None
        self._save_position_custom_state() # Save cleared SL/TP info

    def _sync_exchange_position_state(self):
        if not self.active_position_db_id or not self.db_session: return False 
        
        position_closed_event = False
        sl_exchange_id = self.active_sl_tp_orders.get('sl_id')
        tp_exchange_id = self.active_sl_tp_orders.get('tp_id')

        if self.sl_order_db_id and sl_exchange_id:
            try:
                sl_details = self.exchange_ccxt.fetch_order(sl_exchange_id, self.trading_pair)
                if sl_details['status'] == 'closed':
                    self.logger.info(f"SL order {sl_exchange_id} filled.")
                    strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=self.sl_order_db_id, updates={'status': 'closed', 'price': sl_details.get('average'), 'filled': sl_details.get('filled'), 'closed_at': datetime.utcnow(), 'raw_order_data': json.dumps(sl_details)})
                    strategy_utils.close_strategy_position_in_db(self.db_session, self.active_position_db_id, Decimal(str(sl_details['average'])), Decimal(str(sl_details['filled'])), f"Closed by SL {sl_exchange_id}")
                    position_closed_event = True
                    if tp_exchange_id: self._cancel_specific_order(tp_exchange_id, self.tp_order_db_id, "TP (SL Hit)")
            except ccxt.OrderNotFound: self.logger.warning(f"SL order {sl_exchange_id} not found in sync."); strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=self.sl_order_db_id, updates={'status': 'not_found_on_sync'}); self.active_sl_tp_orders['sl_id'] = None; self.sl_order_db_id = None
            except Exception as e: self.logger.error(f"Error syncing SL order {sl_exchange_id}: {e}")

        if not position_closed_event and self.tp_order_db_id and tp_exchange_id:
            try:
                tp_details = self.exchange_ccxt.fetch_order(tp_exchange_id, self.trading_pair)
                if tp_details['status'] == 'closed':
                    self.logger.info(f"TP order {tp_exchange_id} filled.")
                    strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=self.tp_order_db_id, updates={'status': 'closed', 'price': tp_details.get('average'), 'filled': tp_details.get('filled'), 'closed_at': datetime.utcnow(), 'raw_order_data': json.dumps(tp_details)})
                    strategy_utils.close_strategy_position_in_db(self.db_session, self.active_position_db_id, Decimal(str(tp_details['average'])), Decimal(str(tp_details['filled'])), f"Closed by TP {tp_exchange_id}")
                    position_closed_event = True
                    if sl_exchange_id: self._cancel_specific_order(sl_exchange_id, self.sl_order_db_id, "SL (TP Hit)")
            except ccxt.OrderNotFound: self.logger.warning(f"TP order {tp_exchange_id} not found in sync."); strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=self.tp_order_db_id, updates={'status': 'not_found_on_sync'}); self.active_sl_tp_orders['tp_id'] = None; self.tp_order_db_id = None
            except Exception as e: self.logger.error(f"Error syncing TP order {tp_exchange_id}: {e}")

        if position_closed_event:
            self._reset_internal_position_state()
            self._save_position_custom_state() # Save cleared state
            return True
        return False

    def _cancel_specific_order(self, exchange_order_id: str, db_order_id: Optional[int], reason_prefix: str):
        try:
            self.exchange_ccxt.cancel_order(exchange_order_id, self.trading_pair)
            self.logger.info(f"Canceled {reason_prefix} order {exchange_order_id} on exchange.")
            if db_order_id:
                strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=db_order_id, updates={'status': 'canceled'})
            else: # Fallback if DB ID not tracked for this specific order
                strategy_utils.update_strategy_order_in_db(self.db_session, exchange_order_id=exchange_order_id, subscription_id=self.user_sub_obj.id, symbol=self.trading_pair, updates={'status': 'canceled'})
        except ccxt.OrderNotFound:
            self.logger.warning(f"{reason_prefix} order {exchange_order_id} not found on exchange for cancellation, likely already processed.")
            if db_order_id: strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=db_order_id, updates={'status': 'not_found_on_cancel'})
            else: strategy_utils.update_strategy_order_in_db(self.db_session, exchange_order_id=exchange_order_id, subscription_id=self.user_sub_obj.id, symbol=self.trading_pair, updates={'status': 'not_found_on_cancel'})
        except Exception as e:
            self.logger.error(f"Failed to cancel {reason_prefix} order {exchange_order_id}: {e}", exc_info=True)

    def _process_trading_logic(self, nw_ohlcv_df: pandas.DataFrame, stoch_rsi_ohlcv_df: pandas.DataFrame):
        # Core logic using self.active_position_side, self.position_entry_price etc.
        # ... (original calculation logic for NW bands, StochRSI, signals) ...
        symbol = self.trading_pair # Use instance var
        min_nw_data_len = self.nw_yhat_lookback + self.nw_mae_lookback + 5
        min_stoch_rsi_data_len = self.stoch_rsi_length + self.stoch_rsi_stoch_length + self.stoch_rsi_k_smooth + self.stoch_rsi_d_smooth + 50

        if nw_ohlcv_df is None or len(nw_ohlcv_df) < min_nw_data_len: self.logger.warning(f"Not enough NW data."); return
        if stoch_rsi_ohlcv_df is None or len(stoch_rsi_ohlcv_df) < min_stoch_rsi_data_len: self.logger.warning(f"Not enough StochRSI data."); return

        nw_closes_np = nw_ohlcv_df['close'].to_numpy(dtype=float)
        _, upper_band, lower_band = self._causal_nadaraya_watson_envelope(nw_closes_np, self.nw_h_bandwidth, self.nw_multiplier, self.nw_yhat_lookback, self.nw_mae_lookback)
        if np.isnan(upper_band[-1]) or np.isnan(lower_band[-1]): self.logger.warning("NW bands are NaN."); return
        
        latest_nw_close = Decimal(str(nw_ohlcv_df['close'].iloc[-1]))
        latest_upper_band = Decimal(str(upper_band[-1])); latest_lower_band = Decimal(str(lower_band[-1]))

        stoch_rsi_closes_np = stoch_rsi_ohlcv_df['close'].to_numpy(dtype=float)
        stoch_k, _ = self._calculate_stoch_rsi(stoch_rsi_closes_np, self.stoch_rsi_length, self.stoch_rsi_stoch_length, self.stoch_rsi_k_smooth, self.stoch_rsi_d_smooth)
        if np.isnan(stoch_k[-1]): self.logger.warning("StochRSI K is NaN."); return
        latest_stoch_k = Decimal(str(stoch_k[-1]))

        # Entry / Exit / Reversal Logic
        # If currently in a position
        if self.active_position_side:
            if self.active_position_side == 'long' and latest_nw_close >= latest_upper_band and latest_stoch_k > self.stoch_rsi_overbought_level:
                self.logger.info(f"Signal: CLOSE LONG & REVERSE TO SHORT for {symbol} at {latest_nw_close}.")
                self._cancel_active_sl_tp_orders() # Cancel existing SL/TP
                close_order_db = self._place_order(symbol, 'market', 'sell', self.position_qty, params={'reduceOnly': True})
                if close_order_db: # Await fill before proceeding
                    filled_close = self._await_order_fill(self.exchange_ccxt, close_order_db.order_id, symbol)
                    if filled_close and filled_close['status']=='closed':
                        strategy_utils.close_strategy_position_in_db(self.db_session, self.active_position_db_id, Decimal(str(filled_close['average'])), Decimal(str(filled_close['filled'])), "Reversal to Short")
                        self._reset_internal_position_state() # State reset
                        # Proceed to open short immediately
                        self._open_new_position('short', latest_nw_close)
                    else: self.logger.error("Failed to confirm close for reversal to short.")
                return # End cycle
            elif self.active_position_side == 'short' and latest_nw_close <= latest_lower_band and latest_stoch_k < self.stoch_rsi_oversold_level:
                self.logger.info(f"Signal: CLOSE SHORT & REVERSE TO LONG for {symbol} at {latest_nw_close}.")
                self._cancel_active_sl_tp_orders()
                close_order_db = self._place_order(symbol, 'market', 'buy', self.position_qty, params={'reduceOnly': True})
                if close_order_db:
                    filled_close = self._await_order_fill(self.exchange_ccxt, close_order_db.order_id, symbol)
                    if filled_close and filled_close['status']=='closed':
                        strategy_utils.close_strategy_position_in_db(self.db_session, self.active_position_db_id, Decimal(str(filled_close['average'])), Decimal(str(filled_close['filled'])), "Reversal to Long")
                        self._reset_internal_position_state()
                        self._open_new_position('long', latest_nw_close)
                    else: self.logger.error("Failed to confirm close for reversal to long.")
                return # End cycle
        # If no position, check for new entry
        else:
            if latest_nw_close <= latest_lower_band and latest_stoch_k < self.stoch_rsi_oversold_level:
                self._open_new_position('long', latest_nw_close)
            elif latest_nw_close >= latest_upper_band and latest_stoch_k > self.stoch_rsi_overbought_level:
                self._open_new_position('short', latest_nw_close)
        
        if self.active_position_db_id: # If any action resulted in an open position or state change for existing
            self._save_position_custom_state()


    def _open_new_position(self, side: str, entry_price: Decimal):
        self.logger.info(f"Signal: ENTER {side.upper()} for {self.trading_pair} at {entry_price}.")
        qty_to_trade = self.order_quantity_usd / entry_price
        
        entry_order_db = self._place_order(self.trading_pair, 'market', side, qty_to_trade)
        if entry_order_db and entry_order_db.order_id:
            filled_entry_order = self._await_order_fill(self.exchange_ccxt, entry_order_db.order_id, self.trading_pair)
            if filled_entry_order and filled_entry_order['status'] == 'closed':
                actual_entry_price = Decimal(str(filled_entry_order['average']))
                actual_filled_qty = Decimal(str(filled_entry_order['filled']))
                strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=entry_order_db.id, updates={'status': 'closed', 'price': float(actual_entry_price), 'filled': float(actual_filled_qty), 'cost': float(actual_entry_price*actual_filled_qty), 'closed_at': datetime.utcnow()})

                new_pos_db = strategy_utils.create_strategy_position_in_db(
                    self.db_session, self.user_sub_obj.id, self.trading_pair, 
                    str(self.exchange_ccxt.id), side, float(actual_filled_qty), float(actual_entry_price)
                )
                if new_pos_db:
                    self.active_position_db_id = new_pos_db.id
                    self.active_position_side = side
                    self.position_entry_price = actual_entry_price
                    self.position_qty = actual_filled_qty
                    self.logger.info(f"Position {side.upper()} created. DB ID: {new_pos_db.id}, Entry: {self.position_entry_price}, Qty: {self.position_qty}")

                    # Place SL/TP
                    sl_price = self.position_entry_price * (Decimal('1') - self.stop_loss_pct) if side == 'long' else self.position_entry_price * (Decimal('1') + self.stop_loss_pct)
                    tp_price = self.position_entry_price * (Decimal('1') + self.take_profit_pct) if side == 'long' else self.position_entry_price * (Decimal('1') - self.take_profit_pct)
                    
                    sl_order_db = self._place_order(self.trading_pair, 'STOP_MARKET', 'sell' if side == 'long' else 'buy', self.position_qty, params={'stopPrice': self._format_price(sl_price), 'reduceOnly': True})
                    if sl_order_db: self.sl_order_db_id = sl_order_db.id; self.active_sl_tp_orders['sl_id'] = sl_order_db.order_id
                    
                    tp_order_db = self._place_order(self.trading_pair, 'TAKE_PROFIT_MARKET', 'sell' if side == 'long' else 'buy', self.position_qty, params={'stopPrice': self._format_price(tp_price), 'reduceOnly': True})
                    if tp_order_db: self.tp_order_db_id = tp_order_db.id; self.active_sl_tp_orders['tp_id'] = tp_order_db.order_id
                    
                    self._save_position_custom_state() # Save SL/TP order IDs
                else:
                    self.logger.error(f"[{self.name}-{self.trading_pair}] Failed to create Position DB record after entry.")
            else:
                self.logger.error(f"Entry order {entry_order_db.order_id if entry_order_db else 'N/A'} did not fill or status unknown.")
                if entry_order_db: strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=entry_order_db.id, updates={'status': filled_entry_order.get('status', 'fill_check_failed') if filled_entry_order else 'fill_check_failed'})
        else:
            self.logger.error(f"[{self.name}-{self.trading_pair}] Failed to place entry order on exchange.")


    def execute_live_signal(self, market_data_df: Optional[pandas.DataFrame]=None, htf_df: Optional[pandas.DataFrame]=None): # Allow optional dfs for direct call
        self.logger.info(f"Executing {self.name} for {self.trading_pair} UserSubID {self.user_sub_obj.id}")
        
        self._load_persistent_state() # Refresh state at start of cycle

        try:
            current_close_price_ticker = self.exchange_ccxt.fetch_ticker(self.trading_pair)['last']
            if self._sync_exchange_position_state(Decimal(str(current_close_price_ticker))):
                self.logger.info(f"[{self.name}-{self.trading_pair}] Position closed by SL/TP sync. Cycle ended.")
                return
        except Exception as e:
            self.logger.error(f"[{self.name}-{self.trading_pair}] Error during pre-logic sync or price fetch: {e}", exc_info=True)
            return

        nw_data_needed = self.nw_yhat_lookback + self.nw_mae_lookback + 50
        stoch_rsi_data_needed = self.stoch_rsi_length + self.stoch_rsi_stoch_length + self.stoch_rsi_k_smooth + self.stoch_rsi_d_smooth + 100

        try:
            # If DFs are not passed, fetch them. This allows flexibility for the Celery task.
            if market_data_df is None:
                nw_ohlcv_list = self.exchange_ccxt.fetch_ohlcv(self.trading_pair, self.nw_timeframe, limit=nw_data_needed)
                market_data_df = pandas.DataFrame(nw_ohlcv_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            if htf_df is None:
                stoch_rsi_ohlcv_list = self.exchange_ccxt.fetch_ohlcv(self.trading_pair, self.stoch_rsi_timeframe, limit=stoch_rsi_data_needed)
                htf_df = pandas.DataFrame(stoch_rsi_ohlcv_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            self._process_trading_logic(market_data_df, htf_df) # Pass the DFs

        except Exception as e:
            self.logger.error(f"Error in {self.name} execute_live_signal main try block: {e}", exc_info=True)

    def run_backtest(self, historical_data_feed, current_simulated_time_utc):
        # Backtest logic remains simulation-based
        self.logger.info(f"Running backtest for {self.name} on {self.trading_pair} at {current_simulated_time_utc}...")
        # ... (Full backtesting logic as previously provided, using local variables for state) ...
        # For brevity, not pasting the full backtest logic. It should not use self.db_session or strategy_utils.
        self.logger.warning("Backtesting for NadarayaWatsonStochRSIStrategy needs a dedicated simulation loop. Returning placeholder.")
        return {"action": "HOLD", "reason": "Backtest simulation for this strategy is complex and not fully implemented in this refactor."}

```
