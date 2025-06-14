import time
import math
import logging
import pandas as pd
import datetime 
import json 
from sqlalchemy.orm import Session 
from backend.models import Position, Order, UserStrategySubscription # Added UserStrategySubscription
from backend import strategy_utils # Import the new strategy_utils
import ccxt # For ccxt.OrderNotFound

logger = logging.getLogger(__name__)

class DCAStrategy:
    def __init__(self, symbol: str, timeframe: str, db_session: Session, user_sub_obj: UserStrategySubscription, capital: float = 10000, **custom_parameters):
        self.name = "DCA Strategy"
        self.symbol = symbol
        self.timeframe = timeframe 
        
        # Store db_session and user_sub_obj
        self.db_session = db_session
        self.user_sub_obj = user_sub_obj
        self.logger = logger # Use module-level logger

        defaults = {
            "base_order_size_usdt": 10.0,
            "safety_order_size_usdt": 10.0,
            "tp1_percent": 1.0,
            "tp1_sell_percent": 40.0,
            "tp2_percent": 3.0,
            "tp2_sell_percent": 40.0,
            "tp3_percent": 20.0, 
            "stop_loss_percent": 5.0,
            "safety_order_deviation_percent": 1.0,
            "safety_order_scale_factor": 1.0, 
            "max_safety_orders": 5,
            # New parameters
            "order_fill_timeout_seconds": 60,
            "order_fill_check_interval_seconds": 3,
            "min_asset_qty_for_closure": 0.000001
        }
        
        self_params = {**defaults, **custom_parameters}
        for key, value in self_params.items():
            setattr(self, key, value)

        self.tp1_sell_multiplier = self.tp1_sell_percent / 100.0
        self.tp2_sell_multiplier = self.tp2_sell_percent / 100.0

        # Initialize new parameters
        self.order_fill_timeout_seconds = int(self_params.get("order_fill_timeout_seconds", 60))
        self.order_fill_check_interval_seconds = int(self_params.get("order_fill_check_interval_seconds", 3))
        self.min_asset_qty_for_closure = float(self_params.get("min_asset_qty_for_closure", 0.000001))
        
        self.price_precision = 8
        self.quantity_precision = 8
        self._precisions_fetched_ = False

        # State attributes
        self.active_position_db_id = None
        self.position_db: Optional[Position] = None # Store the ORM object
        self.dca_state = { # Initialize with defaults
            'entry_price_avg': 0.0, 'current_position_size_asset': 0.0,
            'total_usdt_invested': 0.0, 'safety_orders_placed_count': 0, # Count of *filled* SOs
            'take_profit_prices': [], 'current_stop_loss_price': 0.0,
            'tp_levels_hit': [False, False, False], 
            'open_safety_order_ids': {} # Stores {db_order_id: exchange_order_id} for open SOs
        }

        logger.info(f"[{self.name}-{self.symbol}] Initializing for SubID {self.user_sub_obj.id} with params: {self_params}")
        self._load_dca_persistent_state()

    def _load_dca_persistent_state(self):
        if not self.db_session or not self.user_sub_obj:
            self.logger.warning(f"[{self.name}-{self.symbol}] DB session or user_sub_obj not available for loading state.")
            return

        open_position = strategy_utils.get_open_strategy_position(
            self.db_session, self.user_sub_obj.id, self.symbol
        )
        if open_position:
            self.logger.info(f"[{self.name}-{self.symbol}] Loading persistent state for open position ID: {open_position.id}")
            self.active_position_db_id = open_position.id
            self.position_db = open_position 
            
            loaded_dca_state = json.loads(open_position.custom_data) if open_position.custom_data else {}
            
            default_state_keys = {
                'entry_price_avg': open_position.entry_price, 
                'current_position_size_asset': open_position.amount,
                'total_usdt_invested': loaded_dca_state.get('total_usdt_invested', open_position.entry_price * open_position.amount), # Prefer loaded, fallback to approximation
                'safety_orders_placed_count': 0, 
                'take_profit_prices': [], 
                'current_stop_loss_price': 0.0,
                'tp_levels_hit': [False, False, False],
                'open_safety_order_ids': {} 
            }
            self.dca_state = {**default_state_keys, **loaded_dca_state}
            
            # Ensure essential fields are present and correctly typed after loading
            self.dca_state['entry_price_avg'] = float(self.dca_state.get('entry_price_avg', open_position.entry_price or 0.0))
            self.dca_state['current_position_size_asset'] = float(self.dca_state.get('current_position_size_asset', open_position.amount or 0.0))
            self.dca_state['total_usdt_invested'] = float(self.dca_state.get('total_usdt_invested', 0.0))
            self.dca_state['safety_orders_placed_count'] = int(self.dca_state.get('safety_orders_placed_count', 0))
            self.dca_state['tp_levels_hit'] = self.dca_state.get('tp_levels_hit', [False,False,False])
            self.dca_state['open_safety_order_ids'] = self.dca_state.get('open_safety_order_ids', {})


            if not self.dca_state.get('take_profit_prices') or not self.dca_state.get('current_stop_loss_price') or not self.dca_state['take_profit_prices']:
                prices, sl = self._calculate_take_profits_and_sl(self.dca_state['entry_price_avg'])
                self.dca_state['take_profit_prices'] = prices
                self.dca_state['current_stop_loss_price'] = sl
            
            self.logger.info(f"[{self.name}-{self.symbol}] Loaded DCA state: {self.dca_state}")
        else:
            self.logger.info(f"[{self.name}-{self.symbol}] No active persistent DCA position found. Initializing default state.")
            self.active_position_db_id = None
            self.position_db = None 
            # Default state already initialized in __init__

    @classmethod
    def validate_parameters(cls, params: dict) -> dict:
        """Validates strategy-specific parameters."""
        definition = cls.get_parameters_definition()
        validated_params = {}

        # Check for unknown parameters first
        for key in params:
            if key not in definition:
                # Allow 'symbol', 'timeframe', 'capital' as they might be passed by backtesting engine
                # but are not part of user-configurable DCA specific params.
                # However, the DCA strategy __init__ directly uses symbol, timeframe, capital.
                # For custom_parameters specifically, these should not be present if they are fixed.
                # This validation is for the `custom_parameters` part of strategy_params.
                # For this DCA strategy, it's simpler if __init__ receives all params and validator checks all.
                # Let's assume for now 'params' to validate are ONLY the ones in definition.
                # If other params like 'symbol' are passed at a higher level, they are not part of this dict.
                pass # Silently ignore extra params not in definition for now, or raise error if strictness needed.
                # raise ValueError(f"Unknown parameter '{key}' provided for DCA strategy.")


        for key, def_value in definition.items():
            val_type_str = def_value.get("type")
            choices = def_value.get("options") # "options" not "choices"
            min_val = def_value.get("min")
            max_val = def_value.get("max")
            default_val = def_value.get("default")

            user_val = params.get(key)

            if user_val is None: # Parameter not provided by user
                if default_val is not None:
                    user_val = default_val # Apply default
                else:
                    # This case implies a required parameter without a default is missing.
                    # Depending on design, could raise error or let it pass if __init__ handles it.
                    # For now, let __init__ handle it if it's truly optional without default.
                    # If it's always required, get_parameters_definition should not have it as optional.
                    # Or, raise ValueError(f"Required parameter '{key}' is missing and has no default.")
                    pass # Let it pass if not strictly required by definition to have a value here

            if user_val is not None: # If value is present (either user-provided or default)
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
                elif val_type_str == "string":
                    if not isinstance(user_val, str):
                        raise ValueError(f"Parameter '{key}' must be a string. Got: {user_val}")
                elif val_type_str == "bool": # DCA strategy doesn't have bools in current def
                    if not isinstance(user_val, bool):
                        if str(user_val).lower() in ['true', 'yes', '1']: user_val = True
                        elif str(user_val).lower() in ['false', 'no', '0']: user_val = False
                        else: raise ValueError(f"Parameter '{key}' must be a boolean. Got: {user_val}")
                
                # Choice validation (DCA strategy doesn't have 'choices' in current def)
                if choices and user_val not in choices:
                    raise ValueError(f"Parameter '{key}' value '{user_val}' is not in valid choices: {choices}")
                
                # Min/Max validation
                if val_type_str in ["int", "float"]:
                    if min_val is not None and user_val < min_val:
                        raise ValueError(f"Parameter '{key}' value {user_val} is less than min {min_val}.")
                    if max_val is not None and user_val > max_val:
                        raise ValueError(f"Parameter '{key}' value {user_val} is greater than max {max_val}.")
            
            validated_params[key] = user_val
            
        # Final check for any params passed in `params` that were not in `definition`
        # This is important if we don't raise error for unknown params at the beginning.
        for key_param in params:
            if key_param not in definition:
                # This is where you decide if extra parameters are truly an error.
                # For DCA, if 'symbol', 'timeframe', 'capital' are passed in this dict, they are unexpected
                # as they are explicit __init__ args.
                # For now, let's assume they are handled by the calling context of validate_parameters.
                # If validate_parameters is meant to receive ONLY custom_parameters, then this check is good.
                # If it receives ALL params that __init__ would, then need to adjust.
                # Given the task, this validator should work on the `custom_parameters` part.
                cls.logger.warning(f"Parameter '{key_param}' was provided but is not in DCA strategy definition. It will be ignored by strategy logic if not an explicit __init__ arg.")
                # To be stricter, one might add: validated_params[key_param] = params[key_param]
                # or raise ValueError here. For now, it means they are just passed through if not defined.


        return validated_params

    @classmethod
    def get_parameters_definition(cls):
        return { # Parameters as defined previously
            "base_order_size_usdt": {"type": "float", "default": 10.0, "label": "Base Order Size (USDT)"},
            "safety_order_size_usdt": {"type": "float", "default": 10.0, "label": "Safety Order Size (USDT)"},
            "tp1_percent": {"type": "float", "default": 1.0, "label": "Take Profit 1 (%)"},
            "tp1_sell_percent": {"type": "float", "default": 40.0, "label": "TP1 Sell (%)"},
            "tp2_percent": {"type": "float", "default": 3.0, "label": "Take Profit 2 (%)"},
            "tp2_sell_percent": {"type": "float", "default": 40.0, "label": "TP2 Sell (%)"},
            "tp3_percent": {"type": "float", "default": 20.0, "label": "Take Profit 3 (%)"}, # TP3 sells 100% of remaining
            "stop_loss_percent": {"type": "float", "default": 5.0, "label": "Stop Loss (%)"},
            "safety_order_deviation_percent": {"type": "float", "default": 1.0, "label": "Safety Order Deviation (%)"},
            "safety_order_scale_factor": {"type": "float", "default": 1.0, "label": "Safety Order Scale Factor"},
            "max_safety_orders": {"type": "int", "default": 5, "label": "Max Safety Orders"},
            # Definitions for new parameters
            "order_fill_timeout_seconds": {"type": "int", "default": 60, "min":10, "max":300, "label": "Order Fill Timeout (s)"},
            "order_fill_check_interval_seconds": {"type": "int", "default": 3, "min":1, "max":30, "label": "Order Fill Check Interval (s)"},
            "min_asset_qty_for_closure": {"type": "float", "default": 0.000001, "min":0.0, "label": "Min Asset Qty for Closure Check (Base Asset)"}
        }

    def _get_precisions(self, exchange_ccxt):
        if not self._precisions_fetched_:
            try:
                exchange_ccxt.load_markets(True)
                market = exchange_ccxt.market(self.symbol)
                self.price_precision = market['precision']['price']
                self.quantity_precision = market['precision']['amount']
                self._precisions_fetched_ = True
                self.logger.info(f"[{self.name}-{self.symbol}] Precisions: Price={self.price_precision}, Qty={self.quantity_precision}")
            except Exception as e:
                self.logger.error(f"[{self.name}-{self.symbol}] Error fetching precision: {e}", exc_info=True)

    def _format_price(self, price, exchange_ccxt):
        self._get_precisions(exchange_ccxt)
        return float(exchange_ccxt.price_to_precision(self.symbol, price))

    def _format_quantity(self, quantity, exchange_ccxt):
        self._get_precisions(exchange_ccxt)
        return float(exchange_ccxt.amount_to_precision(self.symbol, quantity))

    def _await_order_fill(self, exchange_ccxt, order_id: str, symbol: str): # Removed defaults, will use instance vars
        # Same as provided, but uses instance vars for timeout/interval
        start_time = time.time()
        self.logger.info(f"[{self.name}-{self.symbol}] Awaiting fill for order {order_id} (timeout: {self.order_fill_timeout_seconds}s)")
        while time.time() - start_time < self.order_fill_timeout_seconds:
            try:
                order = exchange_ccxt.fetch_order(order_id, symbol)
                self.logger.debug(f"[{self.name}-{self.symbol}] Order {order_id} status: {order['status']}")
                if order['status'] == 'closed':
                    self.logger.info(f"[{self.name}-{self.symbol}] Order {order_id} confirmed filled. Avg Price: {order.get('average')}, Filled Qty: {order.get('filled')}")
                    return order
                elif order['status'] in ['canceled', 'rejected', 'expired']:
                    self.logger.warning(f"[{self.name}-{self.symbol}] Order {order_id} is {order['status']}, will not be filled.")
                    return order
            except ccxt.OrderNotFound: self.logger.warning(f"[{self.name}-{self.symbol}] Order {order_id} not found. Retrying.")
            except Exception as e: self.logger.error(f"[{self.name}-{self.symbol}] Error fetching order {order_id}: {e}. Retrying.", exc_info=True)
            time.sleep(self.order_fill_check_interval_seconds)
        self.logger.warning(f"[{self.name}-{self.symbol}] Timeout waiting for order {order_id} to fill. Final check.")
        try:
            final_order_status = exchange_ccxt.fetch_order(order_id, symbol)
            self.logger.info(f"[{self.name}-{self.symbol}] Final status for order {order_id}: {final_order_status['status']}")
            return final_order_status
        except Exception as e: self.logger.error(f"[{self.name}-{self.symbol}] Final check for order {order_id} failed: {e}", exc_info=True); return None
            
    def _update_dca_state_in_db(self):
        if not self.position_db or not self.db_session:
            self.logger.error(f"[{self.name}-{self.symbol}] Cannot update DCA state: position_db or db_session not set.")
            return

        self.position_db.custom_data = json.dumps(self.dca_state)
        self.position_db.entry_price = self.dca_state['entry_price_avg']
        self.position_db.amount = self.dca_state['current_position_size_asset']
        self.position_db.updated_at = datetime.datetime.utcnow()
        try:
            self.db_session.commit()
            self.logger.info(f"[{self.name}-{self.symbol}] DCA state and position {self.position_db.id} updated in DB. AvgPrice: {self.position_db.entry_price}, Amt: {self.position_db.amount}")
        except Exception as e:
            self.logger.error(f"[{self.name}-{self.symbol}] Error committing DCA state to DB for pos {self.position_db.id}: {e}", exc_info=True)
            self.db_session.rollback()

    def _calculate_take_profits_and_sl(self, entry_price_avg: float):
        if entry_price_avg == 0: return [], 0.0 # Avoid division by zero if entry_price_avg is not set
        tp1 = entry_price_avg * (1 + self.tp1_percent / 100)
        tp2 = entry_price_avg * (1 + self.tp2_percent / 100)
        tp3 = entry_price_avg * (1 + self.tp3_percent / 100)
        current_stop_loss_price = entry_price_avg * (1 - self.stop_loss_percent / 100)
        return [tp1, tp2, tp3], current_stop_loss_price

    def _check_and_process_safety_order_fills(self, exchange_ccxt):
        if not self.position_db or not self.dca_state.get('open_safety_order_ids'):
            return

        filled_so_db_ids_to_remove = []
        for db_so_id_str, exch_so_id in list(self.dca_state['open_safety_order_ids'].items()): # list() for safe iteration
            db_so_id = int(db_so_id_str)
            try:
                so_details_exc = exchange_ccxt.fetch_order(exch_so_id, self.symbol)
                if so_details_exc['status'] == 'closed':
                    self.logger.info(f"[{self.name}-{self.symbol}] Safety order {exch_so_id} (DB ID: {db_so_id}) found filled.")
                    updates = {'status': 'closed', 'filled': so_details_exc.get('filled'), 
                               'price': so_details_exc.get('average'), 'closed_at': datetime.datetime.utcnow(),
                               'raw_order_data': json.dumps(so_details_exc)}
                    strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=db_so_id, updates=updates)
                    
                    filled_qty = float(so_details_exc['filled'])
                    fill_price = float(so_details_exc['average'])
                    
                    self.dca_state['total_usdt_invested'] += filled_qty * fill_price
                    self.dca_state['current_position_size_asset'] += filled_qty
                    if self.dca_state['current_position_size_asset'] > 0 : # Avoid division by zero
                         self.dca_state['entry_price_avg'] = self.dca_state['total_usdt_invested'] / self.dca_state['current_position_size_asset']
                    else: # Should not happen if SO filled adds to position
                         self.dca_state['entry_price_avg'] = fill_price # Or some other fallback

                    self.dca_state['safety_orders_placed_count'] += 1 # This counts filled SOs
                    
                    new_tps, new_sl = self._calculate_take_profits_and_sl(self.dca_state['entry_price_avg'])
                    self.dca_state['take_profit_prices'] = new_tps
                    self.dca_state['current_stop_loss_price'] = new_sl
                    self.dca_state['tp_levels_hit'] = [False, False, False] # Reset TP levels on new avg price

                    filled_so_db_ids_to_remove.append(db_so_id_str)
                    self.logger.info(f"[{self.name}-{self.symbol}] Updated DCA state after SO fill: AvgPrice {self.dca_state['entry_price_avg']}, Size {self.dca_state['current_position_size_asset']}, New SL {new_sl}")

            except ccxt.OrderNotFound:
                self.logger.warning(f"[{self.name}-{self.symbol}] Safety order {exch_so_id} (DB ID: {db_so_id}) not found. Removing from open list.")
                strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=db_so_id, updates={'status': 'not_found_on_exchange'})
                filled_so_db_ids_to_remove.append(db_so_id_str) # Remove from tracking
            except Exception as e:
                self.logger.error(f"[{self.name}-{self.symbol}] Error checking SO {exch_so_id}: {e}", exc_info=True)

        for db_id_str in filled_so_db_ids_to_remove:
            del self.dca_state['open_safety_order_ids'][db_id_str]
        
        if filled_so_db_ids_to_remove: # If any SO was processed
            self._update_dca_state_in_db()


    def execute_live_signal(self, exchange_ccxt): # db_session and user_sub_obj are now instance variables
        if not exchange_ccxt: self.logger.error(f"[{self.name}-{self.symbol}] Exchange object not provided. SubID: {self.user_sub_obj.id}."); return
        self.logger.debug(f"[{self.name}-{self.symbol}] Executing live signal for SubID: {self.user_sub_obj.id}...")
        self._get_precisions(exchange_ccxt)

        if not self.db_session or not self.user_sub_obj:
            self.logger.error(f"[{self.name}-{self.symbol}] DB session or user_sub_obj not configured. Cannot execute."); return

        # Sync Safety Order Fills first
        if self.position_db and self.position_db.is_open: # Only if a position is active
            self._check_and_process_safety_order_fills(exchange_ccxt)

        try:
            ticker = exchange_ccxt.fetch_ticker(self.symbol)
            current_price = ticker['last']
            if not current_price: self.logger.warning(f"[{self.name}-{self.symbol}] Could not fetch current price."); return
        except Exception as e: self.logger.error(f"[{self.name}-{self.symbol}] Error fetching ticker: {e}", exc_info=True); return

        self.logger.debug(f"[{self.name}-{self.symbol}] Current Price: {current_price}, Pos Active: {bool(self.position_db and self.position_db.is_open)}, Avg Entry: {self.dca_state['entry_price_avg']}, Size: {self.dca_state['current_position_size_asset']}")

        # --- Initial Position Entry ---
        if not self.position_db or not self.position_db.is_open: # No active position, or last one closed
            self.logger.info(f"[{self.name}-{self.symbol}] No active position. Attempting base order at {current_price}.")
            base_qty_asset = self.base_order_size_usdt / current_price
            formatted_base_qty = self._format_quantity(base_qty_asset, exchange_ccxt)
            if formatted_base_qty <= 0: self.logger.warning(f"[{self.name}-{self.symbol}] Base order qty zero. Skipping."); return

            db_base_order = strategy_utils.create_strategy_order_in_db(self.db_session, self.user_sub_obj.id, self.symbol, 'limit', 'buy', formatted_base_qty, self._format_price(current_price, exchange_ccxt), notes="DCA Base Order")
            if not db_base_order: self.logger.error(f"[{self.name}-{self.symbol}] Failed to create DB record for base order."); return
            
            try:
                base_order_receipt = exchange_ccxt.create_limit_buy_order(self.symbol, formatted_base_qty, self._format_price(current_price, exchange_ccxt))
                strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=db_base_order.id, updates={'order_id': base_order_receipt['id'], 'status': base_order_receipt.get('status', 'open'), 'raw_order_data': json.dumps(base_order_receipt)})
                
                filled_base_order = self._await_order_fill(exchange_ccxt, base_order_receipt['id'], self.symbol)
                if not filled_base_order or filled_base_order['status'] != 'closed':
                    self.logger.error(f"[{self.name}-{self.symbol}] Base order {base_order_receipt['id']} failed to fill. Status: {filled_base_order.get('status') if filled_base_order else 'Unknown'}")
                    strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=db_base_order.id, updates={'status': filled_base_order.get('status', 'fill_check_failed') if filled_base_order else 'fill_check_failed'})
                    return
                
                strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=db_base_order.id, updates={'status': 'closed', 'price': filled_base_order['average'], 'filled': filled_base_order['filled'], 'cost': filled_base_order['cost'], 'closed_at': datetime.datetime.utcnow(), 'raw_order_data': json.dumps(filled_base_order)})
                
                # Initialize DCA state for the new position
                self.dca_state['entry_price_avg'] = filled_base_order['average']
                self.dca_state['current_position_size_asset'] = filled_base_order['filled']
                self.dca_state['total_usdt_invested'] = filled_base_order.get('cost', filled_base_order['average'] * filled_base_order['filled'])
                self.dca_state['safety_orders_placed_count'] = 0
                self.dca_state['tp_levels_hit'] = [False, False, False]
                self.dca_state['open_safety_order_ids'] = {} # Reset open SO IDs
                prices, sl = self._calculate_take_profits_and_sl(self.dca_state['entry_price_avg'])
                self.dca_state['take_profit_prices'] = prices; self.dca_state['current_stop_loss_price'] = sl
                
                self.position_db = strategy_utils.create_strategy_position_in_db(self.db_session, self.user_sub_obj.id, self.symbol, str(exchange_ccxt.id), "long", self.dca_state['current_position_size_asset'], self.dca_state['entry_price_avg'], status_message="DCA Base Order Opened")
                if not self.position_db: self.logger.error(f"[{self.name}-{self.symbol}] Failed to create Position DB record."); return
                self.active_position_db_id = self.position_db.id
                self._update_dca_state_in_db() # Save initial DCA state to Position.custom_data
                
                # Place Safety Orders
                initial_entry_for_so = self.dca_state['entry_price_avg']
                for i in range(self.max_safety_orders):
                    so_price = initial_entry_for_so * (1 - self.safety_order_deviation_percent / 100 * (i + 1))
                    so_price_fmt = self._format_price(so_price, exchange_ccxt)
                    so_size_usdt = self.safety_order_size_usdt * (self.safety_order_scale_factor ** i)
                    so_qty_asset = so_size_usdt / so_price if so_price > 0 else 0
                    so_qty_fmt = self._format_quantity(so_qty_asset, exchange_ccxt)

                    if so_qty_fmt > 0:
                        db_so = strategy_utils.create_strategy_order_in_db(self.db_session, self.user_sub_obj.id, self.symbol, 'limit', 'buy', so_qty_fmt, so_price_fmt, notes=f"DCA SO {i+1}")
                        if not db_so: continue # Failed to create DB record for SO

                        try:
                            so_receipt = exchange_ccxt.create_limit_buy_order(self.symbol, so_qty_fmt, so_price_fmt)
                            strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=db_so.id, updates={'order_id': so_receipt['id'], 'status': so_receipt.get('status', 'open'), 'raw_order_data': json.dumps(so_receipt)})
                            self.dca_state['open_safety_order_ids'][str(db_so.id)] = so_receipt['id'] # Track open SO
                            self.logger.info(f"[{self.name}-{self.symbol}] Safety order {i+1} (ExchID {so_receipt['id']}, DBID {db_so.id}) placed for PosID {self.position_db.id} at {so_price_fmt}, Qty {so_qty_fmt}")
                        except Exception as e_so: 
                            self.logger.error(f"[{self.name}-{self.symbol}] Failed to place SO {i+1} for PosID {self.position_db.id}: {e_so}", exc_info=True)
                            strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=db_so.id, updates={'status': 'failed_creation'})
                    else: self.logger.warning(f"[{self.name}-{self.symbol}] SO {i+1} qty zero. USDT: {so_size_usdt}, Price: {so_price_fmt}")
                self._update_dca_state_in_db() # Save open_safety_order_ids
                self.logger.info(f"[{self.name}-{self.symbol}] Base order filled. Pos ID {self.position_db.id}. Entry: {self.dca_state['entry_price_avg']}, Size: {self.dca_state['current_position_size_asset']}. Safety orders placed.")

            except Exception as e_base: 
                self.logger.error(f"[{self.name}-{self.symbol}] Error during base order placement/fill: {e_base}", exc_info=True)
                if db_base_order: strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=db_base_order.id, updates={'status': 'error'})
            return 

        # --- Position Management (TP / SL) ---
        if self.position_db and self.position_db.is_open:
            # Check Take Profits
            for i in range(len(self.dca_state['take_profit_prices'])):
                if not self.dca_state['tp_levels_hit'][i] and current_price >= self.dca_state['take_profit_prices'][i]:
                    sell_qty_asset = 0
                    if i == 0: sell_qty_asset = self.dca_state['current_position_size_asset'] * self.tp1_sell_multiplier
                    elif i == 1: sell_qty_asset = self.dca_state['current_position_size_asset'] * self.tp2_sell_multiplier
                    elif i == 2: sell_qty_asset = self.dca_state['current_position_size_asset'] # Sell all remaining for TP3
                    
                    formatted_sell_qty = self._format_quantity(sell_qty_asset, exchange_ccxt)
                    if formatted_sell_qty <= 0: continue # Avoid selling zero or negative

                    self.logger.info(f"[{self.name}-{self.symbol}] TP{i+1} hit at {current_price}. Attempting to sell {formatted_sell_qty}.")
                    db_tp_order = strategy_utils.create_strategy_order_in_db(self.db_session, self.user_sub_obj.id, self.symbol, 'market', 'sell', formatted_sell_qty, notes=f"DCA TP{i+1}")
                    if not db_tp_order: self.logger.error(f"[{self.name}-{self.symbol}] Failed to create DB record for TP{i+1} order."); continue

                    try:
                        tp_order_receipt = exchange_ccxt.create_market_sell_order(self.symbol, formatted_sell_qty)
                        strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=db_tp_order.id, updates={'order_id': tp_order_receipt['id'], 'status': 'open', 'raw_order_data': json.dumps(tp_order_receipt)})
                        
                        filled_tp_order = self._await_order_fill(exchange_ccxt, tp_order_receipt['id'], self.symbol)
                        if filled_tp_order and filled_tp_order['status'] == 'closed':
                            strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=db_tp_order.id, updates={'status': 'closed', 'price': filled_tp_order['average'], 'filled': filled_tp_order['filled'], 'cost': filled_tp_order['cost'], 'closed_at': datetime.datetime.utcnow(), 'raw_order_data': json.dumps(filled_tp_order)})
                            
                            self.dca_state['current_position_size_asset'] -= filled_tp_order['filled']
                            # total_usdt_invested is tricky to adjust perfectly without knowing which buys were sold.
                            # A simpler approximation: reduce by proportion of size sold * avg entry.
                            proportion_sold = filled_tp_order['filled'] / (self.dca_state['current_position_size_asset'] + filled_tp_order['filled']) # before subtraction
                            self.dca_state['total_usdt_invested'] *= (1 - proportion_sold)
                            if self.dca_state['total_usdt_invested'] < 0: self.dca_state['total_usdt_invested'] = 0
                            self.dca_state['tp_levels_hit'][i] = True
                            self.logger.info(f"[{self.name}-{self.symbol}] TP{i+1} sold {filled_tp_order['filled']}. Remaining size: {self.dca_state['current_position_size_asset']}")
                            
                            if i < 2 and self.dca_state['current_position_size_asset'] > 0: # TP1 or TP2, and position still open
                                if i == 0: self.dca_state['current_stop_loss_price'] = self.dca_state['entry_price_avg'] # SL to BE
                                elif i == 1: self.dca_state['current_stop_loss_price'] = self.dca_state['take_profit_prices'][0] # SL to TP1
                                self.logger.info(f"[{self.name}-{self.symbol}] TP{i+1} hit. Adjusted SL to {self.dca_state['current_stop_loss_price']}")
                            
                            # Use parameterized negligible amount for closure check
                            if self.dca_state['current_position_size_asset'] <= self._format_quantity(self.min_asset_qty_for_closure, exchange_ccxt) or i == 2 : 
                                self.logger.info(f"[{self.name}-{self.symbol}] Position fully closed by TP{i+1} or negligible amount left.")
                                strategy_utils.close_strategy_position_in_db(self.db_session, self.active_position_db_id, filled_tp_order['average'], filled_tp_order['filled'], f"Closed by TP{i+1}")
                                self.active_position_db_id = None; self.position_db = None
                                self.dca_state = {'entry_price_avg': 0.0, 'current_position_size_asset': 0.0, 'total_usdt_invested': 0.0, 'safety_orders_placed_count': 0, 'take_profit_prices': [], 'current_stop_loss_price': 0.0, 'tp_levels_hit': [False,False,False], 'open_safety_order_ids': {}}
                                # Cancel any remaining open safety orders
                                for db_so_id_str_cancel, exch_so_id_cancel in list(self.dca_state['open_safety_order_ids'].items()):
                                    try: exchange_ccxt.cancel_order(exch_so_id_cancel, self.symbol); strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=int(db_so_id_str_cancel), updates={'status': 'canceled_due_to_tp_close'})
                                    except Exception as e_cancel_so: self.logger.warning(f"Failed to cancel SO {exch_so_id_cancel} after full TP: {e_cancel_so}")
                                self.dca_state['open_safety_order_ids'] = {}
                            
                            self._update_dca_state_in_db() # Save state if position still open or for final closed state details
                        else: 
                            self.logger.error(f"[{self.name}-{self.symbol}] TP{i+1} market sell order failed. ID: {tp_order_receipt['id'] if tp_order_receipt else 'N/A'}")
                            strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=db_tp_order.id, updates={'status': filled_tp_order.get('status', 'fill_check_failed') if filled_tp_order else 'fill_check_failed'})
                    except Exception as e_tp_sell: 
                        self.logger.error(f"[{self.name}-{self.symbol}] Error placing TP{i+1} sell: {e_tp_sell}", exc_info=True)
                        if db_tp_order: strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=db_tp_order.id, updates={'status': 'error'})
                    break # Process one TP level per cycle
            
            # Check Stop Loss (only if position still open)
            if self.position_db and self.position_db.is_open and self.dca_state.get('current_stop_loss_price', 0) > 0 and current_price <= self.dca_state['current_stop_loss_price']:
                self.logger.info(f"[{self.name}-{self.symbol}] Stop Loss hit at {current_price}. SL price: {self.dca_state['current_stop_loss_price']}. Selling remaining.")
                sl_qty_asset = self._format_quantity(self.dca_state['current_position_size_asset'], exchange_ccxt)
                if sl_qty_asset > 0:
                    db_sl_order = strategy_utils.create_strategy_order_in_db(self.db_session, self.user_sub_obj.id, self.symbol, 'market', 'sell', sl_qty_asset, notes="DCA SL Hit")
                    if not db_sl_order: self.logger.error(f"[{self.name}-{self.symbol}] Failed to create DB record for SL order."); return

                    try:
                        sl_order_receipt = exchange_ccxt.create_market_sell_order(self.symbol, sl_qty_asset)
                        strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=db_sl_order.id, updates={'order_id': sl_order_receipt['id'], 'status': 'open', 'raw_order_data': json.dumps(sl_order_receipt)})
                        
                        filled_sl_order = self._await_order_fill(exchange_ccxt, sl_order_receipt['id'], self.symbol)
                        if filled_sl_order and filled_sl_order['status'] == 'closed':
                            strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=db_sl_order.id, updates={'status': 'closed', 'price': filled_sl_order['average'], 'filled': filled_sl_order['filled'], 'cost': filled_sl_order['cost'], 'closed_at': datetime.datetime.utcnow(), 'raw_order_data': json.dumps(filled_sl_order)})
                            self.logger.info(f"[{self.name}-{self.symbol}] Stop loss executed. Sold {filled_sl_order['filled']}.")
                            
                            strategy_utils.close_strategy_position_in_db(self.db_session, self.active_position_db_id, filled_sl_order['average'], filled_sl_order['filled'], "Closed by Stop Loss")
                            self.active_position_db_id = None; self.position_db = None
                            self.dca_state = {'entry_price_avg': 0.0, 'current_position_size_asset': 0.0, 'total_usdt_invested': 0.0, 'safety_orders_placed_count': 0, 'take_profit_prices': [], 'current_stop_loss_price': 0.0, 'tp_levels_hit': [False,False,False], 'open_safety_order_ids': {}}
                             # Cancel any remaining open safety orders
                            for db_so_id_str_cancel, exch_so_id_cancel in list(self.dca_state['open_safety_order_ids'].items()): # Ensure list for safe iteration if modifying dict
                                try: exchange_ccxt.cancel_order(exch_so_id_cancel, self.symbol); strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=int(db_so_id_str_cancel), updates={'status': 'canceled_due_to_sl'})
                                except Exception as e_cancel_so: self.logger.warning(f"Failed to cancel SO {exch_so_id_cancel} after SL: {e_cancel_so}")
                            self.dca_state['open_safety_order_ids'] = {}

                        else: 
                            self.logger.error(f"[{self.name}-{self.symbol}] SL market sell order failed. ID: {sl_order_receipt['id'] if sl_order_receipt else 'N/A'}")
                            strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=db_sl_order.id, updates={'status': filled_sl_order.get('status', 'fill_check_failed') if filled_sl_order else 'fill_check_failed'})
                    except Exception as e_sl_sell: 
                        self.logger.error(f"[{self.name}-{self.symbol}] Error placing SL sell: {e_sl_sell}", exc_info=True)
                        if db_sl_order: strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=db_sl_order.id, updates={'status': 'error'})
                else: # No position size left to SL, ensure state is reset and position closed
                    if self.position_db and self.position_db.is_open : # Should not happen if sl_qty_asset is 0, but as safeguard
                         strategy_utils.close_strategy_position_in_db(self.db_session, self.active_position_db_id, current_price, 0, "Closed due to zero size before SL execution")
                    self.active_position_db_id = None; self.position_db = None
                    self.dca_state = {'entry_price_avg': 0.0, 'current_position_size_asset': 0.0, 'total_usdt_invested': 0.0, 'safety_orders_placed_count': 0, 'take_profit_prices': [], 'current_stop_loss_price': 0.0, 'tp_levels_hit': [False,False,False], 'open_safety_order_ids': {}}
            
            # Persist state if position still open and not just closed by SL/TP
            if self.position_db and self.position_db.is_open:
                self._update_dca_state_in_db()

        self.logger.debug(f"[{self.name}-{self.symbol}] Live signal execution cycle finished for SubID {self.user_sub_obj.id}.")
