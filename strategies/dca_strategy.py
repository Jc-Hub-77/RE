import time
import math
import logging
import pandas as pd
import datetime 
import json 
from sqlalchemy.orm import Session 
from backend.models import Position, Order 

logger = logging.getLogger(__name__)

class DCAStrategy:
    def __init__(self, symbol: str, timeframe: str, capital: float = 10000, **custom_parameters):
        self.name = "DCA Strategy"
        self.symbol = symbol
        self.timeframe = timeframe 
        self.capital = capital 

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
            "max_safety_orders": 5
        }
        
        self_params = {**defaults, **custom_parameters}
        for key, value in self_params.items():
            setattr(self, key, value)

        self.tp1_sell_multiplier = self.tp1_sell_percent / 100.0
        self.tp2_sell_multiplier = self.tp2_sell_percent / 100.0
        
        # Precisions - will be fetched
        self.price_precision = 8
        self.quantity_precision = 8
        self._precisions_fetched_ = False

        logger.info(f"[{self.name}-{self.symbol}] Initialized with effective params: {self_params}")

    @classmethod
    def get_parameters_definition(cls):
        return {
            "base_order_size_usdt": {"type": "float", "default": 10.0, "label": "Base Order Size (USDT)"},
            "safety_order_size_usdt": {"type": "float", "default": 10.0, "label": "Safety Order Size (USDT)"},
            "tp1_percent": {"type": "float", "default": 1.0, "label": "Take Profit 1 (%)"},
            "tp1_sell_percent": {"type": "float", "default": 40.0, "label": "TP1 Sell (%)"},
            "tp2_percent": {"type": "float", "default": 3.0, "label": "Take Profit 2 (%)"},
            "tp2_sell_percent": {"type": "float", "default": 40.0, "label": "TP2 Sell (%)"},
            "tp3_percent": {"type": "float", "default": 20.0, "label": "Take Profit 3 (%)"},
            "stop_loss_percent": {"type": "float", "default": 5.0, "label": "Stop Loss (%)"},
            "safety_order_deviation_percent": {"type": "float", "default": 1.0, "label": "Safety Order Deviation (%)"},
            "safety_order_scale_factor": {"type": "float", "default": 1.0, "label": "Safety Order Scale Factor"},
            "max_safety_orders": {"type": "int", "default": 5, "label": "Max Safety Orders"}
        }

    def _get_precisions(self, exchange_ccxt):
        if not self._precisions_fetched_:
            try:
                exchange_ccxt.load_markets(True)
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
        logger.warning(f"[{self.name}-{self.symbol}] Timeout waiting for order {order_id} to fill. Final check.")
        try:
            final_order_status = exchange_ccxt.fetch_order(order_id, symbol)
            logger.info(f"[{self.name}-{self.symbol}] Final status for order {order_id}: {final_order_status['status']}")
            return final_order_status
        except Exception as e:
            logger.error(f"[{self.name}-{self.symbol}] Final check for order {order_id} failed: {e}", exc_info=True)
            return None
            
    def _create_db_order(self, db_session: Session, subscription_id: int, position_id: int = None, **kwargs):
        db_order = Order(subscription_id=subscription_id, **kwargs)
        # Note: 'position_id' is not in the current Order model schema.
        # If it were, you would set it: db_order.position_id = position_id
        db_session.add(db_order)
        db_session.commit()
        return db_order

    def _update_dca_state_in_db(self, db_session: Session, position_db: Position, dca_state: dict):
        position_db.custom_data = json.dumps(dca_state)
        position_db.entry_price = dca_state['entry_price_avg'] # Update position's main entry price
        position_db.amount = dca_state['current_position_size_asset'] # Update position's main amount
        position_db.updated_at = datetime.datetime.utcnow()
        db_session.commit()

    def _calculate_take_profits_and_sl(self, entry_price_avg: float):
        tp1 = entry_price_avg * (1 + self.tp1_percent / 100)
        tp2 = entry_price_avg * (1 + self.tp2_percent / 100)
        tp3 = entry_price_avg * (1 + self.tp3_percent / 100)
        current_stop_loss_price = entry_price_avg * (1 - self.stop_loss_percent / 100)
        return [tp1, tp2, tp3], current_stop_loss_price

    def execute_live_signal(self, db_session: Session, subscription_id: int, exchange_ccxt, user_sub_obj: UserStrategySubscription):
        if not exchange_ccxt: logger.error(f"[{self.name}-{self.symbol}] Exchange object not provided for sub {subscription_id}."); return
        logger.debug(f"[{self.name}-{self.symbol}] Executing live signal for sub {subscription_id}...")
        self._get_precisions(exchange_ccxt)

        position_db = db_session.query(Position).filter(Position.subscription_id == subscription_id, Position.symbol == self.symbol, Position.is_open == True).first()
        
        dca_state = {"tp_levels_hit": [False, False, False]} # Default for new position
        if position_db:
            dca_state = json.loads(position_db.custom_data) if position_db.custom_data else dca_state
            dca_state.setdefault('tp_levels_hit', [False,False,False]) # Ensure it exists
            # Load core attributes from position_db into dca_state for consistent handling
            dca_state['entry_price_avg'] = position_db.entry_price
            dca_state['current_position_size_asset'] = position_db.amount
            dca_state['total_usdt_invested'] = position_db.entry_price * position_db.amount # Approximation
            dca_state['safety_orders_placed_count'] = dca_state.get('safety_orders_placed_count', 0) # Filled SOs
            
            if not dca_state.get('take_profit_prices') or not dca_state.get('current_stop_loss_price'):
                prices, sl = self._calculate_take_profits_and_sl(dca_state['entry_price_avg'])
                dca_state['take_profit_prices'] = prices
                dca_state['current_stop_loss_price'] = sl
            logger.debug(f"[{self.name}-{self.symbol}] Loaded state for Pos ID {position_db.id}: {dca_state}")
        else: # Initialize for potential new position
            dca_state['entry_price_avg'] = 0.0
            dca_state['current_position_size_asset'] = 0.0
            dca_state['total_usdt_invested'] = 0.0
            dca_state['safety_orders_placed_count'] = 0
            dca_state['take_profit_prices'] = []
            dca_state['current_stop_loss_price'] = 0.0
            logger.debug(f"[{self.name}-{self.symbol}] No active DB position found for sub {subscription_id}.")


        try:
            ticker = exchange_ccxt.fetch_ticker(self.symbol)
            current_price = ticker['last']
            if not current_price: logger.warning(f"[{self.name}-{self.symbol}] Could not fetch current price."); return
        except Exception as e: logger.error(f"[{self.name}-{self.symbol}] Error fetching ticker: {e}", exc_info=True); return

        logger.debug(f"[{self.name}-{self.symbol}] Current Price: {current_price}, Pos Active: {bool(position_db)}, Avg Entry: {dca_state['entry_price_avg']}, Size: {dca_state['current_position_size_asset']}")

        # --- Initial Position Entry ---
        if not position_db:
            logger.info(f"[{self.name}-{self.symbol}] No active position. Attempting base order at {current_price}.")
            base_qty_asset = self.base_order_size_usdt / current_price
            formatted_base_qty = self._format_quantity(base_qty_asset, exchange_ccxt)
            if formatted_base_qty <= 0: logger.warning(f"[{self.name}-{self.symbol}] Base order qty zero. Skipping."); return

            db_base_order = self._create_db_order(db_session, subscription_id, symbol=self.symbol, order_type='limit', side='buy', amount=formatted_base_qty, price=self._format_price(current_price, exchange_ccxt), status='pending_creation')
            try:
                base_order_receipt = exchange_ccxt.create_limit_buy_order(self.symbol, formatted_base_qty, self._format_price(current_price, exchange_ccxt))
                db_base_order.order_id = base_order_receipt['id']; db_base_order.status = base_order_receipt.get('status', 'open'); db_session.commit()
                
                filled_base_order = self._await_order_fill(exchange_ccxt, base_order_receipt['id'], self.symbol)
                if not filled_base_order or filled_base_order['status'] != 'closed':
                    logger.error(f"[{self.name}-{self.symbol}] Base order {base_order_receipt['id']} failed to fill. Status: {filled_base_order.get('status') if filled_base_order else 'Unknown'}")
                    db_base_order.status = filled_base_order.get('status', 'fill_check_failed') if filled_base_order else 'fill_check_failed'; db_session.commit()
                    return
                
                db_base_order.status = 'closed'; db_base_order.price = filled_base_order['average']; db_base_order.filled = filled_base_order['filled']; db_base_order.cost = filled_base_order['cost']; db_base_order.updated_at = datetime.datetime.utcnow(); db_session.commit()
                
                dca_state['entry_price_avg'] = filled_base_order['average']
                dca_state['current_position_size_asset'] = filled_base_order['filled']
                dca_state['total_usdt_invested'] = filled_base_order['cost'] if filled_base_order.get('cost') else filled_base_order['average'] * filled_base_order['filled']
                dca_state['safety_orders_placed_count'] = 0 # Reset for new trade cycle
                dca_state['tp_levels_hit'] = [False, False, False]
                prices, sl = self._calculate_take_profits_and_sl(dca_state['entry_price_avg'])
                dca_state['take_profit_prices'] = prices; dca_state['current_stop_loss_price'] = sl
                
                position_db = Position(subscription_id=subscription_id, symbol=self.symbol, exchange_name=str(exchange_ccxt.id), side="long", amount=dca_state['current_position_size_asset'], entry_price=dca_state['entry_price_avg'], current_price=current_price, is_open=True, created_at=datetime.datetime.utcnow(), updated_at=datetime.datetime.utcnow())
                db_session.add(position_db); db_session.commit() # Commit position first to get ID
                self._update_dca_state_in_db(db_session, position_db, dca_state) # Now save DCA state with position_id
                
                # Place Safety Orders
                initial_entry_for_so = dca_state['entry_price_avg']
                for i in range(self.max_safety_orders):
                    so_price = initial_entry_for_so * (1 - self.safety_order_deviation_percent / 100 * (i + 1))
                    so_price_fmt = self._format_price(so_price, exchange_ccxt)
                    so_size_usdt = self.safety_order_size_usdt * (self.safety_order_scale_factor ** i)
                    so_qty_asset = so_size_usdt / so_price if so_price > 0 else 0
                    so_qty_fmt = self._format_quantity(so_qty_asset, exchange_ccxt)

                    if so_qty_fmt > 0:
                        db_so = self._create_db_order(db_session, subscription_id, position_id=position_db.id, symbol=self.symbol, order_type='limit', side='buy', amount=so_qty_fmt, price=so_price_fmt, status='pending_creation')
                        try:
                            so_receipt = exchange_ccxt.create_limit_buy_order(self.symbol, so_qty_fmt, so_price_fmt)
                            db_so.order_id = so_receipt['id']; db_so.status = so_receipt.get('status', 'open'); db_session.commit()
                            logger.info(f"[{self.name}-{self.symbol}] Safety order {i+1} (ID {so_receipt['id']}) placed for Pos ID {position_db.id} at {so_price_fmt}, Qty {so_qty_fmt}")
                        except Exception as e_so: logger.error(f"[{self.name}-{self.symbol}] Failed to place SO {i+1} for Pos ID {position_db.id}: {e_so}", exc_info=True); db_so.status = 'failed_creation'; db_session.commit()
                    else: logger.warning(f"[{self.name}-{self.symbol}] SO {i+1} qty zero. USDT: {so_size_usdt}, Price: {so_price_fmt}")
                logger.info(f"[{self.name}-{self.symbol}] Base order filled. Pos ID {position_db.id}. Entry: {dca_state['entry_price_avg']}, Size: {dca_state['current_position_size_asset']}. Safety orders placed.")

            except Exception as e_base: logger.error(f"[{self.name}-{self.symbol}] Error during base order placement/fill: {e_base}", exc_info=True); db_base_order.status = 'error'; db_session.commit()
            return # End cycle after entry attempt

        # --- Position Management ---
        if position_db: # Position is active
            # TODO: Implement safety order fill check: Iterate open 'limit' 'buy' orders linked to this position_db.id.
            # If exchange_ccxt.fetch_order(id) shows 'closed', then:
            # 1. Update that Order in DB (status, filled qty, avg price).
            # 2. Recalculate dca_state['entry_price_avg'], 'current_position_size_asset', 'total_usdt_invested'.
            # 3. dca_state['safety_orders_placed_count'] += 1.
            # 4. Recalculate TPs/SL: prices, sl = self._calculate_take_profits_and_sl(dca_state['entry_price_avg']); dca_state['take_profit_prices']=prices; dca_state['current_stop_loss_price']=sl
            # 5. Persist dca_state: self._update_dca_state_in_db(db_session, position_db, dca_state)
            # 6. Cancel and replace existing TP/SL orders with new ones based on updated avg entry. This is complex.
            # For now, TPs/SLs are based on initial entry + all potential SOs.

            # Check Take Profits
            for i in range(len(dca_state['take_profit_prices'])):
                if not dca_state['tp_levels_hit'][i] and current_price >= dca_state['take_profit_prices'][i]:
                    sell_qty_asset = 0
                    if i == 0: sell_qty_asset = dca_state['current_position_size_asset'] * self.tp1_sell_multiplier
                    elif i == 1: sell_qty_asset = dca_state['current_position_size_asset'] * self.tp2_sell_multiplier
                    elif i == 2: sell_qty_asset = dca_state['current_position_size_asset'] 
                    
                    formatted_sell_qty = self._format_quantity(sell_qty_asset, exchange_ccxt)
                    if formatted_sell_qty <= 0: continue

                    logger.info(f"[{self.name}-{self.symbol}] TP{i+1} hit at {current_price}. Selling {formatted_sell_qty}.")
                    db_tp_order = self._create_db_order(db_session, subscription_id, position_id=position_db.id, symbol=self.symbol, order_type='market', side='sell', amount=formatted_sell_qty, status='pending_creation')
                    try:
                        tp_order_receipt = exchange_ccxt.create_market_sell_order(self.symbol, formatted_sell_qty)
                        db_tp_order.order_id = tp_order_receipt['id']; db_tp_order.status = 'open'; db_session.commit()
                        filled_tp_order = self._await_order_fill(exchange_ccxt, tp_order_receipt['id'], self.symbol)
                        if filled_tp_order and filled_tp_order['status'] == 'closed':
                            db_tp_order.status = 'closed'; db_tp_order.price = filled_tp_order['average']; db_tp_order.filled = filled_tp_order['filled']; db_tp_order.cost = filled_tp_order['cost']; db_tp_order.updated_at = datetime.datetime.utcnow(); db_session.commit()
                            
                            dca_state['current_position_size_asset'] -= filled_tp_order['filled']
                            dca_state['total_usdt_invested'] -= (filled_tp_order['filled'] * filled_tp_order['average']) # Adjust cost basis
                            if dca_state['total_usdt_invested'] < 0: dca_state['total_usdt_invested'] = 0
                            dca_state['tp_levels_hit'][i] = True
                            logger.info(f"[{self.name}-{self.symbol}] TP{i+1} sold {filled_tp_order['filled']}. Remaining size: {dca_state['current_position_size_asset']}")
                            
                            if i < 2: # TP1 or TP2, adjust SL
                                prices, sl = self._calculate_take_profits_and_sl(dca_state['entry_price_avg']) # SL based on original avg entry
                                if i == 0: dca_state['current_stop_loss_price'] = dca_state['entry_price_avg'] # SL to BE
                                elif i == 1: dca_state['current_stop_loss_price'] = dca_state['take_profit_prices'][0] # SL to TP1
                                logger.info(f"[{self.name}-{self.symbol}] TP{i+1} hit. Adjusted SL to {dca_state['current_stop_loss_price']}")
                            else: # TP3 hit
                                logger.info(f"[{self.name}-{self.symbol}] TP3 hit. Closing position fully.")
                                position_db.is_open = False; position_db.closed_at = datetime.datetime.utcnow()
                                dca_state['current_position_size_asset'] = 0 # Ensure it's zero
                            self._update_dca_state_in_db(db_session, position_db, dca_state)
                        else: logger.error(f"[{self.name}-{self.symbol}] TP{i+1} market sell order failed to fill. Order ID: {tp_order_receipt['id'] if tp_order_receipt else 'N/A'}"); db_tp_order.status = 'fill_check_failed'; db_session.commit()
                    except Exception as e_tp_sell: logger.error(f"[{self.name}-{self.symbol}] Error placing TP{i+1} sell: {e_tp_sell}", exc_info=True); db_tp_order.status = 'error'; db_session.commit()
                    break 
            
            # Check Stop Loss
            if dca_state.get('current_stop_loss_price', 0) > 0 and current_price <= dca_state['current_stop_loss_price']:
                logger.info(f"[{self.name}-{self.symbol}] Stop Loss hit at {current_price}. SL price: {dca_state['current_stop_loss_price']}. Selling remaining.")
                sl_qty_asset = self._format_quantity(dca_state['current_position_size_asset'], exchange_ccxt)
                if sl_qty_asset > 0:
                    db_sl_order = self._create_db_order(db_session, subscription_id, position_id=position_db.id, symbol=self.symbol, order_type='market', side='sell', amount=sl_qty_asset, status='pending_creation')
                    try:
                        sl_order_receipt = exchange_ccxt.create_market_sell_order(self.symbol, sl_qty_asset)
                        db_sl_order.order_id = sl_order_receipt['id']; db_sl_order.status = 'open'; db_session.commit()
                        filled_sl_order = self._await_order_fill(exchange_ccxt, sl_order_receipt['id'], self.symbol)
                        if filled_sl_order and filled_sl_order['status'] == 'closed':
                            db_sl_order.status = 'closed'; db_sl_order.price = filled_sl_order['average']; db_sl_order.filled = filled_sl_order['filled']; db_sl_order.cost = filled_sl_order['cost']; db_sl_order.updated_at = datetime.datetime.utcnow(); db_session.commit()
                            logger.info(f"[{self.name}-{self.symbol}] Stop loss executed. Sold {filled_sl_order['filled']}.")
                            position_db.is_open = False; position_db.closed_at = datetime.datetime.utcnow()
                            # PNL calculation would be based on cost basis vs sell proceeds.
                            # For simplicity, PNL is not explicitly calculated and stored on Position here.
                            self._update_dca_state_in_db(db_session, position_db, {'entry_price_avg': 0, 'current_position_size_asset': 0, 'total_usdt_invested': 0, 'safety_orders_placed_count': 0, 'take_profit_prices': [], 'current_stop_loss_price': 0, 'tp_levels_hit': [False,False,False]}) # Reset state
                        else: logger.error(f"[{self.name}-{self.symbol}] SL market sell order failed. Order ID: {sl_order_receipt['id'] if sl_order_receipt else 'N/A'}"); db_sl_order.status = 'fill_check_failed'; db_session.commit()
                    except Exception as e_sl_sell: logger.error(f"[{self.name}-{self.symbol}] Error placing SL sell: {e_sl_sell}", exc_info=True); db_sl_order.status = 'error'; db_session.commit()
                else: # No position left to SL, ensure state is reset
                    position_db.is_open = False; position_db.closed_at = datetime.datetime.utcnow() if not position_db.closed_at else position_db.closed_at
                    self._update_dca_state_in_db(db_session, position_db, {'entry_price_avg': 0, 'current_position_size_asset': 0, 'total_usdt_invested': 0, 'safety_orders_placed_count': 0, 'take_profit_prices': [], 'current_stop_loss_price': 0, 'tp_levels_hit': [False,False,False]})
            
            # Persist any changes to dca_state (like SL adjustment after TP) if position still open
            if position_db.is_open:
                self._update_dca_state_in_db(db_session, position_db, dca_state)

        logger.debug(f"[{self.name}-{self.symbol}] Live signal execution cycle finished for sub {subscription_id}.")
