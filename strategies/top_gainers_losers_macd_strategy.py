import logging
import numpy as np
import talib
import ccxt
import json
from datetime import datetime # Ensure datetime is imported correctly
from decimal import Decimal, ROUND_DOWN, ROUND_UP, getcontext
from typing import Optional, Dict, Any, List

import pandas
from sqlalchemy.orm import Session
from backend.models import Position, Order, UserStrategySubscription
from backend import strategy_utils

# Ensure high precision for Decimal calculations
getcontext().prec = 18

class TopGainersLosersMACD:
    def __init__(self, db_session: Session, user_sub_obj: UserStrategySubscription, strategy_params: dict, exchange_ccxt, logger_obj=None): # Renamed logger
        self.db_session = db_session
        self.user_sub_obj = user_sub_obj
        self.params = strategy_params
        self.exchange_ccxt = exchange_ccxt
        self.logger = logger_obj if logger_obj else logging.getLogger(__name__) # Use passed logger
        self.name = "TopGainersLosersMACD"

        # Extract and validate parameters
        self.leverage = int(self.params.get("leverage", 3))
        self.stop_loss_percent = float(self.params.get("stop_loss_percent", 0.05)) # Store as float for calculations
        self.risk_per_trade_percent = float(self.params.get("risk_per_trade_percent", 0.05))
        self.macd_fast_period = int(self.params.get("macd_fast_period", 34))
        self.macd_slow_period = int(self.params.get("macd_slow_period", 144))
        self.macd_signal_period = int(self.params.get("macd_signal_period", 9))
        self.top_n_symbols_to_scan = int(self.params.get("top_n_symbols_to_scan", 10))
        self.max_concurrent_trades = int(self.params.get("max_concurrent_trades", 2))
        self.kline_interval = self.params.get("kline_interval", "15m")
        self.min_volume_usdt_24h = float(self.params.get("min_volume_usdt_24h", 1000000.0))
        self.min_price_change_percent_filter = float(self.params.get("min_price_change_percent_filter", 3.0))
        self.min_candles_for_macd = int(self.params.get("min_candles_for_macd", 144))
        
        self.active_trades: Dict[str, Dict[str, Any]] = {} 
        # Structure: {symbol: {'position_db_id': int, 'sl_order_db_id': Optional[int], 
        #                      'sl_exchange_id': Optional[str], 'entry_price': Decimal, 
        #                      'qty': Decimal, 'side': str}}

        self.logger.info(f"{self.name} strategy initializing for UserSubID {self.user_sub_obj.id}...")
        self._load_all_persistent_positions() # Load state at initialization
        self.logger.info(f"{self.name} strategy initialized. Active trades loaded: {len(self.active_trades)}")


    @staticmethod
    def get_parameters_definition(): # Same as original
        return {
            "leverage": {"type": "int", "label": "Leverage", "default": 3, "min": 1, "max": 25},
            "stop_loss_percent": {"type": "float", "label": "Stop Loss % (e.g., 0.05 for 5%)", "default": 0.05},
            "risk_per_trade_percent": {"type": "float", "label": "Risk Per Trade % (of balance)", "default": 0.05},
            "macd_fast_period": {"type": "int", "label": "MACD Fast Period", "default": 34},
            "macd_slow_period": {"type": "int", "label": "MACD Slow Period", "default": 144},
            "macd_signal_period": {"type": "int", "label": "MACD Signal Period", "default": 9},
            "top_n_symbols_to_scan": {"type": "int", "label": "Top N Symbols to Scan", "default": 10},
            "max_concurrent_trades": {"type": "int", "label": "Max Concurrent Trades", "default": 2},
            "kline_interval": {"type": "str", "label": "Kline Interval for MACD", "default": "15m"},
            "min_volume_usdt_24h": {"type": "float", "label": "Min. 24h QuoteVolume (USDT)", "default": 1000000.0},
            "min_price_change_percent_filter": {"type": "float", "label": "Min. Price Change % (for G/L)", "default": 3.0},
            "min_candles_for_macd": {"type": "int", "label": "Min. Candles for MACD", "default": 144},
        }

    def _load_all_persistent_positions(self):
        if not self.db_session or not self.user_sub_obj:
            self.logger.warning(f"[{self.name}] DB session/user_sub_obj not available for loading state.")
            return

        open_positions = self.db_session.query(Position).filter(
            Position.subscription_id == self.user_sub_obj.id,
            Position.is_open == True
        ).all()

        self.active_trades = {} # Reset before loading
        for pos in open_positions:
            self.logger.info(f"[{self.name}] Loading persistent state for open position ID: {pos.id} on {pos.symbol}")
            trade_state = {
                'position_db_id': pos.id,
                'entry_price': Decimal(str(pos.entry_price)) if pos.entry_price is not None else Decimal("0"),
                'qty': Decimal(str(pos.amount)) if pos.amount is not None else Decimal("0"),
                'side': pos.side,
                'sl_order_db_id': None,
                'sl_exchange_id': None
            }
            # Try to find associated open SL order
            sl_orders_db = strategy_utils.get_open_orders_for_subscription(
                self.db_session, self.user_sub_obj.id, pos.symbol, order_type='stop_market' # Standardized
            )
            if sl_orders_db: # Assuming one SL per position for this strategy
                trade_state['sl_order_db_id'] = sl_orders_db[0].id
                trade_state['sl_exchange_id'] = sl_orders_db[0].order_id
            self.active_trades[pos.symbol] = trade_state
        
        self.logger.info(f"[{self.name}] Loaded {len(self.active_trades)} active trades from DB: {list(self.active_trades.keys())}")


    def _get_top_gainer_loser_candidates(self): # Same as original
        self.logger.info("Fetching top gainer/loser candidates...")
        try:
            all_tickers = self.exchange_ccxt.fetch_tickers()
            futures_usdt_tickers = []
            for symbol, ticker_data in all_tickers.items():
                if '/USDT' not in symbol or not self.exchange_ccxt.markets.get(symbol, {}).get('future', False): continue
                if not ticker_data or 'quoteVolume' not in ticker_data or 'percentage' not in ticker_data: continue
                if ticker_data['quoteVolume'] >= self.min_volume_usdt_24h and abs(ticker_data['percentage']) >= self.min_price_change_percent_filter:
                    futures_usdt_tickers.append({'symbol': symbol, 'priceChangePercent': ticker_data['percentage'], 'quoteVolume': ticker_data['quoteVolume'], 'type': 'gainer' if ticker_data['percentage'] > 0 else 'loser'})
            if not futures_usdt_tickers: self.logger.info("No tickers met initial criteria."); return []
            sorted_tickers = sorted(futures_usdt_tickers, key=lambda x: abs(x['priceChangePercent']), reverse=True)
            candidates = sorted_tickers[:self.top_n_symbols_to_scan]
            self.logger.info(f"Found {len(candidates)} candidates: {[c['symbol'] for c in candidates]}")
            return candidates
        except Exception as e: self.logger.error(f"Error fetching tickers: {e}", exc_info=True); return []

    def _get_open_position_details(self, symbol: str) -> Optional[Dict[str, Any]]:
        return self.active_trades.get(symbol)

    def _get_active_positions_count(self) -> int:
        # More accurate count directly from DB for this subscription
        return self.db_session.query(Position).filter(
            Position.subscription_id == self.user_sub_obj.id,
            Position.is_open == True
        ).count()


    def _calculate_trade_qty(self, symbol: str, entry_price: float, stop_loss_price: float, side: str) -> Decimal:
        # Uses float for calculations, returns Decimal
        try:
            balance_response = self.exchange_ccxt.fetch_balance(params={'type': 'future'})
            usdt_balance = float(balance_response.get('USDT', {}).get('free', 0))
            if usdt_balance == 0: usdt_balance = float(balance_response.get('USDT', {}).get('total', 0))
            if usdt_balance <= 0: self.logger.warning("Insufficient USDT balance."); return Decimal("0")

            risk_amount_usdt = usdt_balance * self.risk_per_trade_percent
            price_diff_abs = abs(entry_price - stop_loss_price)
            if price_diff_abs == 0: return Decimal("0")
            
            # Position value based on risk per trade and stop loss distance
            base_position_value_usdt = risk_amount_usdt / (price_diff_abs / entry_price)
            quantity_asset = base_position_value_usdt / entry_price # This is the quantity if leverage=1x
            
            # The actual quantity to trade on exchange considering leverage
            # Notional value = quantity_asset_leveraged * entry_price
            # Margin required = Notional value / leverage
            # For now, let's assume order_quantity_usd from params IS the notional value
            # Or if we use risk_per_trade_percent, the quantity should be for NOTIONAL value
            # Let's use a fixed notional based on a portion of equity, e.g. 10% of equity as notional per trade
            # For simplicity, using the fixed USD amount from params for now, as per original logic for _place_order_with_sl's quantity calc
            # This means self.order_quantity_usd is the NOTIONAL value of the position.
            # quantity = self.order_quantity_usd / entry_price

            # Re-evaluating qty based on risk:
            # Risk amount = (entry_price - stop_loss_price) * quantity_base_asset * leverage_factor (if SL on notional)
            # OR Risk amount = ( (entry_price - stop_loss_price) / entry_price ) * Position_Notional_Value
            # Let Position_Notional_Value = X
            # risk_amount_usdt = (abs(entry_price - stop_loss_price) / entry_price) * X
            # X = risk_amount_usdt * entry_price / abs(entry_price - stop_loss_price)
            # quantity_asset_leveraged = X / entry_price
            
            position_notional_value_usdt = risk_amount_usdt * entry_price / price_diff_abs
            quantity = position_notional_value_usdt / entry_price # This is the actual quantity to trade (base asset)

            market = self.exchange_ccxt.markets[symbol]
            qty_precision_str = str(market['precision']['amount'])
            places = abs(Decimal(qty_precision_str).as_tuple().exponent)
            
            rounded_qty = Decimal(str(quantity)).quantize(Decimal('1e-' + str(places)), rounding=ROUND_DOWN)
            
            min_qty = Decimal(str(market['limits']['amount']['min']))
            if rounded_qty < min_qty:
                self.logger.warning(f"Calc qty {rounded_qty} for {symbol} < min_qty {min_qty}. Setting to 0.")
                return Decimal("0")
            return rounded_qty
        except Exception as e: self.logger.error(f"Error calculating position size for {symbol}: {e}", exc_info=True); return Decimal("0")

    def _place_order_with_sl(self, symbol: str, side: str, quantity: Decimal, entry_price: float, stop_loss_price: float) -> bool:
        self.logger.info(f"Attempting to place {side} order for {quantity} of {symbol} at market. Approx Entry: {entry_price}, SL: {stop_loss_price}")
        db_entry_order, db_sl_order = None, None
        try:
            if hasattr(self.exchange_ccxt, 'set_leverage'): self.exchange_ccxt.set_leverage(self.leverage, symbol)
            # if hasattr(self.exchange_ccxt, 'cancel_all_orders'): self.exchange_ccxt.cancel_all_orders(symbol) # Risky if other bots run

            order_side_exch = 'buy' if side == 'long' else 'sell'
            db_entry_order = strategy_utils.create_strategy_order_in_db(self.db_session, self.user_sub_obj.id, symbol, 'market', order_side_exch, float(quantity), price=entry_price, notes="Entry Order")
            if not db_entry_order: return False

            market_order_receipt = self.exchange_ccxt.create_order(symbol, 'MARKET', order_side_exch, float(quantity))
            strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=db_entry_order.id, updates={'order_id': market_order_receipt['id'], 'status': 'open', 'raw_order_data': json.dumps(market_order_receipt)})
            
            # Await fill for market order to get actual entry price
            filled_entry_order = self._await_order_fill(market_order_receipt['id'], symbol) # Assume _await_order_fill is available
            if not (filled_entry_order and filled_entry_order['status'] == 'closed'):
                self.logger.error(f"Market entry order {market_order_receipt['id']} failed to fill for {symbol}.")
                strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=db_entry_order.id, updates={'status': filled_entry_order.get('status', 'fill_check_failed') if filled_entry_order else 'fill_check_failed'})
                return False
            
            actual_entry_price = Decimal(str(filled_entry_order['average']))
            actual_filled_qty = Decimal(str(filled_entry_order['filled']))
            strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=db_entry_order.id, updates={'status': 'closed', 'price': float(actual_entry_price), 'filled': float(actual_filled_qty), 'cost': float(actual_entry_price * actual_filled_qty), 'closed_at': datetime.utcnow()})

            new_pos_db = strategy_utils.create_strategy_position_in_db(self.db_session, self.user_sub_obj.id, symbol, str(self.exchange_ccxt.id), side, float(actual_filled_qty), float(actual_entry_price), status_message=f"{self.name} Entry")
            if not new_pos_db: self.logger.error(f"Failed to create Position DB record for {symbol}."); return False
            
            sl_side_exch = 'sell' if side == 'long' else 'buy'
            sl_params = {'stopPrice': self.exchange_ccxt.price_to_precision(symbol, stop_loss_price)} # Ensure price_to_precision is used
            
            db_sl_order = strategy_utils.create_strategy_order_in_db(self.db_session, self.user_sub_obj.id, symbol, 'stop_market', sl_side_exch, float(actual_filled_qty), price=stop_loss_price, notes="SL Order")
            if not db_sl_order: self.logger.error(f"Failed to create DB record for SL order for {symbol}."); return False # Position is open without SL protection

            sl_order_receipt = self.exchange_ccxt.create_order(symbol, 'STOP_MARKET', sl_side_exch, float(actual_filled_qty), params=sl_params)
            strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=db_sl_order.id, updates={'order_id': sl_order_receipt['id'], 'status': 'open', 'raw_order_data': json.dumps(sl_order_receipt)})
            
            self.active_trades[symbol] = {
                'position_db_id': new_pos_db.id, 'entry_price': actual_entry_price, 
                'qty': actual_filled_qty, 'side': side,
                'sl_order_db_id': db_sl_order.id, 'sl_exchange_id': sl_order_receipt['id']
            }
            self.logger.info(f"Successfully placed {side} for {symbol}. PosID: {new_pos_db.id}, SL OrderID: {sl_order_receipt['id']}.")
            return True
        except Exception as e:
            self.logger.error(f"Error in _place_order_with_sl for {symbol}: {e}", exc_info=True)
            if db_entry_order and db_entry_order.id and db_entry_order.status != 'closed': strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=db_entry_order.id, updates={'status': 'error_on_exchange'})
            if db_sl_order and db_sl_order.id and db_sl_order.status != 'open': strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=db_sl_order.id, updates={'status': 'error_on_exchange'})
        return False

    def _close_open_position(self, symbol: str, current_pos_details: Dict[str, Any]):
        self.logger.info(f"Attempting to close {current_pos_details['side']} position for {symbol} (DB ID: {current_pos_details['position_db_id']})")
        db_close_order = None
        try:
            sl_exchange_id = current_pos_details.get('sl_exchange_id')
            sl_order_db_id = current_pos_details.get('sl_order_db_id')
            if sl_exchange_id:
                try: 
                    self.exchange_ccxt.cancel_order(sl_exchange_id, symbol)
                    if sl_order_db_id: strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=sl_order_db_id, updates={'status': 'canceled', 'status_message': 'Canceled for position close'})
                except ccxt.OrderNotFound: self.logger.warning(f"SL order {sl_exchange_id} not found to cancel during position close for {symbol}.")
                except Exception as e_cancel: self.logger.error(f"Error cancelling SL order {sl_exchange_id} for {symbol}: {e_cancel}")

            close_side_exch = 'sell' if current_pos_details['side'] == 'long' else 'buy'
            qty_to_close = current_pos_details['qty']
            
            db_close_order = strategy_utils.create_strategy_order_in_db(self.db_session, self.user_sub_obj.id, symbol, 'market', close_side_exch, float(qty_to_close), notes="Position Close Order")
            if not db_close_order: self.logger.error(f"Failed to create DB record for close order {symbol}."); return False

            market_close_receipt = self.exchange_ccxt.create_order(symbol, 'MARKET', close_side_exch, float(qty_to_close), params={'reduceOnly': True})
            strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=db_close_order.id, updates={'order_id': market_close_receipt['id'], 'status': 'open', 'raw_order_data': json.dumps(market_close_receipt)})

            filled_close_order = self._await_order_fill(market_close_receipt['id'], symbol) # Assume _await_order_fill is available
            if filled_close_order and filled_close_order['status'] == 'closed':
                actual_close_price = Decimal(str(filled_close_order['average']))
                actual_filled_qty = Decimal(str(filled_close_order['filled']))
                strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=db_close_order.id, updates={'status': 'closed', 'price': float(actual_close_price), 'filled': float(actual_filled_qty), 'cost': float(actual_close_price * actual_filled_qty), 'closed_at': datetime.utcnow()})
                
                strategy_utils.close_strategy_position_in_db(self.db_session, current_pos_details['position_db_id'], actual_close_price, actual_filled_qty, "Closed by strategy signal")
                self.logger.info(f"Position for {symbol} (DB ID: {current_pos_details['position_db_id']}) closed successfully.")
            else:
                self.logger.error(f"Market close order {market_close_receipt['id']} for {symbol} failed to fill or status unknown.")
                strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=db_close_order.id, updates={'status': filled_close_order.get('status', 'fill_check_failed') if filled_close_order else 'fill_check_failed'})
                return False # Indicate close might not be fully confirmed

            if symbol in self.active_trades: del self.active_trades[symbol]
            return True
        except Exception as e:
            self.logger.error(f"Error closing position for {symbol}: {e}", exc_info=True)
            if db_close_order and db_close_order.id: strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=db_close_order.id, updates={'status': 'error_on_exchange'})
        return False

    def _sync_individual_position_state(self, symbol: str) -> bool:
        if symbol not in self.active_trades: return False
        
        trade_state = self.active_trades[symbol]
        sl_exchange_id = trade_state.get('sl_exchange_id')
        sl_order_db_id = trade_state.get('sl_order_db_id')
        position_db_id = trade_state['position_db_id']

        if sl_exchange_id and sl_order_db_id:
            try:
                sl_details = self.exchange_ccxt.fetch_order(sl_exchange_id, symbol)
                if sl_details['status'] == 'closed':
                    self.logger.info(f"SL order {sl_exchange_id} for {symbol} found filled on exchange.")
                    strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=sl_order_db_id, updates={'status': 'closed', 'price': sl_details.get('average'), 'filled': sl_details.get('filled'), 'closed_at': datetime.utcnow(), 'raw_order_data': json.dumps(sl_details)})
                    strategy_utils.close_strategy_position_in_db(self.db_session, position_db_id, Decimal(str(sl_details['average'])), Decimal(str(sl_details['filled'])), f"Closed by SL {sl_exchange_id} (synced)")
                    if symbol in self.active_trades: del self.active_trades[symbol]
                    self.logger.info(f"Position for {symbol} (DB ID: {position_db_id}) marked closed due to synced SL fill.")
                    return True # Position was closed
            except ccxt.OrderNotFound:
                self.logger.warning(f"SL order {sl_exchange_id} for {symbol} not found during sync. Assuming manually canceled or already processed.")
                strategy_utils.update_strategy_order_in_db(self.db_session, order_db_id=sl_order_db_id, updates={'status': 'not_found_on_sync'})
                # Remove from active tracking if not found, as it can't be managed
                trade_state['sl_exchange_id'] = None; trade_state['sl_order_db_id'] = None 
            except Exception as e:
                self.logger.error(f"Error syncing SL order {sl_exchange_id} for {symbol}: {e}", exc_info=True)
        return False # Position not closed by this sync check

    def execute_live_signal(self): # Removed market_data_df as it fetches internally
        self.logger.info(f"Executing {self.name} strategy for UserSubID {self.user_sub_obj.id}...")
        
        # Load/Refresh all active positions for this subscription at the start of each cycle
        self._load_all_persistent_positions()

        # Sync status of individual positions (e.g. check if SL hit)
        for symbol_to_sync in list(self.active_trades.keys()): # list() for safe iteration if modifying dict
            if self._sync_individual_position_state(symbol_to_sync):
                self.logger.info(f"Position for {symbol_to_sync} was closed during sync. Re-evaluating trade counts.")
        
        active_candidates = self._get_top_gainer_loser_candidates()
        if not active_candidates: self.logger.info("No suitable gainer/loser candidates found."); return

        current_active_trades_count = len(self.active_trades) # Use length of our synced dict
        allowed_new_trades = self.max_concurrent_trades - current_active_trades_count
        
        processed_symbols_in_cycle = set()

        for candidate in active_candidates:
            symbol = candidate['symbol']
            if symbol in processed_symbols_in_cycle: continue
            processed_symbols_in_cycle.add(symbol)

            self.logger.info(f"Processing candidate: {symbol} ({candidate['type']})")
            current_pos_details = self.active_trades.get(symbol) # Use our synced state

            try:
                ohlcv = self.exchange_ccxt.fetch_ohlcv(symbol, timeframe=self.kline_interval, limit=self.min_candles_for_macd + 50)
                if len(ohlcv) < self.min_candles_for_macd: self.logger.warning(f"Not enough kline data for {symbol}."); continue

                closes = np.array([c[4] for c in ohlcv]); opens = np.array([c[1] for c in ohlcv])
                macd, signal, hist = talib.MACD(closes, fastperiod=self.macd_fast_period, slowperiod=self.macd_slow_period, signalperiod=self.macd_signal_period)
                if hist is None or len(hist) < 2: self.logger.warning(f"Could not calculate MACD for {symbol}."); continue
                
                current_macd_hist = hist[-1]; prev_macd_hist = hist[-2]
                is_green_candle = closes[-1] > opens[-1]; is_red_candle = closes[-1] < opens[-1]
                current_price = float(self.exchange_ccxt.fetch_ticker(symbol)['last'])
                if current_price is None: self.logger.warning(f"No current price for {symbol}."); continue

                # Buy Signal
                if current_macd_hist > prev_macd_hist and is_green_candle:
                    if current_pos_details and current_pos_details['side'] == 'long': self.logger.info(f"Already long {symbol}.")
                    else:
                        if current_pos_details and current_pos_details['side'] == 'short':
                            self.logger.info(f"Closing short {symbol} for reversal to long.")
                            self._close_open_position(symbol, current_pos_details) # This updates self.active_trades
                            current_pos_details = None # Mark as closed for this cycle
                        
                        current_active_trades_count = len(self.active_trades) # Re-check after potential close
                        allowed_new_trades = self.max_concurrent_trades - current_active_trades_count
                        if not current_pos_details and allowed_new_trades > 0:
                            sl_price = current_price * (1 - self.stop_loss_percent)
                            trade_qty = self._calculate_trade_qty(symbol, current_price, sl_price, 'long')
                            if trade_qty > Decimal("0"): self._place_order_with_sl(symbol, 'long', trade_qty, current_price, sl_price)
                            else: self.logger.warning(f"Calculated qty 0 for long {symbol}.")
                        else: self.logger.info(f"Conditions not met for new long on {symbol} (Max trades or pos not fully closed). Active: {current_active_trades_count}")
                
                # Sell Signal
                elif current_macd_hist < prev_macd_hist and is_red_candle:
                    if current_pos_details and current_pos_details['side'] == 'short': self.logger.info(f"Already short {symbol}.")
                    else:
                        if current_pos_details and current_pos_details['side'] == 'long':
                            self.logger.info(f"Closing long {symbol} for reversal to short.")
                            self._close_open_position(symbol, current_pos_details)
                            current_pos_details = None
                        
                        current_active_trades_count = len(self.active_trades)
                        allowed_new_trades = self.max_concurrent_trades - current_active_trades_count
                        if not current_pos_details and allowed_new_trades > 0:
                            sl_price = current_price * (1 + self.stop_loss_percent)
                            trade_qty = self._calculate_trade_qty(symbol, current_price, sl_price, 'short')
                            if trade_qty > Decimal("0"): self._place_order_with_sl(symbol, 'short', trade_qty, current_price, sl_price)
                            else: self.logger.warning(f"Calculated qty 0 for short {symbol}.")
                        else: self.logger.info(f"Conditions not met for new short on {symbol} (Max trades or pos not fully closed). Active: {current_active_trades_count}")
                else:
                    self.logger.info(f"No actionable MACD signal for {symbol}")
            except Exception as e: self.logger.error(f"Error processing symbol {symbol}: {e}", exc_info=True)
        self.logger.info(f"{self.name} execution cycle finished.")

    def run_backtest(self, historical_data_feed, current_simulated_time_utc):
        # Backtesting logic remains largely unchanged as it's simulation-focused
        # and does not directly interact with the live DB state or exchange in the same way.
        # However, it would need to simulate the self.active_trades dictionary management.
        self.logger.warning("Backtesting for TopGainersLosersMACD needs significant adaptation for multi-symbol state. Placeholder.")
        return {"action": "HOLD", "reason": "Backtest for multi-symbol not fully adapted in this refactor."}

```
