# trading_platform/strategies/top_gainers_losers_macd_strategy.py
import datetime
import logging
import time
import pandas as pd
import ta
import numpy as np
import json # For UserStrategySubscription parameters
from sqlalchemy.orm import Session
from backend.models import Position, Order, UserStrategySubscription

logger = logging.getLogger(__name__)

class TopGainersLosersMACDStrategy:
    def __init__(self, symbol: str, timeframe: str, capital: float = 10000, **custom_parameters):
        self.name = "TopGainersLosersMACDStrategy" # Ensure class name matches
        self.symbol_param_default = symbol # This strategy dynamically selects symbols
        self.timeframe = timeframe # Timeframe for MACD signals on selected symbols
        self.capital_total_allocation = capital # Total capital allocated to this strategy instance

        defaults = {
            "top_n_symbols": 3, "scan_interval_minutes": 15,
            "macd_fast_period": 12, "macd_slow_period": 26, "macd_signal_period": 9,
            "leverage": 3, "stop_loss_percent": 5.0, "risk_per_trade_percent": 1.0,
            "min_volume_usdt_24h": 1000000, "min_candles_for_signal": 50,
            "symbol_eval_period_hours": 6, "symbol_min_volatility_percent": 3.0
        }
        self_params = {**defaults, **custom_parameters}
        for key, value in self_params.items(): setattr(self, key, value)

        self.stop_loss_decimal = self.stop_loss_percent / 100.0
        self.risk_per_trade_decimal = self.risk_per_trade_percent / 100.0
        self.symbol_min_volatility_decimal = self.symbol_min_volatility_percent / 100.0
        
        # In-memory state for this strategy instance (not per symbol)
        self.last_scan_time_utc = None
        self.precisions_cache = {} 
        
        # Tracked symbols and their specific states will be managed via DB Position/Order custom_data or separate state model
        # For this pass, we'll simplify by keeping some operational state in memory, acknowledging it's not perfectly resilient.
        # A more robust solution would use UserStrategySubscription.custom_data or a dedicated StrategyState model.
        self.currently_tracked_symbols_operational_state = {} # e.g. { 'BTC/USDT': {'volatility_ok_until': datetime, ...} }

        logger.info(f"[{self.name}] Initialized with effective params: {self_params}")

    @classmethod
    def get_parameters_definition(cls):
        return {
            "top_n_symbols": {"type": "int", "default": 3, "min": 1, "max": 10, "label": "Top N Gainers/Losers to Track"},
            "scan_interval_minutes": {"type": "int", "default": 15, "min": 5, "max": 120, "label": "Scan Interval (minutes)"},
            "macd_fast_period": {"type": "int", "default": 12, "min": 2, "label": "MACD Fast Period"},
            "macd_slow_period": {"type": "int", "default": 26, "min": 2, "label": "MACD Slow Period"},
            "macd_signal_period": {"type": "int", "default": 9, "min": 2, "label": "MACD Signal Period"},
            "leverage": {"type": "int", "default": 3, "min": 1, "max": 20, "label": "Leverage (Informational)"},
            "stop_loss_percent": {"type": "float", "default": 5.0, "min": 0.1, "step": 0.1, "label": "Stop Loss %"},
            "risk_per_trade_percent": {"type": "float", "default": 1.0, "min": 0.1, "step": 0.1, "label": "Risk per Trade (% of allocated capital per symbol)"},
            "min_volume_usdt_24h": {"type": "float", "default": 1000000, "min": 10000, "label": "Min 24h Volume (USDT) for Scan"},
            "min_candles_for_signal": {"type": "int", "default": 50, "min": 30, "label": "Min Candles for MACD Signal"},
            "symbol_eval_period_hours": {"type": "int", "default": 6, "min":1, "label": "Symbol Volatility Check Period (Hours)"},
            "symbol_min_volatility_percent": {"type": "float", "default": 3.0, "min":0.1, "label": "Min Volatility % in Eval Period"}
        }

    def _get_precisions(self, symbol, exchange_ccxt):
        if symbol not in self.precisions_cache:
            try:
                exchange_ccxt.load_markets(True)
                market = exchange_ccxt.market(symbol)
                self.precisions_cache[symbol] = {'price': market['precision']['price'], 'amount': market['precision']['amount']}
                logger.info(f"[{self.name}] Precisions for {symbol}: {self.precisions_cache[symbol]}")
            except Exception as e: logger.error(f"[{self.name}] Error fetching precision for {symbol}: {e}", exc_info=True); self.precisions_cache[symbol] = {'price': 8, 'amount': 8}
        return self.precisions_cache[symbol]

    def _format_quantity(self, symbol, quantity, exchange_ccxt): prec = self._get_precisions(symbol, exchange_ccxt); return float(exchange_ccxt.amount_to_precision(symbol, quantity, precision=prec['amount']))
    def _format_price(self, symbol, price, exchange_ccxt): prec = self._get_precisions(symbol, exchange_ccxt); return float(exchange_ccxt.price_to_precision(symbol, price, precision=prec['price']))

    def _await_order_fill(self, exchange_ccxt, order_id: str, symbol: str, timeout_seconds: int = 60, check_interval_seconds: int = 3):
        start_time = time.time()
        logger.info(f"[{self.name}-{symbol}] Awaiting fill for order {order_id} (timeout: {timeout_seconds}s)")
        while time.time() - start_time < timeout_seconds:
            try:
                order = exchange_ccxt.fetch_order(order_id, symbol)
                logger.debug(f"[{self.name}-{symbol}] Order {order_id} status: {order['status']}")
                if order['status'] == 'closed': logger.info(f"[{self.name}-{symbol}] Order {order_id} filled. AvgPrice: {order.get('average')}, Qty: {order.get('filled')}"); return order
                if order['status'] in ['canceled', 'rejected', 'expired']: logger.warning(f"[{self.name}-{symbol}] Order {order_id} is {order['status']}."); return order
            except ccxt.OrderNotFound: logger.warning(f"[{self.name}-{symbol}] Order {order_id} not found. Retrying.")
            except Exception as e: logger.error(f"[{self.name}-{symbol}] Error fetching order {order_id}: {e}. Retrying.", exc_info=True)
            time.sleep(check_interval_seconds)
        logger.warning(f"[{self.name}-{self.symbol}] Timeout for order {order_id}. Final check.")
        try: final_status = exchange_ccxt.fetch_order(order_id, symbol); logger.info(f"[{self.name}-{self.symbol}] Final status for order {order_id}: {final_status['status']}"); return final_status
        except Exception as e: logger.error(f"[{self.name}-{self.symbol}] Final check for order {order_id} failed: {e}", exc_info=True); return None
        
    def _create_db_order(self, db_session: Session, subscription_id: int, symbol: str, **kwargs): # Added symbol
        db_order = Order(subscription_id=subscription_id, symbol=symbol, **kwargs); db_session.add(db_order); db_session.commit(); return db_order

    def _calculate_position_size_asset(self, symbol, entry_price, stop_loss_price, exchange_ccxt, capital_for_this_trade_usdt):
        if entry_price == 0 or abs(entry_price - stop_loss_price) == 0: return 0
        usdt_to_risk_on_this_trade = capital_for_this_trade_usdt * self.risk_per_trade_decimal
        price_diff_per_unit_at_sl = abs(entry_price - stop_loss_price)
        quantity_asset = usdt_to_risk_on_this_trade / price_diff_per_unit_at_sl
        return self._format_quantity(symbol, quantity_asset, exchange_ccxt)

    def _scan_top_gainers_losers(self, exchange_ccxt):
        logger.info(f"[{self.name}] Scanning for top gainers/losers...")
        try:
            all_tickers = exchange_ccxt.fetch_tickers()
            eligible_symbols = []
            for symbol, data in all_tickers.items():
                market = exchange_ccxt.market(symbol) # Ensures market data is loaded for the symbol
                if market.get('quote', '').upper() == 'USDT' and market.get('active', True) and \
                   (market.get('type','').lower() in ['future', 'swap'] or (market.get('spot') and not market.get('margin')) ) and \
                   data.get('quoteVolume', 0) >= self.min_volume_usdt_24h and data.get('change') is not None: # Use 'change' for %
                    eligible_symbols.append({'symbol': symbol, 'priceChangePercent': data['change'], 'volume_usdt': data['quoteVolume']}) # CCXT uses 'change' not 'percentage'
            
            if not eligible_symbols: logger.warning(f"[{self.name}] No eligible symbols found after scan filters."); return []
            
            sorted_by_change = sorted(eligible_symbols, key=lambda x: x['priceChangePercent'], reverse=True)
            top_gainers = sorted_by_change[:self.top_n_symbols]
            top_losers = sorted_by_change[-self.top_n_symbols:] # These will be least negative, or smallest positive if all are up.
            # Filter out losers that are actually gainers if all symbols are up
            true_losers = [s for s in top_losers if s['priceChangePercent'] < 0]
            
            logger.info(f"[{self.name}] Scan results: Gainers - {[s['symbol'] for s in top_gainers]}, Losers - {[s['symbol'] for s in true_losers]}")
            return top_gainers + true_losers # Combine, duplicates handled by caller
        except Exception as e: logger.error(f"[{self.name}] Error scanning tickers: {e}", exc_info=True); return []

    def run_backtest(self, historical_df: pd.DataFrame, htf_historical_df: pd.DataFrame = None):
        logger.warning(f"[{self.name}] Backtesting is not applicable for this multi-symbol scanner strategy via this interface.")
        return {"pnl": 0, "trades": [], "message": "Backtesting not implemented."}

    def execute_live_signal(self, db_session: Session, subscription_id: int, market_data_df: pd.DataFrame, exchange_ccxt, user_sub_obj: UserStrategySubscription):
        logger.debug(f"[{self.name}] Executing live signal for sub {subscription_id}...")
        now_utc = datetime.datetime.now(datetime.timezone.utc)

        # --- Scanner Logic ---
        if self.last_scan_time_utc is None or (now_utc - self.last_scan_time_utc) >= datetime.timedelta(minutes=self.scan_interval_minutes):
            scanned_symbols_info = self._scan_top_gainers_losers(exchange_ccxt)
            self.last_scan_time_utc = now_utc
            
            newly_scanned_symbols_set = set(item['symbol'] for item in scanned_symbols_info)
            current_operational_symbols = set(self.currently_tracked_symbols_operational_state.keys())

            for item in scanned_symbols_info: # Add new symbols
                symbol = item['symbol']
                if symbol not in self.currently_tracked_symbols_operational_state:
                    self.currently_tracked_symbols_operational_state[symbol] = { 'volatility_ok_until': now_utc + datetime.timedelta(hours=self.symbol_eval_period_hours) }
                    logger.info(f"[{self.name}] Added new symbol to track: {symbol}")
            
            for symbol in list(current_operational_symbols - newly_scanned_symbols_set): # Remove symbols no longer in scan
                # Before removing, check if there's an open DB position for this sub_id and symbol. If so, don't remove from operational state yet.
                # This check is complex as it requires querying DB positions for *this specific subscription* for *each removed symbol*.
                # For simplicity, we assume if a symbol is removed from scan, we stop tracking it operationally unless a DB position exists (checked later).
                logger.info(f"[{self.name}] Symbol {symbol} no longer in scan. Will be pruned if no active position.")
                # Actual pruning will happen in the next block if no active DB position.

        # --- Symbol Pruning & Trading Logic ---
        active_db_positions_symbols = {p.symbol for p in db_session.query(Position.symbol).filter(Position.subscription_id == subscription_id, Position.is_open == True).all()}
        
        # Prune symbols from operational state if they are past volatility check and have no active DB position
        for symbol in list(self.currently_tracked_symbols_operational_state.keys()):
            if symbol not in active_db_positions_symbols and \
               self.currently_tracked_symbols_operational_state[symbol].get('volatility_ok_until', now_utc) < now_utc :
                del self.currently_tracked_symbols_operational_state[symbol]
                logger.info(f"[{self.name}] Pruned symbol (stale/non-volatile, no active DB position): {symbol}")
        
        if not self.currently_tracked_symbols_operational_state and not active_db_positions_symbols: logger.info(f"[{self.name}] No symbols to trade or manage."); return

        capital_per_symbol = (self.capital_total_allocation / len(self.currently_tracked_symbols_operational_state)) if self.currently_tracked_symbols_operational_state else self.capital_total_allocation
        if capital_per_symbol == 0 and not active_db_positions_symbols : logger.warning(f"[{self.name}] Capital per symbol is zero."); return

        # Iterate through all symbols that are either in operational state or have active DB positions
        symbols_to_process = set(self.currently_tracked_symbols_operational_state.keys()).union(active_db_positions_symbols)

        for symbol in symbols_to_process:
            self._get_precisions(symbol, exchange_ccxt) # Ensure precisions are loaded for this symbol
            position_db = db_session.query(Position).filter(Position.subscription_id == subscription_id, Position.symbol == symbol, Position.is_open == True).first()
            
            try:
                ohlcv = exchange_ccxt.fetch_ohlcv(symbol, self.timeframe, limit=self.min_candles_for_signal + 5)
                if not ohlcv or len(ohlcv) < self.min_candles_for_signal: logger.warning(f"[{self.name}-{symbol}] Insufficient candles."); continue
                
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms'); df.set_index('timestamp', inplace=True)
                
                macd_obj = ta.trend.MACD(close=df['close'], fast=self.macd_fast_period, slow=self.macd_slow_period, signal=self.macd_signal_period)
                if macd_obj is None or macd_obj.empty() or len(macd_obj) < 2: logger.warning(f"[{self.name}-{symbol}] MACD calculation failed or insufficient length."); continue
                
                hist = macd_obj.macd_diff(); latest_hist = hist.iloc[-1]; prev_hist = hist.iloc[-2]
                latest_close = df['close'].iloc[-1]; latest_open = df['open'].iloc[-1]
                is_green = latest_close > latest_open; is_red = latest_close < latest_open

                # Exit Logic
                if position_db:
                    sl_price = position_db.entry_price * (1 - self.stop_loss_decimal if position_db.side == 'long' else 1 + self.stop_loss_decimal)
                    if (position_db.side == 'long' and latest_close <= sl_price) or \
                       (position_db.side == 'short' and latest_close >= sl_price):
                        logger.info(f"[{self.name}-{symbol}] SL hit for {position_db.side} Pos ID {position_db.id} at {latest_close}.")
                        close_side = 'sell' if position_db.side == 'long' else 'buy'; close_qty = self._format_quantity(symbol, position_db.amount, exchange_ccxt)
                        db_sl_exit_order = self._create_db_order(db_session, subscription_id, symbol=symbol, order_type='market', side=close_side, amount=close_qty, status='pending_creation')
                        try:
                            sl_exit_receipt = exchange_ccxt.create_market_order(symbol, close_side, close_qty, params={'reduceOnly': True})
                            db_sl_exit_order.order_id=sl_exit_receipt['id']; db_sl_exit_order.status='open'; db_session.commit()
                            filled_sl_exit = self._await_order_fill(exchange_ccxt, sl_exit_receipt['id'], symbol)
                            if filled_sl_exit and filled_sl_exit['status'] == 'closed':
                                db_sl_exit_order.status='closed'; db_sl_exit_order.price=filled_sl_exit['average']; db_sl_exit_order.filled=filled_sl_exit['filled']; db_sl_exit_order.cost=filled_sl_exit['cost']; db_sl_exit_order.updated_at=datetime.datetime.utcnow()
                                position_db.is_open=False; position_db.closed_at=datetime.datetime.utcnow()
                                pnl = (filled_sl_exit['average'] - position_db.entry_price) * filled_sl_exit['filled'] if position_db.side == 'long' else (position_db.entry_price - filled_sl_exit['average']) * filled_sl_exit['filled']
                                position_db.pnl=pnl; position_db.updated_at = datetime.datetime.utcnow()
                                logger.info(f"[{self.name}-{symbol}] {position_db.side} Pos ID {position_db.id} SL closed. PnL: {pnl:.2f}")
                            else: logger.error(f"[{self.name}-{symbol}] SL Exit order {sl_exit_receipt['id']} fail. Pos {position_db.id} open."); db_sl_exit_order.status = filled_sl_exit.get('status', 'fill_check_failed') if filled_sl_exit else 'fill_check_failed'
                            db_session.commit()
                        except Exception as e_sl_exit: logger.error(f"[{self.name}-{symbol}] Error closing SL for Pos {position_db.id}: {e_sl_exit}", exc_info=True); db_sl_exit_order.status='error'; db_session.commit()
                        continue # Move to next symbol after action

                # Entry Logic
                if not position_db and (symbol in self.currently_tracked_symbols_operational_state): # Only enter if still actively tracked
                    entry_side = None
                    if latest_hist > 0 and prev_hist <= 0 and is_green: entry_side = 'buy'  # Long signal
                    elif latest_hist < 0 and prev_hist >= 0 and is_red: entry_side = 'sell' # Short signal

                    if entry_side:
                        sl_price_for_sizing = latest_close * (1 - self.stop_loss_decimal) if entry_side == 'buy' else latest_close * (1 + self.stop_loss_decimal)
                        qty = self._calculate_position_size_asset(symbol, latest_close, sl_price_for_sizing, exchange_ccxt, capital_per_symbol)
                        if qty <= 0: logger.warning(f"[{self.name}-{symbol}] Calculated qty zero or less. Skipping."); continue

                        logger.info(f"[{self.name}-{symbol}] {entry_side.upper()} SIGNAL at {latest_close}. Qty: {qty}, SL: {sl_price_for_sizing}")
                        db_entry_order = self._create_db_order(db_session, subscription_id, symbol=symbol, order_type='market', side=entry_side, amount=qty, status='pending_creation')
                        try:
                            entry_receipt = exchange_ccxt.create_market_order(symbol, entry_side, qty)
                            db_entry_order.order_id=entry_receipt['id']; db_entry_order.status='open'; db_session.commit()
                            filled_entry = self._await_order_fill(exchange_ccxt, entry_receipt['id'], symbol)
                            if filled_entry and filled_entry['status'] == 'closed':
                                db_entry_order.status='closed'; db_entry_order.price=filled_entry['average']; db_entry_order.filled=filled_entry['filled']; db_entry_order.cost=filled_entry['cost']; db_entry_order.updated_at=datetime.datetime.utcnow()
                                
                                new_pos = Position(subscription_id=subscription_id, symbol=symbol, exchange_name=str(exchange_ccxt.id), side=('long' if entry_side=='buy' else 'short'), amount=filled_entry['filled'], entry_price=filled_entry['average'], current_price=filled_entry['average'], is_open=True, created_at=datetime.datetime.utcnow(), updated_at=datetime.datetime.utcnow())
                                db_session.add(new_pos); db_session.commit()
                                logger.info(f"[{self.name}-{symbol}] {new_pos.side.upper()} Pos ID {new_pos.id} created. Entry: {new_pos.entry_price}, Size: {new_pos.amount}")
                                # Place SL order (TP is not used by this strategy's original logic)
                                sl_order_side = 'sell' if new_pos.side == 'long' else 'buy'
                                sl_price_actual = new_pos.entry_price * (1 - self.stop_loss_decimal if new_pos.side == 'long' else 1 + self.stop_loss_decimal)
                                db_sl_order = self._create_db_order(db_session, subscription_id, symbol=symbol, order_type='stop_market', side=sl_order_side, amount=new_pos.amount, price=self._format_price(symbol, sl_price_actual, exchange_ccxt), status='pending_creation')
                                try:
                                    sl_receipt = exchange_ccxt.create_order(symbol, 'stop_market', sl_order_side, new_pos.amount, params={'stopPrice': self._format_price(symbol, sl_price_actual, exchange_ccxt), 'reduceOnly':True})
                                    db_sl_order.order_id=sl_receipt['id']; db_sl_order.status='open'; logger.info(f"[{self.name}-{symbol}] SL order {sl_receipt['id']} for Pos {new_pos.id}")
                                except Exception as e_sl: logger.error(f"[{self.name}-{symbol}] Error SL for Pos {new_pos.id}: {e_sl}", exc_info=True); db_sl_order.status='error'
                                db_session.commit()
                            else: logger.error(f"[{self.name}-{symbol}] Entry order {entry_receipt['id']} failed. Pos not opened."); db_entry_order.status = filled_entry.get('status', 'fill_check_failed') if filled_entry else 'fill_check_failed'
                            db_session.commit()
                        except Exception as e_entry: logger.error(f"[{self.name}-{symbol}] Error during {entry_side} entry for {symbol}: {e_entry}", exc_info=True); db_entry_order.status='error'; db_session.commit()
            except Exception as e_sym_proc: logger.error(f"[{self.name}] Error processing symbol {symbol}: {e_sym_proc}", exc_info=True)
        logger.debug(f"[{self.name}] Live signal execution cycle finished for sub {subscription_id}.")
