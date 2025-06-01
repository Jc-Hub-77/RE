import logging
import numpy as np
import talib
import ccxt
import json
import datetime
from decimal import Decimal, ROUND_DOWN, ROUND_UP

# Assuming platform models are accessible for type hinting if needed,
# but the strategy itself won't import them directly for creation/update.
# from backend.models import Order, Position

class TopGainersLosersMACD:
    def __init__(self, db_session, user_sub_obj, strategy_params: dict, exchange_ccxt, logger=None):
        self.db_session = db_session # SQLAlchemy session
        self.user_sub_obj = user_sub_obj # UserStrategySubscription ORM object
        self.params = strategy_params
        self.exchange_ccxt = exchange_ccxt
        self.logger = logger if logger else logging.getLogger(__name__)
        self.name = "TopGainersLosersMACD"

        # Extract and validate parameters
        self.leverage = int(self.params.get("leverage", 3))
        self.stop_loss_percent = float(self.params.get("stop_loss_percent", 0.05))
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
        # self.margin_mode = self.params.get("margin_mode", "ISOLATED").upper()

        self.logger.info(f"{self.name} strategy initialized for UserSubID {self.user_sub_obj.id} with params: {self.params}")

    @staticmethod
    def get_parameters_definition():
        return {
            "leverage": {"type": "int", "label": "Leverage", "default": 3, "min": 1, "max": 25},
            "stop_loss_percent": {"type": "float", "label": "Stop Loss % (e.g., 0.05 for 5%)", "default": 0.05, "min": 0.001, "max": 0.5, "step": 0.001},
            "risk_per_trade_percent": {"type": "float", "label": "Risk Per Trade % (of balance)", "default": 0.05, "min": 0.001, "max": 0.1, "step": 0.001},
            "macd_fast_period": {"type": "int", "label": "MACD Fast Period", "default": 34, "min": 5, "max": 200},
            "macd_slow_period": {"type": "int", "label": "MACD Slow Period", "default": 144, "min": 20, "max": 500},
            "macd_signal_period": {"type": "int", "label": "MACD Signal Period", "default": 9, "min": 3, "max": 50},
            "top_n_symbols_to_scan": {"type": "int", "label": "Top N Symbols to Scan (Gainers+Losers)", "default": 10, "min": 2, "max": 50},
            "max_concurrent_trades": {"type": "int", "label": "Max Concurrent Trades", "default": 2, "min": 1, "max": 10},
            "kline_interval": {"type": "str", "label": "Kline Interval for MACD", "default": "15m", "choices": ["5m", "15m", "30m", "1h", "4h"]},
            "min_volume_usdt_24h": {"type": "float", "label": "Min. 24h QuoteVolume (USDT)", "default": 1000000.0, "min": 0.0},
            "min_price_change_percent_filter": {"type": "float", "label": "Min. Price Change % (for G/L)", "default": 3.0, "min": 0.0},
            "min_candles_for_macd": {"type": "int", "label": "Min. Candles for MACD", "default": 144, "min": 50, "max": 500},
            # "margin_mode": {"type": "str", "label": "Margin Mode", "default": "ISOLATED", "choices": ["ISOLATED", "CROSSED"]}
        }

    def _get_top_gainer_loser_candidates(self):
        self.logger.info("Fetching top gainer/loser candidates...")
        try:
            all_tickers = self.exchange_ccxt.fetch_tickers() # Fetches for all available markets

            futures_usdt_tickers = []
            for symbol, ticker_data in all_tickers.items():
                if '/USDT' not in symbol or not self.exchange_ccxt.markets[symbol].get('future', False): # Filter for USDT futures
                    continue
                if not ticker_data or 'quoteVolume' not in ticker_data or 'percentage' not in ticker_data:
                    continue

                if ticker_data['quoteVolume'] >= self.min_volume_usdt_24h and \
                   abs(ticker_data['percentage']) >= self.min_price_change_percent_filter:
                    futures_usdt_tickers.append({
                        'symbol': symbol,
                        'priceChangePercent': ticker_data['percentage'],
                        'quoteVolume': ticker_data['quoteVolume'],
                        'type': 'gainer' if ticker_data['percentage'] > 0 else 'loser'
                    })

            if not futures_usdt_tickers:
                self.logger.info("No tickers met initial volume and price change criteria.")
                return []

            # Sort by absolute percentage change to get top movers overall
            sorted_tickers = sorted(futures_usdt_tickers, key=lambda x: abs(x['priceChangePercent']), reverse=True)

            candidates = sorted_tickers[:self.top_n_symbols_to_scan]
            self.logger.info(f"Found {len(candidates)} candidates: {[c['symbol'] for c in candidates]}")
            return candidates

        except ccxt.NetworkError as e:
            self.logger.error(f"Network error fetching tickers: {e}")
        except ccxt.ExchangeError as e:
            self.logger.error(f"Exchange error fetching tickers: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error fetching tickers: {e}", exc_info=True)
        return []

    def _get_open_position_details(self, symbol):
        '''Checks if an open position exists for the symbol for THIS subscription.'''
        # This requires querying the database for positions linked to self.user_sub_obj.id
        # For now, this is a simplified version. A real implementation would use:
        # open_pos = self.db_session.query(Position).filter(
        #     Position.subscription_id == self.user_sub_obj.id,
        #     Position.symbol == symbol,
        #     Position.is_open == True
        # ).first()
        # if open_pos:
        #     return {'side': open_pos.side, 'qty': open_pos.amount, 'entry_price': open_pos.entry_price, 'id': open_pos.id}
        # return None
        
        # Simplified CCXT check (may not be specific enough if multiple bots use same account)
        try:
            positions = self.exchange_ccxt.fetch_positions([symbol])
            for p in positions:
                # CCXT position amount can be positive (long) or negative (short). Zero means no position.
                # 'side' might be 'long'/'short' or not present; 'contracts' or 'info' for quantity.
                # This needs to be adapted based on specific exchange CCXT response for futures.
                # Example for Binance-like:
                if p['symbol'] == symbol and float(p.get('contracts', p.get('unrealizedPnl', 0)) != 0): # Check if 'contracts' exists and is non-zero
                    pos_qty = float(p.get('contracts', 0))
                    if pos_qty == 0 and 'info' in p and 'positionAmt' in p['info']: # Fallback for some exchanges
                         pos_qty = float(p['info']['positionAmt'])

                    if pos_qty != 0:
                        side = 'long' if pos_qty > 0 else 'short'
                        return {'side': side, 'qty': abs(pos_qty), 'entry_price': float(p.get('entryPrice', 0))}
        except Exception as e:
            self.logger.error(f"Error fetching position for {symbol}: {e}")
        return None


    def _get_active_positions_count(self):
        # Placeholder: Counts open positions associated with this subscription_id in the DB
        # count = self.db_session.query(Position).filter(
        #     Position.subscription_id == self.user_sub_obj.id,
        #     Position.is_open == True
        # ).count()
        # return count
        # Simplified for now, as _get_open_position_details is also simplified
        # This count should ideally come from DB to be accurate for *this strategy instance*
        # For now, we are limited by not having DB write access in this subtask
        # We will track open_positions_symbols based on what this current execution cycle tries to trade
        # This is not perfect but a step.
        return 0 # This will be refined or rely on DB queries in a real scenario.


    def _calculate_trade_qty(self, symbol, entry_price, stop_loss_price, side):
        try:
            balance_response = self.exchange_ccxt.fetch_balance(params={'type': 'future'}) # Check your exchange's specific param for futures balance
            usdt_balance = balance_response['USDT']['free'] if 'USDT' in balance_response and 'free' in balance_response['USDT'] else 0
            if usdt_balance == 0: # Try total if free is 0
                 usdt_balance = balance_response['USDT']['total'] if 'USDT' in balance_response and 'total' in balance_response['USDT'] else 0

            if usdt_balance <= 0:
                self.logger.warning("Insufficient USDT balance for trading.")
                return 0

            risk_amount_usdt = usdt_balance * self.risk_per_trade_percent
            price_diff_abs = abs(entry_price - stop_loss_price)
            if price_diff_abs == 0: return 0

            # Quantity in base asset
            # quantity = risk_amount_usdt / price_diff_abs

            # Adjust for leverage: The position value is qty * entry_price.
            # The actual margin used is (qty * entry_price) / leverage.
            # The risk calculation above is about how much capital to lose, not position size directly with leverage.
            # Position size should be ( (equity * risk_per_trade) / (entry_price - stop_loss_price) ) * entry_price for value, then / entry_price for qty.
            # The above calculation is for quantity of base asset if 1x leverage was used for that risk.
            # If we want leveraged position:
            # Max position value we can afford to lose `risk_amount_usdt` on with `stop_loss_percent` move:
            # position_value_at_risk = risk_amount_usdt / self.stop_loss_percent
            # quantity = position_value_at_risk / entry_price
            # This quantity is then leveraged by self.leverage, but margin is (position_value_at_risk / self.leverage)

            # Simpler: original script's base_position_value = risk_amount / (price_difference / entry_price)
            # This base_position_value is the notional size in USDT.
            # So, quantity = base_position_value / entry_price
            base_position_value = risk_amount_usdt / (price_diff_abs / entry_price)
            quantity = base_position_value / entry_price


            market = self.exchange_ccxt.markets[symbol]
            qty_precision = market['precision']['amount']
            
            # Use Decimal for precision rounding
            decimal_qty = Decimal(str(quantity))
            rounded_qty = decimal_qty.quantize(Decimal(str(qty_precision)), rounding=ROUND_DOWN)
            
            min_qty = market['limits']['amount']['min']
            if float(rounded_qty) < min_qty:
                self.logger.warning(f"Calculated quantity {rounded_qty} for {symbol} is less than min_qty {min_qty}. Setting to 0.")
                return 0
            
            return float(rounded_qty)

        except Exception as e:
            self.logger.error(f"Error calculating position size for {symbol}: {e}", exc_info=True)
            return 0

    def _place_order_with_sl(self, symbol, side, quantity, entry_price, stop_loss_price):
        self.logger.info(f"Attempting to place {side} order for {quantity} of {symbol} at market. Entry: {entry_price}, SL: {stop_loss_price}")
        try:
            # 1. Set Leverage (if needed and supported)
            if hasattr(self.exchange_ccxt, 'set_leverage'):
                 self.exchange_ccxt.set_leverage(self.leverage, symbol) # Removed marginMode, assume default or user set
                 self.logger.info(f"Set leverage to {self.leverage}x for {symbol}")

            # 2. Cancel existing orders for the symbol (optional, but good practice if opening new trade)
            if hasattr(self.exchange_ccxt, 'cancel_all_orders'):
                self.exchange_ccxt.cancel_all_orders(symbol)
                self.logger.info(f"Cancelled all existing open orders for {symbol}")

            # 3. Open Market Order
            order_side = 'buy' if side == 'long' else 'sell'
            market_order = self.exchange_ccxt.create_order(symbol, 'MARKET', order_side, quantity)
            self.logger.info(f"Market {order_side} order placed for {symbol}: {market_order['id'] if market_order else 'Failed'}")
            # TODO: Record market_order in DB, link to self.user_sub_obj.id

            # 4. Place Stop Loss Order
            sl_side = 'sell' if side == 'long' else 'buy'
            sl_params = {'stopPrice': self.exchange_ccxt.price_to_precision(symbol, stop_loss_price)}
            
            # Ensure stopPrice is correctly formatted for the exchange
            # Some exchanges require 'reduceOnly' for SL orders not to open new positions
            # sl_params['reduceOnly'] = True # Add if appropriate for your exchange and logic

            stop_loss_order = self.exchange_ccxt.create_order(symbol, 'STOP_MARKET', sl_side, quantity, params=sl_params)
            self.logger.info(f"Stop loss ({sl_side}) order placed for {symbol} at {stop_loss_price}: {stop_loss_order['id'] if stop_loss_order else 'Failed'}")
            # TODO: Record stop_loss_order in DB

            # TODO: Create/Update Position record in DB for this new trade
            # new_position = Position(subscription_id=self.user_sub_obj.id, symbol=symbol, side=side, amount=quantity, entry_price=entry_price, ...)
            # self.db_session.add(new_position)
            # self.db_session.commit()
            self.logger.info(f"DB: Placeholder for creating Position record for {symbol} {side} {quantity} @ {entry_price}")

            return True
        except ccxt.InsufficientFunds as e:
            self.logger.error(f"Insufficient funds for {symbol} {side} order: {e}")
        except ccxt.NetworkError as e:
            self.logger.error(f"Network error placing order for {symbol}: {e}")
        except ccxt.ExchangeError as e: # Catch other specific CCXT errors if needed
            self.logger.error(f"Exchange error placing order for {symbol}: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error placing order for {symbol}: {e}", exc_info=True)
        return False

    def _close_open_position(self, symbol, current_pos_details):
        self.logger.info(f"Attempting to close {current_pos_details['side']} position for {symbol} of qty {current_pos_details['qty']}")
        try:
            close_side = 'sell' if current_pos_details['side'] == 'long' else 'buy'
            
            # Cancel existing SL/TP orders first
            if hasattr(self.exchange_ccxt, 'cancel_all_orders'):
                self.exchange_ccxt.cancel_all_orders(symbol)
                self.logger.info(f"Cancelled all orders for {symbol} before closing position.")

            market_close_order = self.exchange_ccxt.create_order(symbol, 'MARKET', close_side, current_pos_details['qty'])
            self.logger.info(f"Market {close_side} (close) order placed for {symbol}: {market_close_order['id'] if market_close_order else 'Failed'}")

            # TODO: Update Position record in DB to closed
            # pos_to_update = self.db_session.query(Position).filter_by(id=current_pos_details['id']).first()
            # if pos_to_update: pos_to_update.is_open = False; pos_to_update.closed_at = datetime.datetime.utcnow(); # calculate PNL
            # self.db_session.commit()
            self.logger.info(f"DB: Placeholder for updating Position record for {symbol} to closed.")
            return True
        except Exception as e:
            self.logger.error(f"Error closing position for {symbol}: {e}", exc_info=True)
            return False

    def execute_live_signal(self, market_data_df=None): # market_data_df for a primary symbol is not used here
        self.logger.info(f"Executing {self.name} strategy for UserSubID {self.user_sub_obj.id}...")

        # 1. Fetch Top Gainers and Losers
        active_candidates = self._get_top_gainer_loser_candidates()
        if not active_candidates:
            self.logger.info("No suitable gainer/loser candidates found in this cycle.")
            return

        # 2. Manage Positions & Trades
        # This count should ideally come from DB specific to this subscription's trades
        current_active_trades_count = self._get_active_positions_count() # Simplified for now
        allowed_new_trades = self.max_concurrent_trades - current_active_trades_count
        
        processed_symbols_in_cycle = set() # To avoid processing same symbol if it's both gainer & loser (rare)

        for candidate in active_candidates:
            symbol = candidate['symbol']
            if symbol in processed_symbols_in_cycle:
                continue
            processed_symbols_in_cycle.add(symbol)

            self.logger.info(f"Processing candidate: {symbol} ({candidate['type']})")

            current_pos_details = self._get_open_position_details(symbol)

            try:
                ohlcv = self.exchange_ccxt.fetch_ohlcv(symbol, timeframe=self.kline_interval, limit=self.min_candles_for_macd + 50) # +50 buffer for TA lib
                if len(ohlcv) < self.min_candles_for_macd:
                    self.logger.warning(f"Not enough kline data for {symbol} ({len(ohlcv)} candles). Need {self.min_candles_for_macd}.")
                    continue

                closes = np.array([candle[4] for candle in ohlcv])
                opens = np.array([candle[1] for candle in ohlcv])

                macd, signal, hist = talib.MACD(
                    closes, fastperiod=self.macd_fast_period, slowperiod=self.macd_slow_period, signalperiod=self.macd_signal_period
                )
                if hist is None or len(hist) < 2:
                    self.logger.warning(f"Could not calculate MACD or not enough values for {symbol}.")
                    continue
                
                current_macd_hist = hist[-1]
                prev_macd_hist = hist[-2]
                
                # Last closed candle color
                last_closed_candle_close = closes[-1] # Most recent fully closed candle
                last_closed_candle_open = opens[-1]
                
                is_green_candle = last_closed_candle_close > last_closed_candle_open
                is_red_candle = last_closed_candle_close < last_closed_candle_open

                ticker = self.exchange_ccxt.fetch_ticker(symbol)
                current_price = ticker['last']
                if current_price is None:
                    self.logger.warning(f"Could not fetch current price for {symbol}.")
                    continue

                # Buy Signal Logic
                if current_macd_hist > prev_macd_hist and is_green_candle:
                    self.logger.info(f"Buy signal detected for {symbol} at {current_price}")
                    if current_pos_details and current_pos_details['side'] == 'long':
                        self.logger.info(f"Already in a long position for {symbol}.")
                    else:
                        if current_pos_details and current_pos_details['side'] == 'short':
                            self.logger.info(f"Closing existing short position for {symbol} before opening long.")
                            self._close_open_position(symbol, current_pos_details)
                            current_pos_details = None # Position is now closed

                        if not current_pos_details and allowed_new_trades > 0:
                            sl_price = current_price * (1 - self.stop_loss_percent)
                            trade_qty = self._calculate_trade_qty(symbol, current_price, sl_price, 'long')
                            if trade_qty > 0:
                                if self._place_order_with_sl(symbol, 'long', trade_qty, current_price, sl_price):
                                    allowed_new_trades -= 1
                            else:
                                self.logger.warning(f"Calculated quantity is 0 for {symbol}. Skipping trade.")
                        elif current_pos_details: # implies it was a short, now closed
                             sl_price = current_price * (1 - self.stop_loss_percent)
                             trade_qty = self._calculate_trade_qty(symbol, current_price, sl_price, 'long')
                             if trade_qty > 0: self._place_order_with_sl(symbol, 'long', trade_qty, current_price, sl_price)
                        else:
                            self.logger.info(f"Max concurrent trades reached or no position to close for {symbol}. Cannot open new long.")

                # Sell Signal Logic
                elif current_macd_hist < prev_macd_hist and is_red_candle:
                    self.logger.info(f"Sell signal detected for {symbol} at {current_price}")
                    if current_pos_details and current_pos_details['side'] == 'short':
                        self.logger.info(f"Already in a short position for {symbol}.")
                    else:
                        if current_pos_details and current_pos_details['side'] == 'long':
                            self.logger.info(f"Closing existing long position for {symbol} before opening short.")
                            self._close_open_position(symbol, current_pos_details)
                            current_pos_details = None # Position is now closed

                        if not current_pos_details and allowed_new_trades > 0:
                            sl_price = current_price * (1 + self.stop_loss_percent)
                            trade_qty = self._calculate_trade_qty(symbol, current_price, sl_price, 'short')
                            if trade_qty > 0:
                                if self._place_order_with_sl(symbol, 'short', trade_qty, current_price, sl_price):
                                    allowed_new_trades -=1
                            else:
                                self.logger.warning(f"Calculated quantity is 0 for {symbol}. Skipping trade.")
                        elif current_pos_details: # implies it was a long, now closed
                            sl_price = current_price * (1 + self.stop_loss_percent)
                            trade_qty = self._calculate_trade_qty(symbol, current_price, sl_price, 'short')
                            if trade_qty > 0: self._place_order_with_sl(symbol, 'short', trade_qty, current_price, sl_price)
                        else:
                            self.logger.info(f"Max concurrent trades reached or no position to close for {symbol}. Cannot open new short.")
                else:
                    self.logger.info(f"No actionable MACD signal for {symbol} (Hist: {current_macd_hist:.8f} vs {prev_macd_hist:.8f}, Candle: {'Green' if is_green_candle else 'Red' if is_red_candle else 'Neutral'})")

            except ccxt.NetworkError as e:
                self.logger.error(f"CCXT NetworkError processing symbol {symbol}: {e}")
            except ccxt.ExchangeError as e:
                self.logger.error(f"CCXT ExchangeError processing symbol {symbol}: {e}")
            except Exception as e:
                self.logger.error(f"Unexpected error processing symbol {symbol}: {e}", exc_info=True)

        self.logger.info(f"{self.name} execution cycle finished.")

    # run_backtest method would be similar but use historical data feeder
    # For now, we focus on execute_live_signal
