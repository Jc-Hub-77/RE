# backend/strategy_utils.py
import datetime
import json
import logging
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from backend.models import Position, Order, UserStrategySubscription # Ensure these are correct paths

logger = logging.getLogger(__name__)

def get_open_strategy_position(db_session: Session, subscription_id: int, symbol: str) -> Optional[Position]:
    """Queries and returns an open Position for a given subscription and symbol, or None."""
    try:
        return db_session.query(Position).filter(
            Position.subscription_id == subscription_id,
            Position.symbol == symbol,
            Position.is_open == True
        ).first()
    except Exception as e:
        logger.error(f"Error fetching open position for sub {subscription_id}, sym {symbol}: {e}", exc_info=True)
        return None

def get_open_orders_for_subscription(db_session: Session, subscription_id: int, symbol: str, order_type: Optional[str] = None) -> List[Order]:
    """
    Queries for open Order records for a given subscription and symbol.
    Optionally filters by a specific order_type (e.g., 'stop_market', 'limit').
    """
    try:
        query = db_session.query(Order).filter(
            Order.subscription_id == subscription_id,
            Order.symbol == symbol,
            Order.status == 'open' # Or consider other non-terminal statuses like 'new'
        )
        if order_type:
            query = query.filter(Order.order_type == order_type)
        return query.all()
    except Exception as e:
        logger.error(f"Error fetching open orders for sub {subscription_id}, sym {symbol}: {e}", exc_info=True)
        return []

def create_strategy_order_in_db(db_session: Session, subscription_id: int, symbol: str, 
                                order_type: str, side: str, amount: float, price: Optional[float] = None, 
                                status: str = 'pending_creation', exchange_order_id: Optional[str] = None, 
                                raw_order_data: Optional[dict] = None, notes: Optional[str] = None) -> Optional[Order]:
    """Creates, commits, and returns a new Order record, or None on error."""
    try:
        db_order = Order(
            subscription_id=subscription_id,
            symbol=symbol,
            order_type=order_type,
            side=side,
            amount=amount,
            price=price,
            status=status,
            order_id=exchange_order_id,
            raw_order_data=json.dumps(raw_order_data) if raw_order_data else None,
            notes=notes,
            created_at=datetime.datetime.utcnow(),
            updated_at=datetime.datetime.utcnow()
        )
        db_session.add(db_order)
        db_session.commit()
        db_session.refresh(db_order)
        logger.info(f"Strategy order created in DB: ID {db_order.id}, ExchID {exchange_order_id} for SubID {subscription_id}")
        return db_order
    except Exception as e:
        logger.error(f"Error creating strategy order in DB for SubID {subscription_id}: {e}", exc_info=True)
        db_session.rollback()
        return None

def update_strategy_order_in_db(db_session: Session, order_db_id: Optional[int] = None, 
                                exchange_order_id: Optional[str] = None, 
                                subscription_id: Optional[int] = None, # Required if using exchange_order_id
                                symbol: Optional[str] = None, # Required if using exchange_order_id
                                updates: dict) -> Optional[Order]:
    """
    Updates an existing Order record identified by DB ID or exchange_order_id (scoped to subscription & symbol).
    'updates' dict contains fields to update, e.g., {'status': 'closed', 'filled': 0.1, ...}.
    Returns the updated order or None on error/not found.
    """
    try:
        order_to_update = None
        if order_db_id:
            order_to_update = db_session.query(Order).filter(Order.id == order_db_id).first()
        elif exchange_order_id and subscription_id and symbol:
            order_to_update = db_session.query(Order).filter(
                Order.order_id == exchange_order_id, # Corrected from Order.exchange_order_id to Order.order_id
                Order.subscription_id == subscription_id,
                Order.symbol == symbol
            ).first()
        
        if not order_to_update:
            logger.warning(f"Order not found for update. DB_ID: {order_db_id}, ExchID: {exchange_order_id}, SubID: {subscription_id}")
            return None

        for key, value in updates.items():
            if hasattr(order_to_update, key):
                setattr(order_to_update, key, value)
        order_to_update.updated_at = datetime.datetime.utcnow()
        
        db_session.commit()
        db_session.refresh(order_to_update)
        logger.info(f"Strategy order DB ID {order_to_update.id} (ExchID {order_to_update.order_id}) updated. Status: {order_to_update.status}")
        return order_to_update
    except Exception as e:
        logger.error(f"Error updating strategy order DB ID {order_db_id} / ExchID {exchange_order_id}: {e}", exc_info=True)
        db_session.rollback()
        return None

def create_strategy_position_in_db(db_session: Session, subscription_id: int, symbol: str, 
                                   exchange_name: str, side: str, amount: float, entry_price: float, 
                                   status_message: Optional[str] = "Position Opened", 
                                   current_price_override: Optional[float] = None) -> Optional[Position]:
    """Creates, commits, and returns a new Position record, or None on error."""
    try:
        new_pos = Position(
            subscription_id=subscription_id,
            symbol=symbol,
            exchange_name=exchange_name,
            side=side,
            amount=amount,
            entry_price=entry_price,
            current_price=current_price_override if current_price_override is not None else entry_price,
            is_open=True,
            status_message=status_message,
            created_at=datetime.datetime.utcnow(),
            updated_at=datetime.datetime.utcnow()
        )
        db_session.add(new_pos)
        db_session.commit()
        db_session.refresh(new_pos)
        logger.info(f"Strategy position created in DB: ID {new_pos.id} for SubID {subscription_id}, {side} {amount} {symbol} @ {entry_price}")
        return new_pos
    except Exception as e:
        logger.error(f"Error creating strategy position in DB for SubID {subscription_id}: {e}", exc_info=True)
        db_session.rollback()
        return None

def close_strategy_position_in_db(db_session: Session, position_db_id: int, close_price: float, 
                                  filled_amount_at_close: float, reason: str, 
                                  pnl_override: Optional[float] = None) -> Optional[Position]:
    """Closes an existing Position record. Calculates PnL if not overridden. Returns updated position or None."""
    try:
        position_to_close = db_session.query(Position).filter(Position.id == position_db_id).first()
        if not position_to_close:
            logger.warning(f"Position DB ID {position_db_id} not found for closing.")
            return None
        
        if not position_to_close.is_open:
            logger.info(f"Position DB ID {position_db_id} is already closed.")
            return position_to_close

        position_to_close.is_open = False
        position_to_close.closed_at = datetime.datetime.utcnow()
        position_to_close.status_message = reason
        position_to_close.updated_at = datetime.datetime.utcnow()
        position_to_close.current_price = close_price # Last known price at close

        if pnl_override is not None:
            position_to_close.pnl = pnl_override
        elif position_to_close.entry_price is not None:
            if position_to_close.side == 'long':
                calculated_pnl = (close_price - position_to_close.entry_price) * filled_amount_at_close
            else: # short
                calculated_pnl = (position_to_close.entry_price - close_price) * filled_amount_at_close
            position_to_close.pnl = calculated_pnl
        
        db_session.commit()
        db_session.refresh(position_to_close)
        logger.info(f"Strategy position DB ID {position_db_id} closed. Reason: {reason}. PnL: {position_to_close.pnl}")
        return position_to_close
    except Exception as e:
        logger.error(f"Error closing strategy position DB ID {position_db_id}: {e}", exc_info=True)
        db_session.rollback()
        return None

# Placeholder for strategy-specific state on Position model (if needed in future)
# def update_position_strategy_state(db_session: Session, position_db_id: int, state_data: dict) -> Optional[Position]:
#     pass
# def load_position_strategy_state(db_session: Session, position_db_id: int) -> Optional[dict]:
#     pass
