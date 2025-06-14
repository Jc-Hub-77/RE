"""add order and position tables

Revision ID: 9feeb80d8c05
Revises: None
Create Date: 2025-05-21 11:32:59.140217

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import datetime # Added for default datetimes


# revision identifiers, used by Alembic.
revision: str = '9feeb80d8c05'
revises: Union[str, None] = None
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('orders',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('subscription_id', sa.Integer(), nullable=False),
        sa.Column('order_id', sa.String(), nullable=True),
        sa.Column('symbol', sa.String(), nullable=False),
        sa.Column('order_type', sa.String(), nullable=False),
        sa.Column('side', sa.String(), nullable=False),
        sa.Column('amount', sa.Float(), nullable=False),
        sa.Column('price', sa.Float(), nullable=True),
        sa.Column('cost', sa.Float(), nullable=True),
        sa.Column('filled', sa.Float(), nullable=True),
        sa.Column('remaining', sa.Float(), nullable=True),
        sa.Column('status', sa.String(), nullable=True, default='open'),
        sa.Column('created_at', sa.DateTime(), nullable=True, default=datetime.datetime.utcnow),
        sa.Column('updated_at', sa.DateTime(), nullable=True, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow),
        sa.Column('closed_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.Index(op.f('ix_orders_id'), 'id', unique=False),
        sa.Index(op.f('ix_orders_order_id'), 'order_id', unique=False), # Assuming exchange order_id might not be unique across all records if non-null
        sa.Index(op.f('ix_orders_status'), 'status', unique=False)
    )

    op.create_table('positions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('subscription_id', sa.Integer(), nullable=False),
        sa.Column('symbol', sa.String(), nullable=False),
        sa.Column('exchange_name', sa.String(), nullable=False),
        sa.Column('side', sa.String(), nullable=False),
        sa.Column('amount', sa.Float(), nullable=False),
        sa.Column('entry_price', sa.Float(), nullable=True),
        sa.Column('current_price', sa.Float(), nullable=True),
        sa.Column('is_open', sa.Boolean(), nullable=True, default=True),
        sa.Column('created_at', sa.DateTime(), nullable=True, default=datetime.datetime.utcnow),
        sa.Column('updated_at', sa.DateTime(), nullable=True, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow),
        sa.Column('closed_at', sa.DateTime(), nullable=True),
        sa.Column('pnl', sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.Index(op.f('ix_positions_id'), 'id', unique=False),
        sa.Index(op.f('ix_positions_is_open'), 'is_open', unique=False)
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_positions_is_open'), table_name='positions')
    op.drop_index(op.f('ix_positions_id'), table_name='positions')
    op.drop_table('positions')

    op.drop_index(op.f('ix_orders_status'), table_name='orders')
    op.drop_index(op.f('ix_orders_order_id'), table_name='orders')
    op.drop_index(op.f('ix_orders_id'), table_name='orders')
    op.drop_table('orders')
