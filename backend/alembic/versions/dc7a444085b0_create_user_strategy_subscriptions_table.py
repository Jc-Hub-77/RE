"""create user strategy subscriptions table

Revision ID: dc7a444085b0
Revises: 6ca05d1600df
Create Date: 2025-06-01 15:00:00.000000 # Adjusted Create Date

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import datetime


# revision identifiers, used by Alembic.
revision: str = 'dc7a444085b0'
down_revision: Union[str, None] = '6ca05d1600df' # Corrected
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('user_strategy_subscriptions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('strategy_id', sa.Integer(), nullable=False),
        sa.Column('api_key_id', sa.Integer(), nullable=False),
        sa.Column('custom_parameters', sa.Text(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True, default=False),
        sa.Column('subscribed_at', sa.DateTime(), nullable=True, default=datetime.datetime.utcnow),
        sa.Column('expires_at', sa.DateTime(), nullable=True),
        sa.Column('backtest_results_id', sa.Integer(), nullable=True),
        sa.Column('status_message', sa.String(), nullable=True),
        sa.Column('celery_task_id', sa.String(), nullable=True),
        sa.ForeignKeyConstraint(['api_key_id'], ['api_keys.id'], ),
        sa.ForeignKeyConstraint(['backtest_results_id'], ['backtest_results.id'], ),
        sa.ForeignKeyConstraint(['strategy_id'], ['strategies.id'], ),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_user_strategy_subscriptions_celery_task_id'), 'user_strategy_subscriptions', ['celery_task_id'], unique=False)
    op.create_index(op.f('ix_user_strategy_subscriptions_id'), 'user_strategy_subscriptions', ['id'], unique=False)

    # Add foreign key constraints from other tables to user_strategy_subscriptions
    op.create_foreign_key(
        "fk_orders_subscription_id", "orders",
        "user_strategy_subscriptions", ["subscription_id"], ["id"]
    )
    op.create_foreign_key(
        "fk_positions_subscription_id", "positions",
        "user_strategy_subscriptions", ["subscription_id"], ["id"]
    )
    op.create_foreign_key(
        "fk_payment_transactions_subscription_id", "payment_transactions",
        "user_strategy_subscriptions", ["user_strategy_subscription_id"], ["id"]
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint("fk_payment_transactions_subscription_id", "payment_transactions", type_="foreignkey")
    op.drop_constraint("fk_positions_subscription_id", "positions", type_="foreignkey")
    op.drop_constraint("fk_orders_subscription_id", "orders", type_="foreignkey")
    
    op.drop_index(op.f('ix_user_strategy_subscriptions_id'), table_name='user_strategy_subscriptions')
    op.drop_index(op.f('ix_user_strategy_subscriptions_celery_task_id'), table_name='user_strategy_subscriptions')
    op.drop_table('user_strategy_subscriptions')
