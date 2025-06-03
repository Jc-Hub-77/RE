"""create user strategy subscriptions table

Revision ID: dc7a444085b0
Revises: 6ca05d1600df
Create Date: 2025-05-30 03:01:35.418944

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import datetime # Added for default datetimes


# revision identifiers, used by Alembic.
revision: str = 'dc7a444085b0'
revises: Union[str, None] = '6ca05d1600df'
down_revision: Union[str, None] = '6ca05d1600df'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('user_strategy_subscriptions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('strategy_id', sa.Integer(), nullable=False),
        sa.Column('status', sa.String(), nullable=False, default='active'),
        sa.Column('created_at', sa.DateTime(), nullable=True, default=datetime.datetime.utcnow),
        sa.Column('updated_at', sa.DateTime(), nullable=True, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow),
        sa.PrimaryKeyConstraint('id')
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('user_strategy_subscriptions')
