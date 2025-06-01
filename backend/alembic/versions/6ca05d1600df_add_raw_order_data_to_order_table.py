"""add_raw_order_data_to_order_table

Revision ID: 6ca05d1600df
Revises: 9feeb80d8c05
Create Date: 2025-05-27 00:37:27.838641

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '6ca05d1600df'
down_revision: Union[str, None] = '9feeb80d8c05'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('orders', sa.Column('raw_order_data', sa.Text(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('orders', 'raw_order_data')
