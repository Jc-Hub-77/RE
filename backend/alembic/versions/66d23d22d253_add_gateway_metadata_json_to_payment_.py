"""add_gateway_metadata_json_to_payment_transaction

Revision ID: 66d23d22d253
Revises: dc7a444085b0
Create Date: 2025-06-02 21:27:38.081742 # Original Create Date

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '66d23d22d253'
down_revision: Union[str, None] = 'dc7a444085b0' # Corrected
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('payment_transactions', sa.Column('gateway_metadata_json', sa.Text(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('payment_transactions', 'gateway_metadata_json')
