"""Rename prompt transform request user column to user_id_hash."""

from alembic import op
import sqlalchemy as sa


revision = "20260427_0004"
down_revision = "20260418_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("prompt_transform_requests") as batch_op:
        batch_op.alter_column("user_id", new_column_name="user_id_hash", existing_type=sa.String(length=255))


def downgrade() -> None:
    with op.batch_alter_table("prompt_transform_requests") as batch_op:
        batch_op.alter_column("user_id_hash", new_column_name="user_id", existing_type=sa.String(length=255))
