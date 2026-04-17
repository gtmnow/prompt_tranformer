"""add prompt enforcement and request log fields"""

from alembic import op
import sqlalchemy as sa


revision = "20260417_0002"
down_revision = "20260415_0001"
branch_labels = None
depends_on = None


PROFILE_TABLES = (
    "final_profile",
    "type_detail",
    "brain_chemistry",
    "environment_details",
    "behaviorial_adj",
)


def upgrade() -> None:
    for table_name in PROFILE_TABLES:
        op.add_column(
            table_name,
            sa.Column("prompt_enforcement_level", sa.String(length=20), nullable=False, server_default="none"),
        )
        op.add_column(
            table_name,
            sa.Column("compliance_check_enabled", sa.Boolean(), nullable=False, server_default=sa.false()),
        )
        op.add_column(
            table_name,
            sa.Column("pii_check_enabled", sa.Boolean(), nullable=False, server_default=sa.false()),
        )

    with op.batch_alter_table("prompt_transform_requests") as batch_op:
        batch_op.add_column(
            sa.Column("conversation_id", sa.String(length=255), nullable=False, server_default="")
        )
        batch_op.add_column(
            sa.Column("result_type", sa.String(length=50), nullable=False, server_default="transformed")
        )
        batch_op.add_column(sa.Column("coaching_tip", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("blocking_message", sa.Text(), nullable=True))
        batch_op.add_column(
            sa.Column("enforcement_level", sa.String(length=20), nullable=False, server_default="none")
        )
        batch_op.add_column(
            sa.Column("compliance_check_enabled", sa.Boolean(), nullable=False, server_default=sa.false())
        )
        batch_op.add_column(
            sa.Column("pii_check_enabled", sa.Boolean(), nullable=False, server_default=sa.false())
        )
        batch_op.add_column(
            sa.Column("conversation_json", sa.JSON(), nullable=False, server_default=sa.text("'{}'"))
        )
        batch_op.add_column(
            sa.Column("findings_json", sa.JSON(), nullable=False, server_default=sa.text("'[]'"))
        )
        batch_op.alter_column("transformed_prompt", existing_type=sa.Text(), nullable=True)
    op.create_index("ix_prompt_transform_requests_conversation_id", "prompt_transform_requests", ["conversation_id"])


def downgrade() -> None:
    op.drop_index("ix_prompt_transform_requests_conversation_id", table_name="prompt_transform_requests")
    with op.batch_alter_table("prompt_transform_requests") as batch_op:
        batch_op.alter_column("transformed_prompt", existing_type=sa.Text(), nullable=False)
        batch_op.drop_column("findings_json")
        batch_op.drop_column("conversation_json")
        batch_op.drop_column("pii_check_enabled")
        batch_op.drop_column("compliance_check_enabled")
        batch_op.drop_column("enforcement_level")
        batch_op.drop_column("blocking_message")
        batch_op.drop_column("coaching_tip")
        batch_op.drop_column("result_type")
        batch_op.drop_column("conversation_id")

    for table_name in PROFILE_TABLES:
        op.drop_column(table_name, "pii_check_enabled")
        op.drop_column(table_name, "compliance_check_enabled")
        op.drop_column(table_name, "prompt_enforcement_level")
