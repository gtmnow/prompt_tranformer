"""initial schema"""

from alembic import op
import sqlalchemy as sa


revision = "20260415_0001"
down_revision = None
branch_labels = None
depends_on = None


def _create_profile_table(name: str) -> None:
    op.create_table(
        name,
        sa.Column("user_id_hash", sa.String(length=255), primary_key=True),
        sa.Column("structure", sa.Float(), nullable=False),
        sa.Column("answer_first", sa.Float(), nullable=False),
        sa.Column("tone_directness", sa.Float(), nullable=False),
        sa.Column("detail_level", sa.Float(), nullable=False),
        sa.Column("ambiguity_reduction", sa.Float(), nullable=False),
        sa.Column("exploration_level", sa.Float(), nullable=False),
        sa.Column("context_loading", sa.Float(), nullable=False),
        sa.Column("profile_version", sa.String(length=50), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )


def upgrade() -> None:
    for table_name in (
        "final_profile",
        "type_detail",
        "brain_chemistry",
        "environment_details",
        "behaviorial_adj",
    ):
        _create_profile_table(table_name)

    op.create_table(
        "prompt_transform_requests",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("session_id", sa.String(length=255), nullable=False),
        sa.Column("user_id_hash", sa.String(length=255), nullable=False),
        sa.Column("raw_prompt", sa.Text(), nullable=False),
        sa.Column("transformed_prompt", sa.Text(), nullable=False),
        sa.Column("task_type", sa.String(length=100), nullable=False),
        sa.Column("target_provider", sa.String(length=100), nullable=False),
        sa.Column("target_model", sa.String(length=100), nullable=False),
        sa.Column("persona_source", sa.String(length=100), nullable=False),
        sa.Column("used_fallback_model", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_prompt_transform_requests_session_id", "prompt_transform_requests", ["session_id"])
    op.create_index("ix_prompt_transform_requests_user_id_hash", "prompt_transform_requests", ["user_id_hash"])


def downgrade() -> None:
    op.drop_index("ix_prompt_transform_requests_user_id_hash", table_name="prompt_transform_requests")
    op.drop_index("ix_prompt_transform_requests_session_id", table_name="prompt_transform_requests")
    op.drop_table("prompt_transform_requests")
    for table_name in (
        "behaviorial_adj",
        "environment_details",
        "brain_chemistry",
        "type_detail",
        "final_profile",
    ):
        op.drop_table(table_name)
