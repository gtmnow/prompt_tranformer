"""add conversation prompt scoring table"""

from alembic import op
import sqlalchemy as sa


revision = "20260418_0003"
down_revision = "20260417_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "conversation_prompt_scores",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("conversation_id", sa.String(length=255), nullable=False),
        sa.Column("user_id_hash", sa.String(length=255), nullable=False),
        sa.Column("task_type", sa.String(length=100), nullable=False, server_default="unknown"),
        sa.Column("conversation_started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("conversation_ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_scored_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("conversation_deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("enforcement_level", sa.String(length=20), nullable=False, server_default="none"),
        sa.Column("initial_score", sa.Integer(), nullable=False),
        sa.Column("best_score", sa.Integer(), nullable=False),
        sa.Column("final_score", sa.Integer(), nullable=False),
        sa.Column("improvement_score", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("best_improvement_score", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("passed_without_coaching", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("reached_policy_complete", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("coaching_turn_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("blocked_turn_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("transformed_turn_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("who_status", sa.String(length=20), nullable=False, server_default="missing"),
        sa.Column("task_status", sa.String(length=20), nullable=False, server_default="missing"),
        sa.Column("context_status", sa.String(length=20), nullable=False, server_default="missing"),
        sa.Column("output_status", sa.String(length=20), nullable=False, server_default="missing"),
        sa.Column("score_details_json", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("scoring_version", sa.String(length=50), nullable=False, server_default="v1"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("conversation_id"),
    )
    op.create_index(
        "ix_conversation_prompt_scores_conversation_id",
        "conversation_prompt_scores",
        ["conversation_id"],
    )
    op.create_index(
        "ix_conversation_prompt_scores_user_id_hash",
        "conversation_prompt_scores",
        ["user_id_hash"],
    )
    op.create_index(
        "ix_conversation_prompt_scores_last_scored_at",
        "conversation_prompt_scores",
        ["last_scored_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_conversation_prompt_scores_last_scored_at", table_name="conversation_prompt_scores")
    op.drop_index("ix_conversation_prompt_scores_user_id_hash", table_name="conversation_prompt_scores")
    op.drop_index("ix_conversation_prompt_scores_conversation_id", table_name="conversation_prompt_scores")
    op.drop_table("conversation_prompt_scores")
