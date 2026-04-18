"""add llm score columns to conversation prompt scores"""

from alembic import op
import sqlalchemy as sa


revision = "20260418_0004"
down_revision = "20260418_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("conversation_prompt_scores", sa.Column("initial_llm_score", sa.Integer(), nullable=True))
    op.add_column("conversation_prompt_scores", sa.Column("best_llm_score", sa.Integer(), nullable=True))
    op.add_column("conversation_prompt_scores", sa.Column("final_llm_score", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("conversation_prompt_scores", "final_llm_score")
    op.drop_column("conversation_prompt_scores", "best_llm_score")
    op.drop_column("conversation_prompt_scores", "initial_llm_score")
