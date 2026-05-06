from __future__ import annotations

import logging
from pathlib import Path

from alembic import command
from alembic.config import Config

from app.core.config import get_settings
from app.db.seed import run_seed
from app.db.session import engine
from app.schema_contract import validate_schema_contract


logger = logging.getLogger(__name__)


def _alembic_config() -> Config:
    root_dir = Path(__file__).resolve().parent.parent.parent
    config = Config(str(root_dir / "alembic.ini"))
    config.set_main_option("script_location", str(root_dir / "alembic"))
    config.set_main_option("sqlalchemy.url", get_settings().database_url)
    return config


def bootstrap_database() -> None:
    settings = get_settings()
    if settings.effective_herman_db_canonical_mode:
        try:
            validate_schema_contract(
                engine=engine,
                version_table=settings.herman_db_version_table,
                allowed_revisions=settings.herman_db_allowed_revisions,
            )
        except Exception as exc:
            message = str(exc).lower()
            if (
                settings.railway_auto_migrate
                and (("version table" in message and "missing" in message) or "empty" in message)
            ):
                logger.warning(
                    "Canonical schema contract check could not run, continuing with local migrations. Reason=%s",
                    exc,
                )
                command.upgrade(_alembic_config(), "head")
            else:
                raise
    elif settings.railway_auto_migrate:
        command.upgrade(_alembic_config(), "head")
    if settings.railway_seed_on_start:
        run_seed()


if __name__ == "__main__":
    bootstrap_database()
