from __future__ import annotations

import re

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine


_TABLE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class SchemaContractError(RuntimeError):
    pass


def validate_schema_contract(*, engine: Engine, version_table: str, allowed_revisions: set[str]) -> str:
    if not allowed_revisions:
        raise SchemaContractError("No allowed herman-db revisions are configured.")
    if not _TABLE_NAME_RE.match(version_table):
        raise SchemaContractError(f"Invalid schema version table name: {version_table}")

    with engine.connect() as connection:
        inspector = inspect(connection)
        if not inspector.has_table(version_table):
            raise SchemaContractError(f"Required herman-db version table '{version_table}' is missing.")
        revision = connection.execute(text(f"SELECT version_num FROM {version_table} LIMIT 1")).scalar_one_or_none()

    if revision is None:
        raise SchemaContractError(f"Version table '{version_table}' is empty.")
    if revision not in allowed_revisions:
        supported = ", ".join(sorted(allowed_revisions))
        raise SchemaContractError(
            f"Incompatible herman-db revision '{revision}'. Supported revisions: {supported}."
        )
    return str(revision)
