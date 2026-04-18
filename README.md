# Prompt Transformer

Prompt Transformer is a FastAPI service that rewrites user prompts using:

- a precomputed `final_profile` stored in PostgreSQL
- deterministic task inference
- local model policies
- optional summary persona overrides
- optional LLM-assisted structure evaluation for prompt enforcement

The transformed prompt itself is built deterministically. Prompt structure evaluation may optionally use a small LLM evaluator to classify `who`, `task`, `context`, and `output` as `present`, `derived`, or `missing`.

## MVP scope

This repository implements the MVP runtime transformer only.

Included:

- FastAPI API layer
- PostgreSQL-backed profile lookup
- Alembic migrations
- seed data for sample users
- deterministic YAML rule loading
- Railway-ready deployment path

Not included:

- profile generation or learning
- conversation management
- prompt execution
- merging profile layers at runtime

## Repo map

```text
app/
  api/         HTTP routes
  core/        config and rule loading
  db/          session, bootstrap, seeding
  models/      SQLAlchemy ORM models
  rules/       YAML rule files, including scoring calibration
  schemas/     request/response schemas
  services/    runtime transformer services
  main.py      FastAPI app factory
  run_server.py  Railway/local startup entrypoint
alembic/       migrations
docs/          handoff and operator docs
tests/         API tests
```

## Runtime flow

1. Receive `POST /api/transform_prompt`
2. Validate payload
3. Resolve persona from `summary_type`, `final_profile`, or generic default
4. Infer task type from deterministic rules
5. Resolve model policy from local YAML
6. Evaluate conversation enforcement plus optional compliance and PII checks
7. Build the transformed prompt only when the request is allowed to proceed
8. Optionally log the request and decision result
9. Return either a transformed prompt, coaching guidance, or a blocked result

See [docs/architecture.md](./docs/architecture.md) for the detailed flow and ownership boundaries.

## Quick start

1. Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
```

2. Set environment variables:

```bash
cp .env.example .env
```

3. Run migrations:

```bash
alembic upgrade head
```

4. Seed sample data:

```bash
python -m app.db.seed
```

5. Start the API:

```bash
uvicorn app.main:app --reload
```

Alternative startup path that mirrors Railway:

```bash
python3 -m app.run_server
```

## API

Primary endpoint:

- `POST /api/transform_prompt`

Score read endpoint:

- `GET /api/conversation_scores/{conversation_id}?user_id=<user_id>`

Health endpoint:

- `GET /api/health`

Authentication headers for production/shared-service use:

- `Authorization: Bearer <PROMPT_TRANSFORMER_API_KEY>`
- `X-Client-Id: <approved-client-id>`

Example request:

```json
{
  "session_id": "sess_123",
  "conversation_id": "conv_123",
  "user_id": "user_1",
  "raw_prompt": "Explain this concept simply",
  "target_llm": {
    "provider": "openai",
    "model": "gpt-4.1"
  }
}
```

Example successful response:

```json
{
  "session_id": "sess_123",
  "conversation_id": "conv_123",
  "user_id": "user_1",
  "result_type": "transformed",
  "transformed_prompt": "Explain the topic according to the guidance below.\nStart with the direct answer before supporting detail.\n...",
  "task_type": "explanation",
  "conversation": null,
  "findings": [],
  "metadata": {
    "persona_source": "db_profile",
    "rules_applied": [
      "task:explanation:keyword"
    ],
    "profile_version": "v1",
    "requested_model": "gpt-4.1",
    "resolved_model": "gpt-4.1",
    "used_fallback_model": false
  }
}
```

See [docs/api_contract.md](./docs/api_contract.md) for request/response rules and expected behaviors.

Planned feature work for conversation-level prompt enforcement, compliance checks, and PII checks is documented in [docs/prompt_enforcement_implementation_spec.md](./docs/prompt_enforcement_implementation_spec.md).

The current runtime now supports conversation-level enforcement outcomes through `result_type` as part of the API contract.

Prompt score data is persisted in the transformer database and should be read through the score endpoint instead of being reconstructed in the UI.

The scoring model itself is loaded from [app/rules/prompt_scoring.yaml](./app/rules/prompt_scoring.yaml) so calibration changes live with the transformer build rather than in the database.

The top-level `version` in that YAML is the scoring model version of record and is persisted with each conversation score row as `scoring_version`.

The long-term scoring direction is hybrid:

- heuristics provide the baseline score and fallback behavior
- an internal LLM evaluator provides semantic scoring judgments for the four dimensions
- transformer-owned fusion logic produces the final score shown in the UI

This hybrid methodology is documented in [docs/prompt_scoring_implementation_spec.md](./docs/prompt_scoring_implementation_spec.md).

The enforcement ladder is:

- `none`
  - no prompt structure coaching
- `low`
  - process the prompt and optionally return light coaching
- `moderate`
  - require `who`, `task`, `context`, and `output`
- `full`
  - require all four elements plus labeled `Who:`, `Task:`, `Context:`, and `Output:` sections

Prompt scoring design and implementation planning is documented in [docs/prompt_scoring_implementation_spec.md](./docs/prompt_scoring_implementation_spec.md).

## Seeded users

The MVP ships with nine sample users in the database. Each one mirrors the default persona values for the matching summary personality type:

- `user_1` -> summary type `1`
- `user_2` -> summary type `2`
- `user_3` -> summary type `3`
- `user_4` -> summary type `4`
- `user_5` -> summary type `5`
- `user_6` -> summary type `6`
- `user_7` -> summary type `7`
- `user_8` -> summary type `8`
- `user_9` -> summary type `9`

`user_missing` is intentionally absent and should exercise generic fallback behavior.

These seeded IDs stand in for future hashed user IDs. The service assumes `user_id == user_id_hash`.

## Railway

Create a Railway project with:

1. A PostgreSQL service
2. An app service pointing at this repo

Set these environment variables on the app service:

```bash
DATABASE_URL=<railway postgres url in SQLAlchemy form>
APP_ENV=production
LOG_LEVEL=INFO
PORT=8000
REQUIRE_SERVICE_AUTH=true
PROMPT_TRANSFORMER_API_KEY=<shared service credential>
ALLOWED_CLIENT_IDS=hermanprompt
ENABLE_REQUEST_LOGGING=false
RAILWAY_AUTO_MIGRATE=true
RAILWAY_SEED_ON_START=true
HOST=0.0.0.0
```

Notes:

- `DATABASE_URL` must be set on the app service, not only on the Postgres service.
- `DATABASE_URL` should use `postgresql+psycopg://...`, not raw `postgresql://...`.
- When `REQUIRE_SERVICE_AUTH=true`, callers must send both `Authorization: Bearer <PROMPT_TRANSFORMER_API_KEY>` and an allowed `X-Client-Id`.
- `RAILWAY_AUTO_MIGRATE=true` runs `alembic upgrade head` during startup.
- `RAILWAY_SEED_ON_START=true` is useful for the first MVP deploy. After the sample profiles are loaded, switch it to `false`.
- `railway.json` starts the service through `python3 -m app.run_server`, which bootstraps the database and then launches Uvicorn.

See [docs/operations.md](./docs/operations.md) for deployment and troubleshooting steps.

## Documentation index

- [docs/architecture.md](./docs/architecture.md)
- [docs/api_contract.md](./docs/api_contract.md)
- [docs/operations.md](./docs/operations.md)
- [docs/prompt_enforcement_implementation_spec.md](./docs/prompt_enforcement_implementation_spec.md)
- [docs/prompt_scoring_implementation_spec.md](./docs/prompt_scoring_implementation_spec.md)
