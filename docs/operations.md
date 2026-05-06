# Operations

## Environment variables

Required in most deployments:

- `DATABASE_URL`
- `APP_ENV`
- `LOG_LEVEL`
- `PORT`
- `ENABLE_REQUEST_LOGGING`
- `REQUIRE_SERVICE_AUTH`
- `PROMPT_TRANSFORMER_API_KEY`
- `ALLOWED_CLIENT_IDS`

Startup/deployment helpers:

- `HOST`

Optional structure evaluator settings:

- `STRUCTURE_EVALUATOR_ENABLED`
- `STRUCTURE_EVALUATOR_API_KEY`
- `OPENAI_API_KEY`
- `STRUCTURE_EVALUATOR_BASE_URL`
- `STRUCTURE_EVALUATOR_MODEL`
- `STRUCTURE_EVALUATOR_TIMEOUT_SECONDS`

## Local development

1. Set `.env`
2. Run `alembic upgrade head`
3. Run `python -m app.db.seed`
4. Start with `uvicorn app.main:app --reload`

The local Alembic flow is for local app-managed databases. Shared Herman DB environments should be migrated by the canonical `herman-db` migration stream.

## Railway startup behavior

`python3 -m app.run_server` does this:

1. read environment config
2. launch Uvicorn

Schema migration and seeding are owned by `projects/herman-db` only, not by this service.

## First Railway deploy checklist

1. Set `DATABASE_URL` on the app service
2. Set `REQUIRE_SERVICE_AUTH=true`
3. Set `PROMPT_TRANSFORMER_API_KEY=<shared service credential>`
4. Set `ALLOWED_CLIENT_IDS=hermanprompt`
5. Deploy
6. Verify `GET /api/health`
7. Verify authenticated `POST /api/transform_prompt`

## Smoke tests

### Health check

```bash
curl https://<service-domain>/api/health
```

### Transform test

```bash
curl -X POST "https://<service-domain>/api/transform_prompt" \
  -H "Content-Type: application/json" \
  -H "X-Client-Id: hermanprompt" \
  -H "Authorization: Bearer <PROMPT_TRANSFORMER_API_KEY>" \
  -d '{
    "session_id": "sess_123",
    "conversation_id": "conv_123",
    "user_id": "user_1",
    "raw_prompt": "Explain this concept simply",
    "target_llm": {
      "provider": "openai",
      "model": "gpt-4.1"
    }
  }'
```

Expected behavior for `user_1`:

- `persona_source` should be `db_profile`
- `result_type` should be present in the JSON response
- `conversation_id` should echo back `conv_123`

## Prompt enforcement deploy checklist

1. Confirm Railway is deploying the repo on `main`
2. If using LLM-based prompt evaluation, set `STRUCTURE_EVALUATOR_ENABLED=true`
3. Set either `STRUCTURE_EVALUATOR_API_KEY` or `OPENAI_API_KEY`
4. Optionally set `STRUCTURE_EVALUATOR_MODEL=gpt-4.1-mini`
5. After deploy, verify `POST /api/transform_prompt` returns `coaching` for a minimal `full`-enforcement prompt such as `tell me a joke`

## Troubleshooting

### `ModuleNotFoundError: No module named 'uvicorn'`

Cause:

- Railway did not install dependencies

Fix:

- ensure `requirements.txt` exists in repo root

### App connects to `localhost:5432`

Cause:

- app service does not have the correct `DATABASE_URL`

Fix:

- set `DATABASE_URL` directly on the app service
- use SQLAlchemy format: `postgresql+psycopg://...`

### `persona_source` returns `generic_default` for seeded users

Cause:

- seed data was not loaded
 
Fix:

- run seed through `projects/herman-db` migration/seed path
- verify the shared DB seed run completed

### App crashes during migration/boot

Check:

- `DATABASE_URL`
- Postgres service availability
- migration logs in Railway runtime output

### `401` or `403` from `POST /api/transform_prompt`

Check:

- `REQUIRE_SERVICE_AUTH=true`
- `PROMPT_TRANSFORMER_API_KEY` matches HermanPrompt backend `PROMPT_TRANSFORMER_API_KEY`
- `ALLOWED_CLIENT_IDS` includes `hermanprompt`

## Data notes

Current seeded users:

- `user_1`
- `user_2`
- `user_3`
- `user_4`

`user_missing` is not stored and should continue to exercise the fallback path.
