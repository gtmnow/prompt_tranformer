# Operations

## Environment variables

Required in most deployments:

- `DATABASE_URL`
- `APP_ENV`
- `LOG_LEVEL`
- `PORT`
- `ENABLE_REQUEST_LOGGING`

Startup/deployment helpers:

- `RAILWAY_AUTO_MIGRATE`
- `RAILWAY_SEED_ON_START`
- `HOST`

## Local development

1. Set `.env`
2. Run `alembic upgrade head`
3. Run `python -m app.db.seed`
4. Start with `uvicorn app.main:app --reload`

## Railway startup behavior

`python3 -m app.run_server` does this:

1. read environment config
2. run migrations if `RAILWAY_AUTO_MIGRATE=true`
3. run seed if `RAILWAY_SEED_ON_START=true`
4. launch Uvicorn

## First Railway deploy checklist

1. Set `DATABASE_URL` on the app service
2. Set `RAILWAY_AUTO_MIGRATE=true`
3. Set `RAILWAY_SEED_ON_START=true`
4. Deploy
5. Verify `GET /api/health`
6. Verify `POST /api/transform_prompt`
7. Set `RAILWAY_SEED_ON_START=false`

## Smoke tests

### Health check

```bash
curl https://<service-domain>/api/health
```

### Transform test

```bash
curl -X POST "https://<service-domain>/api/transform_prompt" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "sess_123",
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
- `profile_version` should be `v1`

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

- set `RAILWAY_SEED_ON_START=true`
- redeploy once
- test again
- switch it back to `false`

### App crashes during migration/boot

Check:

- `DATABASE_URL`
- Postgres service availability
- migration logs in Railway runtime output

## Data notes

Current seeded users:

- `user_1`
- `user_2`
- `user_3`
- `user_4`

`user_missing` is not stored and should continue to exercise the fallback path.
