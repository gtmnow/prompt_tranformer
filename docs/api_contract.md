# API Contract

## Endpoints

### `GET /api/health`

Returns:

```json
{
  "status": "ok"
}
```

### `POST /api/transform_prompt`

Request body:

```json
{
  "session_id": "sess_123",
  "user_id": "user_1",
  "raw_prompt": "Explain this concept simply",
  "target_llm": {
    "provider": "openai",
    "model": "gpt-4.1"
  },
  "summary_type": 3
}
```

## Field meanings

- `session_id`
  - opaque caller-provided request/session identifier
- `user_id`
  - non-PII identifier used directly as profile lookup key
- `raw_prompt`
  - original user prompt to transform
- `target_llm.provider`
  - model provider key used for local policy lookup
- `target_llm.model`
  - requested model name used for local policy lookup
- `summary_type`
  - optional override in the range `1..9`

## Successful response

```json
{
  "session_id": "sess_123",
  "user_id": "user_1",
  "transformed_prompt": "Explain the topic according to the guidance below.\n...",
  "task_type": "explanation",
  "metadata": {
    "persona_source": "db_profile",
    "rules_applied": [
      "task:explanation:keyword",
      "persona:answer_first:enabled"
    ],
    "profile_version": "v1",
    "requested_model": "gpt-4.1",
    "resolved_model": "gpt-4.1",
    "used_fallback_model": false
  }
}
```

## Supported task types

- `summarization`
- `explanation`
- `writing`
- `planning`
- `analysis`
- `recommendation`
- `decision_support`
- `unknown`

## Persona source meanings

- `summary_override`
  - `summary_type` was provided and mapped to a local persona
- `db_profile`
  - profile found in `final_profile`
- `generic_default`
  - no override and no DB row found

## Error behavior

- invalid payload fields: `400`
- invalid `summary_type`: `400`
- database unavailable: `503`
- user not found: no error, falls back to `generic_default`
- unknown model: no error, falls back to the configured provider/default model policy

## Notes for integrators

- `user_id` is treated as a hashed external identifier by convention.
- The service is deterministic and side-effect free unless request logging is enabled.
- Callers should not depend on the exact wording of `transformed_prompt`; they should depend on the contract fields and general deterministic behavior.
