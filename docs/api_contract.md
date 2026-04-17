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

Required headers when service auth is enabled:

- `Authorization: Bearer <PROMPT_TRANSFORMER_API_KEY>`
- `X-Client-Id: <approved-client-id>`

Request body:

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

## Field meanings

- `session_id`
  - opaque caller-provided request/session identifier
- `user_id`
  - non-PII identifier used directly as profile lookup key
- `conversation_id`
  - caller-provided conversation/thread identifier
- `raw_prompt`
  - original user prompt to transform
- `target_llm.provider`
  - model provider key used for local policy lookup
- `target_llm.model`
  - requested model name used for local policy lookup
- `conversation`
  - optional shared conversation object for conversation-level enforcement work
- `summary_type`
  - optional override in the range `1..9`

## Successful response

```json
{
  "session_id": "sess_123",
  "conversation_id": "conv_123",
  "user_id": "user_1",
  "result_type": "transformed",
  "transformed_prompt": "Explain the topic according to the guidance below.\n...",
  "task_type": "explanation",
  "conversation": null,
  "findings": [],
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

The service can return:

- `result_type: "transformed"` when the request passes enforcement and no blocking findings exist
- `result_type: "coaching"` when required prompt structure is missing for the active enforcement level
- `result_type: "blocked"` when compliance or PII findings are severe enough to stop execution

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
- mismatched `conversation_id` and `conversation.conversation_id`: `400`
- invalid `summary_type`: `400`
- missing client identity or missing service credentials: `401`
- invalid service credentials: `403`
- database unavailable: `503`
- user not found: no error, falls back to `generic_default`
- unknown model: no error, falls back to the configured provider/default model policy

## Notes for integrators

- `user_id` is treated as a hashed external identifier by convention.
- The service is deterministic and side-effect free unless request logging is enabled.
- Callers should branch on `result_type` and only forward `transformed_prompt` to the target LLM when `result_type == "transformed"`.
- Callers should not depend on the exact wording of `transformed_prompt`, `coaching_tip`, or `blocking_message`; they should depend on the contract fields and general deterministic behavior.
