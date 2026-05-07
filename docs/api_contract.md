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
  "user_id_hash": "user_1",
  "raw_prompt": "Explain this concept simply",
  "target_llm": {
    "provider": "openai",
    "model": "gpt-4.1"
  }
}
```

Optional request fields:

- `conversation`
  - prior transformer-owned conversation state for the same `conversation_id`
- `summary_type`
  - optional persona override in the range `1..9`
- `enforcement_level`
  - optional enforcement override: `none`, `low`, `moderate`, or `full`
- `request_live_web_search`
  - optional explicit override for live web retrieval intent (`true` or `false`)
  - when omitted, heuristics fall back to transformed/raw prompt signals

### `POST /api/chat/execute`

Required headers when service auth is enabled:

- `Authorization: Bearer <PROMPT_TRANSFORMER_API_KEY>`
- `X-Client-Id: <approved-client-id>`

Request body:

```json
{
  "session_id": "sess_123",
  "conversation_id": "conv_123",
  "user_id_hash": "user_1",
  "raw_prompt": "Explain this concept simply",
  "target_llm": {
    "provider": "openai",
    "model": "gpt-4.1"
  },
  "conversation_history": [],
  "attachments": [],
  "transform_enabled": true,
  "request_live_web_search": true
}
```

Response body:

```json
{
  "session_id": "sess_123",
  "conversation_id": "conv_123",
  "user_id_hash": "user_1",
  "result_type": "transformed",
  "assistant_text": "..."
}
```

## Field meanings

- `session_id`
  - opaque caller-provided request/session identifier
- `user_id_hash`
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
  - optional prior transformer conversation state for the same thread
- `summary_type`
  - optional override in the range `1..9`
- `enforcement_level`
  - optional override for prompt-structure enforcement

## Conversation requirement contract

Every field under `conversation.requirements` uses this shape:

```json
{
  "value": "You are an experienced recruiter...",
  "status": "present",
  "heuristic_score": 25,
  "llm_score": 22,
  "max_score": 25,
  "reason": "Role is specific and audience-appropriate.",
  "improvement_hint": null
}
```

Fields always returned by Prompt Transformer:

- `who`
- `task`
- `context`
- `output`

Field meanings:

- `value`
  - the best transformer-owned value for that section
- `status`
  - fused section status used by downstream clients
- `heuristic_score`
  - deterministic section score from transformer heuristics
- `llm_score`
  - semantic section score from the internal evaluator when available, otherwise `null`
- `max_score`
  - section maximum, currently `25`
- `reason`
  - transformer-owned explanation of the current section quality
- `improvement_hint`
  - short transformer-owned suggestion for improving that section, or `null`

## Successful response

```json
{
  "session_id": "sess_123",
  "conversation_id": "conv_123",
  "user_id_hash": "user_1",
  "result_type": "transformed",
  "transformed_prompt": "Explain the topic according to the guidance below.\n...",
  "task_type": "explanation",
  "conversation": {
    "conversation_id": "conv_123",
    "requirements": {
      "who": {
        "value": "You are an experienced recruiter...",
        "status": "present",
        "heuristic_score": 25,
        "llm_score": 22,
        "max_score": 25,
        "reason": "Role is specific and audience-appropriate.",
        "improvement_hint": null
      },
      "task": {
        "value": "Create an action plan to reduce unqualified applicants.",
        "status": "derived",
        "heuristic_score": 14,
        "llm_score": 12,
        "max_score": 25,
        "reason": "Task is present but too broad.",
        "improvement_hint": "State the exact outcome and decision criteria."
      },
      "context": {
        "value": "This is for an executive hiring review...",
        "status": "present",
        "heuristic_score": 25,
        "llm_score": 24,
        "max_score": 25,
        "reason": "Context is clear.",
        "improvement_hint": null
      },
      "output": {
        "value": "Respond with a summary and bullet points.",
        "status": "present",
        "heuristic_score": 20,
        "llm_score": 18,
        "max_score": 25,
        "reason": "Output is defined but not precise enough.",
        "improvement_hint": "Add exact structure and length."
      }
    },
    "enforcement": {
      "level": "moderate",
      "status": "passes",
      "missing_fields": [],
      "last_evaluated_at": "2026-04-25T12:00:00+00:00"
    }
  },
  "scoring": {
    "scoring_version": "v4",
    "initial_score": 62,
    "final_score": 81,
    "initial_llm_score": 58,
    "final_llm_score": 76,
    "structural_score": 81
  },
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
- `result_type: "transformed"` may still include a non-blocking `coaching_tip` under `low` enforcement
- `result_type: "coaching"` when required prompt structure is missing for the active enforcement level
- `result_type: "blocked"` when compliance or PII findings are severe enough to stop execution

### `GET /api/conversation_scores/{conversation_id}`

Required query params:

- `user_id_hash`

Required headers when service auth is enabled:

- `Authorization: Bearer <PROMPT_TRANSFORMER_API_KEY>`
- `X-Client-Id: <approved-client-id>`

Response body:

```json
{
  "conversation_id": "conv_123",
  "user_id_hash": "user_1",
  "scoring_version": "v4",
  "initial_score": 62,
  "best_score": 81,
  "final_score": 81,
  "initial_llm_score": 58,
  "best_llm_score": 76,
  "final_llm_score": 76,
  "structural_score": 81,
  "improvement_score": 19,
  "best_improvement_score": 19,
  "last_scored_at": "2026-04-25T12:00:00+00:00",
  "conversation": {
    "conversation_id": "conv_123",
    "requirements": {
      "who": {
        "value": "You are an experienced recruiter...",
        "status": "present",
        "heuristic_score": 25,
        "llm_score": 22,
        "max_score": 25,
        "reason": "Role is specific and audience-appropriate.",
        "improvement_hint": null
      },
      "task": {
        "value": "Create an action plan to reduce unqualified applicants.",
        "status": "derived",
        "heuristic_score": 14,
        "llm_score": 12,
        "max_score": 25,
        "reason": "Task is present but too broad.",
        "improvement_hint": "State the exact outcome and decision criteria."
      },
      "context": {
        "value": "This is for an executive hiring review...",
        "status": "present",
        "heuristic_score": 25,
        "llm_score": 24,
        "max_score": 25,
        "reason": "Context is clear.",
        "improvement_hint": null
      },
      "output": {
        "value": "Respond with a summary and bullet points.",
        "status": "present",
        "heuristic_score": 20,
        "llm_score": 18,
        "max_score": 25,
        "reason": "Output is defined but not precise enough.",
        "improvement_hint": "Add exact structure and length."
      }
    },
    "enforcement": {
      "level": "moderate",
      "status": "passes",
      "missing_fields": [],
      "last_evaluated_at": "2026-04-25T12:00:00+00:00"
    }
  }
}
```

This endpoint is the preferred read path for UI scoring displays.

Downstream clients should treat this endpoint as the reload path for previously persisted field-level prompt scoring data.

Conversation requirement statuses use:

- `present`
- `derived`
- `missing`

Meaning:

- `present`
  - the prompt clearly contains the element in natural language
- `derived`
  - the transformer could reasonably infer the element even though it was not clearly stated
- `missing`
  - the prompt did not provide enough information

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

- `user_id_hash` is treated as a hashed external identifier by convention.
- The service is deterministic and side-effect free unless request logging is enabled.
- Callers should branch on `result_type` and only forward `transformed_prompt` to the target LLM when `result_type == "transformed"`.
- Callers should not depend on the exact wording of `transformed_prompt`, `coaching_tip`, or `blocking_message`; they should depend on the contract fields and general deterministic behavior.
- Prompt Transformer is the source of truth for both conversation-level scores and field-level requirement scoring.
- Downstream clients should not recompute `who`, `task`, `context`, or `output` scores locally.
- Downstream clients should render and route from `conversation.requirements.<field>` and the top-level `scoring` object returned by Prompt Transformer.
