# Prompt Enforcement Implementation Spec

## Purpose

This document defines the implementation plan for conversation-level prompt structure enforcement, compliance checks, and PII checks in Prompt Transformer.

The feature adds a coaching and gating layer before prompt transformation. Depending on user profile settings and the conversation's current setup state, the transformer will either:

- transform the prompt normally
- return coaching guidance for missing prompt structure
- return warnings or blocking findings for compliance or PII risks

Prompt structure evaluation may use an LLM-assisted evaluator for semantic extraction, but enforcement decisions remain transformer-owned and deterministic at the contract level.

## Goals

- enforce prompt structure requirements at the conversation level
- keep enforcement policy in the user's prompt profile
- include `conversation_id` in every transformer request and response
- let the UI store conversation setup state in thread memory
- avoid re-asking for the same setup data on every message
- prevent coached or blocked prompts from being sent to the target LLM
- add deterministic compliance and PII screening before transformation
- use enforcement as a training aid that improves user prompting discipline over time

## Non-goals

- persistent storage of per-thread `who`, `task`, `context`, and `output` in the transformer database
- full DLP or enterprise compliance coverage
- automatic policy enforcement after the prompt has already been sent to a model
- changing the prompt score directly based on enforcement level

## Core concepts

### User profile settings

These settings are persistent and should be stored in the profile tables.

- `prompt_enforcement_level`
  - `none`
  - `low`
  - `moderate`
  - `full`
- `compliance_check_enabled`
  - `true`
  - `false`
- `pii_check_enabled`
  - `true`
  - `false`

These values are runtime policy controls, not per-message content.

### Conversation setup state

These values are conversation-scoped and should be stored by the Prompt UI in memory with each conversation thread.

- `conversation_id`
- `who`
- `task`
- `context`
- `output`

Each field should also track how it was established.

- `present`
- `derived`
- `missing`

The transformer evaluates the conversation state provided by the UI together with the current raw prompt.

### Shared conversation object

The request contract should include a shared conversation object shape that both the UI and transformer can read and write.

Recommended top-level request field:

- `conversation`

Recommended responsibilities:

- the UI is the system of record for persisting this object with the conversation thread
- the transformer reads the object on each request
- the transformer returns an updated version of the object in each response

This keeps the object shared across both systems without requiring the transformer to become the storage owner for thread state.

## Ownership split

### Transformer responsibilities

- read user enforcement settings from the profile
- read `conversation_id` and conversation state from the request
- validate prompt structure using semantic extraction plus deterministic enforcement rules
- evaluate compliance and PII checks when enabled
- decide whether the request is transformable, coachable, warn-only, or blocked
- return structured results and updated conversation state for the UI to act on

### Prompt UI responsibilities

- store conversation setup state in thread memory
- persist the shared conversation object for each conversation thread
- send the current thread state to the transformer on each request
- update thread memory from transformer results
- show coaching or warning messages to the user
- never forward a `coaching` or `blocked` result to the target LLM

## Required prompt structure

The conversation setup model is:

- `who`
  - the role the LLM should fulfill
  - examples: doctor, lawyer, expert programmer, product strategist
- `task`
  - the unambiguous job to perform
  - examples: summarize, draft, explain, compare, debug, plan
- `context`
  - why the user needs the answer, how it will be used, or any important constraints
  - examples: for a board deck, for a legal intake note, for an email to prospects, for a landing page
- `output`
  - the expected deliverable or format
  - examples: plain response, bullet list, JSON, markdown file, image prompt, code patch

The transformer should validate presence of these elements against the combined conversation state, not just the latest user message.

Presence means the prompt clearly contains the element in natural language. Users should not need rigid labels to receive structural credit.

## Conversation object contract

The shared conversation object should let both systems reason about:

- the conversation identity
- the known conversation setup values
- how those values were established
- whether the current conversation satisfies the active enforcement policy

Recommended shape:

- `conversation_id`
- `requirements`
  - `who`
  - `task`
  - `context`
  - `output`
- `enforcement`
  - `level`
  - `status`
  - `missing_fields`
  - `last_evaluated_at`

The `enforcement.status` value can be:

- `not_evaluated`
- `passes`
- `needs_coaching`
- `blocked`

Recommended rule:

- the transformer computes enforcement status
- the UI stores the latest returned status with the conversation

## Enforcement levels

### `none`

- no structure gating
- raw prompt proceeds to transformation
- compliance and PII checks may still run if enabled

### `low`

- process the prompt even when one or more structure elements are missing
- return light coaching feedback when helpful
- do not block execution for ordinary missing structure
- reserve blocking for compliance or PII findings

Recommended product behavior:

- use low enforcement as a gentle reminder layer
- allow users to continue even when the prompt is incomplete
- surface a compact coaching tip such as "Coaching: include role, context, and output next time for a stronger result."

### `moderate`

- require all four ideas:
  - `who`
  - `task`
  - `context`
  - `output`
- do not require labeled formatting
- allow natural-language prompts to pass when the four elements are clearly present
- if one or more elements are still missing after semantic evaluation, return coaching and do not produce a transformed prompt

Recommended product behavior:

- use moderate enforcement after users no longer need rigid structure training
- coach users on missing elements, not on prompt formatting style

### `full`

- require all four elements:
  - `who`
  - `task`
  - `context`
  - `output`
- require deliberate labeled structure using:
  - `Who:`
  - `Task:`
  - `Context:`
  - `Output:`
- if any required element is missing, or if the prompt does not use the labeled format, return coaching and do not produce a transformed prompt for model execution

Recommended product behavior:

- use full enforcement as a training mode for deliberate prompt construction
- treat it as a discipline-building layer, not a different scoring algorithm
- allow users to graduate to less strict enforcement over time as their starting scores improve

## Semantic extraction and derivation rules

The first implementation should recognize three structural states:

- `present`
  - the prompt clearly contains the element
- `derived`
  - the element is reasonably inferable from the prompt or conversation state, but is not clearly stated
- `missing`
  - the prompt does not give enough information

Examples:

- `tell me a joke`
  - `task`: `present`
  - `output`: `derived`
  - `who`: `missing`
  - `context`: `missing`
- `You are a senior Python engineer. Explain rate limiting for a SaaS API. I am studying for a system design interview. Answer in the chat with an overview, components, flow, tradeoffs, and one example.`
  - all four fields: `present`

The service should not require users to write explicit labels in order to receive structural credit.

Labels are an enforcement rule only for `full`, not a scoring rule.

The extraction layer should favor semantic interpretation over brittle exact-match formatting.

Examples:

- detect `task` from imperative verbs, direct asks, or clearly described work
- detect `output` from phrases like:
  - "write an email"
  - "give me bullet points"
  - "return JSON"
  - "draft a memo"
  - "generate an image prompt"
- detect `context` from phrases like:
  - "for my boss"
  - "for a presentation"
  - "for a client"
  - "to send to prospects"
- detect `who` from phrases like:
  - "act as"
  - "you are a"
  - "as an expert"

When the evaluator cannot determine a field confidently, it should leave the field as `missing` rather than inventing a default.

## Compliance checks

When `compliance_check_enabled` is true, the transformer should run a deterministic screening pass for risky or regulated prompt content.

Initial scope can include pattern-based findings such as:

- legal, medical, or financial professional advice framing
- requests involving regulated records or confidential business data
- security-sensitive instructions
- policy-sensitive instructions involving customer, employee, or contract data

Each finding should include:

- `type`
  - `compliance`
- `severity`
  - `low`
  - `medium`
  - `high`
- `code`
  - stable identifier such as `regulated_advice` or `confidential_business_data`
- `message`
  - user-facing explanation

Recommended behavior:

- `low` or `medium` severity can be warning-only
- `high` severity can produce a `blocked` result

Severity mapping should be deterministic and documented in code.

## PII checks

When `pii_check_enabled` is true, the transformer should run a deterministic screening pass for personally identifiable information exposure.

Initial scope can include:

- email addresses
- phone numbers
- street addresses
- full names combined with contact details
- account or policy identifiers
- SSN-like patterns
- uploaded or pasted lists containing person-level records

Examples of content that should trigger findings:

- a prospect list with names and emails
- a patient or client roster
- pasted HR or payroll data

Each finding should include:

- `type`
  - `pii`
- `severity`
  - `low`
  - `medium`
  - `high`
- `code`
  - stable identifier such as `email_list_detected` or `sensitive_identifier_detected`
- `message`
  - user-facing explanation

Recommended behavior:

- isolated low-risk strings may be warning-only
- repeated person-level records or highly sensitive identifiers should produce a `blocked` result

## Result contract

The transformer should move from a single-shape success response to a typed result contract.

Recommended top-level field:

- `result_type`
  - `transformed`
  - `coaching`
  - `blocked`

### `transformed`

Use when prompt structure passes and no blocking compliance or PII findings exist.

Required fields:

- `transformed_prompt`
- `task_type`
- `conversation`
- `findings`
- `metadata`

### `coaching`

Use when prompt structure is incomplete for the current enforcement level.

Required fields:

- `coaching_tip`
- `conversation`
- `findings`
- `metadata`

No usable `transformed_prompt` should be returned for execution in this state.

### `blocked`

Use when compliance or PII findings are severe enough to stop the request.

Required fields:

- `blocking_message`
- `conversation`
- `findings`
- `metadata`

No usable `transformed_prompt` should be returned for execution in this state.

## Recommended request schema changes

Add a conversation object to the request payload.

```json
{
  "session_id": "sess_123",
  "conversation_id": "conv_123",
  "user_id": "user_1",
  "raw_prompt": "Draft an outreach email for this prospect list.",
  "target_llm": {
    "provider": "openai",
    "model": "gpt-4.1"
  },
  "conversation": {
    "conversation_id": "conv_123",
    "requirements": {
      "who": {
        "value": "sales copywriter",
        "status": "present"
      },
      "task": {
        "value": "draft an outreach email",
        "status": "present"
      },
      "context": {
        "value": "for B2B outbound prospecting",
        "status": "derived"
      },
      "output": {
        "value": "email draft",
        "status": "derived"
      }
    },
    "enforcement": {
      "level": "moderate",
      "status": "passes",
      "missing_fields": [],
      "last_evaluated_at": "2026-04-17T12:34:56Z"
    }
  },
  "summary_type": 3
}
```

Notes:

- `conversation_id` should be a first-class top-level field for tracing, logging, and request correlation
- `conversation.conversation_id` duplicates the same identifier inside the shared object so the object is self-contained when persisted or inspected independently

If backward compatibility is required, `conversation` should be optional at first and default to all fields missing.

## Recommended response schema

```json
{
  "session_id": "sess_123",
  "conversation_id": "conv_123",
  "user_id": "user_1",
  "result_type": "coaching",
  "task_type": "writing",
  "coaching_tip": "Add the role, the intended use, and the output format so I can transform this safely.",
  "conversation": {
    "conversation_id": "conv_123",
    "requirements": {
      "who": {
        "value": null,
        "status": "missing"
      },
      "task": {
        "value": "draft something",
        "status": "derived"
      },
      "context": {
        "value": null,
        "status": "missing"
      },
      "output": {
        "value": null,
        "status": "missing"
      }
    },
    "enforcement": {
      "level": "full",
      "status": "needs_coaching",
      "missing_fields": ["who", "context", "output"],
      "last_evaluated_at": "2026-04-17T12:34:56Z"
    }
  },
  "findings": [
    {
      "type": "pii",
      "severity": "high",
      "code": "email_list_detected",
      "message": "This prompt appears to include a list of personal contact records."
    }
  ],
  "metadata": {
    "persona_source": "db_profile",
    "rules_applied": [
      "task:writing:keyword",
      "policy:enforcement:full",
      "check:pii:enabled"
    ],
    "profile_version": "v2",
    "requested_model": "gpt-4.1",
    "resolved_model": "gpt-4.1",
    "used_fallback_model": false
  }
}
```

## UI flow

### New conversation

1. UI starts with an empty conversation object for the thread.
2. UI submits user prompt and current empty state to transformer.
3. Transformer derives what it can and returns either:
   - `transformed`
   - `coaching`
   - `blocked`
4. UI updates in-memory thread state from the returned `conversation` object.

### Existing conversation

1. UI reads the stored conversation object for the thread.
2. UI submits the new user prompt plus current thread state.
3. Transformer validates the combined conversation setup.
4. UI updates thread memory from the returned `conversation` object.

### LLM dispatch rule

The UI must only send a prompt to the target LLM when:

- `result_type == "transformed"`

The UI must not send:

- `coaching_tip`
- `blocking_message`
- `findings`

to the target LLM as part of the user request payload.

## Database changes

Add new columns to the profile tables:

- `prompt_enforcement_level`
- `compliance_check_enabled`
- `pii_check_enabled`

Recommended types:

- `prompt_enforcement_level`
  - string or enum
- `compliance_check_enabled`
  - boolean
- `pii_check_enabled`
  - boolean

Tables to update:

- `final_profile`
- `type_detail`
- `brain_chemistry`
- `environment_details`
- `behaviorial_adj`

If runtime only reads `final_profile`, adding the fields to all profile-layer tables is optional for the first release. Add them everywhere only if profile composition will use them later.

No conversation object persistence is required in the transformer database for the first implementation. The UI should remain the source of truth for per-conversation storage.

## Service changes

### New or updated schema models

- update `TransformPromptRequest`
  - add `conversation_id`
  - add optional `conversation`
- update `TransformPromptResponse`
  - replace single fixed success shape with typed result fields
  - include `conversation_id`
  - include returned `conversation`
- add schema models for:
  - conversation requirement item
  - conversation object
  - enforcement status
  - finding
  - result type

### New services

- `PromptRequirementService`
  - merges request prompt and thread state
  - derives `who`, `task`, `context`, `output`
  - validates required elements for the effective enforcement level
- `ComplianceCheckService`
  - returns deterministic compliance findings
- `PIICheckService`
  - returns deterministic PII findings

### Updated orchestration

`TransformerEngine.transform()` should:

1. resolve profile
2. resolve effective enforcement and check flags
3. merge and validate conversation requirements
4. run compliance and PII checks if enabled
5. decide `result_type`
6. build transformed prompt only for `transformed`
7. return updated conversation state with enforcement status
8. log the request and result

## Logging changes

Request logs should capture:

- `conversation_id`
- `result_type`
- effective enforcement level
- whether compliance check ran
- whether PII check ran
- structured findings summary
- conversation requirements summary

Avoid logging raw sensitive content beyond what is already intentionally logged by the service.

## Backward compatibility plan

Recommended rollout:

1. add optional request fields and new profile settings
2. add `conversation_id` to all callers
3. support old callers that do not send `conversation`
4. default old callers to empty conversation state
5. return updated conversation state in responses
6. introduce `result_type`
7. update UI to branch on `result_type`
8. once all callers are updated, make stricter assumptions if needed

If a hard schema break is acceptable, this can be done in one release instead.

## Testing plan

### API tests

Add tests for:

- `none` enforcement passes through
- `low` enforcement blocks only highly ambiguous prompts
- `moderate` enforcement derives missing fields when possible
- `moderate` enforcement returns coaching when required fields remain missing
- `full` enforcement requires all four elements
- compliance warning-only response
- compliance blocked response
- PII warning-only response
- PII blocked response
- mixed findings with `coaching`
- mixed findings with `blocked`
- backward-compatible requests without `conversation`
- responses that echo or update `conversation_id`

### Service tests

Add deterministic unit tests for:

- `who` derivation
- `task` derivation
- `context` derivation
- `output` derivation
- severity thresholds
- stable finding codes

## Open decisions

The implementation should explicitly confirm these decisions before coding starts:

- whether compliance findings are warning-only, blocking, or severity-based
- whether PII findings are always blocking above a threshold
- whether `moderate` requires `context` or `output`, or just one of them
- whether `who` may default to a generic assistant role in `moderate`
- whether request logging should redact findings tied to raw sensitive prompt content

## Recommended first iteration

To keep the first release manageable:

- store enforcement settings in `final_profile`
- let the UI own all per-thread conversation memory
- require `conversation_id`
- make `conversation` optional
- implement heuristic derivation for `who`, `task`, `context`, and `output`
- return typed `result_type`
- treat high-severity PII as blocking
- treat compliance as warning-only first, unless a clear blocking rule exists

This gives the Prompt UI a reliable contract while keeping the transformer deterministic and implementation scope controlled.
