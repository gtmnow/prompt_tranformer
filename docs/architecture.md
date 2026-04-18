# Architecture

## Purpose

Prompt Transformer is a runtime prompt construction service. It accepts a raw prompt and rewrites it using:

- a profile from `final_profile`
- task inference rules
- model policy rules
- optional summary persona overrides
- optional LLM-assisted prompt structure evaluation and scoring inputs

The transformed prompt construction path remains deterministic. The service may optionally call a small evaluator model for structure extraction and future hybrid scoring, but it does not generate or update user profiles.

## Request lifecycle

1. `app.api.routes.transform_prompt` receives the HTTP request.
2. `TransformPromptRequest` validates payload structure.
3. `TransformerEngine.transform()` orchestrates the runtime path.
4. `ProfileResolver` determines the persona source:
   - summary override
   - database profile
   - generic default
5. `TaskInferenceService` classifies the prompt from local keywords/phrases.
6. `LLMPolicyService` resolves the target model policy or fallback model.
7. `PromptRequirementService` merges conversation state, derives missing fields, and evaluates enforcement.
8. Optional compliance and PII check services add findings.
9. `PromptScoringService` computes the structural score and persists the conversation rollup.
10. `TransformerEngine._build_prompt()` constructs the transformed prompt only when the request is allowed to proceed.
11. `RequestLogger` optionally persists a debug log row.
12. The API returns a typed result containing either a transformed prompt, coaching guidance, or a blocked outcome.

## Design principles

- Deterministic prompt construction: user-facing prompt rewriting should not depend on stochastic generation.
- Database-first: runtime personalization comes only from `final_profile`.
- Stateless: no session memory outside request logging.
- Explicit fallback paths: missing users and unknown models degrade safely.

For scoring and structure extraction, the long-term design is hybrid:

- deterministic rules provide baseline stability
- LLM evaluation provides semantic judgment
- transformer-owned fusion logic produces the final score and field states

## Module ownership

### API layer

- `app/main.py`
  - FastAPI app factory
  - validation error mapping to `400`
  - lifespan startup for rule preload
- `app/api/routes.py`
  - health route
  - transform route
  - HTTP error translation

### Core

- `app/core/config.py`
  - env-driven runtime settings
- `app/core/rules.py`
  - YAML rule loading and caching

### Database

- `app/db/session.py`
  - SQLAlchemy engine and session factory
- `app/db/bootstrap.py`
  - startup migrations and optional seeding
- `app/db/seed.py`
  - sample data load for MVP users

### Models

- `app/models/profile.py`
  - runtime and future profile-layer tables
- `app/models/request_log.py`
  - optional request log table

### Services

- `app/services/profile_resolver.py`
  - summary override, DB lookup, generic fallback
- `app/services/task_inference.py`
  - deterministic task detection
- `app/services/llm_policy.py`
  - model policy lookup with fallback
- `app/services/prompt_requirements.py`
  - conversation-state merging, derivation, and enforcement evaluation
- `app/services/compliance_checks.py`
  - deterministic compliance findings
- `app/services/pii_checks.py`
  - deterministic PII findings
- `app/services/transformer_engine.py`
  - orchestration, gating, and prompt construction
- `app/services/request_logger.py`
  - optional request persistence
- `app/services/prompt_scoring.py`
  - structural score calculation and rollup persistence
- `app/services/structure_evaluator.py`
  - optional LLM-backed semantic extraction

## Database boundaries

Runtime only reads:

- `final_profile`

Future-compatible but not used at runtime:

- `type_detail`
- `brain_chemistry`
- `environment_details`
- `behaviorial_adj`

This separation is intentional. Any future Profile Builder should own profile calculation and layer merging outside this service.

## Rule system

Rules live in `app/rules/` and are loaded at app startup:

- `summary_personas.yaml`
- `llm_policies.yaml`
- `task_rules.yaml`
- `prompt_scoring.yaml`

The runtime assumes these files are valid mappings. If you change rule structure, update the corresponding service code.

`prompt_scoring.yaml` should be treated as the source of truth for score calibration and future hybrid scoring thresholds.

## Extension guidance

Safe changes:

- add new seeded users
- add new task keywords/phrases
- add new model policy entries
- tune threshold behavior in `TransformerEngine._build_prompt()`

Changes that need care:

- altering request/response schema
- changing rule file structure
- modifying the precedence order of task/persona/model rules
- adding new persistence behavior during request handling

## Conversation Enforcement

Conversation-level prompt enforcement, compliance checks, and PII checks are now part of the runtime path.

See [prompt_enforcement_implementation_spec.md](./prompt_enforcement_implementation_spec.md) for the implementation design, API shape changes, profile changes, and UI integration contract.

## Prompt Scoring

Prompt scoring is transformer-owned analytics.

Current behavior:

- deterministic field classification
- YAML-backed score calibration
- database-backed conversation rollups

Planned behavior:

- heuristic baseline score
- LLM-evaluated semantic score
- fused final score persisted in the transformer database
- UI reads the final score through the score endpoint

See [prompt_scoring_implementation_spec.md](./prompt_scoring_implementation_spec.md) for the detailed methodology.
