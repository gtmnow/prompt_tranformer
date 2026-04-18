# Prompt Scoring Implementation Spec

## Purpose

This document defines the implementation plan for prompt scoring in Prompt Transformer.

The goal is to measure prompt quality at the conversation level and track whether a user's prompting skill improves over time.

Prompt scoring must remain separate from prompt enforcement.

## Core rule

Scoring measures prompt quality.

Enforcement measures whether the system is currently requiring more prompt structure before proceeding.

That means:

- the score should come only from prompt structure quality
- enforcement should not change the score
- enforcement should only affect whether the user is coached, allowed through, or blocked

In practice:

- the same prompt should receive the same score regardless of enforcement level
- changing enforcement from `none` to `full` should not lower or raise the score
- higher enforcement may help users improve later prompts, which can lead to higher future scores

## Goals

- score prompt quality at the conversation level
- reuse the existing `who`, `task`, `context`, and `output` model
- distinguish between:
  - `present`
  - `derived`
  - `missing`
- normalize scores on a 0 to 100 scale
- track `starting`, `ending`, `best`, and `improvement` score moments
- preserve score history for future admin-tool analysis
- keep score history even if a user-facing conversation is deleted

## Non-goals

- replacing enforcement with scoring
- using enforcement level as a scoring multiplier or penalty
- grading the factual quality of the final LLM answer
- building a public ranking system

## Product use case

Prompt scoring is intended to support:

- internal analytics on prompt quality and coaching effectiveness
- a future admin tool that reviews a user's prompting skill over time

Prompt scoring is therefore persistent analytics, not ephemeral conversation UI state.

The UI should read conversation score data from a transformer-owned score endpoint backed by the transformer database.

The scoring model itself should live in transformer-owned rule files, not in database tables, so scoring can be recalibrated without adding per-request database overhead.

## Scoring unit

The primary scoring unit is the conversation.

Reasoning:

- enforcement already works at the conversation level
- key prompt dimensions may be established across multiple turns
- users may improve their prompt after coaching
- conversation-level scoring better reflects skill development than single-message scoring

Per-turn snapshots are optional telemetry, not the primary user metric.

## Inputs to scoring

Prompt scoring uses the canonical conversation state returned by enforcement:

- `conversation_id`
- `user_id_hash`
- `requirements.who`
- `requirements.task`
- `requirements.context`
- `requirements.output`

Each requirement has one of three statuses:

- `present`
- `derived`
- `missing`

These statuses are the minimum required inputs to the score.

Enforcement level may be stored as context for analytics, but it must not change the score math.

## Planned hybrid scoring methodology

The target scoring model is a hybrid of:

- deterministic heuristics
- LLM-assisted semantic evaluation

This hybrid model is intended for internal scoring only. Users do not see the hidden evaluator pass.

### Deterministic layer

The deterministic layer provides:

- a stable baseline classification
- obvious pattern detection
- a safe fallback when the evaluator is unavailable
- score explainability for debugging and audits

The deterministic layer should continue to classify `who`, `task`, `context`, and `output` as:

- `present`
- `derived`
- `missing`

### LLM scoring layer

The LLM scoring layer should evaluate the same four dimensions semantically and return structured output.

Recommended per-field output:

- `status`
  - `present`
  - `derived`
  - `missing`
- `confidence`
  - normalized confidence such as `0.0` to `1.0`
- `reason`
  - short internal explanation for why the field was classified that way

The evaluator must be internal-only and schema-constrained. It should not generate the final user-visible score explanation directly.

### Fusion layer

Prompt Transformer should combine the heuristic result and the LLM result into one final field status per dimension.

Recommended fusion approach:

- heuristics set a conservative floor
- high-confidence LLM judgments may upgrade or downgrade field status
- low-confidence LLM judgments should not override deterministic signals

Example fusion rules:

- if both layers agree, accept the shared result
- if heuristics say `missing` and the LLM says `present` with high confidence, upgrade to `present`
- if heuristics say `present` and the LLM says `derived` with low confidence, keep `present`
- if the evaluator fails entirely, use the deterministic result only

### Final score

The final score shown to users should come from the fused field statuses, not from heuristics alone and not from the raw LLM judgment alone.

Recommended stored analytics fields:

- `heuristic_score`
- `llm_score`
- `final_score`
- `scoring_version`
- `scoring_method`
  - for example `heuristic_only` or `hybrid_llm_v1`

This supports comparison over time as the scoring model evolves.

## Scoring principles

### 1. Clearly present prompt content scores highest

If the prompt clearly contains a field in natural language, that should score higher than a field the system merely infers.

Labels like `Who:` or `Task:` are not required for full structural credit.

### 2. Enforcement does not affect the score

The score should be stable across enforcement settings.

The purpose of enforcement is to coach the user toward a better prompt, not to redefine what the current prompt is worth.

### 3. Improvement matters

The system should track how a prompt improves during the conversation, not just whether it was strong on the first attempt.

### 4. Scores should be easy to explain

The model should remain transparent enough that users and admins can understand what increased or decreased a score.

## Current baseline scoring model

Each requirement contributes 25 points.

- `who`: 25
- `task`: 25
- `context`: 25
- `output`: 25

Per-field points:

- `present`: 25
- `derived`: 5
- `missing`: 0

Formula:

- `structural_score = who + task + context + output`

Range:

- minimum: `0`
- maximum: `100`

This is the primary and only score shown to users in the Herman Prompt experience.

Enforcement level may influence whether the prompt is coached or allowed through, but it does not change these points.

Recommended implementation detail:

- keep the scoring model in `app/rules/prompt_scoring.yaml`
- load it with the same rule-registry path used for task and model policy rules
- store the applied scoring version in analytics rows for comparability over time

Versioning rule:

- `app/rules/prompt_scoring.yaml` must include a top-level `version`
- that YAML `version` is the scoring model version of record
- every persisted score row must store that exact `scoring_version`
- score read responses should return `scoring_version` so downstream systems can tell which model produced the score

This YAML-backed model is the calibration layer for the current deterministic baseline and also the right place to store future hybrid-scoring weights or thresholds.

## Scoring moments

Each conversation should track these score moments:

### Starting score

The score from the first substantive prompt in the conversation.

Purpose:

- baseline prompt quality before coaching

### Ending score

The score from the latest known conversation state.

Purpose:

- where the conversation currently stands

### Best score

The highest score reached at any point in the conversation.

Purpose:

- the strongest prompt state achieved in the conversation

### Improvement score

Formula:

- `improvement_score = ending_score - starting_score`

Recommended additional metric:

- `best_improvement_score = best_score - starting_score`

Purpose:

- measure whether coaching and revision improved prompt quality

## Additional conversation metrics

Each scored conversation should also track:

- `coaching_turn_count`
- `blocked_turn_count`
- `transformed_turn_count`
- `passed_without_coaching`
- `reached_policy_complete`
- `scoring_version`

Important note:

- `reached_policy_complete` is an enforcement outcome
- it is useful for analytics
- it is not part of the score

## Persistence design

Prompt scoring should live in the transformer database because Prompt Transformer already owns:

- canonical requirement state
- result typing
- request logging
- conversation-level enforcement evaluation

## Read path

The Herman Prompt UI should not reconstruct scores from conversation memory and should not treat score data as an in-band chat artifact.

Recommended read model:

1. Prompt Transformer evaluates and stores the score rollup.
2. Herman Prompt fetches the score rollup by `conversation_id` and `user_id_hash`.
3. UI surfaces `initial`, `final`, and later `best` or `improvement` values from that database-backed read path.

Recommended endpoint:

- `GET /api/conversation_scores/{conversation_id}?user_id=<user_id_hash>`

When hybrid scoring is introduced, this read path should continue to return the final fused score used by the UI.

### Recommended table: `conversation_prompt_scores`

Suggested columns:

- `id`
- `conversation_id`
- `user_id_hash`
- `task_type`
- `conversation_started_at`
- `conversation_ended_at`
- `last_scored_at`
- `conversation_deleted_at`
- `enforcement_level`
- `initial_score`
- `best_score`
- `final_score`
- `improvement_score`
- `best_improvement_score`
- `passed_without_coaching`
- `reached_policy_complete`
- `coaching_turn_count`
- `blocked_turn_count`
- `transformed_turn_count`
- `who_status`
- `task_status`
- `context_status`
- `output_status`
- `score_details_json`
- `scoring_version`
- timestamps

Recommended uniqueness rule:

- unique on `conversation_id`

Important design note:

- this is an analytics table, not a conversation-sidecar table
- it should remain available for reporting even if the user-facing conversation is later deleted

## Optional event history

If needed later, add a supporting table such as `conversation_prompt_score_events`.

Suggested columns:

- `id`
- `conversation_id`
- `user_id_hash`
- `task_type`
- `turn_index`
- `structural_score`
- `result_type`
- `who_status`
- `task_status`
- `context_status`
- `output_status`
- `missing_fields_json`
- `score_details_json`
- `created_at`

Purpose:

- preserve score progression over time
- support auditing
- support future admin charts without recomputing from request logs

## Retention model

Prompt score retention should be independent from user-facing conversation retention.

Recommended behavior:

- if a conversation exists, scores link to it by `conversation_id`
- if a conversation is deleted from the user-facing app, the score row remains
- when a conversation is deleted, the analytics row should record:
  - `conversation_deleted_at`
- do not delete score rollups by default
- do not delete score events by default

Recommended distinction:

- standard conversation delete:
  - removes or hides the user-facing conversation
  - preserves analytics
- privacy or legal hard delete:
  - removes both conversation content and analytics

## Relationship to enforcement

Enforcement should remain visible in analytics, but only as separate metadata.

Recommended enforcement-related fields:

- `enforcement_level`
- `passed_without_coaching`
- `reached_policy_complete`
- `coaching_turn_count`

These are useful for analysis such as:

- does stricter enforcement improve later prompt quality?
- which users improve after coaching?
- which tasks frequently require coaching?

But none of these fields should change the score itself.

## Relationship to request logging

Request logging remains the detailed per-request event record.

Scoring tables remain the analytics rollup layer.

These responsibilities should stay separate.

## Scoring algorithm

### Current implementation

- classify each field as `present`, `derived`, or `missing`
- map status to points through `app/rules/prompt_scoring.yaml`
- sum the four dimension scores

### Planned implementation

- compute a heuristic field status and score
- compute an LLM-evaluated field status and score
- fuse the two into a final field status per dimension
- compute the final score from the fused statuses
- persist both component scores plus the final score for analytics

### Step 1

Evaluate conversation requirements using the existing enforcement path.

### Step 2

Map each requirement status to points:

- `present` => `25`
- `derived` => `15`
- `missing` => `0`

### Step 3

Compute:

- `structural_score`

### Step 4

Upsert the conversation rollup row.

Rules:

- if no prior row exists, set `initial_score = current_score`
- always set `final_score = current_score`
- set `best_score = max(prior_best_score, current_score)`
- update `improvement_score`
- update `best_improvement_score`
- update turn counters using the current `result_type`

### Step 5

Optionally append a score event row.

## Suggested score detail payload

Recommended `score_details_json` shape:

- `field_points`
  - `who`
  - `task`
  - `context`
  - `output`
- `field_statuses`
  - `who`
  - `task`
  - `context`
  - `output`
- `weights`
  - `who`
  - `task`
  - `context`
  - `output`
- `missing_fields`
- `enforcement_level`
- `enforcement_status`
- `result_type`
- `calculated_at`

This makes debugging easier without allowing enforcement to influence the score.

## User-level analytics

User skill should be measured from aggregates over many conversations.

Recommended derived metrics per user:

- `conversation_count`
- `average_initial_score`
- `average_final_score`
- `average_best_score`
- `average_improvement_score`
- `average_coaching_turn_count`
- `pass_without_coaching_rate`
- `policy_completion_rate`
- rolling averages over the last:
  - `7 conversations`
  - `30 conversations`

Recommended additional slices:

- by `task_type`
- by `enforcement_level`
- excluding conversations where `conversation_deleted_at is not null`
- including all conversations regardless of deletion state

## Admin-tool use case

The future admin tool should be able to answer questions like:

- how has a user's prompt quality changed over time?
- how often does a user improve after coaching?
- which task types is a user strongest or weakest at?
- how often does a user reach policy completion after starting low?

To support that, each score row should include at minimum:

- `user_id_hash`
- `task_type`
- `conversation_started_at`
- `last_scored_at`
- `initial_score`
- `best_score`
- `final_score`
- `improvement_score`
- `conversation_deleted_at`

## API considerations

The transformer may expose a compact score summary for testing or internal tooling.

Recommended summary:

- `structural_score`

If enforcement state is needed, it should be exposed separately from scoring.

## Testing plan

### Unit tests

Add tests for:

- per-field score mapping
- structural score calculation
- improvement calculation
- score detail payload generation

### Integration tests

Add tests for:

- first scored turn creates a rollup row
- later turns update `final_score`
- later stronger turns update `best_score`
- `coaching` conversations still generate score updates
- `blocked` conversations still record progression appropriately
- conversation deletion does not remove score history by default once delete-handling exists

### Example test cases

Case 1:

- prompt has only `task`
- expect low structural score

Case 2:

- prompt has explicit `who`, `task`, `context`, `output`
- expect score `100`

Case 3:

- prompt has `task`, inferred `context`, inferred `output`, missing `who`
- expect partial score

Case 4:

- first turn is weak, second turn completes missing fields after coaching
- expect positive improvement

## Rollout plan

### Phase 1

- implement structural scoring and persistence
- backfill nothing
- compute scores only for new conversations going forward

### Phase 2

- add internal reporting queries
- validate whether score distributions look reasonable

### Phase 3

- add admin-tool read APIs or reporting queries for user score history over time

## Open decisions

These should be finalized before broader rollout:

1. Should `derived` always receive the same partial credit?
2. Should blocked conversations count toward prompt-skill analytics, or only toward safety analytics?
3. Should conversations with `enforcement_level = none` be excluded from trend metrics?
4. Do we want only conversation rollups in v1, or both rollups and per-turn events?
5. Should admin reporting include deleted conversations by default, or hide them unless explicitly requested?

## Recommendation

Implement prompt scoring as a transformer-owned structural score built from `who`, `task`, `context`, and `output` only.

Use enforcement as a coaching layer that helps users reach higher scores over time, but never as a direct input to the score itself.
