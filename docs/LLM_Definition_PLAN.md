# Plan: Org-Assigned LLM Wiring Across Herman Admin, Herman Prompt, and Herman Transform

## Summary

Implement tenant-scoped LLM resolution end-to-end using the existing Herman Admin tenant LLM config as the system of record, then have Herman Prompt and Herman Transform consume that org-assigned configuration per user. The build should preserve the current canonical `user_id_hash` flow and keep Herman Prompt out of direct Herman Admin app-to-app HTTP calls for bootstrap. Instead, all three services should read the authoritative shared data layer.

This plan assumes:
- “organization” in the intended behavior maps to the current `tenant` model in Herman Admin.
- one effective LLM config exists per tenant and all users in that tenant inherit it.
- the shared database used today for `auth_users`, `final_profile`, and analytics is also where `tenants` and `tenant_llm_config` live for runtime resolution.
- the existing Herman Admin platform-managed LLM pool is already built and should be treated as the implementation of the “HermanScience master LLM list,” even if the product wording is still catching up.

## Concrete Gap List

### Herman Admin
- Present today:
  - tenant/org LLM config CRUD already exists in `tenant_llm_config`
  - vault-backed secret references already exist
  - platform-managed LLM entries already exist
  - activation already requires a validated tenant LLM config
- Missing:
  - a service-facing read contract for “effective user/org LLM config by `user_id_hash`”
  - explicit inheritance documentation that users get their tenant’s single LLM
  - runtime-ready status guarantees beyond admin-side validation
  - clear distinction between admin-only masked config views and internal runtime secret resolution

### Herman Prompt
- Present today:
  - bootstrap resolves `user_id_hash`, `tenant_id`, profile version, and enforcement level
  - transformer request already sends a `target_llm`
  - final LLM execution already goes through a provider adapter boundary
- Missing:
  - bootstrap load of the user’s inherited org LLM config
  - session/bootstrap response fields exposing the effective provider/model for the current user
  - replacement of global env `LLM_PROVIDER` / `LLM_MODEL` / `LLM_API_KEY` as the per-request source of truth
  - request-time lookup of tenant LLM credentials and endpoint
  - graceful failure when a user has no valid tenant LLM assignment
  - propagation of tenant-assigned LLM config into both transformer calls and final answer generation

### Herman Transform
- Present today:
  - accepts `target_llm.provider` and `target_llm.model`
  - resolves model policy locally with fallback
  - caches resolved persona/profile by `user_id_hash`
  - optional LLM evaluator path exists
- Missing:
  - lookup of effective tenant LLM runtime config by `user_id_hash`
  - cache for resolved org LLM credentials/config
  - use of tenant-assigned credentials for all actual LLM-backed scoring/evaluation work
  - validation that the incoming requested model matches the tenant’s assigned model
  - clear policy for mismatches between caller-sent model and authoritative tenant model
  - removal of env-only `STRUCTURE_EVALUATOR_*` as the runtime source of truth for tenant-scoped scoring

## Key Implementation Changes

### 1. Herman Admin becomes the authoritative runtime config owner
- Keep `tenant_llm_config` and `platform_managed_llm_configs` as the source of truth for org-assigned LLM selection.
- Add a backend-only resolver in Herman Admin service code that can return an effective runtime LLM record for a given `user_id_hash` or `tenant_id`.
- The resolver should:
  - resolve `user_id_hash -> user_tenant_membership.tenant_id` via the primary active membership, with `auth_users.tenant_id` treated as a derived compatibility pointer
  - load the tenant’s `tenant_llm_config`
  - require `credential_status == "valid"`
  - resolve the real API key from `secret_reference`
  - return provider, model, endpoint URL, secret reference metadata, transformation/scoring toggles, and tenant id
- Add an internal schema for this resolved runtime payload. It must be backend-facing only and must never expose masked/admin UI fields as if they were usable runtime credentials.
- Update Admin docs to state that all users in a tenant inherit the same effective LLM config and that platform-managed LLMs are already implemented as the shared pool.

### 2. Herman Prompt loads and uses the effective tenant LLM at bootstrap and send time
- Extend bootstrap loading so Herman Prompt resolves the user’s effective tenant LLM config from the shared data layer alongside `auth_users` and `final_profile`.
- Extend `SessionBootstrapResponse` to include a non-secret `llm` object:
  - `provider`
  - `model`
  - `configured` or `status`
  - optionally `transformation_enabled` and `scoring_enabled`
- Do not include secrets in bootstrap.
- If the user profile exists but no valid tenant LLM config exists, bootstrap should fail with a blocking error message consistent with the current “contact your administrator” pattern.
- Introduce a runtime LLM config resolver/service in Herman Prompt and make it the only source of truth for:
  - transformer `target_llm`
  - final downstream provider selection
  - API key / base URL used by the provider adapter
- Refactor the provider adapter boundary so request methods accept a resolved runtime config object instead of reading `settings.llm_*` directly.
- Refactor `TransformerClient.transform_prompt` so `target_llm` comes from the resolved tenant config, not env.
- Preserve the env config only as local/dev fallback when explicitly enabled, not as the default production path.

### 3. Herman Transform resolves authoritative tenant LLM config and uses it for LLM-backed actions
- Add a runtime LLM config resolver in Herman Transform keyed by `user_id_hash`, parallel to the existing profile resolver.
- Cache the resolved tenant LLM config by `user_id_hash` with TTL and explicit invalidation hooks similar to the profile cache.
- On `transform_prompt`:
  - resolve the tenant-assigned config from the shared data layer
  - compare it to incoming `target_llm`
  - if provider/model mismatch, prefer the authoritative tenant config and mark metadata to indicate override
- Extend transform metadata to report:
  - `requested_provider`
  - `requested_model`
  - `resolved_provider`
  - `resolved_model`
  - `used_fallback_model`
  - `used_authoritative_tenant_llm`
- Keep local model policy resolution, but run it against the authoritative tenant provider/model.
- Replace env-based evaluator config for tenant-scoped scoring with the resolved tenant config:
  - scoring/evaluator API key comes from the tenant’s resolved secret
  - model comes from the tenant’s assigned model unless product explicitly chooses a separate evaluator model later
- Keep the deterministic transform path unchanged where no LLM call is required; only LLM-backed evaluation/scoring should use tenant credentials.

### 4. Contracts and data interfaces
- Add a shared internal “resolved runtime LLM config” shape across services with:
  - `tenant_id`
  - `user_id_hash`
  - `provider`
  - `model`
  - `endpoint_url`
  - `api_key`
  - `transformation_enabled`
  - `scoring_enabled`
  - `credential_status`
  - `source_kind` such as `customer_managed` or `platform_managed`
- Update Herman Prompt bootstrap API contract to expose only the safe subset.
- Update Herman Transform request/response docs to clarify:
  - caller still sends `target_llm`
  - transformer validates against authoritative tenant config
  - authoritative tenant config wins on mismatch
- Document one failure contract across Prompt and Transform:
  - missing tenant assignment
  - invalid tenant credentials
  - disabled transformation
  - disabled scoring
  - unresolved vault secret

### 5. Operational behavior and edge cases
- Tenant assignment changes should affect the next request after cache expiry or explicit invalidation.
- User moved to a new tenant should inherit the new tenant’s LLM automatically through the primary `user_tenant_membership`, with `auth_users.tenant_id` staying in sync as a derived pointer.
- If transformation is disabled for a tenant:
  - Herman Prompt should bypass Prompt Transformer intentionally and record the bypass reason from tenant config.
- If scoring is disabled for a tenant:
  - Herman Transform should skip evaluator-backed scoring and return deterministic-only scoring metadata.
- If a platform-managed LLM record is deleted or becomes invalid:
  - tenant bootstrap and send paths should block with a clear admin-action-needed error rather than falling back silently to env defaults.
- Unknown model policy in Herman Transform should still use local fallback policy rules, but only after the authoritative tenant provider/model has been established.

## Public API / Interface Changes

- Herman Prompt `GET /session/bootstrap`
  - add `llm` object with effective non-secret provider/model/status fields
- Herman Prompt internal provider interface
  - generation methods should accept a resolved runtime LLM config instead of reading globals
- Herman Transform `TransformMetadata`
  - expand requested vs resolved provider/model reporting and authoritative-resolution indicator
- Herman Admin internal service layer
  - add runtime resolver API/function for effective tenant LLM config by `user_id_hash` or `tenant_id`

## Test Plan

- Herman Admin
  - tenant with customer-managed key resolves a usable runtime config
  - tenant with platform-managed config resolves inherited provider/model/secret correctly
  - invalid tenant config does not resolve as runtime-usable
  - user in tenant A resolves tenant A config; after tenant reassignment resolves tenant B config

- Herman Prompt
  - bootstrap returns effective provider/model for a valid tenant LLM
  - bootstrap blocks when `auth_users` exists but tenant LLM is missing or invalid
  - transformer request uses tenant-assigned provider/model, not env defaults
  - final provider adapter call uses tenant-assigned model/key/endpoint
  - transformation-disabled tenant bypasses transform cleanly
  - scoring-disabled tenant still allows final answer generation

- Herman Transform
  - matching incoming `target_llm` passes through as authoritative
  - mismatched incoming `target_llm` is overridden by tenant config and flagged in metadata
  - profile cache and tenant-llm cache operate independently
  - evaluator uses tenant key/model when scoring is enabled
  - evaluator is skipped when scoring is disabled
  - unknown model still resolves against local fallback policy after tenant config lookup
  - missing tenant config returns a clear service error, not `generic_default` behavior for credentials

- End-to-end
  - create tenant LLM config in Admin, assign user, launch Herman Prompt, verify bootstrap shows assigned model, transformer sees same model, and final response uses same tenant config
  - switch tenant model in Admin, expire/invalidate caches, verify next Prompt/Transform requests use the new model
  - platform-managed shared LLM selected in Admin flows through to Prompt and Transform without exposing secrets in UI payloads

## Assumptions and Defaults

- Use the existing Herman Admin `tenant` model as the org abstraction.
- Use the existing `tenant_llm_config` and `platform_managed_llm_configs` tables instead of inventing new org-LLM tables.
- Keep the shared database as the integration point; do not add direct Herman Prompt -> Herman Admin HTTP dependency for bootstrap.
- Keep one effective LLM per tenant for v1; do not add per-user LLM overrides.
- Treat tenant config as authoritative over caller-supplied `target_llm`.
- Preserve env-based LLM settings only for local development fallback, not production runtime selection.
- Keep transform construction deterministic; only LLM-backed evaluator/scoring paths need tenant credentials.
