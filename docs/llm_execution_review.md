## Targeted LLM Execution Review

This review focused only on the runtime LLM execution path used by chat, helper generation, and provider adapters.

### Pre-refactor issues

- `FinalResponseService` used a bespoke `httpx` client path instead of the shared `LlmGatewayService`.
- Live web search was enabled by prompt substring matching against the transformed prompt only.
- Final-response requests could drift between the caller model and the provider profile's resolved model.
- Provider 4xx responses were remapped into generic 5xx errors, which obscured actionable request bugs.

### Refactor direction

- Chat/final-response execution now flows through the shared gateway and OpenAI-compatible adapter stack.
- Shared request objects now support structured messages, responses API input items, and tool payloads.
- Final-response intent is resolved explicitly from the raw prompt plus transformed prompt, then passed into the final-response layer.
- Provider HTTP status codes are preserved when available so invalid requests stay visible as invalid requests.

### Remaining architectural follow-up

- Move intent resolution into a dedicated policy/service layer if web search and image generation continue expanding.
- Add provider capability metadata for tool support instead of hard-coded provider checks.
- Consider unifying helper, structure-evaluator, and final-response request construction behind one higher-level orchestration layer.
