# Feature Specification: Runtime Configuration and Bootstrap

**Feature Branch**: `feature/time-machine-runtime-configuration`  
**Created**: 2026-08-09  
**Status**: Draft  
**Input**: Start the service with validated local configuration, paths, resource limits, and dependencies for Apple Silicon inference.

## Clarifications

### Session 2026-08-09

- Q: If an optional inference runtime (such as image generation) is unavailable at startup, how should NodeMLX behave? → A: Start core service; disable only the unavailable optional capability.
- Q: When a resource-limit setting is numeric but outside its supported range, should NodeMLX correct it to a safe bound or refuse startup? → A: Clamp out-of-range numeric values to documented safe bounds and log a warning.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Start With Safe Local Defaults (Priority: P1)

As a local NodeMLX operator, I can start the service with no optional configuration so that a supported local setup works predictably without exposing the service beyond the local machine.

**Affected clients**: none  
**Why this priority**: A dependable safe startup is required before authentication, model selection, or chat can be used.  
**Independent Test**: Start the service with only required local prerequisites and verify it reports a clear ready state using safe defaults.

**Acceptance Scenarios**:

1. **Given** a supported local machine with no optional configuration, **When** the operator starts NodeMLX, **Then** the service starts using documented safe local defaults.
2. **Given** a machine that does not meet the local inference requirements, **When** the operator starts NodeMLX, **Then** startup stops with an actionable explanation rather than running in a degraded or unsafe state.

### User Story 2 - Correct Invalid Configuration Before Service Use (Priority: P2)

As a local NodeMLX operator, I receive clear feedback for invalid or conflicting configuration so that I can correct it before users begin a chat session.

**Affected clients**: none  
**Why this priority**: Early validation prevents avoidable failed requests, inaccessible models, and resource exhaustion during use.  
**Independent Test**: Provide invalid startup values and verify startup identifies the affected setting and the acceptable value range or format.

**Acceptance Scenarios**:

1. **Given** an invalid local setting, **When** the operator starts NodeMLX, **Then** the service identifies the setting and explains how to correct it.
2. **Given** a configuration that exceeds supported resource limits, **When** the operator starts NodeMLX, **Then** the service applies documented bounds or refuses to start with a clear reason.

### User Story 3 - Use Configured Local Model Resources (Priority: P3)

As a local NodeMLX operator, I can direct the service to supported local model and runtime locations so that models and optional image-generation capabilities are discovered without modifying source files.

**Affected clients**: none  
**Why this priority**: Local inference setups vary, but configuration must remain predictable and safe.  
**Independent Test**: Start with a valid custom local runtime or model location and verify the service uses it; start with an unavailable location and verify the remediation message.

**Acceptance Scenarios**:

1. **Given** a valid configured local runtime or model location, **When** NodeMLX starts, **Then** it uses that location for the applicable capability.
2. **Given** an unavailable or invalid configured location, **When** NodeMLX starts, **Then** it does not silently substitute an unexpected location.

## Edge Cases *(mandatory)*

- Required local inference support is unavailable or incompatible with the machine; startup fails with an actionable explanation.
- An optional runtime is unavailable; core service starts, the affected optional capability is disabled, and the operator receives an actionable diagnostic.
- A missing or malformed numeric resource setting causes actionable startup validation failure; an out-of-range numeric setting is clamped to a documented safe bound with an operator warning.
- A configured model cache, temporary-image location, or optional runtime location is unavailable, unreadable, or not executable.
- A generated local secret or data directory cannot be created or persisted.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST start with documented local-safe defaults when optional configuration is absent.
- **FR-002**: The system MUST validate startup configuration before accepting user traffic.
- **FR-003**: The system MUST report invalid, unsupported, or conflicting settings with the affected setting and a corrective action.
- **FR-004**: The system MUST keep service binding local by default unless the operator explicitly changes it.
- **FR-005**: The system MUST clamp numeric resource-sensitive generation settings outside documented safe bounds and warn the operator; missing or malformed numeric settings MUST produce an actionable startup validation failure.
- **FR-005a**: The system MUST allow core local chat service to start when an optional inference runtime is unavailable, disable only that capability, and report an actionable diagnostic.
- **FR-006**: The system MUST allow supported local model and runtime locations to be configured without source modification.
- **FR-007**: The system MUST preserve explicit operator configuration and must not silently replace an unavailable configured location with an unrelated location.
- **FR-008**: The system MUST document all user-configurable startup settings, defaults, accepted values, and failure behavior.

### Persistence and Runtime Impact

- **Stores affected**: Existing locally persisted application state only; this feature does not change user, session, or conversation data schemas.
- **Migration/backfill/compatibility**: No migration or backfill is required. Existing valid local configuration must remain supported.
- **Inference runtime affected**: Local text, vision, and optional image-generation runtime discovery and validation.
- **Configuration/environment variables affected**: Existing startup, local network binding, model location, resource limit, temporary-file location, and runtime-location settings; the final plan must enumerate each changed setting.

### Key Entities

- **Runtime Configuration**: The operator-supplied and defaulted settings required to safely start and operate the local service.
- **Runtime Prerequisite**: A machine capability, local path, executable, or storage location that must be available for a configured capability.

## Non-Goals

- Adding remote hosting, multi-tenant deployment, or public-network exposure.
- Redesigning authentication, conversation persistence, model selection UI, or chat protocol behavior.
- Changing model-generation algorithms or introducing new inference providers.
- Automatically installing missing external model runtimes or model weights during startup.

## Success Criteria *(mandatory)*

- **SC-001**: An operator on a supported local machine can start the service with no optional settings and reach a ready state on the first attempt.
- **SC-002**: For each invalid startup setting covered by the feature, the operator receives a specific corrective message before any user session can begin.
- **SC-003**: All documented supported startup settings have an observable default or validation outcome.
- **SC-004**: Existing supported local setups retain their current startup behavior after the feature is released.

## Assumptions and Open Questions

- **Assumption**: NodeMLX remains a local-first application for supported Apple Silicon macOS machines.
- **Assumption**: Optional external runtimes and model files are installed and maintained by the local operator rather than automatically provisioned by the service.
- **Assumption**: Existing environment-based configuration remains the operator interface unless the implementation plan identifies a compatibility-preserving improvement.

## Verification Plan

- **Automated**: Add focused startup/configuration tests and run `npm test`.
- **Build**: No web or native client build is required unless implementation changes those modules; run their applicable builds if scope expands.
- **Smoke test**: Start with defaults, a valid custom local location, and representative invalid resource/path settings; verify ready-state or actionable failure behavior for each.
