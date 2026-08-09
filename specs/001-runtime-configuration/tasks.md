# Tasks: Runtime Configuration and Bootstrap

**Input**: Design documents from `/specs/001-runtime-configuration/`  
**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `quickstart.md`

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Tasks in different files with no prerequisite relationship.
- **[Story]**: `US1`, `US2`, and `US3` map to the prioritized user stories in `spec.md`.
- Backend configuration behavior is testable, so focused Node tests are required.

## Phase 1: Contract and Setup

- [X] T001 Record the no-protocol/no-persistence scope and the configuration compatibility matrix in `specs/001-runtime-configuration/plan.md`.
- [X] T002 [P] Document the effective startup defaults, variables, and local-only exposure guidance in `README.md`.
- [X] T003 [P] Extend focused startup-configuration coverage in `test/config.test.js` for defaults, malformed input, and bounded values.

## Phase 2: Foundational Backend and Runtime Work

- [X] T004 Implement explicit configuration parsing, validation, bounds, and safe diagnostic helpers in `src/config.js`.
- [X] T005 Integrate validated configuration and required-versus-optional readiness behavior into `src/bootstrap.js` without blocking Fastify or changing HTTP/WebSocket payloads.
- [X] T006 Add focused readiness tests in `test/config.test.js` or a new `test/runtime-configuration.test.js` without requiring installed model runtimes.

## Phase 3: User Story 1 - Start With Safe Local Defaults (Priority: P1)

**Goal**: An operator can start core NodeMLX on a supported local machine using documented safe defaults.  
**Independent Test**: Configuration construction from an empty environment returns documented effective defaults and a supported local bind address.

- [X] T010 [US1] Update `src/config.js` so empty optional configuration produces one validated, documented configuration object with a localhost default.
- [X] T011 [US1] Update `src/bootstrap.js` to consume the validated configuration before it assembles application resources and accepts traffic.
- [X] T012 [US1] Add default/local-binding assertions to `test/config.test.js`.
- [X] T013 [US1] Align `README.md` startup defaults with `src/config.js`, including the actual default port.

## Phase 4: User Story 2 - Correct Invalid Configuration Before Service Use (Priority: P2)

**Goal**: An operator receives an actionable correction for malformed configuration and safe clamping for finite out-of-range resource values.  
**Independent Test**: Representative malformed values return a setting-specific validation outcome; below/above-bound values return effective safe bounds and a warning result.

- [X] T020 [US2] Implement setting-specific malformed-value errors and warning metadata in `src/config.js`.
- [X] T021 [US2] Ensure `src/bootstrap.js` fails before Fastify listens when required configuration validation fails.
- [X] T022 [US2] Add malformed-port/resource and bound-clamping tests in `test/config.test.js`.
- [X] T023 [US2] Document accepted forms, bounds, clamping, and failure behavior in `README.md`.

## Phase 5: User Story 3 - Use Configured Local Model Resources (Priority: P3)

**Goal**: An operator can use supported custom local runtime locations, while an unavailable optional runtime disables only that capability with remediation guidance.  
**Independent Test**: A valid custom location is retained; an unavailable optional runtime yields a disabled-capability diagnostic without a core-startup-failure result.

- [X] T030 [US3] Add runtime-location normalization and optional-capability readiness helpers in `src/config.js` using the existing adapter contracts.
- [X] T031 [US3] Integrate optional capability diagnostics in `src/bootstrap.js` without auto-installing, silently falling back, or merging DiffusionKit/z-image environments.
- [X] T032 [US3] Add valid-location and unavailable-optional-capability tests in `test/config.test.js` or `test/runtime-configuration.test.js`.
- [X] T033 [US3] Document optional runtime readiness, remediation, and separate environment requirements in `README.md`.

## Phase 6: Verification and Handoff

- [X] T901 Run `npm test` from the repository root.
- [X] T902 Execute the default, malformed configuration, out-of-range numeric, and unavailable optional-runtime smoke scenarios from `specs/001-runtime-configuration/quickstart.md`.
- [X] T903 Record test/smoke evidence and unavailable runtime prerequisites in `specs/001-runtime-configuration/quickstart.md` or the implementation handoff.
- [X] T904 Run `git diff --check` and confirm no Angular, NativeMac, HTTP/WebSocket, SQLite, or Mongo changes were introduced.

## Dependencies and Parallelism

- T004 must complete before T005, T010, T020, or T030.
- T003/T006 may be prepared independently but their assertions must match the finalized configuration contract.
- User Story 1 is the MVP. User Story 2 depends on the same validation boundary. User Story 3 depends on the configuration normalization and existing runtime adapter contracts.
- README tasks can be drafted in parallel with tests once the configuration contract is fixed, but must be finalized after behavior is implemented.
- No story is complete until its focused tests and the final `npm test` pass.

## Implementation Strategy

1. Establish one configuration boundary and preserve existing supported defaults.
2. Make malformed and out-of-range behavior deterministic and independently tested.
3. Add optional runtime readiness without broadening runtime scope or touching client protocols.
4. Finish documentation and execute the bounded local smoke matrix.
