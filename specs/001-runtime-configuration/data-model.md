# Data Model: Runtime Configuration and Bootstrap

## Persisted Data Impact

This feature does not add, remove, or alter persisted SQLite or Mongo records.

- **SQLite auth/session/model registry/application secrets**: unchanged; existing database and generated-secret behavior remain compatible.
- **Mongo conversations**: unchanged.
- **Migration/backfill**: none.

## Runtime-Only Entities

### Runtime Configuration

| Attribute | Purpose | Validation / lifecycle |
|---|---|---|
| setting name | Identifies an operator-facing setting | Included in any startup error or warning. |
| supplied value | Raw local environment/configuration value | Parsed once at process startup. |
| effective value | Defaulted, normalized, or clamped value | Passed to application assembly after validation. |
| source | Whether value is default or operator-supplied | Used for diagnostic clarity; never persisted as a secret. |

### Capability Readiness

| Attribute | Purpose | Validation / lifecycle |
|---|---|---|
| capability | Core chat or named optional inference capability | Evaluated during startup readiness. |
| state | available, unavailable, or required-prerequisite-failed | Determines whether the service can continue. |
| diagnostic | Safe corrective message for operators | Logged at startup; must not include credentials or sensitive local paths. |
| remediation | Operator action to restore capability | Documented and surfaced with unavailable optional capability. |

## State Rules

1. Required local Apple Silicon/node-mlx prerequisite failure stops startup.
2. Malformed required startup configuration stops startup before traffic is accepted.
3. Finite out-of-range resource values become documented safe bounds and produce a warning.
4. Optional runtime unavailability leaves core service available and marks only that capability unavailable.
5. No runtime configuration or readiness state is stored in user/session/chat persistence by this feature.
