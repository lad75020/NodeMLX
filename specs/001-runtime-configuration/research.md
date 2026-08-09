# Research: Runtime Configuration and Bootstrap

## Decision: Validate all startup settings in one configuration boundary

**Rationale**: `src/config.js` already owns defaults and numeric clamping, while `server.js` intentionally initializes configuration before importing the application graph. Keeping parsing, validation, and messages together makes behavior testable without booting Fastify or model workers.

**Alternatives considered**:
- Rely solely on downstream Fastify and runtime errors — rejected because operators need corrective feedback before traffic is accepted.
- Spread validation through inference adapters — rejected because configuration rules would be duplicated and hard to test coherently.

## Decision: Clamp finite out-of-range resource numbers; reject missing or malformed required numbers

**Rationale**: This records clarification Q2 and preserves the existing bounded-generation design. Clamping keeps a local service usable while enforcing safe limits; malformed configuration cannot be interpreted safely and should fail early with a setting-specific message.

**Alternatives considered**:
- Fail startup for all out-of-range values — rejected by user clarification.
- Accept unsafe values unchanged — rejected because local resource exhaustion is a core risk.

## Decision: Treat optional runtime absence as capability unavailability

**Rationale**: This records clarification Q1. NodeMLX can still provide core chat where an optional image runtime is absent, but must communicate that the capability is disabled and must not silently choose a different runtime.

**Alternatives considered**:
- Fail entire startup — rejected by user clarification.
- Hide or ignore the missing runtime — rejected because it obscures operator remediation and causes misleading feature behavior.

## Decision: Preserve separate DiffusionKit and z-image environments

**Rationale**: The constitution and existing runtime resolution tests establish intentional separate virtual-environment behavior. Startup readiness checks must query existing resolver behavior rather than introduce a shared interpreter or automatic installation.

**Alternatives considered**:
- Merge Python environments — rejected as incompatible with repository governance and existing runtime design.
- Install missing dependencies at startup — rejected as unsafe, network-dependent, and outside local operator control.

## Decision: Fix documentation to match actual defaults

**Rationale**: The README currently states `PORT=3000`, while `createConfig` defaults to `18956`. A startup configuration feature must have one truthful operator contract.

**Alternatives considered**:
- Change code to port 3000 — deferred; preserve current implementation compatibility unless a deliberate port-policy change is requested.
- Leave documentation unchanged — rejected because it invalidates startup smoke expectations.
