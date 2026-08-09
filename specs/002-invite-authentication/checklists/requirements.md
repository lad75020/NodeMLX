# Requirements Checklist: Invite-Only Authentication

- [ ] CLI-only provisioning remains; no registration, invitation code, or admin web UI is added. (FR-001)
- [ ] Password login issues the established HTTP cookie and WebSocket credential. (FR-002)
- [ ] Valid authenticated HTTP and WebSocket activity renews a seven-day inactivity lifetime; logout and expiry invalidate access. (FR-003)
- [ ] Local administrator CLI reset never prints secret values and atomically invalidates that user’s sessions. (FR-004)
- [ ] Login failures remain generic and do not reveal account existence. (FR-005)
- [ ] Angular and NativeMac restore, clear, and use authentication state consistently. (FR-006)
- [ ] API and OpenAPI docs describe public auth behavior.
- [ ] Focused Node tests and all applicable build verification pass.
