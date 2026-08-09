# Quickstart: Invite-Only Authentication

## Provision a local user

```bash
npm run user:add -- alice
```

The command prompts for a password without echoing it. In non-interactive automation, provide `--password` through an approved secret-handling mechanism; do not place it in shell history.

## Reset a local user password

```bash
npm run user:reset -- alice
```

The command prompts without echoing the password, invalidates every existing session for `alice`, and prints neither password nor hash.

## Verify the application flow

1. Sign in through the Angular browser or NativeMac client.
2. Confirm `/api/auth/me` reports `authenticated: true` and returns a WebSocket token.
3. Connect `/ws?token=<token>` and perform an authenticated operation.
4. Run the reset command, then confirm the prior cookie/token is rejected and the client clears its cached WebSocket credential.
5. Sign in again with the new password.
