# Data Model: Invite-Only Authentication

## Existing SQLite tables used

### `users`

| Column | Feature use |
|---|---|
| `id` | Stable session owner and JWT subject. |
| `username` | CLI-provisioned, case-insensitive unique sign-in identifier. |
| `password_hash` | scrypt salt/hash value; replaced on administrator reset. |
| `last_login_at` | Updated only after successful password login. |

### `sessions`

| Column | Feature use |
|---|---|
| `id` | Opaque HTTP cookie value and JWT session identifier. |
| `user_id` | Target of transactional revocation on reset. |
| `expires_at` | Seven-day rolling inactivity deadline. |
| `last_seen_at` | Audit/activity timestamp updated with expiry renewal. |

## Compatibility and migration

No schema migration is required. Existing rows remain readable. Sessions created before deployment keep their recorded expiry until they next authenticate successfully; thereafter their expiry is renewed under the seven-day policy. Password reset is an atomic `UPDATE users` plus `DELETE sessions WHERE user_id = ?` transaction. A failed update rolls back the deletion.
