---
name: xiaomi-home
description: List and inspect Xiaomi smart home devices via the Mi Cloud API. Retrieves device tokens, local IPs, models and online status.
env:
  - XIAOMI_USERNAME
  - XIAOMI_PASSWORD
  - XIAOMI_COUNTRY
triggers:
  - xiaomi
  - mi home
  - smart home
  - devices
---

# Xiaomi Smart Home Skill

Connects to the Xiaomi Mi Cloud API to list devices on the account.

## Requirements

| Variable | Required | Default | Description |
|---|---|---|---|
| `XIAOMI_USERNAME` | ✅ | — | Mi account email or phone |
| `XIAOMI_PASSWORD` | ✅ | — | Mi account password |
| `XIAOMI_COUNTRY` | optional | `de` | Server region: `de` (Europe), `us`, `cn`, `sg`, `ru`, `tw`, `i2` |

## Usage

```bash
# List all devices (name, model, IP, token, online status)
venv/bin/python scripts/xiaomi_home.py devices

# Dump raw JSON from cloud
venv/bin/python scripts/xiaomi_home.py raw
```

## Workflow

When the user asks about Xiaomi/Mi Home devices:

1. Run `xiaomi_home.py devices`
2. Report device names, online status, and models
3. Include local IPs and tokens only if user explicitly asks (tokens are sensitive)

**Important:** Do not expose tokens in group chats or shared contexts. Report them only in direct messages with the owner.

## Notes

- Default server is `de` (Europe) — change `XIAOMI_COUNTRY` if devices are on a different regional account
- Tokens retrieved here are used by `python-miio` for local LAN control
- This skill does not control devices — it retrieves discovery info only
