# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 1.x | ✅ |

## Reporting a vulnerability

**Do not open a public issue for security problems.**

Email **moh123mon456@gmail.com** with:

- Description and potential impact
- Reproduction steps / proof of concept
- Affected module (`src/*.py`, web console, agent)
- Version or commit hash

| Commitment | |
|---|---|
| Acknowledgement | within 72 hours |
| Initial assessment | within 7 days |
| Fix for critical issues | within 30 days |

## Scope

Testing is allowed **only against a local copy on your own machine**. The Flask
console (`src/webapp.py`) binds to localhost by default and has **no
authentication layer** — it is a local demo console, not a hardened service.
Do not expose it to the internet.

## By-design behaviour (not vulnerabilities)

| Behaviour | Why |
|---|---|
| Web console has no auth | It is a local demo/teaching console, bound to localhost |
| Models load from `models/` at import | They ship in-repo so the console works without retraining |
| Datasets auto-download on first retrain | CICIDS2017 (~300 MB) and Malimg (~1.2 GB) go to gitignored caches, with a documented offline fallback |
| Some numbers are reported as "unknown" | No metric is published before ground truth is filled |

## Secrets

- No API keys, tokens, or `.env` files are part of this repository.
- Datasets are public research corpora (CICIDS2017, Malimg, UCI SMS Spam).
