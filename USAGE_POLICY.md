# Usage Policy

## In one line

SentinelAI is a **defensive detection and research platform**. Use it on data
and systems you own or are explicitly authorized to analyze.

## Allowed

| Use | |
|---|---|
| Security research and academic work | ✅ |
| Building/benchmarking detection models on public corpora | ✅ |
| Internal SOC tooling prototype on your own telemetry | ✅ |
| Teaching and lab exercises | ✅ |

## Prohibited

| Use | |
|---|---|
| Processing third-party traffic without authorization | ❌ |
| Deploying the demo console on a public network | ❌ |
| Using the response policy to take destructive actions on production systems without human approval | ❌ |
| Republishing client telemetry or incident data | ❌ |

## Notes on the response agent

The reinforcement-learning policy proposes an action (block / isolate / monitor).
**It is a recommendation engine, not an autonomous actuator.** Any real action must
pass a human decision and an approved change process.

## Data handling

- Training corpora in this repo are **public research datasets** — no private data.
- Processed datasets and trained models are committed so the demo runs instantly;
  they contain no personal information.
- Do not feed production packet captures into this repo.

## Disclaimer

Provided "as is" under the MIT License, without warranty of any kind. The author is
not liable for misuse or for any direct or indirect damage. **Legal compliance is
the responsibility of the user alone.**
