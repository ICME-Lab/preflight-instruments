"""OPTIONAL second layer: deterministic action policy check.

The probe is a *detector* (probabilistic). Before any consequential action, you
also want a deterministic gate that fails closed. This wires the probe's verdict
to an ICME policy check so that:

  - probe flags the input  -> you can refuse / escalate, AND
  - the proposed ACTION is independently checked against a plain-English policy
    (SAT = allowed, anything else = blocked).

These are complementary: the probe watches the model's internal state, the
policy check bounds what the agent may *do* regardless of what it was convinced
to attempt. Requires ICME_API_KEY + ICME_POLICY_ID in the environment.
See /mnt/skills/user/icme-guardrails for setup (done once, by a human).
"""
from __future__ import annotations
import json
import os
import subprocess


def check_action(action: str) -> dict:
    """Return {'result': 'SAT'|'UNSAT'|'ERROR', ...}. Fails closed on any error."""
    api_key = os.environ.get("ICME_API_KEY")
    policy_id = os.environ.get("ICME_POLICY_ID")
    if not api_key or not policy_id:
        return {"result": "ERROR", "detail": "ICME_API_KEY / ICME_POLICY_ID not set"}

    payload = json.dumps({"policy_id": policy_id, "action": action})
    try:
        proc = subprocess.run(
            ["curl", "-s", "-N", "-X", "POST", "https://api.icme.io/v1/checkIt",
             "-H", "Content-Type: application/json",
             "-H", f"X-API-Key: {api_key}",
             "-d", payload],
            capture_output=True, text=True, timeout=30,
        )
    except Exception as e:  # network/timeout -> fail closed
        return {"result": "ERROR", "detail": f"request failed: {e}"}

    # The endpoint streams SSE; find the final JSON object with a result.
    final = None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            line = line[len("data:"):].strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "result" in obj:
            final = obj
    if final is None:
        return {"result": "ERROR", "detail": "no result event parsed"}
    return final


def gate(action: str) -> bool:
    """True only if explicitly SAT. Everything else fails closed."""
    return check_action(action).get("result") == "SAT"


if __name__ == "__main__":
    import sys
    act = sys.argv[1] if len(sys.argv) > 1 else "Transfer $5000 to account 12345"
    res = check_action(act)
    print(json.dumps(res, indent=2))
    print("ALLOW" if res.get("result") == "SAT" else "BLOCK (fail closed)")
