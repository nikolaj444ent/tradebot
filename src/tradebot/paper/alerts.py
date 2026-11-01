"""Reaching a human.

DESIGN.md section 6: the failure that actually costs money is the process
dying silently while holding a position. This module is the "reach a human"
half of that defence (the heartbeat file is the other half).

Deliberately tiny: stderr always, plus an optional webhook POST if
``TRADELAB_ALERT_WEBHOOK`` is set (Slack/Discord-style ``{"text": ...}``
payload). An alert failing must never take the daemon down with it.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request


def send_alert(subject: str, body: str = "") -> None:
    print(f"[ALERT] {subject}" + (f"\n{body}" if body else ""), file=sys.stderr)
    url = os.environ.get("TRADELAB_ALERT_WEBHOOK")
    if not url:
        return
    try:
        payload = json.dumps({"text": f"{subject}\n{body}".strip()}).encode()
        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception as exc:  # the alert is best-effort by design
        print(f"[ALERT] webhook delivery failed: {exc}", file=sys.stderr)
