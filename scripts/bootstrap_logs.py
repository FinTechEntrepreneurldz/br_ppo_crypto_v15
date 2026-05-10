#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
FILES = [
    LOGS / "portfolio" / "portfolio.csv",
    LOGS / "decisions" / "latest_decision.csv",
    LOGS / "decisions" / "decisions.csv",
    LOGS / "target_weights" / "latest_target_weights.csv",
    LOGS / "target_weights" / "target_weights.csv",
    LOGS / "positions" / "latest_positions.csv",
    LOGS / "orders" / "latest_planned_orders.csv",
    LOGS / "orders" / "latest_submitted_orders.csv",
    LOGS / "orders" / "submitted_orders.csv",
    LOGS / "health" / "signal_history.csv",
]
for path in FILES:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("")
health = LOGS / "health" / "health_status.json"
health.parent.mkdir(parents=True, exist_ok=True)
if not health.exists():
    health.write_text('{"overall_status":"bootstrap_pending"}\n')
print("bootstrapped logs")
