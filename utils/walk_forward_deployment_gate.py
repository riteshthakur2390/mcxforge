"""
utils/walk_forward_deployment_gate.py
=====================================
Deployment guard for config/model changes.

It evaluates the latest walk-forward/out-of-sample artifact and blocks hot
reload deployment when the latest result does not meet minimum quality gates.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


@dataclass
class GateResult:
    allowed: bool
    reason: str
    metrics: dict[str, Any]
    artifact: str


class WalkForwardDeploymentGate:
    """Blocks live config/model deployment unless OOS metrics pass thresholds."""

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self.base_dir = Path(base_dir or Path(__file__).resolve().parent.parent)
        self.enabled = str(os.getenv("WALK_FORWARD_GATE_ENABLED", "true")).lower() not in {"0", "false", "no"}
        self.min_sharpe = float(os.getenv("WF_MIN_SHARPE", "0.80"))
        self.max_drawdown_pct = float(os.getenv("WF_MAX_DRAWDOWN_PCT", "12.0"))
        self.min_trades = int(os.getenv("WF_MIN_TRADES", "20"))
        self.min_profit_factor = float(os.getenv("WF_MIN_PROFIT_FACTOR", "1.10"))
        self.min_win_rate = float(os.getenv("WF_MIN_WIN_RATE", "45.0"))

    def evaluate_latest(self) -> GateResult:
        if not self.enabled:
            return GateResult(True, "walk_forward_gate_disabled", {}, "")

        artifact = self._latest_artifact()
        if artifact is None:
            return GateResult(False, "no_walk_forward_artifact_found", {}, "")

        try:
            data = json.loads(artifact.read_text())
        except Exception as exc:
            return GateResult(False, f"artifact_read_failed:{exc}", {}, str(artifact))

        metrics = self._extract_metrics(data)
        failures = self._failures(metrics)
        if failures:
            return GateResult(False, ";".join(failures), metrics, str(artifact))
        return GateResult(True, "walk_forward_oos_passed", metrics, str(artifact))

    def _latest_artifact(self) -> Optional[Path]:
        patterns = [
            "backtesting/results/walk_forward*.json",
            "journal/walk_forward*.json",
            "ml/reports/walk_forward*.json",
            "reports/walk_forward*.json",
        ]
        candidates: list[Path] = []
        for pattern in patterns:
            candidates.extend(self.base_dir.glob(pattern))
        candidates = [p for p in candidates if p.is_file()]
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.stat().st_mtime)

    @staticmethod
    def _extract_metrics(data: dict[str, Any]) -> dict[str, Any]:
        src = data.get("oos") or data.get("out_of_sample") or data.get("summary") or data
        return {
            "sharpe": float(src.get("sharpe", src.get("sharpe_ratio", 0.0)) or 0.0),
            "max_drawdown_pct": abs(float(src.get("max_drawdown_pct", src.get("drawdown_pct", 999.0)) or 999.0)),
            "trades": int(src.get("trades", src.get("trade_count", src.get("n_trades", 0))) or 0),
            "profit_factor": float(src.get("profit_factor", 0.0) or 0.0),
            "win_rate": float(src.get("win_rate", src.get("win_rate_pct", 0.0)) or 0.0),
        }

    def _failures(self, metrics: dict[str, Any]) -> list[str]:
        failures = []
        if float(metrics.get("sharpe", 0.0)) < self.min_sharpe:
            failures.append(f"sharpe<{self.min_sharpe}")
        if float(metrics.get("max_drawdown_pct", 999.0)) > self.max_drawdown_pct:
            failures.append(f"drawdown>{self.max_drawdown_pct}")
        if int(metrics.get("trades", 0)) < self.min_trades:
            failures.append(f"trades<{self.min_trades}")
        if float(metrics.get("profit_factor", 0.0)) < self.min_profit_factor:
            failures.append(f"profit_factor<{self.min_profit_factor}")
        if float(metrics.get("win_rate", 0.0)) < self.min_win_rate:
            failures.append(f"win_rate<{self.min_win_rate}")
        if failures:
            logger.warning(f"[WalkForwardDeploymentGate] Blocked deployment | {failures} | metrics={metrics}")
        return failures
