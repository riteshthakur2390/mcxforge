"""
llm_cost_tracker.py — SignalForge LLM Cost Monitor
====================================================
Tracks token usage and estimated cost per model across all LLM calls.
Logs to console + optional CSV. Check daily spend at any time.

Usage:
    from llm_cost_tracker import tracker
    tracker.record(response)          # pass LLMResponse from router
    tracker.print_summary()           # print current session totals
    tracker.daily_summary()           # grouped by model + task
"""

import csv
import os
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional

# Pricing per 1M tokens (USD) — update if pricing changes
PRICING = {
    "deepseek-chat": {
        "input":  0.28,
        "output": 0.42,
    },
    "claude-haiku-4-5-20251001": {
        "input":  0.25,
        "output": 1.25,
    },
}


@dataclass
class CallRecord:
    timestamp: str
    task_type: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


@dataclass
class CostTracker:
    records: List[CallRecord] = field(default_factory=list)
    log_file: Optional[str] = None  # set to a path to enable CSV logging

    def record(self, llm_response) -> float:
        """
        Record a completed LLM call from an LLMResponse object.
        Returns estimated cost in USD for this call.
        """
        model = llm_response.model_used
        in_tok = llm_response.input_tokens or 0
        out_tok = llm_response.output_tokens or 0

        pricing = PRICING.get(model, {"input": 0, "output": 0})
        cost = (in_tok * pricing["input"] + out_tok * pricing["output"]) / 1_000_000

        rec = CallRecord(
            timestamp=datetime.now().isoformat(),
            task_type=str(llm_response.task_type),
            model=model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost,
        )
        self.records.append(rec)

        if self.log_file:
            self._append_csv(rec)

        return cost

    def _append_csv(self, rec: CallRecord):
        file_exists = os.path.exists(self.log_file)
        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "task_type", "model", "input_tokens", "output_tokens", "cost_usd"])
            writer.writerow([rec.timestamp, rec.task_type, rec.model, rec.input_tokens, rec.output_tokens, f"{rec.cost_usd:.6f}"])

    def session_total(self) -> dict:
        total_cost = sum(r.cost_usd for r in self.records)
        by_model = defaultdict(lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost": 0.0})
        for r in self.records:
            by_model[r.model]["calls"] += 1
            by_model[r.model]["input_tokens"] += r.input_tokens
            by_model[r.model]["output_tokens"] += r.output_tokens
            by_model[r.model]["cost"] += r.cost_usd
        return {"total_cost": total_cost, "by_model": dict(by_model), "total_calls": len(self.records)}

    def print_summary(self):
        summary = self.session_total()
        print("\n" + "═" * 55)
        print("  SignalForge LLM Cost Summary (Session)")
        print("═" * 55)
        print(f"  Total Calls : {summary['total_calls']}")
        print(f"  Total Cost  : ${summary['total_cost']:.4f}")
        print("─" * 55)
        for model, data in summary["by_model"].items():
            short = model.split("-")[0].capitalize()
            print(f"  {short:<12} | {data['calls']:>4} calls | "
                  f"{data['input_tokens']:>7} in | {data['output_tokens']:>6} out | "
                  f"${data['cost']:.4f}")
        print("═" * 55 + "\n")

    def reset(self):
        self.records.clear()


# Singleton — import this across modules
tracker = CostTracker(log_file="llm_costs.csv")