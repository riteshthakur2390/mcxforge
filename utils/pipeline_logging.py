from __future__ import annotations

from loguru import logger


def log_pipeline_stage(
    agent: str,
    stage: str,
    status: str,
    *,
    reason: str = "",
    **values,
) -> None:
    parts = [
        f"[{agent}]",
        f"stage={stage}",
        f"status={status}",
    ]
    if reason:
        parts.append(f"reason={reason}")
    for key, value in values.items():
        if value is None or value == "":
            continue
        parts.append(f"{key}={value}")
    logger.bind(pipeline_trace=True).info(" | ".join(parts))
