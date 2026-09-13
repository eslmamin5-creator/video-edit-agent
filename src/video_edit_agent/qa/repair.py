"""Bounded auto-repair loop (spec section 33): applies safe, deterministic
fixes for `auto_repairable=True` QAIssues and re-runs QA, capped at a small
number of iterations so a stubborn issue can never spin the pipeline forever.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from video_edit_agent.core.schemas import QAIssue, QAReport, QASeverity

MAX_REPAIR_ITERATIONS = 3


@dataclass
class RepairResult:
    report: QAReport
    iterations_used: int
    repairs_applied: list[str]


RepairFn = Callable[[QAIssue], bool]
"""A repair function attempts to fix one issue in place and returns True if
it applied a fix (the issue should then be re-checked), False if it could
not handle this issue category."""


def run_repair_loop(
    initial_report: QAReport,
    rerun_qa: Callable[[], QAReport],
    repair_handlers: dict[str, RepairFn],
) -> RepairResult:
    """Repeatedly applies matching repair handlers to auto-repairable issues
    and re-runs the full QA pass, stopping when either no repairable issues
    remain, no handler made progress, or `MAX_REPAIR_ITERATIONS` is reached."""
    report = initial_report
    repairs_applied: list[str] = []
    iterations = 0

    while iterations < MAX_REPAIR_ITERATIONS:
        repairable = [i for i in report.issues if i.auto_repairable]
        if not repairable:
            break

        made_progress = False
        for issue in repairable:
            handler = repair_handlers.get(issue.category)
            if handler is None:
                continue
            if handler(issue):
                repairs_applied.append(f"{issue.category}: {issue.message}")
                made_progress = True

        iterations += 1
        if not made_progress:
            break

        report = rerun_qa()

    return RepairResult(report=report, iterations_used=iterations, repairs_applied=repairs_applied)


def has_blocking_errors(report: QAReport) -> bool:
    return any(i.severity == QASeverity.ERROR and not i.auto_repairable for i in report.issues)
