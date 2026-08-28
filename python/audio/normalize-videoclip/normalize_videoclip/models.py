"""Plain data types shared by core, CLI, and MCP layers.

Everything here is a dataclass or enum with a ``to_dict`` method — no
behavior, no I/O. That's what makes it safe to hand these objects straight
to ``json.dumps`` for the ``--json`` CLI flag or an MCP tool response.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class TaskStatus(str, Enum):
    SKIPPED = "skipped"
    SUCCESS = "success"
    FAILED = "failed"
    DRY_RUN = "dry_run"


@dataclass(slots=True)
class VideoTask:
    source: Path
    destination: Path
    requires_reencode: bool  # True: legacy container, full re-encode. False: stream-copy + audio normalize.

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": str(self.source),
            "destination": str(self.destination),
            "requires_reencode": self.requires_reencode,
        }


@dataclass(slots=True)
class TaskResult:
    task: VideoTask
    status: TaskStatus
    message: str = ""
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.task.to_dict(),
            "status": self.status.value,
            "message": self.message,
            "duration_seconds": round(self.duration_seconds, 2),
        }


@dataclass(slots=True)
class NormalizeSummary:
    results: list[TaskResult] = field(default_factory=list)

    @property
    def succeeded(self) -> int:
        return sum(1 for r in self.results if r.status == TaskStatus.SUCCESS)

    @property
    def skipped(self) -> int:
        return sum(1 for r in self.results if r.status == TaskStatus.SKIPPED)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status == TaskStatus.FAILED)

    @property
    def dry_run_count(self) -> int:
        return sum(1 for r in self.results if r.status == TaskStatus.DRY_RUN)

    @property
    def exit_code(self) -> int:
        """0 if nothing failed, 1 otherwise. Suitable for sys.exit()."""
        return 1 if self.failed else 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": len(self.results),
            "succeeded": self.succeeded,
            "skipped": self.skipped,
            "failed": self.failed,
            "dry_run": self.dry_run_count,
            "results": [r.to_dict() for r in self.results],
        }
