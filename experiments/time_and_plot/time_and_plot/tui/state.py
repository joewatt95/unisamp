"""This module contains the TuiState class."""

import queue
from collections import defaultdict
from typing import Any, DefaultDict

from ..config import Config


class TuiState:
    """
    Manages the state of the TUI application.
    """

    def __init__(self, config: Config):
        self.config = config
        self.job_id_to_value_index: dict[int, int] = {}
        self.value_statuses: list[dict[str, Any]] = []
        self.job_id_counter = 0
        self.status_queue: queue.Queue = queue.Queue()
        self.job_results_by_val: DefaultDict[str, list] = defaultdict(list)
        self.total_jobs = 0
        self.jobs_completed = 0

        self.keep_running = True

    def setup_jobs(self) -> None:
        """
        Sets up job data structures and initial TUI state.
        """
        for i, val in enumerate(self.config.values):
            self.value_statuses.append(
                {
                    "val": val,
                    "status": "Pending",
                    "reps_done": 0,
                    "reps_total": self.config.repetitions,
                    "timings": [],
                    "has_failed_reps": False,
                }
            )
            for _ in range(self.config.repetitions):
                self.job_id_to_value_index[self.job_id_counter] = i
                self.job_id_counter += 1
        self.total_jobs = self.job_id_counter