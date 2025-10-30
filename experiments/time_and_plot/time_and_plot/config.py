"""This module contains the configuration data structure for the application."""

from dataclasses import dataclass
from typing import Optional, Sequence



@dataclass
class Config:
    """
    Holds the configuration for the application.
    """

    command: str
    values: Sequence[str]
    repetitions: int
    show_plot: bool
    output_plot: Optional[str]
    output_stats: Optional[str]
    no_tui: bool
    verbose: int
    parallel_workers: int = 1
