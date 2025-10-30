"""This module contains the configuration data structure for the application."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """
    Holds the configuration for the application.
    """

    command: str
    values: list[str]
    repetitions: int
    parallel_workers: Optional[int]
    show_plot: bool
    output_plot: Optional[str]
    output_stats: Optional[str]
    no_tui: bool
    verbose: int
