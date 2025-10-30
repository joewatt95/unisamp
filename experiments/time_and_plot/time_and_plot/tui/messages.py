"""This module contains the message classes for the TUI."""

from typing import Any
from textual.message import Message


class JobUpdate(Message):
    """A message to update the TUI with job status."""

    def __init__(self, update: dict[str, Any]) -> None:
        self.update = update
        super().__init__()
