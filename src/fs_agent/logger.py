"""Shared logging helpers."""

from __future__ import annotations

import logging
from functools import lru_cache

from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install as install_rich_traceback

_console = Console()
install_rich_traceback(show_locals=False)


def _configure_root_logger(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=[RichHandler(console=_console, rich_tracebacks=True)],
    )


@lru_cache(maxsize=1)
def _bootstrap_logging() -> None:
    _configure_root_logger()


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger configured with Rich handlers."""

    _bootstrap_logging()
    return logging.getLogger(name)
