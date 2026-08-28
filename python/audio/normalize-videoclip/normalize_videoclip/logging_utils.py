"""Logging configuration for the CLI entry point.

The library only ever calls ``logger.debug/info/warning`` — it never prints.
This function is what turns that into visible output, and only the CLI
calls it (the MCP server configures its own quiet logging so protocol
framing on stdout is never polluted).
"""

from __future__ import annotations

import logging


def setup_logging(verbosity: int, quiet: bool) -> None:
    if quiet:
        level = logging.ERROR
    elif verbosity >= 2:
        level = logging.DEBUG
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
