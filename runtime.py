"""Local output locations and opt-in logging for command-line runners.

Importing the package never creates directories or configures application
logging. Commands call :func:`configure_logging` explicitly; library users can
configure the ``official_neural_np4g`` logger through standard Python logging.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import re
import sys
import threading


_PACKAGE_LOGGER = "official_neural_np4g"
_CONFIGURATION_LOCK = threading.RLock()
_MAX_LOG_BYTES = 5 * 1024 * 1024
_LOG_BACKUPS = 3


def default_output_root() -> Path:
    """Return the local output base without creating it.

    ``NP4G_OUTPUT_DIR`` overrides the base. Source checkouts use their sibling
    ``local/official_neural_np4g`` directory. Installed packages use the user's
    platform state directory, keeping all generated files outside the package.
    """

    override = os.environ.get("NP4G_OUTPUT_DIR")
    if override:
        return Path(override).expanduser().resolve()
    workspace = Path(__file__).resolve().parent.parent
    if (workspace / "pyproject.toml").is_file() or (workspace / "AGENTS.md").is_file():
        return workspace / "local" / "official_neural_np4g"
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state")
    return (base / "official_neural_np4g").expanduser().resolve()


def default_artifact_root(protocol: str | None = None) -> Path:
    """Return the artifact directory, optionally separated by protocol version."""

    destination = default_output_root() / "artifacts"
    if protocol is not None:
        if not protocol or Path(protocol).name != protocol or protocol in (".", ".."):
            raise ValueError("protocol must be a non-empty directory name")
        destination /= protocol
    return destination


def configure_logging(
    command: str,
    log_dir: Path | str | None = None,
    level: int | str = logging.INFO,
) -> Path:
    """Enable bounded UTF-8 logs and stderr diagnostics for this package only.

    Each command keeps one active log up to 5 MiB and three rotated backups.
    Repeated calls reuse the current handlers. Changing commands or destinations
    closes and replaces only the handlers created by this function. Root and
    third-party loggers are unaffected. The returned path names the active log.
    """

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", command):
        raise ValueError("command must be a simple log-file name")
    # Validate before creating directories or changing logger configuration.
    if isinstance(level, str):
        selected_level = logging.getLevelName(level)
        if not isinstance(selected_level, int):
            raise ValueError(f"unknown logging level: {level!r}")
    elif isinstance(level, int):
        selected_level = level
    else:
        raise TypeError("level must be an integer or logging level name")
    directory = (
        default_output_root() / "logs" if log_dir is None else Path(log_dir).expanduser().resolve()
    )
    log_path = directory / f"{command}.log"
    with _CONFIGURATION_LOCK:
        logger = logging.getLogger(_PACKAGE_LOGGER)
        owned = [handler for handler in logger.handlers if getattr(handler, "_np4g_owned", False)]
        reusable = (
            len(owned) == 2
            and any(
                isinstance(handler, RotatingFileHandler)
                and Path(handler.baseFilename) == log_path
                for handler in owned
            )
            and any(
                type(handler) is logging.StreamHandler and handler.stream is sys.stderr
                for handler in owned
            )
        )
        if not reusable:
            directory.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                log_path,
                maxBytes=_MAX_LOG_BYTES,
                backupCount=_LOG_BACKUPS,
                encoding="utf-8",
            )
            console_handler = logging.StreamHandler(sys.stderr)
            formatter = logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s: %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S%z",
            )
            replacement = [file_handler, console_handler]
            for handler in replacement:
                handler.setFormatter(formatter)
                handler._np4g_owned = True  # type: ignore[attr-defined]
            for handler in owned:
                logger.removeHandler(handler)
                handler.close()
            for handler in replacement:
                logger.addHandler(handler)
            owned = replacement
        for handler in owned:
            handler.setLevel(selected_level)
        logger.setLevel(selected_level)
        logger.propagate = False
    return log_path


__all__ = ["configure_logging", "default_artifact_root", "default_output_root"]
