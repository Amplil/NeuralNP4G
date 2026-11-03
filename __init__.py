"""Neural NP4G models, discrete program extraction, and verification."""

import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())

from .config import OFFICIAL_PROTOCOL_VERSION

__all__ = ["OFFICIAL_PROTOCOL_VERSION"]
