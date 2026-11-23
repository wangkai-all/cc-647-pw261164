"""Ticket occurrence volume estimation for operations reporting.

Run ``python -m ticket_ops.server --data data/tickets.csv`` and open the page.
"""

from .data import DataError, Dataset, load_dataset
from .model import (
    DEFAULT_DRAWS,
    DEFAULT_SMOOTH,
    DEFAULT_WINDOW,
    ChannelEstimate,
    DayEstimate,
    DelayProfile,
    aggregate,
    estimate_channel,
    estimate_dataset,
    replay,
    train_profile,
)

__all__ = [
    "DataError",
    "Dataset",
    "load_dataset",
    "DEFAULT_DRAWS",
    "DEFAULT_SMOOTH",
    "DEFAULT_WINDOW",
    "ChannelEstimate",
    "DayEstimate",
    "DelayProfile",
    "aggregate",
    "estimate_channel",
    "estimate_dataset",
    "replay",
    "train_profile",
]
