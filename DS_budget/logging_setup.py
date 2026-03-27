# -*- coding: utf-8 -*-
import logging
import os
import sys


# -------------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------------

log_dir      = os.getenv("EXPERIMENT_DIR", "DS_budget/logs/local")
base_log_dir = os.path.dirname(log_dir)
os.makedirs(log_dir, exist_ok=True)

_paths = {
    "bnb_status":    os.path.join(log_dir,      "bnb_status.log"),
    "bnb_details":   os.path.join(log_dir,      "bnb_nodes.log"),
    "cg_status":     os.path.join(log_dir,      "ccg_status.log"),
    "cg_details":    os.path.join(log_dir,      "ccg_iters.log"),
    "experiments":   os.path.join(base_log_dir, "experiments.csv"),
}
experiment_log_path = _paths["experiments"]

# -------------------------------------------------------------------------
# Formatters
# -------------------------------------------------------------------------

class ConditionalFormatter(logging.Formatter):
    """Plain messages for INFO; prefixes level name for WARNING and above."""
    def format(self, record):
        self._style._fmt = (
            "%(levelname)s: %(message)s" if record.levelno >= logging.WARNING
            else "%(message)s"
        )
        return super().format(record)


class FlushStreamHandler(logging.StreamHandler):
    """StreamHandler that flushes immediately after each emit."""
    def emit(self, record):
        super().emit(record)
        self.flush()


_plain     = logging.Formatter("%(message)s")
_status_fmt = ConditionalFormatter()


# -------------------------------------------------------------------------
# Logger factory
# -------------------------------------------------------------------------

def _make_logger(name: str, path: str, formatter: logging.Formatter,
                 mode: str = "w", console: bool = False) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(path, mode=mode)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if console:
        ch = FlushStreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(_status_fmt)
        logger.addHandler(ch)
    return logger


# -------------------------------------------------------------------------
# Public loggers
# -------------------------------------------------------------------------

BNB_status_logger = _make_logger("StatusLogger",      _paths["bnb_status"],  _status_fmt, console=True)
BNB_details_logger = _make_logger("NodeLogger",       _paths["bnb_details"], _plain)
CG_status_logger   = _make_logger("ColGenLogger",     _paths["cg_status"],   _status_fmt, console=True)
CG_details_logger  = _make_logger("ColGenDetailsLogger", _paths["cg_details"], _plain)
experiment_logger  = _make_logger("stat_logger",      _paths["experiments"], _plain, mode="a")