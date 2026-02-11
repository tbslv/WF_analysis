"""
wf_analysis

Widefield analysis framework.

Submodules (motion, dff, roi, traces, movies, manual_roi) can be used
independently without importing the full pipeline.
"""

__all__ = [
    "run_full_pipeline",
    "load_config",
]

def run_full_pipeline(*args, **kwargs):
    from .pipeline import run_full_pipeline as _run
    return _run(*args, **kwargs)

def load_config(*args, **kwargs):
    from .pipeline import load_config as _load
    return _load(*args, **kwargs)
