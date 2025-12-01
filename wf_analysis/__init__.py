"""
wf_analysis
===========

Pipeline for widefield imaging analysis.

Typical usage from Python:

    from wf_analysis import run_full_pipeline

    run_full_pipeline("path/to/config.yaml")

Or from the CLI (see scripts/run_pipeline.py):

    python scripts/run_pipeline.py -c path/to/config.yaml
"""

from .pipeline import run_full_pipeline, load_config  # re-export main API

__all__ = [
    "run_full_pipeline",
    "load_config",
]

# Optional simple version – update manually or via packaging tools
__version__ = "0.1.0"
