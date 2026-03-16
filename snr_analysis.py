"""
snr_analysis.py — Compatibility shim for poc.py.

The canonical implementation lives in theory/snr_analysis.py.
This module re-exports both public functions so that the validated
poc.py script can import them with a simple top-level import:

    from snr_analysis import compute_attack_snr_upper_bound, compute_attack_snr_tight
"""

from theory.snr_analysis import (  # noqa: F401
    compute_attack_snr_upper_bound,
    compute_attack_snr_tight,
)
