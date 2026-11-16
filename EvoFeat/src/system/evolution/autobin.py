"""Compatibility exports for the shared FraudFeat AutobinSplit implementation.

EvoFeat uses the benchmark's binning, WoE, and IV definitions directly. This module
keeps the historical import path without maintaining a second algorithm copy.
"""

from fraudfeat.autobin import (
    CAT_MIN_RATIO,
    LOW_CARDINALITY_THRESHOLD,
    MIN_NUMERIC_BIN_RATIO,
    NUM_BINS_NUMERIC,
    SPECIAL_MASS_RATIO,
    AutobinSplit,
    BinDef,
    BinDefCategory,
    BinDefMissing,
    BinDefNumeric,
    BinDefOther,
    BinResult,
    FeatureMetric,
)

__all__ = [
    "AutobinSplit", "BinDef", "BinDefNumeric", "BinDefCategory",
    "BinDefMissing", "BinDefOther", "BinResult", "FeatureMetric",
    "NUM_BINS_NUMERIC", "CAT_MIN_RATIO", "MIN_NUMERIC_BIN_RATIO",
    "LOW_CARDINALITY_THRESHOLD", "SPECIAL_MASS_RATIO",
]
