# -*- coding: utf-8 -*-
"""Local pandas/numpy automatic binning and information-value evaluation.

The implementation supports large-mass isolation, small-bin merging, Top-N Other,
missing-value bins, Laplace-smoothed WoE/IV, and approximate quantiles based on the
Greenwald-Khanna algorithm. It operates on an in-memory pandas DataFrame and has no
distributed-computing or graph-service dependency.

Public API::

    splitter = AutobinSplit(smoothing_factor=0.5)
    metrics, bins = splitter.split_evaluation(
        frame, feature_cols=["value"], label_col="label")

``metrics`` contains per-feature IV and summary statistics. ``bins`` contains bin
boundaries, sample counts, WoE, and IV contributions.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    ))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)
    logger.propagate = False


@dataclass
class BinDefNumeric:
    lower: Optional[float]
    upper: Optional[float]
    lower_inclusive: bool
    upper_inclusive: bool
    kind: str = "numeric"


@dataclass
class BinDefCategory:
    value: str
    kind: str = "categorical"


@dataclass
class BinDefMissing:
    kind: str = "missing"


@dataclass
class BinDefOther:
    merged_values: List[str] = field(default_factory=list)
    kind: str = "other"


BinDef = Union[BinDefNumeric, BinDefCategory, BinDefMissing, BinDefOther]
NUM_BINS_NUMERIC = 10
CAT_MIN_RATIO = 0.05
MIN_NUMERIC_BIN_RATIO = 0.01
LOW_CARDINALITY_THRESHOLD = 10
SPECIAL_MASS_RATIO = 0.5


@dataclass
class BinResult:
    bin_index: int
    label: str
    kind: str
    lower: Optional[float]
    upper: Optional[float]
    lower_inclusive: bool
    upper_inclusive: bool
    category_values: Optional[List[str]]
    count: int
    good: int
    bad: int
    event_rate: float
    woe: float
    iv: float
    iv_ratio: float = 0.0


@dataclass
class FeatureMetric:
    feature_name: str
    is_categorical: bool
    iv: float
    n_bins: int
    missing_rate: float
    min: Optional[float]
    max: Optional[float]
    mean: Optional[float]
    std: Optional[float]
    bin_ivs: Optional[List[float]] = None


def _build_initial_numeric_bins(quantiles: List[float]) -> List[BinDefNumeric]:
    bins = []  # type: List[BinDefNumeric]
    k = len(quantiles)
    if k == 0:
        bins.append(BinDefNumeric(lower=None, upper=None, lower_inclusive=False, upper_inclusive=False))
        return bins
    bins.append(BinDefNumeric(lower=None, upper=float(quantiles[0]), lower_inclusive=False, upper_inclusive=True))
    if k == 1:
        bins.append(BinDefNumeric(lower=float(quantiles[0]), upper=None, lower_inclusive=False, upper_inclusive=False))
        return bins
    bins.append(BinDefNumeric(lower=float(quantiles[0]), upper=float(quantiles[1]), lower_inclusive=False, upper_inclusive=False))
    for i in range(2, k):
        bins.append(BinDefNumeric(lower=float(quantiles[i - 1]), upper=float(quantiles[i]), lower_inclusive=True, upper_inclusive=False))
    bins.append(BinDefNumeric(lower=float(quantiles[k - 1]), upper=None, lower_inclusive=True, upper_inclusive=False))
    return bins


def _merge_small_numeric_bins(
    bins: List[BinDefNumeric], counts: List[int], min_ratio: float,
    reference_total: int = 0,
) -> List[BinDefNumeric]:
    total = sum(counts)
    if total == 0:
        return bins
    cur_bins = list(bins)  # type: List[BinDefNumeric]
    cur_counts = list(counts)  # type: List[int]
    ref = reference_total if reference_total > 0 else total
    threshold = ref * min_ratio
    while len(cur_bins) > 1:
        min_idx = -1
        min_cnt = None  # type: Optional[int]
        for i, c in enumerate(cur_counts):
            if c < threshold and (min_cnt is None or c < min_cnt):
                min_cnt = c
                min_idx = i
        if min_idx < 0:
            break
        if min_idx == len(cur_bins) - 1:
            left, right = min_idx - 1, min_idx
        else:
            left, right = min_idx, min_idx + 1
        merged = BinDefNumeric(
            lower=cur_bins[left].lower,
            upper=cur_bins[right].upper,
            lower_inclusive=cur_bins[left].lower_inclusive,
            upper_inclusive=cur_bins[right].upper_inclusive,
        )
        new_count = cur_counts[left] + cur_counts[right]
        cur_bins = cur_bins[:left] + [merged] + cur_bins[right + 1:]
        cur_counts = cur_counts[:left] + [new_count] + cur_counts[right + 1:]
    return cur_bins


def _format_bin_label(bin_def: dict) -> str:
    kind = bin_def.get("kind")
    if kind == "numeric":
        lo = bin_def.get("lower")
        up = bin_def.get("upper")
        left_b = "[" if bin_def.get("lower_inclusive") else "("
        right_b = "]" if bin_def.get("upper_inclusive") else ")"
        lo_s = "-inf" if lo is None else ("{:g}".format(lo))
        up_s = "+inf" if up is None else ("{:g}".format(up))
        return "{}{}, {}{}".format(left_b, lo_s, up_s, right_b)
    if kind == "categorical":
        return "={}".format(bin_def.get("value"))
    if kind == "missing":
        return "[Missing]"
    if kind == "other":
        vals = bin_def.get("merged_values", [])
        preview = ",".join(vals[:5])
        more = "..." if len(vals) > 5 else ""
        return "[Other: {}{}]".format(preview, more)
    return str(bin_def)


class _QuantileSummaries:
    defaultCompressThreshold = 10000
    defaultHeadSize = 50000

    def __init__(self, relative_error: float):
        self.relative_error = relative_error
        self.compress_threshold = self.defaultCompressThreshold
        self.sampled = []
        self.count = 0
        self.head_sampled = []  # type: List[float]

    def insert_all(self, values: List[float]) -> None:
        idx = 0
        m = len(values)
        while idx < m:
            space = self.defaultHeadSize - len(self.head_sampled)
            take = space if space < (m - idx) else (m - idx)
            self.head_sampled.extend(values[idx:idx + take])
            idx += take
            if len(self.head_sampled) >= self.defaultHeadSize:
                self._with_head_buffer_inserted()
                if len(self.sampled) >= self.compress_threshold:
                    self._compress()

    def _with_head_buffer_inserted(self) -> None:
        if not self.head_sampled:
            return
        current_count = self.count
        sorted_head = sorted(self.head_sampled)
        new_samples = []  # type: List[List[float]]
        sample_idx = 0
        ops_idx = 0
        n_ops = len(sorted_head)
        n_samp = len(self.sampled)
        while ops_idx < n_ops:
            current_sample = sorted_head[ops_idx]
            while sample_idx < n_samp and self.sampled[sample_idx][0] <= current_sample:
                new_samples.append(self.sampled[sample_idx])
                sample_idx += 1
            current_count += 1
            if len(new_samples) == 0 or (sample_idx == n_samp and ops_idx == n_ops - 1):
                delta = 0
            else:
                delta = int(math.floor(2 * self.relative_error * current_count))
            new_samples.append([current_sample, 1, delta])
            ops_idx += 1
        while sample_idx < n_samp:
            new_samples.append(self.sampled[sample_idx])
            sample_idx += 1
        self.sampled = new_samples
        self.count = current_count
        self.head_sampled = []

    def _compress(self) -> None:
        self._with_head_buffer_inserted()
        merge_threshold = 2 * self.relative_error * self.count
        self.sampled = self._compress_immut(self.sampled, merge_threshold)

    @staticmethod
    def _compress_immut(current_samples: List[List[float]], merge_threshold: float) -> List[List[float]]:
        if not current_samples:
            return []
        res_rev = []  # type: List[List[float]]
        head = current_samples[-1]
        i = len(current_samples) - 2
        while i >= 1:
            sample1 = current_samples[i]
            if sample1[1] + head[1] + head[2] < merge_threshold:
                head = [head[0], head[1] + sample1[1], head[2]]
            else:
                res_rev.append(head)
                head = sample1
            i -= 1
        res_rev.append(head)
        curr_head = current_samples[0]
        if curr_head[0] <= head[0] and len(current_samples) > 1:
            res_rev.append(current_samples[0])
        res_rev.reverse()
        return res_rev

    def query(self, quantile: float) -> Optional[float]:
        if self.head_sampled:
            self._compress()
        if not self.sampled:
            return None
        if quantile <= self.relative_error:
            return self.sampled[0][0]
        if quantile >= 1 - self.relative_error:
            return self.sampled[-1][0]
        rank = int(math.ceil(quantile * self.count))
        target_error = self.relative_error * self.count
        min_rank = 0
        i = 0
        n = len(self.sampled)
        while i < n - 1:
            cur_sample = self.sampled[i]
            min_rank += cur_sample[1]
            max_rank = min_rank + cur_sample[2]
            if max_rank - target_error <= rank <= min_rank + target_error:
                return cur_sample[0]
            i += 1
        return self.sampled[-1][0]


def _quantiles(arr: np.ndarray, probs: List[float], relative_error: float = 0.01) -> List[float]:
    if arr.size == 0 or len(probs) == 0:
        return []
    summary = _QuantileSummaries(relative_error)
    summary.insert_all([float(x) for x in arr.tolist()])
    summary._compress()
    out = []  # type: List[float]
    for p in probs:
        q = summary.query(p)
        if q is not None:
            out.append(float(q))
    return out


def _bin_indices(series: pd.Series, bin_defs: List[BinDef]) -> np.ndarray:
    n_rows = len(series)
    result = np.full(n_rows, -1, dtype=np.int64)
    if n_rows == 0:
        return result
    isnull = series.isna().to_numpy()
    notnull = ~isnull
    num = pd.to_numeric(series, errors="coerce").to_numpy()
    str_vals = series.astype(object)
    for idx, bd in enumerate(bin_defs):
        unassigned = result == -1
        if not unassigned.any():
            break
        if isinstance(bd, BinDefNumeric):
            cond = notnull.copy()
            if bd.lower is not None:
                cond &= (num >= bd.lower) if bd.lower_inclusive else (num > bd.lower)
            if bd.upper is not None:
                cond &= (num <= bd.upper) if bd.upper_inclusive else (num < bd.upper)
        elif isinstance(bd, BinDefCategory):
            target = bd.value
            eq = np.array(
                [(notnull[i] and str(str_vals.iat[i]) == target) for i in range(n_rows)],
                dtype=bool,
            )
            cond = eq
        elif isinstance(bd, BinDefMissing):
            cond = isnull
        elif isinstance(bd, BinDefOther):
            cond = notnull.copy()
        else:
            raise ValueError("Unknown BinDef kind: {}".format(bd))
        take = unassigned & cond
        if take.any():
            result[take] = idx
    return result


def _count_defs(series: pd.Series, bin_defs: List[BinDef]) -> List[int]:
    n = len(bin_defs)
    if n == 0:
        return []
    idxs = _bin_indices(series, bin_defs)
    valid = idxs[idxs >= 0]
    if valid.size == 0:
        return [0] * n
    bc = np.bincount(valid, minlength=n)
    return bc[:n].astype(int).tolist()


def _count_bins_with_label(
    series: pd.Series, labels: np.ndarray, bin_defs: List[BinDef],
) -> List[Tuple[int, int]]:
    n = len(bin_defs)
    if n == 0:
        return []
    idxs = _bin_indices(series, bin_defs)
    good = np.zeros(n, dtype=np.int64)
    bad = np.zeros(n, dtype=np.int64)
    good_mask = labels == 0
    bad_mask = labels == 1
    for i in range(n):
        in_bin = idxs == i
        good[i] = int(np.count_nonzero(in_bin & good_mask))
        bad[i] = int(np.count_nonzero(in_bin & bad_mask))
    return list(zip(good.tolist(), bad.tolist()))


class AutobinSplit:
    def __init__(
        self,
        special_mass_ratio: float = SPECIAL_MASS_RATIO,
        min_numeric_bin_ratio: float = MIN_NUMERIC_BIN_RATIO,
        num_bins_numeric: int = NUM_BINS_NUMERIC,
        low_cardinality_threshold: int = LOW_CARDINALITY_THRESHOLD,
        cat_min_ratio: float = CAT_MIN_RATIO,
        smoothing_factor: float = 0.5,
        parallelism: int = 8,
    ):
        self.special_mass_ratio = special_mass_ratio
        self.min_numeric_bin_ratio = min_numeric_bin_ratio
        self.num_bins_numeric = num_bins_numeric
        self.low_cardinality_threshold = low_cardinality_threshold
        self.cat_min_ratio = cat_min_ratio
        self.smoothing_factor = smoothing_factor
        self.parallelism = max(1, int(parallelism))

    def _derive_bin_defs(self, series: pd.Series, is_numeric: bool) -> List[BinDef]:
        non_null = series.dropna()
        has_null = bool(series.isna().any())
        bin_defs = []  # type: List[BinDef]
        if is_numeric:
            arr = pd.to_numeric(non_null, errors="coerce").dropna().to_numpy(dtype=float)
            probs = [i / self.num_bins_numeric for i in range(1, self.num_bins_numeric)]
            raw_qs = _quantiles(arr, probs)
            dominant_counter = Counter(raw_qs)
            dom_count = max(2, int(len(probs) * self.special_mass_ratio))
            mass_values = [float(v) for v, c in dominant_counter.items() if c >= dom_count]
            for mv in mass_values:
                bin_defs.append(BinDefNumeric(lower=mv, upper=mv, lower_inclusive=True, upper_inclusive=True))
            if mass_values:
                mask_rest = ~np.isin(arr, mass_values)
                rest_arr = arr[mask_rest]
            else:
                rest_arr = arr
            rest_count = int(rest_arr.size)

            if rest_count == 0:
                pass
            elif mass_values:
                max_mv = max(mass_values)
                rest_below = rest_arr[rest_arr < max_mv]
                rest_above = rest_arr[rest_arr > max_mv]
                below_cnt = int(rest_below.size)
                above_cnt = int(rest_above.size)
                def _bin_side(side_arr, side_count):
                    if side_count == 0:
                        return []
                    n_bins = max(2, int(self.num_bins_numeric * side_count / rest_count + 0.5))
                    side_probs = [i / n_bins for i in range(1, n_bins)]
                    side_qs = _quantiles(side_arr, side_probs)
                    side_edges = set(side_qs)
                    side_doms = [float(v) for v, c in Counter(side_qs).items() if c >= 2]
                    side_need = (n_bins - 1) - len(side_edges)
                    if side_doms and side_need > 0:
                        side_r2 = side_arr[~np.isin(side_arr, side_doms)]
                        if side_r2.size > 0:
                            extra_p = [(i + 1) / (side_need + 1) for i in range(side_need)]
                            side_edges.update(_quantiles(side_r2, extra_p))
                    sq = sorted(side_edges)
                    if not sq:
                        return []
                    side_bins = _build_initial_numeric_bins(sq)
                    counts_s = _count_defs(pd.Series(side_arr), side_bins)
                    return _merge_small_numeric_bins(side_bins, counts_s, self.min_numeric_bin_ratio)
                below_bins = _bin_side(rest_below, below_cnt)
                if below_bins and below_bins[-1].upper is None:
                    last = below_bins[-1]
                    below_bins[-1] = BinDefNumeric(
                        lower=last.lower, upper=max_mv,
                        lower_inclusive=last.lower_inclusive, upper_inclusive=False,
                    )
                above_bins = _bin_side(rest_above, above_cnt)
                if above_bins and above_bins[0].lower is None:
                    above_bins[0] = BinDefNumeric(
                        lower=max_mv, upper=above_bins[0].upper,
                        lower_inclusive=False, upper_inclusive=above_bins[0].upper_inclusive,
                    )
                bin_defs.extend(below_bins)
                bin_defs.extend(above_bins)
            else:
                rest_qs = _quantiles(rest_arr, probs)
                edges = set(rest_qs)
                dominants = [float(v) for v, c in Counter(rest_qs).items() if c >= 2]
                needed = (self.num_bins_numeric - 1) - len(edges)
                if dominants and needed > 0:
                    rest2 = rest_arr[~np.isin(rest_arr, dominants)]
                    if rest2.size > 0:
                        extra_probs = [(i + 1) / (needed + 1) for i in range(needed)]
                        edges.update(_quantiles(rest2, extra_probs))
                quantiles = sorted(edges)
                initial_bins = _build_initial_numeric_bins(quantiles)
                if len(quantiles) == 0:
                    bin_defs.extend(initial_bins)
                else:
                    counts = _count_defs(pd.Series(rest_arr), initial_bins)
                    bin_defs.extend(_merge_small_numeric_bins(initial_bins, counts, self.min_numeric_bin_ratio))
        else:
            value_counts = non_null.astype(object).map(lambda x: str(x)).value_counts()
            total = int(value_counts.sum())
            if total == 0:
                return [BinDefMissing()] if has_null else []
            threshold = total * self.cat_min_ratio
            kept = []  # type: List[Tuple[str, int]]
            merged = []  # type: List[Tuple[str, int]]
            for v, c in value_counts.items():
                (kept if int(c) >= threshold else merged).append((str(v), int(c)))
            for v, _c in kept:
                bin_defs.append(BinDefCategory(value=v))
            if merged:
                bin_defs.append(BinDefOther(merged_values=[v for v, _ in merged]))
        if has_null:
            bin_defs.append(BinDefMissing())
        return bin_defs

    def _iv_total(self, counts: List[Tuple[int, int]]) -> float:
        total_good = sum(g for g, _ in counts)
        total_bad = sum(b for _, b in counts)
        if total_good == 0 or total_bad == 0:
            return 0.0
        s = self.smoothing_factor
        n_bins = max(1, len(counts))
        iv = 0.0
        for g, b in counts:
            good_pct = (g + s) / (total_good + n_bins * s)
            bad_pct = (b + s) / (total_bad + n_bins * s)
            iv += (bad_pct - good_pct) * float(np.log(bad_pct / good_pct))
        return round(abs(iv), 4)

    def _bin_woe_iv(self, g: int, b: int, total_good: int, total_bad: int,
                    n_bins: int) -> Tuple[float, float]:
        if total_good == 0 or total_bad == 0:
            return 0.0, 0.0
        s = self.smoothing_factor
        n_bins = max(1, n_bins)
        good_pct = (g + s) / (total_good + n_bins * s)
        bad_pct = (b + s) / (total_bad + n_bins * s)
        woe = float(np.log(bad_pct / good_pct))
        iv = (bad_pct - good_pct) * woe
        return round(woe, 6), round(iv, 6)

    @staticmethod
    def _bin_def_to_result(
        idx: int, bd: BinDef, g: int, b: int, woe: float, iv: float,
        iv_ratio: float = 0.0,
    ) -> BinResult:
        n = g + b
        if isinstance(bd, BinDefNumeric):
            kind, lower, upper = "numeric", bd.lower, bd.upper
            li, ui = bd.lower_inclusive, bd.upper_inclusive
            cat_vs = None
        elif isinstance(bd, BinDefCategory):
            kind, lower, upper, li, ui = "categorical", None, None, False, False
            cat_vs = [bd.value]
        elif isinstance(bd, BinDefOther):
            kind, lower, upper, li, ui = "categorical", None, None, False, False
            cat_vs = list(bd.merged_values)
        else:  # BinDefMissing
            kind, lower, upper, li, ui = "missing", None, None, False, False
            cat_vs = None
        label = _format_bin_label(asdict(bd))
        return BinResult(
            bin_index=idx, label=label, kind=kind,
            lower=lower, upper=upper, lower_inclusive=li, upper_inclusive=ui,
            category_values=cat_vs,
            count=n, good=g, bad=b,
            event_rate=round(b / n, 6) if n > 0 else 0.0,
            woe=woe, iv=iv, iv_ratio=iv_ratio,
        )

    def split_evaluation(
        self,
        spark_dataframe,
        feature_cols: Optional[List[str]] = None,
        exclude_cols: Optional[List[str]] = None,
        label_col: str = "label",
    ) -> Tuple[Dict[str, FeatureMetric], Dict[str, List[BinResult]]]:
        df = spark_dataframe  # type: pd.DataFrame
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Local split_evaluation requires pandas.DataFrame, got: {}".format(type(df)))
        available_cols = list(df.columns)
        if label_col not in available_cols:
            raise ValueError("label_col is absent from DataFrame: {}".format(label_col))
        exclude_set = set(exclude_cols or [])
        reserved = {label_col} | exclude_set
        if feature_cols:
            feats = [c for c in feature_cols if c not in exclude_set]
            missing = [c for c in feats if c not in available_cols]
            if missing:
                raise ValueError("Requested feature columns are absent: {}".format(missing))
        else:
            feats = [c for c in available_cols if c not in reserved]
        if not feats:
            raise ValueError("No feature columns are available for evaluation")
        df = df[feats + [label_col]].copy()
        df = df[df[label_col].notna()]
        labels = pd.to_numeric(df[label_col], errors="coerce").to_numpy()
        total_n = len(df)
        numeric_candidates, categorical_features = [], []
        for fname in feats:
            if pd.api.types.is_numeric_dtype(df[fname]) and not pd.api.types.is_bool_dtype(df[fname]):
                numeric_candidates.append(fname)
            else:
                categorical_features.append(fname)
        numeric_features = []
        for c in numeric_candidates:
            ndv = int(df[c].dropna().nunique())
            (categorical_features if ndv <= self.low_cardinality_threshold
             else numeric_features).append(c)
        numeric_set = set(numeric_features)
        targets = [(c, c in numeric_set) for c in feats]
        stats = {}  # type: Dict[str, Dict[str, Optional[float]]]
        for c in feats:
            col = df[c]
            missing_v = float(col.isna().sum())
            entry = {"missing": missing_v}
            if c in numeric_set:
                num = pd.to_numeric(col, errors="coerce")
                entry["min"] = float(num.min()) if num.notna().any() else None
                entry["max"] = float(num.max()) if num.notna().any() else None
                entry["mean"] = float(num.mean()) if num.notna().any() else None
                std_v = num.std()
                entry["std"] = float(std_v) if pd.notna(std_v) else None
            stats[c] = entry

        def _stat(fname, suffix):
            v = stats.get(fname, {}).get(suffix)
            return None if v is None else float(v)

        metrics = {}  # type: Dict[str, FeatureMetric]
        bins = {}     # type: Dict[str, List[BinResult]]

        def _process_one(fname, is_numeric):
            bin_defs = self._derive_bin_defs(df[fname], is_numeric)
            if not bin_defs:
                return fname, [], []
            counts = _count_bins_with_label(df[fname], labels, bin_defs)
            return fname, bin_defs, counts

        results = []
        with ThreadPoolExecutor(max_workers=self.parallelism) as ex:
            futures = {ex.submit(_process_one, c, isn): c for c, isn in targets}
            for fut in as_completed(futures):
                results.append(fut.result())
        for fname, bin_defs, counts in results:
            is_cat = fname not in numeric_set
            missing_v = _stat(fname, "missing")
            missing_rate = round(missing_v / total_n, 6) if (missing_v is not None and total_n > 0) else 0.0
            fmin = _stat(fname, "min") if not is_cat else None
            fmax = _stat(fname, "max") if not is_cat else None
            fmean = _stat(fname, "mean") if not is_cat else None
            fstd = _stat(fname, "std") if not is_cat else None
            fmin = round(fmin, 6) if fmin is not None else None
            fmax = round(fmax, 6) if fmax is not None else None
            fmean = round(fmean, 6) if fmean is not None else None
            fstd = round(fstd, 6) if fstd is not None else None
            if not bin_defs:
                metrics[fname] = FeatureMetric(
                    feature_name=fname, is_categorical=is_cat,
                    iv=0.0, n_bins=0, missing_rate=missing_rate,
                    min=fmin, max=fmax, mean=fmean, std=fstd,
                    bin_ivs=[],
                )
                bins[fname] = []
                continue
            total_good = sum(g for g, _ in counts)
            total_bad = sum(b for _, b in counts)
            iv_total = self._iv_total(counts)
            bin_results = []
            for idx, (bd, (g, b)) in enumerate(zip(bin_defs, counts)):
                woe, iv_c = self._bin_woe_iv(g, b, total_good, total_bad, len(counts))
                iv_ratio = round(iv_c / iv_total, 6) if iv_total > 0 else 0.0
                bin_results.append(self._bin_def_to_result(idx, bd, g, b, woe, iv_c, iv_ratio))
            metrics[fname] = FeatureMetric(
                feature_name=fname, is_categorical=is_cat,
                iv=iv_total, n_bins=len(bin_defs), missing_rate=missing_rate,
                min=fmin, max=fmax, mean=fmean, std=fstd,
                bin_ivs=[b.iv for b in bin_results],
            )
            bins[fname] = bin_results
        return metrics, bins
