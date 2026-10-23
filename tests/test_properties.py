"""Property tests for the metric helpers used by fast_eval.

Profiles (deadline disabled in both):
- ci, the default: 200 examples
- stress, HYPOTHESIS_PROFILE=stress: @settings(max_examples=2000, deadline=None)
"""
import math
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest
import torch
from hypothesis import example, given, settings
from hypothesis import strategies as st
from sklearn.metrics import f1_score, precision_score, recall_score

import eval_f1
from net import RINet_attention

# database=None: do not replay failures from another in-process run.
# derandomize: the same examples every process, so a 50-run loop is stable.
settings.register_profile(
    "ci", max_examples=200, deadline=None, database=None, derandomize=True
)
settings.register_profile(
    "stress", max_examples=2000, deadline=None, database=None, derandomize=True
)
_PROFILE = os.environ.get("HYPOTHESIS_PROFILE", "ci")
if _PROFILE not in ("ci", "stress"):
    _PROFILE = "ci"
settings.load_profile(_PROFILE)

# Stress budget is @settings(max_examples=2000, deadline=None). CI uses 200.
# Explicit settings() does not inherit the loaded profile, so repeat the flags.
if _PROFILE == "stress":
    prop_settings = settings(
        max_examples=2000, deadline=None, database=None, derandomize=True
    )
else:
    prop_settings = settings(
        max_examples=200, deadline=None, database=None, derandomize=True
    )

torch.manual_seed(0)
_SCORE_NET = RINet_attention().eval()


def _close(a, b, rtol=1e-5, atol=1e-8):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return False
    same_nan = np.isnan(a) & np.isnan(b)
    same_posinf = np.isposinf(a) & np.isposinf(b)
    same_neginf = np.isneginf(a) & np.isneginf(b)
    finite = np.isfinite(a) & np.isfinite(b)
    good = same_nan | same_posinf | same_neginf
    if np.any(finite):
        good = good | False
        finite_ok = np.zeros(a.shape, dtype=bool)
        finite_ok[finite] = np.isclose(a[finite], b[finite], rtol=rtol, atol=atol)
        good = good | finite_ok
    return bool(np.all(good))


def _make_desc(n, d, mode, rng):
    if mode == "zeros":
        arr = np.zeros((n, d), dtype=np.float32)
    elif mode == "nan":
        arr = np.full((n, d), np.nan, dtype=np.float32)
    elif mode == "inf":
        sign = 1.0 if rng.rand() < 0.5 else -1.0
        arr = np.full((n, d), sign * np.inf, dtype=np.float32)
    elif mode == "huge":
        val = float(rng.choice(np.array([1e20, -1e20, 1e38, -1e38], dtype=np.float64)))
        arr = np.full((n, d), val, dtype=np.float32)
    elif mode == "mixed":
        arr = rng.uniform(-10.0, 10.0, size=(n, d)).astype(np.float32)
        if n > 0 and d > 0:
            arr.reshape(-1)[0] = np.nan
    else:
        arr = rng.uniform(-10.0, 10.0, size=(n, d)).astype(np.float32)
    return arr


def _reference_inverse_squared_l2(desc1, desc2):
    """Row-wise 1 / ||a - b||^2 in Python floats, independent of eval_f1."""
    left = np.asarray(desc1, dtype=np.float64)
    right = np.asarray(desc2, dtype=np.float64)
    out = []
    for row_a, row_b in zip(left, right):
        acc = 0.0
        invalid = False
        for x, y in zip(row_a.tolist(), row_b.tolist()):
            if math.isnan(x) or math.isnan(y):
                invalid = True
                break
            delta = x - y
            acc += delta * delta
        if invalid or math.isnan(acc):
            out.append(float("nan"))
        elif acc == 0.0:
            out.append(float("inf"))
        else:
            out.append(1.0 / acc)
    return np.asarray(out, dtype=np.float64)


def _reference_max_inverse_squared_l2(desc1, desc2_list):
    per_view = [_reference_inverse_squared_l2(desc1, desc2) for desc2 in desc2_list]
    stacked = np.stack(per_view, axis=1)
    best = []
    for row in stacked:
        finite = [float(v) for v in row.tolist() if not math.isnan(float(v))]
        best.append(max(finite) if finite else float("nan"))
    return np.nan_to_num(np.asarray(best, dtype=np.float64))


def _reference_f1(precision, recall):
    """Harmonic mean in Python floats. 0/0 and NaN become 0 via nan_to_num."""
    ps = np.asarray(precision, dtype=np.float64).reshape(-1)
    rs = np.asarray(recall, dtype=np.float64).reshape(-1)
    vals = []
    for p, r in zip(ps.tolist(), rs.tolist()):
        denom = p + r
        vals.append(float("nan") if denom == 0.0 else (2.0 * p * r) / denom)
    return np.nan_to_num(np.asarray(vals, dtype=np.float64))


def _binary_f1(y_true, pred):
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    pred = np.asarray(pred).astype(int).reshape(-1)
    tp = int(np.sum((y_true == 1) & (pred == 1)))
    fp = int(np.sum((y_true == 0) & (pred == 1)))
    fn = int(np.sum((y_true == 1) & (pred == 0)))
    if tp == 0:
        return 0.0
    prec = tp / float(tp + fp)
    rec = tp / float(tp + fn)
    if prec + rec == 0.0:
        return 0.0
    return 2.0 * prec * rec / (prec + rec)


def _reference_pr_max_f1(y_true, y_score):
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    best = 0.0
    thresholds = [float(v) for v in np.unique(y_score).tolist()]
    thresholds.append(float(np.min(y_score)) - 1.0)
    for threshold in thresholds:
        score = _binary_f1(y_true, y_score >= threshold)
        best = max(best, score)
    return best


@prop_settings
@given(
    n=st.integers(min_value=0, max_value=5),
    d=st.integers(min_value=1, max_value=5),
    mode=st.sampled_from(["moderate", "zeros", "nan", "inf", "huge", "mixed"]),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@example(n=1, d=4, mode="zeros", seed=0)
@example(n=1, d=4, mode="nan", seed=0)
@example(n=1, d=4, mode="inf", seed=0)
@example(n=1, d=4, mode="huge", seed=0)
@example(n=0, d=3, mode="zeros", seed=0)
@example(n=2, d=1, mode="moderate", seed=1)
@example(n=3, d=4, mode="mixed", seed=2)
def test_inverse_squared_l2_matches_reference(n, d, mode, seed):
    rng = np.random.RandomState(seed)
    desc1 = _make_desc(n, d, mode, rng)
    desc2 = _make_desc(n, d, mode, rng)
    got = eval_f1.inverse_squared_l2(desc1, desc2)
    ref = _reference_inverse_squared_l2(desc1, desc2)
    assert got.shape == (n,)
    assert _close(got, ref, rtol=1e-9, atol=1e-9)
    swapped = eval_f1.inverse_squared_l2(desc2, desc1)
    assert _close(got, swapped, rtol=1e-9, atol=1e-9)


@prop_settings
@given(
    n=st.integers(min_value=0, max_value=5),
    d=st.integers(min_value=1, max_value=5),
    n_views=st.integers(min_value=1, max_value=4),
    mode=st.sampled_from(
        ["moderate", "zeros", "nan", "inf", "huge", "mixed", "one_nan_view"]
    ),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@example(n=1, d=4, n_views=2, mode="zeros", seed=0)
@example(n=1, d=4, n_views=2, mode="nan", seed=0)
@example(n=1, d=4, n_views=2, mode="inf", seed=0)
@example(n=1, d=4, n_views=2, mode="huge", seed=0)
@example(n=0, d=3, n_views=2, mode="zeros", seed=0)
@example(n=2, d=1, n_views=1, mode="moderate", seed=1)
@example(n=3, d=4, n_views=3, mode="one_nan_view", seed=2)
@example(n=2, d=3, n_views=2, mode="mixed", seed=3)
def test_max_inverse_squared_l2_invariants(n, d, n_views, mode, seed):
    rng = np.random.RandomState(seed)
    base_mode = "moderate" if mode == "one_nan_view" else mode
    desc1 = _make_desc(n, d, base_mode, rng)
    views = [_make_desc(n, d, base_mode, rng) for _ in range(n_views)]
    if mode == "one_nan_view":
        views[0] = np.full((n, d), np.nan, dtype=np.float32)
    if mode == "zeros":
        # Identical descriptors: every view is a perfect match.
        views = [desc1.copy() for _ in range(n_views)]

    got = eval_f1.max_inverse_squared_l2(desc1, views)
    ref = _reference_max_inverse_squared_l2(desc1, views)
    assert got.shape == (n,)
    assert np.all(np.isfinite(got))
    assert np.all(got >= 0)
    assert _close(got, ref, rtol=1e-9, atol=1e-9)

    reversed_views = list(reversed(views))
    assert _close(got, eval_f1.max_inverse_squared_l2(desc1, reversed_views))
    first_only = eval_f1.max_inverse_squared_l2(desc1, views[:1])
    assert np.all(got >= first_only)
    assert _close(
        eval_f1.max_inverse_squared_l2(desc1, [views[0]]),
        eval_f1.max_inverse_squared_l2(views[0], [desc1]),
    )

    if mode == "zeros" and n > 0:
        assert np.all(got == np.finfo(np.float64).max)
    if mode == "one_nan_view" and n_views >= 2:
        without_nan = eval_f1.max_inverse_squared_l2(desc1, views[1:])
        assert _close(got, without_nan)
    if mode == "moderate" and n > 0:
        scaled = eval_f1.max_inverse_squared_l2(
            desc1 * 2.0, [view * 2.0 for view in views]
        )
        mask = (got > 0) & (got < 1e300)
        if np.any(mask):
            assert np.allclose(scaled[mask], got[mask] / 4.0, rtol=1e-6, atol=1e-8)


def test_inverse_squared_l2_preserves_float64_inputs():
    """Float64 inputs must not be rounded to float32 before the sum."""
    desc1 = np.array([[0.1, 0.2], [1.0, -1.0]], dtype=np.float64)
    desc2 = np.array([[0.3, 0.4], [0.5, 0.25]], dtype=np.float64)
    got = eval_f1.inverse_squared_l2(desc1, desc2)
    ref = _reference_inverse_squared_l2(desc1, desc2)
    assert _close(got, ref, rtol=1e-12, atol=0.0)


def test_f1_preserves_float64_inputs():
    """Casting only recall to float32 still changes the harmonic mean."""
    precision = np.array([0.1, 1.0 / 3.0, 0.2], dtype=np.float64)
    recall = np.array([0.3, 0.7, 1.0 / 7.0], dtype=np.float64)
    got = eval_f1.f1_from_precision_recall(precision, recall)
    ref = _reference_f1(precision, recall)
    assert got.dtype == np.float64
    assert _close(got, ref, rtol=1e-12, atol=0.0)


def test_max_inverse_squared_l2_empty_view_list_raises():
    with pytest.raises(ValueError):
        eval_f1.max_inverse_squared_l2(np.zeros((2, 3), dtype=np.float32), [])


@prop_settings
@given(
    n=st.integers(min_value=0, max_value=8),
    mode=st.sampled_from(["unit", "zeros", "nan", "inf", "mixed"]),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@example(n=0, mode="unit", seed=0)
@example(n=1, mode="zeros", seed=0)
@example(n=1, mode="nan", seed=0)
@example(n=1, mode="inf", seed=0)
@example(n=4, mode="unit", seed=1)
def test_f1_from_precision_recall_invariants(n, mode, seed):
    rng = np.random.RandomState(seed)
    if mode == "zeros":
        precision = np.zeros(n)
        recall = np.zeros(n)
    elif mode == "nan":
        precision = np.full(n, np.nan)
        recall = np.full(n, np.nan)
    elif mode == "inf":
        precision = np.full(n, np.inf)
        recall = rng.uniform(0.0, 1.0, size=n)
    elif mode == "mixed":
        precision = rng.uniform(0.0, 1.0, size=n)
        recall = rng.uniform(0.0, 1.0, size=n)
        if n > 0:
            precision[0] = np.nan
            recall[-1] = 0.0
    else:
        precision = rng.uniform(0.0, 1.0, size=n)
        recall = rng.uniform(0.0, 1.0, size=n)

    got = eval_f1.f1_from_precision_recall(precision, recall)
    ref = _reference_f1(precision, recall)
    assert got.shape == (n,)
    assert got.dtype == np.float64
    assert _close(got, ref)
    swapped = eval_f1.f1_from_precision_recall(recall, precision)
    if mode != "inf":
        # 2*p*r/(p+r) is symmetric. inf * 0 is NaN, so inf mode is not.
        assert _close(got, swapped)
    if mode in ("unit", "zeros", "nan", "mixed"):
        assert np.all(np.isfinite(got))
        assert np.all((got >= 0.0) & (got <= 1.0))
    if n == 0:
        with pytest.raises(ValueError):
            eval_f1.max_f1_from_pr(precision, recall)
    else:
        assert float(eval_f1.max_f1_from_pr(precision, recall)) == pytest.approx(
            float(np.max(ref))
        )


@prop_settings
@given(
    n=st.integers(min_value=1, max_value=10),
    kind=st.sampled_from(["mixed", "all0", "all1", "one"]),
    pred_kind=st.sampled_from(["match", "flip", "zeros", "ones", "random"]),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@example(n=1, kind="all0", pred_kind="zeros", seed=0)
@example(n=1, kind="all1", pred_kind="ones", seed=0)
@example(n=1, kind="one", pred_kind="match", seed=0)
@example(n=4, kind="all0", pred_kind="ones", seed=1)
@example(n=4, kind="all1", pred_kind="zeros", seed=1)
@example(n=5, kind="mixed", pred_kind="random", seed=2)
def test_f1_matches_sklearn(n, kind, pred_kind, seed):
    rng = np.random.RandomState(seed)
    if kind == "all0":
        y_true = np.zeros(n, dtype=int)
    elif kind == "all1":
        y_true = np.ones(n, dtype=int)
    elif kind == "one":
        y_true = np.zeros(n, dtype=int)
        y_true[0] = 1
    else:
        y_true = rng.randint(0, 2, size=n).astype(int)
        if np.unique(y_true).size < 2 and n >= 2:
            y_true[0] = 0
            y_true[1] = 1
    if pred_kind == "match":
        pred = y_true.copy()
    elif pred_kind == "flip":
        pred = 1 - y_true
    elif pred_kind == "zeros":
        pred = np.zeros(n, dtype=int)
    elif pred_kind == "ones":
        pred = np.ones(n, dtype=int)
    else:
        pred = rng.randint(0, 2, size=n).astype(int)

    precision = precision_score(y_true, pred, zero_division=np.nan)
    recall = recall_score(y_true, pred, zero_division=np.nan)
    got = eval_f1.f1_from_precision_recall([precision], [recall])[0]
    expect = f1_score(y_true, pred, zero_division=0)
    manual = _binary_f1(y_true, pred)
    assert got == pytest.approx(float(expect), abs=1e-9)
    assert got == pytest.approx(manual, abs=1e-9)
    assert 0.0 <= float(got) <= 1.0


@prop_settings
@given(
    n=st.integers(min_value=1, max_value=12),
    kind=st.sampled_from(["mixed", "all0", "all1", "one"]),
    score_kind=st.sampled_from(["normal", "zeros", "ties", "huge", "negative"]),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@example(n=1, kind="all0", score_kind="zeros", seed=0)
@example(n=1, kind="all1", score_kind="normal", seed=0)
@example(n=1, kind="one", score_kind="huge", seed=0)
@example(n=6, kind="mixed", score_kind="ties", seed=1)
@example(n=5, kind="mixed", score_kind="negative", seed=2)
@example(n=4, kind="all0", score_kind="huge", seed=3)
def test_pr_max_f1_matches_threshold_sweep(n, kind, score_kind, seed):
    rng = np.random.RandomState(seed)
    if kind == "all0":
        y_true = np.zeros(n, dtype=int)
    elif kind == "all1":
        y_true = np.ones(n, dtype=int)
    elif kind == "one":
        y_true = np.zeros(n, dtype=int)
        y_true[0] = 1
    else:
        y_true = rng.randint(0, 2, size=n).astype(int)
    if score_kind == "zeros":
        scores = np.zeros(n, dtype=np.float64)
    elif score_kind == "ties":
        scores = np.full(n, 0.3, dtype=np.float64)
    elif score_kind == "huge":
        scores = rng.uniform(-1e6, 1e6, size=n).astype(np.float64)
        scores[0] = 1e308
    elif score_kind == "negative":
        scores = rng.uniform(-100.0, -0.1, size=n).astype(np.float64)
    else:
        scores = rng.normal(size=n).astype(np.float64)

    got = float(eval_f1.pr_max_f1(y_true, scores))
    manual = _reference_pr_max_f1(y_true, scores)
    thresholds = [float(v) for v in np.unique(scores).tolist()]
    thresholds.append(float(np.min(scores)) - 1.0)
    best_sklearn = 0.0
    for threshold in thresholds:
        pred = (scores >= threshold).astype(int)
        best_sklearn = max(
            best_sklearn, float(f1_score(y_true, pred, zero_division=0))
        )
    assert 0.0 <= got <= 1.0
    assert got == pytest.approx(manual, abs=1e-8)
    assert got == pytest.approx(best_sklearn, abs=1e-8)
    if kind == "all1":
        assert got == pytest.approx(1.0)
    if kind == "all0":
        assert got == pytest.approx(0.0)


def test_pr_max_f1_rejects_empty_nan_and_inf():
    with pytest.raises(ValueError):
        eval_f1.pr_max_f1(np.array([]), np.array([]))
    with pytest.raises(ValueError):
        eval_f1.pr_max_f1(np.array([0, 1]), np.array([np.nan, 0.2]))
    with pytest.raises(ValueError):
        eval_f1.pr_max_f1(np.array([0, 1]), np.array([np.inf, 0.2]))


@prop_settings
@given(
    n=st.integers(min_value=0, max_value=8),
    n_views=st.integers(min_value=1, max_value=5),
    mode=st.sampled_from(["normal", "zeros", "huge"]),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@example(n=0, n_views=2, mode="zeros", seed=0)
@example(n=1, n_views=1, mode="zeros", seed=0)
@example(n=4, n_views=3, mode="huge", seed=1)
@example(n=3, n_views=8, mode="normal", seed=2)
def test_max_view_scores_matches_reference(n, n_views, mode, seed):
    rng = np.random.RandomState(seed)
    cols = []
    for _ in range(n_views):
        if mode == "zeros":
            cols.append(np.zeros(n, dtype=np.float64))
        elif mode == "huge":
            cols.append(np.full(n, 1e308 if rng.rand() < 0.5 else -1e308))
        else:
            cols.append(rng.normal(size=n))
    got = eval_f1.max_view_scores(cols)
    assert got.shape == (n,)
    if n == 0:
        return
    ref = np.array([max(float(col[i]) for col in cols) for i in range(n)])
    assert np.allclose(got, ref)
    assert np.allclose(got, eval_f1.max_view_scores(list(reversed(cols))))
    assert np.all(got >= eval_f1.max_view_scores(cols[:1]))


def _score_batch(n, mode, rng):
    if mode == "zeros":
        base = np.zeros((n, 288), dtype=np.float32)
    elif mode == "nan":
        base = np.full((n, 288), np.nan, dtype=np.float32)
    elif mode == "inf":
        sign = 1.0 if rng.rand() < 0.5 else -1.0
        base = np.full((n, 288), sign * np.inf, dtype=np.float32)
    elif mode == "huge":
        base = np.full((n, 288), np.float32(1e20), dtype=np.float32)
    else:
        base = rng.normal(0.0, 0.5, size=(n, 288)).astype(np.float32)
    other = rng.normal(0.0, 0.5, size=(n, 288)).astype(np.float32)
    if mode in ("zeros", "nan", "inf", "huge"):
        other = base.copy()
    return torch.from_numpy(base), torch.from_numpy(other)


@prop_settings
@given(
    n=st.integers(min_value=1, max_value=4),
    mode=st.sampled_from(["finite", "zeros", "nan", "inf", "huge"]),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@example(n=1, mode="zeros", seed=0)
@example(n=1, mode="nan", seed=0)
@example(n=1, mode="inf", seed=0)
@example(n=1, mode="huge", seed=0)
@example(n=2, mode="finite", seed=1)
def test_gen_score_range_symmetry_and_norm(n, mode, seed):
    rng = np.random.RandomState(seed)
    fx, fy = _score_batch(n, mode, rng)
    net = _SCORE_NET
    try:
        net.eval()
        with torch.no_grad():
            prob, norm = net.gen_score(fx, fy)
            prob_swap, norm_swap = net.gen_score(fy, fx)
            net.train()
            raw, norm_raw = net.gen_score(fx, fy)
        prob_np = prob.detach().cpu().numpy()
        in_range = np.isnan(prob_np) | ((prob_np >= 0.0) & (prob_np <= 1.0))
        assert np.all(in_range)
        assert _close(prob_np, prob_swap.detach().cpu().numpy(), rtol=1e-5, atol=1e-5)
        assert _close(
            norm.detach().cpu().numpy(),
            norm_swap.detach().cpu().numpy(),
            rtol=1e-5,
            atol=1e-5,
        )
        expected = torch.sigmoid(raw).detach().cpu().numpy()
        assert _close(prob_np, expected, rtol=1e-5, atol=1e-5)
        diff = (fx - fy).detach().cpu().numpy()
        ref_norm = np.sqrt(np.sum(np.float64(diff) * np.float64(diff), axis=1))
        assert _close(norm.detach().cpu().numpy(), ref_norm, rtol=1e-4, atol=1e-5)
        assert _close(norm.detach().cpu().numpy(), norm_raw.detach().cpu().numpy())
        if mode == "zeros":
            assert np.all(norm.detach().cpu().numpy() == 0.0)
            assert np.all((prob_np >= 0.0) & (prob_np <= 1.0))
    finally:
        net.eval()


def test_gen_score_empty_batch():
    net = _SCORE_NET
    empty = torch.zeros(0, 288)
    net.eval()
    with torch.no_grad():
        prob, norm = net.gen_score(empty, empty)
    assert prob.shape == (0,)
    assert norm.shape == (0,)
