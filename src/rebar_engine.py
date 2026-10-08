# rebar_engine.py
#
# Detection engine for the live rebar/stirrup dashboard. Pure numpy/scipy,
# no GUI or hardware imports, so it can be unit-tested offline.
#
# It keeps the detection convention of your offline script (Rebar_detection.py:
# derivative level per component, prominence thresholds, RF sign factor) but
# fixes the things that limit accuracy and robustness in a live run:
#
#   1. Everything is in CENTIMETRES, not samples. Samples are binned onto a
#      uniform position grid (default 1 cm), so results no longer depend on
#      cart speed or BLE rate (your two runs were 7.8 Hz and 23 Hz).
#   2. Gaps / position jumps split the scan into independent segments. Nothing
#      is interpolated across a gap, so a startup jump can no longer create a
#      fake derivative spike (the false "rebar" at 132 cm in your screenshot).
#   3. Hampel filter removes single-sample spikes before differentiation.
#   4. Savitzky-Golay derivative (window given in cm) instead of np.gradient on
#      raw noisy samples.
#   5. Threshold = max(your fixed prominence threshold, k x measured noise),
#      so a noisy run cannot flood the display with false peaks. Channels in
#      `auto_threshold_channels` (the quantum sensor, whose unit is a.u.) use
#      the noise-based threshold only.
#   6. Position = centre of the magnetic bump (zero crossing between the two
#      derivative lobes), not the steepest-slope point. `locate="slope"`
#      restores the original behaviour.
#   7. Sensor fusion: a rebar is CONFIRMED when independent channels agree
#      within a tolerance; every rebar gets a confidence and an uncertainty.

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks, peak_widths, savgol_coeffs, savgol_filter

import Rebar_detection as rd

CHANNELS = ("lis2", "lis3", "quantum")
COMPONENTS = ("Bx", "By", "Bz", "B_total")


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
@dataclass
class EngineConfig:
    # -- method: "original" = your professor's finde_buegelpositionen, unchanged
    #            (raw samples, np.gradient, prominence, distance in SAMPLES, RF sign factor,
    #             position = positionen[peak]);  "robust" = cm-grid / smoothing / centre pipeline
    method: str = "original"
    min_spacing_samples: int = 20   # `distance` of find_peaks in original mode (Mindestbuegelabstand)
    # -- resampling / cleaning
    grid_cm: float = 1.0            # uniform position grid
    max_gap_cm: float = 12.0        # larger position gaps split the scan
    min_segment_cm: float = 20.0    # shorter isolated pieces are discarded
    hampel_cm: float = 7.0          # spike filter window
    hampel_sigma: float = 4.0
    smooth_cm: float = 15.0         # Savitzky-Golay window (derivative); ~ bump width
    min_valid_samples: int = 15
    # -- detection
    min_spacing_cm: float = 25.0    # min distance between two rebars (was: samples)
    sensitivity: float = 1.0        # multiplies all thresholds (lower = more sensitive)
    snr_min: float = 5.0            # noise-based threshold floor (x sigma of signal)
    snr_auto: float = 10.0          # threshold for auto channels (x sigma)
    auto_threshold_channels: tuple = ("quantum",)
    thr_scale: dict = field(default_factory=dict)   # channel -> multiplier for fixed thresholds
    rf_number: int = 1
    use_factors: bool = True
    edge_margin_cm: float = 8.0     # ignore peaks this close to a segment end
    lookahead_cm: float = 20.0      # wait for the trailing lobe before reporting (centre mode)
    # -- localisation
    locate: str = "center"          # "center" | "slope"
    max_lobe_cm: float = 30.0
    lobe_ratio: float = 0.3
    # -- fusion
    fuse_components: tuple = ("Bx",)
    fuse_channels: tuple = CHANNELS
    confirm_tol_cm: float = 4.0
    min_votes: int = 2
    offsets_cm: dict = field(default_factory=dict)  # channel -> cm added to its positions


# ------------------------------------------------------------------
# Result containers
# ------------------------------------------------------------------
@dataclass
class SourceResult:
    channel: str
    component: str
    level: int
    ok: bool = False
    x: np.ndarray | None = None       # grid positions
    y: np.ndarray | None = None       # cleaned values (NaN in unsampled gaps)
    sig: np.ndarray | None = None     # exact signal the peak finder searched
    dets: list = field(default_factory=list)
    noise_raw: float = 0.0            # noise sigma of the raw values (native unit)
    noise_sig: float = 0.0            # noise sigma of `sig`
    thr: float = 0.0
    xmin: float = float("nan")
    xmax: float = float("nan")
    polarity: float = 1.0
    n_samples: int = 0


@dataclass
class EngineResult:
    sources: dict
    rebars: list
    stats: dict
    summary: dict
    cfg: EngineConfig
    compute_ms: float = 0.0
    n_samples: int = 0
    wall_time: float = 0.0
    version: int = 0


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _odd(n, lo=5):
    n = max(int(round(n)), lo)
    return n if n % 2 else n + 1


def _runs(mask):
    """Contiguous True runs of a boolean array as (start, end_exclusive)."""
    if not mask.any():
        return []
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def _grid(pos, val, cfg):
    """Bin samples onto a uniform grid. Returns (x, y, runs, sampled_bins, sampled_vals)
    or None. y is linearly filled across gaps <= max_gap_cm and NaN elsewhere."""
    pos = np.asarray(pos, dtype=float)
    val = np.asarray(val, dtype=float)
    ok = np.isfinite(pos) & np.isfinite(val)
    if ok.sum() < cfg.min_valid_samples:
        return None
    pos, val = pos[ok], val[ok]
    g = cfg.grid_cm
    idx = np.round(pos / g).astype(np.int64)
    i0, i1 = int(idx.min()), int(idx.max())
    n = i1 - i0 + 1
    if n > 400_000 or n < 5:
        return None

    order = np.argsort(idx, kind="stable")
    idx_s, val_s = idx[order], val[order]
    uniq, start, counts = np.unique(idx_s, return_index=True, return_counts=True)
    vals = val_s[start].astype(float)
    for m in np.where(counts > 1)[0]:               # median per multi-sample bin (robust)
        vals[m] = np.median(val_s[start[m]:start[m] + counts[m]])

    xg = (i0 + np.arange(n)) * g
    y = np.interp(np.arange(n), uniq - i0, vals)
    big = np.where(np.diff(uniq) * g > cfg.max_gap_cm)[0]
    for b in big:                                   # never interpolate across a real gap
        y[uniq[b] - i0 + 1: uniq[b + 1] - i0] = np.nan
    runs = [(a, b) for a, b in _runs(np.isfinite(y)) if (b - a) * g >= cfg.min_segment_cm]
    if not runs:
        return None
    return xg, y, runs, uniq, vals


def _noise_from_second_diff(uniq, vals):
    """Robust noise sigma from second differences of neighbouring samples
    (removes slope; var of 2nd difference of white noise = 6 sigma^2)."""
    if len(vals) < 6:
        return 0.0
    ok = (uniq[2:] - uniq[:-2]) <= 4
    if ok.sum() < 5:
        return 0.0
    d2 = vals[2:] - 2 * vals[1:-1] + vals[:-2]
    d2 = d2[ok]
    return float(1.4826 * np.median(np.abs(d2 - np.median(d2))) / math.sqrt(6))


def _hampel(y, half, nsig):
    size = 2 * half + 1
    if len(y) < size:
        return y
    med = median_filter(y, size=size, mode="nearest")
    dev = np.abs(y - med)
    mad = 1.4826 * median_filter(dev, size=size, mode="nearest")
    floor = max(1e-12, 0.3 * float(np.median(mad)))
    bad = dev > nsig * np.maximum(mad, floor)
    if bad.any():
        y = y.copy()
        y[bad] = med[bad]
    return y


def _refine(x, s, i):
    """Parabolic sub-sample refinement of a maximum at index i."""
    if i <= 0 or i >= len(s) - 1:
        return float(x[i])
    y0, y1, y2 = s[i - 1], s[i], s[i + 1]
    den = y0 - 2 * y1 + y2
    if not np.isfinite(den) or den >= 0:
        return float(x[i])
    off = 0.5 * (y0 - y2) / den
    off = max(-1.0, min(1.0, off))
    return float(x[i] + off * (x[i + 1] - x[i]))


def _mutual_partner(s, i, pos_lobes, neg_lobes, max_pts, ratio):
    """Opposite-sign lobe that belongs to the same rebar as lobe i.
    Nearest strong opposite lobe within max_pts, accepted only if the pairing
    is mutual (that lobe's own nearest positive lobe is i). Prevents pairing
    with the trailing lobe of the NEXT rebar when the true partner is weak."""
    cand = [j for j in neg_lobes if abs(j - i) <= max_pts and s[j] <= -ratio * s[i]]
    if not cand:
        return None
    j = min(cand, key=lambda k: abs(k - i))
    if len(pos_lobes):
        nearest_pos = min(pos_lobes, key=lambda k: abs(k - j))
        if nearest_pos != i:
            return None
    return j


def _zero_cross(x, s, i, j):
    a, b = (i, j) if i < j else (j, i)
    seg = s[a:b + 1]
    neg = np.signbit(seg)
    ch = np.where(neg[:-1] != neg[1:])[0]
    if len(ch) == 0:
        return None
    k = ch[np.argmin(np.abs((a + ch + 0.5) - (i + j) / 2.0))]
    s0, s1 = seg[k], seg[k + 1]
    t = s0 / (s0 - s1) if s1 != s0 else 0.5
    return float(x[a + k] + t * (x[a + k + 1] - x[a + k]))


def _detect(x, sig, runs, thr, sigma_sig, cfg):
    g = cfg.grid_cm
    dist = max(1, int(round(cfg.min_spacing_cm / g)))
    margin = int(round(cfg.edge_margin_cm / g))
    look = int(round(cfg.lookahead_cm / g)) if cfg.locate == "center" else 0
    max_pts = int(round(cfg.max_lobe_cm / g))
    dets = []
    for a, b in runs:
        s = sig[a:b]
        xs = x[a:b]
        pk, props = find_peaks(s, prominence=thr, distance=dist)
        if len(pk) == 0:
            continue
        pos_lobes = pk                                           # only real detections compete
        neg_lobes = find_peaks(-s, prominence=thr * 0.5)[0]
        fw = peak_widths(s, pk, rel_height=0.5)[0] * g
        for k, i in enumerate(pk):
            i = int(i)
            if i < margin or i > len(s) - 1 - max(margin, look):
                continue
            x_edge = _refine(xs, s, i)
            xc, paired = x_edge, False
            if cfg.locate == "center":
                j = _mutual_partner(s, i, pos_lobes, neg_lobes, max_pts, cfg.lobe_ratio)
                if j is not None:
                    z = _zero_cross(xs, s, i, j)
                    if z is not None:
                        xc, paired = z, True
            prom = float(props["prominences"][k])
            snr = prom / sigma_sig if sigma_sig > 0 else 1e3
            snr = float(min(snr, 1e3))
            sigma_cm = float(math.hypot(fw[k] / (2.355 * max(snr, 1.0)), 0.3 * g))
            if cfg.locate == "center" and not paired:      # slope position is biased by ~half a lobe
                sigma_cm = float(math.hypot(sigma_cm, 0.5 * fw[k]))
            dets.append({
                "x": xc, "x_edge": x_edge, "paired": paired,
                "prom": prom, "amp": float(s[i]), "snr": snr,
                "fwhm_cm": float(fw[k]), "sigma_cm": sigma_cm,
                "y": float(np.interp(xc, xs, s)),
            })
    return dets


# ------------------------------------------------------------------
# One (channel, component) source
# ------------------------------------------------------------------
def _analyze_original(pos, val, *, channel, component, cfg):
    """Exactly the logic of finde_buegelpositionen():
       derivative level per component (Bx: np.gradient, others: raw), RF sign factor,
       scipy find_peaks(prominence=threshold, distance=Mindestbuegelabstand[samples]),
       rebar position = positionen[peak].
       Only adaptation for live data: samples are sorted by position and repeated
       positions are averaged (np.gradient needs strictly increasing x)."""
    level = rd.DEFAULT_COMPONENT_DERIVATIVE_LEVEL[component]
    res = SourceResult(channel=channel, component=component, level=level)
    pos = np.asarray(pos, dtype=float) + cfg.offsets_cm.get(channel, 0.0)
    val = np.asarray(val, dtype=float)
    ok = np.isfinite(pos) & np.isfinite(val)
    if ok.sum() < max(cfg.min_valid_samples, 5):
        return res
    p, v = rd.dedupe_and_sort(pos[ok], val[ok])
    if len(p) < 5:
        return res
    thr = rd.DEFAULT_THRESHOLDS[component] * cfg.sensitivity * cfg.thr_scale.get(channel, 1.0)
    factor = float(rd.calculate_factor(component, cfg.rf_number, level)) if cfg.use_factors else 1.0
    a = rd.analyze_rebar(p, v, derivative_level=level, threshold=thr,
                         min_spacing=max(1, int(cfg.min_spacing_samples)),
                         sign_factor=factor, use_factors=cfg.use_factors, refine=False)
    dx = float(np.median(np.diff(p))) or 1.0
    d2 = v[2:] - 2 * v[1:-1] + v[:-2]
    sigma_raw = float(1.4826 * np.median(np.abs(d2 - np.median(d2))) / math.sqrt(6))
    gain = {0: 1.0, 1: 1.0 / (math.sqrt(2) * dx), 2: math.sqrt(6) / (4 * dx * dx)}[min(level, 2)]
    noise_sig = sigma_raw * gain
    dets = []
    for j, i in enumerate(a["peak_idx"]):
        i = int(i)
        prom = float(a["prominence"][j])
        snr = float(min(prom / noise_sig, 1e3)) if noise_sig > 0 else 1e3
        dets.append({"x": float(p[i]), "x_edge": float(p[i]), "paired": False, "prom": prom,
                     "amp": float(a["signal"][i]), "snr": snr, "fwhm_cm": float("nan"),
                     "sigma_cm": 0.5 * dx,          # position is a sample: +-half the sample spacing
                     "y": float(a["signal"][i])})
    res.ok, res.x, res.y, res.sig = True, p, v, a["signal"]
    res.dets, res.thr = dets, float(thr)
    res.noise_raw, res.noise_sig = sigma_raw, noise_sig
    res.xmin, res.xmax = float(p[0]), float(p[-1])
    res.polarity, res.n_samples = factor, int(len(p))
    return res


def analyze_source(pos, val, *, channel, component, cfg):
    if cfg.method == "original":
        return _analyze_original(pos, val, channel=channel, component=component, cfg=cfg)
    level = rd.DEFAULT_COMPONENT_DERIVATIVE_LEVEL[component]
    res = SourceResult(channel=channel, component=component, level=level)
    prep = _grid(np.asarray(pos, dtype=float) + cfg.offsets_cm.get(channel, 0.0), val, cfg)
    if prep is None:
        return res
    xg, y, runs, uniq, vals = prep
    g = cfg.grid_cm

    w = _odd(cfg.smooth_cm / g)
    poly = min(3 if w >= 7 else 2, w - 1)
    hk = max(1, int(round(cfg.hampel_cm / g / 2)))

    res.noise_raw = _noise_from_second_diff(uniq, vals)
    gain = float(np.linalg.norm(savgol_coeffs(w, poly, deriv=level, delta=g)))
    res.noise_sig = res.noise_raw * gain

    y_clean = y.copy()
    d_all = np.full_like(y, np.nan)
    good_runs = []
    for a, b in runs:
        if b - a < w + 2:
            continue
        seg = _hampel(y[a:b], hk, cfg.hampel_sigma)
        y_clean[a:b] = seg
        d_all[a:b] = savgol_filter(seg, w, poly, deriv=level, delta=g, mode="interp")
        good_runs.append((a, b))
    if not good_runs:
        return res

    auto = channel in cfg.auto_threshold_channels
    thr_snr = (cfg.snr_auto if auto else cfg.snr_min) * res.noise_sig * cfg.sensitivity
    thr_fix = rd.DEFAULT_THRESHOLDS[component] * cfg.sensitivity * cfg.thr_scale.get(channel, 1.0)
    thr = thr_snr if auto else max(thr_fix, thr_snr)
    if not np.isfinite(thr) or thr <= 0:
        thr = max(thr_fix, 1e-12)

    if cfg.use_factors:
        candidates = [float(rd.calculate_factor(component, cfg.rf_number, level))]
    else:                                   # search both polarities, keep the stronger one
        candidates = [1.0, -1.0]
    best = None
    for sgn in candidates:
        sig = d_all * sgn
        dets = _detect(xg, sig, good_runs, thr, res.noise_sig, cfg)
        score = max((d["prom"] for d in dets), default=-1.0)
        if best is None or score > best[0]:
            best = (score, sgn, sig, dets)
    _, res.polarity, res.sig, res.dets = best

    res.ok = True
    res.x, res.y, res.thr = xg, y_clean, float(thr)
    ok = np.isfinite(res.sig)
    res.xmin, res.xmax = float(xg[ok].min()), float(xg[ok].max())
    res.n_samples = int(len(vals))
    return res


# ------------------------------------------------------------------
# Fusion across sources
# ------------------------------------------------------------------
def _covers(src, x, tol):
    return src.ok and (src.xmin - tol) <= x <= (src.xmax + tol)


def fuse(sources, cfg):
    keys = [(ch, comp) for ch in cfg.fuse_channels for comp in cfg.fuse_components
            if (ch, comp) in sources and sources[(ch, comp)].ok]
    items = []
    for k in keys:
        for d in sources[k].dets:
            items.append((d["x"], k, d))
    items.sort(key=lambda t: t[0])

    tol = cfg.confirm_tol_cm
    max_merge = 0.45 * cfg.min_spacing_cm

    def _stats(c):
        xs = np.array([m[0] for m in c])
        sg = np.maximum(np.array([m[2]["sigma_cm"] for m in c]), 0.3)
        w = 1.0 / sg ** 2
        return float(np.sum(xs * w) / np.sum(w)), float(1.0 / math.sqrt(np.sum(w)))

    clusters = []                        # each: list of (x, key, det)
    for it in items:
        placed = False
        if clusters:
            c = clusters[-1]
            mean, sig_c = _stats(c)
            # a slope-position (unpaired) detection is biased by ~half a lobe; its
            # inflated sigma lets it join the cluster of the correctly centred ones
            reach = min(max(tol, 2.5 * math.hypot(it[2]["sigma_cm"], sig_c)), max_merge)
            if abs(it[0] - mean) <= max(reach, tol):
                same = [m for m in c if m[1] == it[1]]
                if not same:
                    c.append(it)
                elif it[2]["snr"] > same[0][2]["snr"]:
                    c[c.index(same[0])] = it
                placed = True
        if not placed:
            clusters.append([it])

    rebars = []
    for c in clusters:
        xs = np.array([m[0] for m in c])
        snr = np.array([m[2]["snr"] for m in c])
        x, sig_comb = _stats(c)
        spread = float(np.std(xs)) if len(xs) > 1 else 0.0
        available = sum(1 for k in keys if _covers(sources[k], x, tol))
        votes = len(c)
        need = min(cfg.min_votes, max(available, 1))
        status = "confirmed" if votes >= need else "tentative"
        agree = votes / max(available, 1)
        snr_mean = float(np.mean(snr))
        conf = float(agree * (1.0 - math.exp(-min(snr_mean, 60.0) / 6.0)))
        rebars.append({
            "x": x, "sigma_cm": float(math.hypot(sig_comb, spread)),
            "votes": votes, "available": available, "status": status,
            "conf": round(conf, 3), "snr": snr_mean,
            "members": {f"{m[1][0]}/{m[1][1]}": float(m[0]) for m in c},
        })
    for i, r in enumerate(rebars):
        r["spacing_prev"] = (r["x"] - rebars[i - 1]["x"]) if i else None
    return rebars


def channel_stats(sources, rebars, cfg):
    comp = cfg.fuse_components[0]
    confirmed = [r for r in rebars if r["status"] == "confirmed"]
    out = {}
    for ch in CHANNELS:
        s = sources.get((ch, comp))
        if s is None or not s.ok:
            out[ch] = {"ok": False}
            continue
        inside = [r for r in confirmed if _covers(s, r["x"], cfg.confirm_tol_cm)]
        matched = 0
        for r in inside:
            if any(abs(d["x"] - r["x"]) <= cfg.confirm_tol_cm for d in s.dets):
                matched += 1
        snrs = [d["snr"] for d in s.dets]
        out[ch] = {
            "ok": True,
            "noise_raw": s.noise_raw, "noise_sig": s.noise_sig, "thr": s.thr,
            "n_det": len(s.dets), "n_expected": len(inside), "matched": matched,
            "recall": (matched / len(inside)) if inside else None,
            "snr_median": float(np.median(snrs)) if snrs else None,
            "sigma_cm": float(np.median([d["sigma_cm"] for d in s.dets])) if s.dets else None,
        }
    return out


def summarize(rebars, sources, cfg):
    conf = [r for r in rebars if r["status"] == "confirmed"]
    sp = np.diff([r["x"] for r in conf]) if len(conf) > 1 else np.array([])
    xs = [s.xmin for s in sources.values() if s.ok] + [s.xmax for s in sources.values() if s.ok]
    return {
        "n_confirmed": len(conf),
        "n_tentative": len(rebars) - len(conf),
        "spacing_median": float(np.median(sp)) if len(sp) else None,
        "spacing_std": float(np.std(sp)) if len(sp) > 1 else None,
        "conf_mean": float(np.mean([r["conf"] for r in conf])) if conf else None,
        "x_min": min(xs) if xs else None,
        "x_max": max(xs) if xs else None,
    }


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
def run_engine(snap, cfg, version=0):
    """snap: {"pos": array, "lis2": {"x","y","z"}, "lis3": {...}, "quantum": {...}}
    (quantum values are NaN where the sample was stale/absent)."""
    t0 = time.perf_counter()
    pos = np.asarray(snap["pos"], dtype=float)
    sources = {}
    for ch in CHANNELS:
        b = snap[ch]
        bx, by, bz = (np.asarray(b[a], dtype=float) for a in "xyz")
        bt = np.sqrt(bx ** 2 + by ** 2 + bz ** 2)
        for comp, v in zip(COMPONENTS, (bx, by, bz, bt)):
            sources[(ch, comp)] = analyze_source(pos, v, channel=ch, component=comp, cfg=cfg)
    rebars = fuse(sources, cfg)
    return EngineResult(
        sources=sources, rebars=rebars,
        stats=channel_stats(sources, rebars, cfg),
        summary=summarize(rebars, sources, cfg),
        cfg=cfg, compute_ms=(time.perf_counter() - t0) * 1000.0,
        n_samples=int(len(pos)), wall_time=time.time(), version=version,
    )
