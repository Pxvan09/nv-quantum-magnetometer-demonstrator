# ui_common.py -- small helpers shared by the engineer and visitor pages.

from __future__ import annotations

import numpy as np
from nicegui import ui

COLORS = {"x": "#38bdf8", "y": "#fb923c", "z": "#4ade80"}
C_QUANTUM = "#22d3ee"      # quantum sensor
C_CONV = "#a78bfa"         # conventional (LIS) sensor
C_REBAR = "#fb923c"        # detected steel
C_OK = "#4ade80"
C_WARN = "#fbbf24"
C_BAD = "#f87171"

FONT_SANS = "'Inter', 'Segoe UI', system-ui, -apple-system, Roboto, Arial, sans-serif"
FONT_MONO = "'JetBrains Mono', Consolas, 'SF Mono', Menlo, monospace"

HEAD_HTML = f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet">
<style>
  body {{ background: radial-gradient(circle at 10% 0%, #131a2e 0%, #0b0f1a 60%); font-family: {FONT_SANS}; }}
  .dash-card {{ background: rgba(255,255,255,0.035); border: 1px solid rgba(255,255,255,0.08);
               border-radius: 18px; box-shadow: 0 10px 30px rgba(0,0,0,0.35); backdrop-filter: blur(8px); }}
  .kpi-value {{ font-family: {FONT_MONO}; font-weight: 700; font-size: 1.9rem; letter-spacing: -0.5px; }}
  .kpi-label {{ color: #94a3b8; font-size: 0.75rem; letter-spacing: 1px; text-transform: uppercase; }}
  .status-dot {{ width: 9px; height: 9px; border-radius: 50%; display: inline-block; margin-right: 6px; }}
  .status-dot.ok {{ background: #22d3ee; box-shadow: 0 0 8px #22d3ee; }}
  .status-dot.bad {{ background: #f87171; box-shadow: 0 0 8px #f87171; }}
  .status-dot.warn {{ background: #fbbf24; box-shadow: 0 0 8px #fbbf24; }}
  .status-pill {{ background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.08);
                 border-radius: 999px; padding: 4px 12px; font-size: 0.8rem; color: #cbd5e1; }}
  .glow-title {{ background: linear-gradient(90deg, #38bdf8, #a78bfa); -webkit-background-clip: text;
                background-clip: text; color: transparent; font-weight: 700; }}
</style>
"""


def html(content: str = "", **_):
    """ui.html across NiceGUI versions (3.x wants an explicit `sanitize`)."""
    try:
        return ui.html(content, sanitize=False)
    except TypeError:
        return ui.html(content)


def status_dot(ok: bool, warn: bool = False) -> str:
    cls = "warn" if warn else ("ok" if ok else "bad")
    return f'<span class="status-dot {cls}"></span>'


def chart_points(x, y, max_pts=1500):
    """[[x, y], ...] for ECharts; NaN -> None (breaks the line); downsampled."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    stride = max(1, int(np.ceil(n / max_pts)))
    out = []
    for a, b in zip(x[::stride], y[::stride]):
        out.append([float(a), None if not np.isfinite(b) else float(b)])
    return out


def moving_average(values, window):
    """Centered moving average with edge padding (no zero-padding droop at the ends)."""
    arr = np.asarray(values, dtype=float)
    if window <= 1 or len(arr) < 3:
        return arr
    w = min(int(window), len(arr))
    if w % 2 == 0:
        w -= 1
    if w <= 1:
        return arr
    pad = w // 2
    padded = np.pad(arr, pad, mode="edge")
    return np.convolve(padded, np.ones(w) / w, mode="valid")


def download_file(path: str):
    try:
        ui.download.file(path)
    except AttributeError:
        ui.download(path)
