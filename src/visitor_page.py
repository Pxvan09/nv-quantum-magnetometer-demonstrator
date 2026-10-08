# visitor_page.py
#
# Big-screen page for the exhibition:  http://localhost:8080/visitor
# Same live data as the engineer page (shared AppState / engine), but built to be
# understood in five seconds by someone walking past:
#   * a concrete beam "X-ray": steel bars appear at the exact spot where they are found
#   * four big numbers (bars found, spacing, scanned length, confidence)
#   * quantum sensor vs conventional sensor, side by side, on the same scan
#   * one-sentence explanations that rotate, German / English toggle
#   * attract screen when nobody is scanning; "new scan" button

from __future__ import annotations

import math
import random
import time

import numpy as np
from nicegui import ui
from scipy.ndimage import median_filter

import ui_common as C

TR = {
    "en": {
        "title": "Quantum Rebar Scanner",
        "subtitle": "Seeing steel inside concrete with magnetic sensors",
        "found": "STEEL BARS FOUND", "spacing": "TYPICAL SPACING", "scanned": "SCANNED LENGTH",
        "conf": "CONFIDENCE", "verifying": "{n} being verified",
        "quantum": "QUANTUM SENSOR", "conv": "CONVENTIONAL SENSOR",
        "snr": "SIGNAL / NOISE", "det": "BARS DETECTED", "prec": "POSITION ±",
        "offline": "sensor offline", "compare": "Same scan, two sensors - the clearer the trace, the more precise the result",
        "new_scan": "NEW SCAN", "simulated": "SIMULATED DATA", "replay": "REPLAY",
        "pos": "Position", "cm": "cm", "waiting": "scan to begin",
        "attract_title": "Where is the steel hiding?",
        "attract_text": "Roll the cart along the beam and watch the sensors find the reinforcement.",
        "facts": [
            "Steel reinforcement bars distort the Earth's magnetic field.",
            "Our sensors measure this tiny distortion while the cart rolls along the beam.",
            "Every peak in the signal marks a hidden steel bar - no drilling, no X-rays.",
            "Compare a quantum sensor and a conventional magnetometer on exactly the same scan.",
        ],
    },
    "de": {
        "title": "Quanten-Bewehrungsscanner",
        "subtitle": "Stahl im Beton sichtbar machen - mit Magnetsensoren",
        "found": "STAHLSTÄBE GEFUNDEN", "spacing": "TYPISCHER ABSTAND", "scanned": "GESCANNTE LÄNGE",
        "conf": "SICHERHEIT", "verifying": "{n} werden geprüft",
        "quantum": "QUANTENSENSOR", "conv": "HERKÖMMLICHER SENSOR",
        "snr": "SIGNAL / RAUSCHEN", "det": "STÄBE ERKANNT", "prec": "POSITION ±",
        "offline": "Sensor nicht verbunden", "compare": "Gleicher Scan, zwei Sensoren - je klarer die Kurve, desto genauer das Ergebnis",
        "new_scan": "NEUER SCAN", "simulated": "SIMULIERTE DATEN", "replay": "WIEDERGABE",
        "pos": "Position", "cm": "cm", "waiting": "Scan starten",
        "attract_title": "Wo versteckt sich der Stahl?",
        "attract_text": "Rollen Sie den Wagen über den Balken und sehen Sie zu, wie die Sensoren die Bewehrung finden.",
        "facts": [
            "Bewehrungsstahl verzerrt das Erdmagnetfeld.",
            "Unsere Sensoren messen diese winzige Verzerrung, während der Wagen über den Balken rollt.",
            "Jeder Ausschlag im Signal markiert einen verborgenen Stahlstab - ohne Bohren, ohne Röntgen.",
            "Ein Quantensensor und ein herkömmliches Magnetometer im Vergleich - auf exakt demselben Scan.",
        ],
    },
}

W, H, PAD = 1600, 290, 46
BEAM_Y, BEAM_H = 92, 132

VISITOR_CSS = """
<style>
  html, body { overflow: hidden; }
  .nicegui-content { padding: 0 !important; }
  .v-label { color:#94a3b8; font-size:0.85vw; letter-spacing:0.12em; text-transform:uppercase; font-weight:600; }
  .v-big { font-family: 'Inter','Segoe UI',system-ui,sans-serif; font-weight:800; font-size:3.6vw; line-height:1; letter-spacing:-0.02em; }
  .v-mid { font-family: 'Inter','Segoe UI',system-ui,sans-serif; font-weight:700; font-size:1.55vw; line-height:1.1; color:#e2e8f0; }
  .rb { filter: drop-shadow(0 0 7px rgba(251,146,60,0.85)); }
  .ring { transform-box: fill-box; transform-origin: center; animation: ringout 1.3s ease-out forwards; opacity: 0; }
  .ring2 { animation-delay: 0.15s; }
  @keyframes ringout { 0% { transform: scale(0.25); opacity: 0.95; } 100% { transform: scale(1.9); opacity: 0; } }
  .wave { animation: wave 1.6s ease-in-out infinite; }
  .wave2 { animation-delay: 0.25s; } .wave3 { animation-delay: 0.5s; }
  @keyframes wave { 0%,100% { opacity: 0.15; } 50% { opacity: 0.8; } }
  .attract-cart { animation: drive 5s ease-in-out infinite alternate; }
  @keyframes drive { from { transform: translateX(-150px); } to { transform: translateX(150px); } }
  .pulse-slow { animation: pulseslow 2.4s ease-in-out infinite; }
  @keyframes pulseslow { 0%,100% { opacity: 0.55; } 50% { opacity: 1; } }
  .ticker { transition: opacity .5s; }
</style>
"""


def _speckles():
    rnd = random.Random(7)
    out = []
    for _ in range(260):
        x = rnd.uniform(PAD + 6, W - PAD - 6)
        y = rnd.uniform(BEAM_Y + 6, BEAM_Y + BEAM_H - 6)
        r = rnd.uniform(0.8, 3.2)
        a = rnd.uniform(0.05, 0.22)
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="#cbd5e1" opacity="{a:.2f}"/>')
    return "".join(out)


SPECKLES = _speckles()


class Tracker:
    """Gives every fused rebar a stable id so a new bar can pop in once and stay put."""

    def __init__(self):
        self.items, self.next_id = [], 1

    def reset(self):
        self.items, self.next_id = [], 1

    def update(self, rebars, now):
        used, out = set(), []
        for r in rebars:
            best = None
            for it in self.items:
                if it["id"] in used:
                    continue
                d = abs(it["x"] - r["x"])
                if d <= 6.0 and (best is None or d < abs(best["x"] - r["x"])):
                    best = it
            if best is None:
                best = {"id": self.next_id, "first": now, "x": r["x"]}
                self.next_id += 1
                self.items.append(best)
            best["x"] = r["x"]
            used.add(best["id"])
            out.append({**r, "id": best["id"], "first": best["first"]})
        self.items = [it for it in self.items if it["id"] in used]
        return out


def _sx(x, L):
    return PAD + (x / L) * (W - 2 * PAD)


def base_svg(tracked, L, lang, now):
    t = TR[lang]
    p = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="display:block" xmlns="http://www.w3.org/2000/svg">',
         '<defs><linearGradient id="conc" x1="0" y1="0" x2="0" y2="1">'
         '<stop offset="0" stop-color="#4b5563"/><stop offset="1" stop-color="#2f3644"/></linearGradient></defs>',
         f'<rect x="{PAD}" y="{BEAM_Y}" width="{W - 2 * PAD}" height="{BEAM_H}" rx="16" fill="url(#conc)" '
         f'stroke="rgba(255,255,255,0.18)" stroke-width="2"/>', SPECKLES]
    # faint longitudinal bars (structure hint)
    for yy in (BEAM_Y + 24, BEAM_Y + BEAM_H - 24):
        p.append(f'<line x1="{PAD + 10}" y1="{yy}" x2="{W - PAD - 10}" y2="{yy}" stroke="#94a3b8" '
                 f'stroke-opacity="0.22" stroke-width="5" stroke-dasharray="2 10" stroke-linecap="round"/>')
    # ruler
    ry = BEAM_Y + BEAM_H + 14
    step = 50 if L <= 700 else 100
    for cm in range(0, int(L) + 1, 10):
        X = _sx(cm, L)
        major = cm % step == 0
        p.append(f'<line x1="{X:.1f}" y1="{ry}" x2="{X:.1f}" y2="{ry + (14 if major else 7)}" stroke="#64748b" stroke-width="2"/>')
        if major:
            last = cm + step > L
            lab = f'{cm} {t["cm"]}' if last else str(cm)
            p.append(f'<text x="{X:.1f}" y="{ry + 38}" fill="#94a3b8" font-size="22" text-anchor="{"end" if last else "middle"}" '
                     f'font-family="Inter,Segoe UI,sans-serif">{lab}</text>')
    # stirrups
    for r in tracked:
        X = _sx(r["x"], L)
        ok = r["status"] == "confirmed"
        col = C.C_REBAR if ok else C.C_WARN
        dash = "" if ok else 'stroke-dasharray="9 7"'
        op = "1" if ok else "0.6"
        cls = "rb" if ok else ""
        p.append(f'<g class="{cls}" opacity="{op}">'
                 f'<rect x="{X - 9:.1f}" y="{BEAM_Y + 10}" width="18" height="{BEAM_H - 20}" rx="8" fill="none" '
                 f'stroke="{col}" stroke-width="7" {dash}/></g>')
        if ok:
            txt = f'{r["x"]:.0f}'.replace(".", ",") if lang == "de" else f'{r["x"]:.0f}'
            p.append(f'<text x="{X:.1f}" y="{BEAM_Y - 12}" fill="#fed7aa" font-size="26" font-weight="700" '
                     f'text-anchor="middle" font-family="Inter,Segoe UI,sans-serif">{txt}</text>')
    p.append("</svg>")
    return "".join(p)


def fx_svg(tracked, L, now):
    """One-shot 'ping' rings for freshly detected bars (own layer so a re-render of the
    static bars never restarts or cuts the animation)."""
    p = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="display:block" xmlns="http://www.w3.org/2000/svg">']
    for r in tracked:
        if r["status"] == "confirmed" and (now - r["first"]) < 1.4:
            X, Y = _sx(r["x"], L), BEAM_Y + BEAM_H / 2
            p.append(f'<circle class="ring" cx="{X:.1f}" cy="{Y:.1f}" r="46" fill="none" stroke="{C.C_REBAR}" stroke-width="6"/>')
            p.append(f'<circle class="ring ring2" cx="{X:.1f}" cy="{Y:.1f}" r="46" fill="none" stroke="#fed7aa" stroke-width="3"/>')
    p.append("</svg>")
    return "".join(p)


def cart_svg(x_min, x_cur, L, lang):
    t = TR[lang]
    p = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="display:block" xmlns="http://www.w3.org/2000/svg">']
    if x_cur is not None:
        X = _sx(x_cur, L)
        if x_min is not None:
            X0 = _sx(x_min, L)
            p.append(f'<rect x="{X0:.1f}" y="{BEAM_Y}" width="{max(X - X0, 0):.1f}" height="{BEAM_H}" rx="14" '
                     f'fill="rgba(56,189,248,0.10)"/>')
        p.append(f'<line x1="{X:.1f}" y1="60" x2="{X:.1f}" y2="{BEAM_Y + BEAM_H}" stroke="{C.C_QUANTUM}" '
                 f'stroke-opacity="0.55" stroke-width="2" stroke-dasharray="6 6"/>')
        for k, r in enumerate((30, 52, 74)):
            p.append(f'<path class="wave wave{k + 1}" d="M {X - r:.1f} {BEAM_Y - 4} Q {X:.1f} {BEAM_Y + r * 0.9:.1f} '
                     f'{X + r:.1f} {BEAM_Y - 4}" fill="none" stroke="{C.C_QUANTUM}" stroke-width="3" stroke-linecap="round"/>')
        p.append(f'<g><rect x="{X - 52:.1f}" y="8" width="104" height="48" rx="12" fill="#0b1220" stroke="{C.C_QUANTUM}" '
                 f'stroke-width="3"/><circle cx="{X - 24:.1f}" cy="32" r="8" fill="{C.C_QUANTUM}"/>'
                 f'<circle cx="{X + 2:.1f}" cy="32" r="8" fill="{C.C_CONV}"/>'
                 f'<circle cx="{X + 28:.1f}" cy="32" r="8" fill="{C.C_CONV}"/></g>')
        pos_txt = f'{x_cur:.0f} {t["cm"]}'
        anchor, tx = ("start", X + 66) if X < W - 260 else ("end", X - 66)
        p.append(f'<text x="{tx:.1f}" y="42" fill="{C.C_QUANTUM}" font-size="28" font-weight="700" text-anchor="{anchor}" '
                 f'font-family="Inter,Segoe UI,sans-serif">{pos_txt}</text>')
    p.append("</svg>")
    return "".join(p)


def compare_options(color):
    return {
        "backgroundColor": "transparent", "animation": False,
        "grid": {"left": 8, "right": 12, "top": 10, "bottom": 26},
        "xAxis": {"type": "value", "min": 0, "max": 500, "axisLine": {"lineStyle": {"color": "#475569"}},
                  "axisLabel": {"color": "#64748b", "fontSize": 11}, "splitLine": {"show": False}},
        "yAxis": {"type": "value", "min": -0.3, "max": 1.35, "axisLabel": {"show": False},
                  "splitLine": {"lineStyle": {"color": "#1e293b"}}, "axisLine": {"show": False}},
        "series": [
            {"type": "line", "showSymbol": False, "data": [], "lineStyle": {"width": 2, "color": color},
             "areaStyle": {"opacity": 0.15, "color": color}},
            {"type": "scatter", "symbolSize": 13, "z": 5, "data": [],
             "itemStyle": {"color": C.C_REBAR, "borderColor": "#fff", "borderWidth": 1.5}},
        ],
    }


def bump_view(src):
    """Visitor-friendly trace: cleaned signal minus a slow baseline, oriented so the
    bumps point up and scaled to 1 - each detected bar sits on top of a peak."""
    x, y = src.x, src.y
    ok = np.isfinite(y)
    if ok.sum() < 10:
        return None
    yy = np.interp(x, x[ok], y[ok])
    size = min(121, len(yy) - (1 - len(yy) % 2))
    b = yy - median_filter(yy, size=max(size, 3), mode="nearest")
    if src.dets:
        yd = np.interp([d["x"] for d in src.dets], x, b)
        b = b * (1.0 if np.median(yd) >= 0 else -1.0)
    amp = float(np.percentile(np.abs(b[ok]), 99)) or 1.0
    b = b / amp
    marks = [[d["x"], float(np.clip(np.interp(d["x"], x, b), -0.3, 1.35))] for d in src.dets]
    b[~ok] = np.nan
    return x, b, marks


def register(state):
    args = state.args

    @ui.page("/visitor")
    def visitor():
        ui.add_head_html(C.HEAD_HTML + VISITOR_CSS)
        ui.dark_mode().enable()
        lang = {"v": args.lang}
        tracker = Tracker()
        refs = {"scan": state.scan_id}
        cache = {}

        with ui.column().classes("w-full").style(
                "height:100vh; padding:1.2vh 1.6vw; gap:1.2vh; overflow:hidden; box-sizing:border-box;"):
            # ---- header ------------------------------------------------
            with ui.row().classes("w-full items-center justify-between no-wrap").style("height:7vh"):
                with ui.row().classes("items-center no-wrap").style("gap:1.2vw"):
                    if getattr(args, "logo_url", None):
                        ui.image(args.logo_url).style("height:6vh; width:auto")
                    with ui.column().style("gap:0"):
                        refs["title"] = ui.label().classes("glow-title").style("font-size:2.3vw; line-height:1.05")
                        refs["subtitle"] = ui.label().style("color:#94a3b8; font-size:1vw")
                with ui.row().classes("items-center no-wrap").style("gap:1vw"):
                    refs["badge"] = ui.label().style(
                        f"color:{C.C_WARN}; border:1px solid {C.C_WARN}; border-radius:999px; padding:0.2vh 1vw; "
                        "font-size:0.9vw; font-weight:700; letter-spacing:0.1em")
                    refs["lang"] = ui.toggle({"de": "DE", "en": "EN"}, value=lang["v"]).props("unelevated").style("font-size:0.95vw")
                    refs["btn"] = ui.button(on_click=lambda: (state.new_scan(), tracker.reset())).props(
                        "unelevated color=primary").style("font-weight:700; font-size:0.95vw")

            # ---- hero: beam X-ray --------------------------------------
            with ui.card().classes("dash-card w-full").style("height:35vh; padding:1vh 1.2vw; justify-content:center"):
                with ui.element("div").style("position:relative; width:100%"):
                    refs["base"] = C.html("").style("width:100%")
                    refs["fx"] = C.html("").style("position:absolute; top:0; left:0; width:100%; pointer-events:none")
                    refs["cart"] = C.html("").style("position:absolute; top:0; left:0; width:100%; pointer-events:none")

            # ---- big numbers -------------------------------------------
            with ui.row().classes("w-full no-wrap").style("height:11vh; gap:1.2vw"):
                for key, color in (("found", C.C_REBAR), ("spacing", "#e2e8f0"), ("scanned", C.C_QUANTUM), ("conf", C.C_OK)):
                    with ui.card().classes("dash-card").style("flex:1; padding:1vh 1.2vw; justify-content:center; gap:0.4vh"):
                        refs[f"{key}_l"] = ui.label().classes("v-label")
                        with ui.row().classes("items-baseline no-wrap").style("gap:0.8vw"):
                            refs[f"{key}_v"] = ui.label("-").classes("v-big").style(f"color:{color}")
                            refs[f"{key}_s"] = ui.label().style("color:#94a3b8; font-size:0.95vw")

            # ---- comparison --------------------------------------------
            with ui.row().classes("w-full no-wrap").style("height:32vh; gap:1.2vw"):
                for key, color in (("quantum", C.C_QUANTUM), ("conv", C.C_CONV)):
                    with ui.card().classes("dash-card").style("flex:1; padding:1vh 1.2vw; gap:0.6vh; overflow:hidden"):
                        with ui.row().classes("w-full items-center justify-between no-wrap"):
                            with ui.row().classes("items-center no-wrap").style("gap:0.6vw"):
                                ui.element("div").style(f"width:0.9vw; height:0.9vw; border-radius:50%; background:{color}; "
                                                        f"box-shadow:0 0 10px {color}")
                                refs[f"{key}_title"] = ui.label().classes("v-mid")
                            refs[f"{key}_off"] = ui.label().style(f"color:{C.C_BAD}; font-size:0.9vw; font-weight:700")
                        refs[f"{key}_chart"] = ui.echart(compare_options(color)).classes("w-full").style("height:16.5vh")
                        with ui.row().classes("w-full no-wrap").style("gap:1.4vw"):
                            for m in ("det", "snr", "prec"):
                                with ui.column().style("gap:0"):
                                    refs[f"{key}_{m}_l"] = ui.label().classes("v-label").style("font-size:0.7vw")
                                    refs[f"{key}_{m}_v"] = ui.label("-").style(
                                        f"color:{color}; font-size:1.7vw; font-weight:800; font-family:'Inter',sans-serif")

            # ---- ticker / footer ---------------------------------------
            with ui.row().classes("w-full items-center justify-between no-wrap").style("height:6vh"):
                refs["ticker"] = ui.label().classes("ticker").style("color:#cbd5e1; font-size:1.35vw; max-width:68vw")
                with ui.column().style("gap:0; align-items:flex-end"):
                    refs["compare"] = ui.label().style("color:#64748b; font-size:0.8vw; text-align:right")
                    ui.label(args.org).style("color:#94a3b8; font-size:0.9vw; text-align:right")

        # ---- attract overlay ---------------------------------------------
        with ui.element("div").style(
                "position:fixed; inset:0; z-index:100; background:rgba(8,11,20,0.95); display:flex; flex-direction:column; "
                "align-items:center; justify-content:center; gap:3vh; text-align:center") as overlay:
            refs["a_title"] = ui.label().classes("glow-title").style("font-size:5vw; font-weight:800")
            refs["a_sub"] = ui.label().style("color:#94a3b8; font-size:1.8vw")
            C.html(f'<svg viewBox="0 0 400 64" width="60vw" xmlns="http://www.w3.org/2000/svg">'
                   f'<rect x="10" y="42" width="380" height="14" rx="6" fill="#3b4252"/>'
                   f'<g class="attract-cart"><rect x="170" y="6" width="60" height="28" rx="8" fill="#0b1220" '
                   f'stroke="{C.C_QUANTUM}" stroke-width="3"/><circle cx="188" cy="20" r="5" fill="{C.C_QUANTUM}"/>'
                   f'<circle cx="212" cy="20" r="5" fill="{C.C_CONV}"/></g></svg>')
            refs["a_text"] = ui.label().classes("pulse-slow").style("color:#e2e8f0; font-size:2vw; max-width:60vw")
            refs["a_org"] = ui.label(args.org).style("color:#64748b; font-size:1.1vw")

        # -----------------------------------------------------------------
        def set_texts():
            t = TR[lang["v"]]
            refs["title"].set_text(t["title"])
            refs["subtitle"].set_text(t["subtitle"])
            refs["btn"].set_text(t["new_scan"])
            for key in ("found", "spacing", "scanned", "conf"):
                refs[f"{key}_l"].set_text(t[key])
            for key in ("quantum", "conv"):
                refs[f"{key}_title"].set_text(t[key])
                for m in ("det", "snr", "prec"):
                    refs[f"{key}_{m}_l"].set_text(t[m])
            refs["compare"].set_text(t["compare"])
            refs["a_title"].set_text(t["attract_title"])
            refs["a_sub"].set_text(t["subtitle"])
            refs["a_text"].set_text(t["attract_text"])
            cache.clear()

        def on_lang(e):
            lang["v"] = e.value
            set_texts()

        refs["lang"].on_value_change(on_lang)
        set_texts()

        def num(v, nd=0):
            s = f"{v:.{nd}f}"
            return s.replace(".", ",") if lang["v"] == "de" else s

        def tick():
            t = TR[lang["v"]]
            now = time.time()
            if state.scan_id != refs["scan"]:
                refs["scan"] = state.scan_id
                tracker.reset()
                cache.clear()
            res = state.engine_result
            pos = state.history_position

            idle = state.seconds_since_sample() > args.idle_s
            overlay.set_visibility(bool(idle))
            badge = {"live": "", "demo": t["simulated"], "replay": t["replay"]}[state.mode]
            refs["badge"].set_text(badge)
            refs["badge"].set_visibility(bool(badge))

            L = max(args.beam_length_cm, (max(pos) + 10) if pos else 0)
            x_cur = pos[-1] if pos else None
            x_min = min(pos) if pos else None

            rebars = res.rebars if res else []
            tracked = tracker.update(rebars, now)
            sig = (tuple((r["id"], round(r["x"]), r["status"]) for r in tracked), round(L), lang["v"])
            if cache.get("base") != sig:
                cache["base"] = sig
                refs["base"].set_content(base_svg(tracked, L, lang["v"], now))
            fsig = (tuple(r["id"] for r in tracked if r["status"] == "confirmed" and (now - r["first"]) < 1.4), round(L))
            if cache.get("fx") != fsig:
                cache["fx"] = fsig
                refs["fx"].set_content(fx_svg(tracked, L, now))
            csig = (None if x_cur is None else round(x_cur, 0), None if x_min is None else round(x_min, 0), round(L), lang["v"])
            if cache.get("cart") != csig:
                cache["cart"] = csig
                refs["cart"].set_content(cart_svg(x_min, x_cur, L, lang["v"]))

            # big numbers
            s = res.summary if res else None
            n_conf = s["n_confirmed"] if s else 0
            refs["found_v"].set_text(str(n_conf))
            refs["found_s"].set_text(t["verifying"].format(n=s["n_tentative"]) if s and s["n_tentative"] else "")
            refs["spacing_v"].set_text(num(s["spacing_median"]) if s and s["spacing_median"] is not None else "-")
            refs["spacing_s"].set_text(t["cm"] if s and s["spacing_median"] is not None else "")
            scanned = (max(pos) - min(pos)) if pos else 0
            refs["scanned_v"].set_text(num(scanned) if pos else "-")
            refs["scanned_s"].set_text(t["cm"] if pos else "")
            refs["conf_v"].set_text(f'{num(s["conf_mean"] * 100)}' if s and s["conf_mean"] is not None else "-")
            refs["conf_s"].set_text("%" if s and s["conf_mean"] is not None else "")

            # ticker
            facts = t["facts"]
            refs["ticker"].set_text(facts[int(now // 9) % len(facts)])

            # comparison charts (only when the engine produced something new)
            key = (res.version if res else None, lang["v"], round(L))
            if cache.get("cmp") == key:
                return
            cache["cmp"] = key
            comp = (res.cfg.fuse_components[0] if res else "Bx")
            for panel, ch in (("quantum", "quantum"), ("conv", "lis2")):
                chart = refs[f"{panel}_chart"]
                src = res.sources.get((ch, comp)) if res else None
                st = res.stats.get(ch, {"ok": False}) if res else {"ok": False}
                chart.options["xAxis"]["max"] = float(L)
                if src is None or not src.ok:
                    chart.options["series"][0]["data"] = []
                    chart.options["series"][1]["data"] = []
                    refs[f"{panel}_off"].set_text(t["offline"] if res else t["waiting"])
                    refs[f"{panel}_off"].style(f"color:{C.C_BAD if res else '#64748b'}")
                    for m in ("det", "snr", "prec"):
                        refs[f"{panel}_{m}_v"].set_text("-")
                    chart.update()
                    continue
                refs[f"{panel}_off"].set_text("")
                bv = bump_view(src)
                if bv is None:
                    continue
                chart.options["series"][0]["data"] = C.chart_points(bv[0], bv[1], 900)
                chart.options["series"][1]["data"] = bv[2]
                mm = st.get("matched", 0)
                ne = st.get("n_expected", 0)
                refs[f"{panel}_det_v"].set_text(f"{mm}/{ne}" if ne else str(st.get("n_det", 0)))
                snr = st.get("snr_median")
                refs[f"{panel}_snr_v"].set_text(f"{num(snr)}×" if snr else "-")
                sg = st.get("sigma_cm")
                refs[f"{panel}_prec_v"].set_text(f"{num(sg, 1)} {t['cm']}" if sg is not None else "-")
                chart.update()

        failed = {"n": 0}

        def safe_tick():
            try:
                tick()
            except Exception as e:
                failed["n"] += 1
                if failed["n"] in (1, 50, 500):
                    state.log(f"Visitor render error: {e!r}")

        ui.timer(0.25, safe_tick)
