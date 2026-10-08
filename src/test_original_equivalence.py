# Proves that method="original" gives exactly the positions of your professor's
# finde_buegelpositionen (via the faithful port in Rebar_detection.py) on the same data.
import numpy as np
import Rebar_detection as rd
import rebar_engine as eng
from demo_source import DemoScan, as_snapshot

snap = as_snapshot(DemoScan(seed=3, artifacts=True))
ok_all = True
for rf in (1, 2):
    for use_factors in (True, False):
        for ch in ("lis2", "lis3", "quantum"):
            b = snap[ch]
            bx, by, bz = b["x"], b["y"], b["z"]
            comps = {"Bx": bx, "By": by, "Bz": bz, "B_total": np.sqrt(bx**2 + by**2 + bz**2)}
            for comp, vals in comps.items():
                m = np.isfinite(vals) & np.isfinite(snap["pos"])
                if m.sum() < 20:
                    continue
                p, v = rd.dedupe_and_sort(snap["pos"][m], vals[m])
                thr = {"Bx": 0.02, "By": 0.05, "Bz": 0.1, "B_total": 0.05}[comp]
                ref = rd.find_rebar_positions_from_measurements(
                    f"RF{rf}_{comp}", {f"RF{rf}_{comp}": v}, p, {comp: thr}, 20,
                    rd.DEFAULT_COMPONENT_DERIVATIVE_LEVEL, use_factors=use_factors)["Sensor_1"]
                cfg = eng.EngineConfig(method="original", rf_number=rf, use_factors=use_factors, min_spacing_samples=20)
                res = eng.analyze_source(snap["pos"], vals, channel=ch, component=comp, cfg=cfg)
                mine = np.array([d["x"] for d in res.dets])
                same = len(ref) == len(mine) and np.allclose(ref, mine, atol=0, rtol=0)
                ok_all &= same
                if not same:
                    print("MISMATCH", rf, use_factors, ch, comp, ref, mine)
print("original method identical to finde_buegelpositionen on all cases:", ok_all)
