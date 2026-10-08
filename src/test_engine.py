# test_engine.py -- validates rebar_engine on simulated scans with known ground truth.
# Uses the "robust" pipeline (cm grid, Hampel, Savitzky-Golay, centre localisation, fusion);
# the dashboard default "original" reproduces the institute's reference algorithm instead.
#   python test_engine.py
import numpy as np

import rebar_engine as eng
from demo_source import DemoScan, as_snapshot


def score(res, truth, tol=4.0):
    rb = [r for r in res.rebars if r["status"] == "confirmed"]
    xs = np.array([r["x"] for r in rb])
    hit, err = 0, []
    for t in truth:
        if len(xs):
            j = int(np.argmin(np.abs(xs - t)))
            if abs(xs[j] - t) <= tol:
                hit += 1
                err.append(xs[j] - t)
    fp = len(rb) - hit
    return hit, fp, np.array(err)


def run(name, scan, cfg=None, expect_last_cm=None):
    cfg = cfg or eng.EngineConfig(method="robust")
    res = eng.run_engine(as_snapshot(scan), cfg)
    truth = scan.truth
    if expect_last_cm is not None:   # rebars beyond the scanned range can't be found
        truth = truth[truth < expect_last_cm]
    hit, fp, err = score(res, truth)
    rms = float(np.sqrt(np.mean(err ** 2))) if len(err) else float("nan")
    bias = float(np.mean(err)) if len(err) else float("nan")
    print(f"{name:34s} found {hit}/{len(truth)}  false+ {fp}  "
          f"rms {rms:5.2f} cm  bias {bias:+5.2f} cm  ({res.compute_ms:.0f} ms)")
    return res, hit, len(truth), fp, rms


if __name__ == "__main__":
    print("== clean scan, centre vs slope localisation ==")
    for mode in ("center", "slope"):
        cfg = eng.EngineConfig(method="robust", locate=mode)
        run(f"clean, locate={mode}", DemoScan(seed=1), cfg)

    print("== artefacts: sparse startup, 35 cm dropout, spikes ==")
    tot_h = tot_n = tot_fp = 0
    for seed in range(8):
        r, h, n, fp, rms = run(f"artefacts seed {seed}", DemoScan(seed=seed, artifacts=True))
        tot_h += h; tot_n += n; tot_fp += fp
    print(f"TOTAL recall {tot_h}/{tot_n} = {100*tot_h/tot_n:.1f}%   false positives: {tot_fp}")

    print("== noisier sensors (2.5x) ==")
    for seed in range(3):
        run(f"noisy seed {seed}", DemoScan(seed=seed, artifacts=True, lis_noise=0.05, quantum_noise=1.0))

    print("== quantum offline ==")
    r, *_ = run("quantum offline", DemoScan(seed=2, quantum_online=False))
    print("   quantum stats:", r.stats["quantum"], "| lis2 ok:", r.stats["lis2"]["ok"])

    print("== different speed / rate ==")
    run("slow 8 Hz", DemoScan(seed=3, hz=8, v_cm_s=12))
    run("fast 25 Hz", DemoScan(seed=3, hz=25, v_cm_s=35))

    print("== channel comparison (clean scan) ==")
    r, *_ = run("stats", DemoScan(seed=5))
    for ch, s in r.stats.items():
        print("  ", ch, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()})
