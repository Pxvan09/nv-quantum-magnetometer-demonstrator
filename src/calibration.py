# calibration.py
#
# Center + slope calibration of the AQ quantum sensor over the UART, ported from
# device_app.py so that dashboard.py can run it.  Differences to device_app.py:
#   * nothing is written to the device until the user presses "Accept"
#   * the raw sweep + fit + initial guess are kept so the UI can plot them
#   * everything runs in a worker thread (call `cal.run(...)` via asyncio.to_thread)
#
# Needs ensemble_guesser.py and slope_determination.py next to this file.

import datetime
import pickle
import re
import threading
import time

import numpy as np

try:
    from ensemble_guesser import find_center, find_groups, fit_odmr_signal
    from ensemble_guesser import model as center_model
    from slope_determination import fit_odmr as slope_fit_odmr
    from slope_determination import fit_slopes
    from slope_determination import odmr_model as slope_model
    from slope_determination import LINEAR_FIT_RANGE
    CAL_IMPORT_ERROR = None
except Exception as _e:                      # keep the dashboard usable without them
    CAL_IMPORT_ERROR = _e
    LINEAR_FIT_RANGE = 1.2e6

CENTER_POINTS = 1000      # 2 * N_FREQS in device_app.handle_calibration_center_data_reception
SLOPE_POINTS = 500        # N_FREQS in device_app.handle_calibration_slope_data_reception
RX_TIMEOUT_S = 30.0       # abort if the device sends nothing for this long
PLOT_REF_HZ = 2.87e9      # x axis of the plots is (f - PLOT_REF_HZ) in MHz

_NUM = r'[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?'
DATA_RE = re.compile(rf'^\s*({_NUM}),\s*({_NUM})\s*$')
ZFS_FIT       = 2.87e9



def _pts(x_hz, y):
    return [[(float(a) - PLOT_REF_HZ) / 1e6, float(b)] for a, b in zip(x_hz, y)]


class CalibrationManager:
    def __init__(self, state):
        self.state = state
        self._lock = threading.Lock()
        self.abort_event = threading.Event()
        self.busy = False
        self.stage = "idle"          # idle | acquiring | analysing | review | applying | error
        self.kind = None             # "center" | "slope"
        self.message = "Idle"
        self.progress = 0.0          # 0..1
        self.version = 0             # bumped whenever the plot must be redrawn
        self.freqs = []
        self.values = []
        self.result = None
        self.atten = 20.0
        self.n_ens = 3
        # accepted center calibration (needed by the slope calibration)
        self.zfs = None
        self.hf = None
        self.splittings = None
        self.center_applied = False
        self.slope_applied = False
        self.elapsed = 0.0
        self.time_left = None    
                # ODMR measurement
        self.odmr_stage = "idle"         # idle | acquiring | done | error
        self.odmr_message = "Idle"
        self.odmr_progress = 0.0
        self.odmr_elapsed = 0.0
        self.odmr_time_left = None
        self.odmr_freqs = []
        self.odmr_values = []
        self.odmr_version = 0

    # ---------------------------------------------------------------
    def _bump(self):
        self.version += 1

    def _set(self, stage=None, message=None):
        if stage is not None:
            self.stage = stage
        if message is not None:
            self.message = message
        self._bump()

    def discard(self):
        if self.busy:
            return
        self.result = None
        self._set("idle", "Discarded - run the calibration again.")

    # ---------------------------------------------------------------
    def run(self, kind, atten, n_ens):
        """Blocking: stop measurement, sweep, fit.  Leaves the result in stage 'review'."""
        with self._lock:
            if self.busy:
                return
            self.busy = True
        st = self.state
        r = st.reader
        try:
            if CAL_IMPORT_ERROR is not None:
                raise RuntimeError(f"ensemble_guesser / slope_determination not importable: {CAL_IMPORT_ERROR}")
            if kind == "slope":
                if self.splittings is None:
                    raise RuntimeError("Run and accept the center calibration first.")
                if len(self.splittings) != int(n_ens):
                    raise RuntimeError(f"N subensembles ({n_ens}) differs from the accepted center "
                                       f"calibration ({len(self.splittings)}). Repeat the center calibration.")
            self.abort_event.clear()
            self.kind, self.atten, self.n_ens = kind, float(atten), int(n_ens)
            self.freqs, self.values, self.result, self.progress = [], [], None, 0.0
            self.elapsed, self.time_left = 0.0, None
            self._set("acquiring", "Stopping measurement…")
            if st.measuring:
                st.measuring = False
            r.stop_measurement()
            st.mag_status = "calibrating"
            st.log(f"{kind} calibration: MW attenuation {self.atten:g} dB, N subensembles {self.n_ens}")

            self._acquire(kind)
            if self.abort_event.is_set():
                self._set("idle", "Aborted.")
                return
            self._set("analysing", "Fitting…")
            self._save_raw(kind)
            self.result = self._analyse(kind)
            if self.result["ok"]:
                self._set("review", f"{kind.capitalize()} calibration finished - check the plot, "
                                    f"then accept or discard.")
            else:
                self._set("review", f"Fit failed: {self.result['error']}  (raw sweep shown; discard and repeat)")
            st.log(f"{kind} calibration done, fit ok={self.result['ok']}")
        except Exception as e:                       # noqa
            st.log(f"{kind} calibration error: {e}")
            self._set("error", f"{kind.capitalize()} calibration failed: {e}")
            if isinstance(e, OSError):               # serial port died
                st.mag_status = "error"
        finally:
            if st.mag_status == "calibrating":
                st.mag_status = "connected"
            self.busy = False
            self._bump()

    # ---------------------------------------------------------------
    def _acquire(self, kind):
        r = self.state.reader
        ser = r._ser
        if ser is None:
            raise RuntimeError("Quantum sensor port is not open.")
        if kind == "center":
            total = CENTER_POINTS
            cmds = [f"setCenterFreq {ZFS_FIT}","disableHyperfine", f"setMWAttenuation {self.atten:g}", "runCalibrationCenterFreq"]
        else:
            total = SLOPE_POINTS
            cmds = ["enableHyperfine", f"setMWAttenuation {self.atten:g}", "runCalibrationSlopeFreq"]
        ser.reset_input_buffer()
        for c in cmds:
            r.send(c)
        self.message = f"{kind.capitalize()} sweep running…"
        t0 = last_rx = time.monotonic()
        while len(self.freqs) < total:
            if self.abort_event.is_set():
                r.send("stop")
                return
            raw = ser.readline()
            now = time.monotonic()
            if not raw:
                if now - last_rx > RX_TIMEOUT_S:
                    raise TimeoutError(f"no data from the device for {RX_TIMEOUT_S:.0f} s")
                continue
            last_rx = now
            m = DATA_RE.match(raw.decode("ascii", errors="ignore"))
            if not m:                                # status lines, leftovers of the measurement
                continue
            f_hz = float(m.group(1))
            v = float(m.group(2)) / 0xFFFFFFFF * 10          # same scaling as device_app
            self.freqs.append(f_hz)
            self.values.append(v)
            n = len(self.freqs)
            self.progress = n / total
            self.elapsed = now - t0
            if n > 5:
                self.time_left = self.elapsed / n * (total - n)
            self.message = f"{kind.capitalize()} sweep: {n}/{total} points"
            print(f"[{kind} cal] {n:4d}/{total}  f = {f_hz / 1e9:.6f} GHz   value = {v:+.6f}")

    def _save_raw(self, kind):
        """Same .pys files as device_app, so ensemble_guesser.py / slope_determination.py can replay them."""
        try:
            stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            payload = {"freqs": self.freqs, "values": self.values}
            if kind == "slope":
                payload.update({"zfs_fit": self.zfs, "hf_fit": self.hf, "splittings": self.splittings})
            with open(f"{stamp}_calibration_{kind}.pys", "wb") as f:
                pickle.dump(payload, f)
        except Exception as e:                       # noqa
            self.state.log(f"could not save raw calibration data: {e}")

    # ---------------------------------------------------------------
    def _analyse(self, kind):
        freqs = np.asarray(self.freqs, dtype=float)
        sig = np.asarray(self.values, dtype=float)
        n = self.n_ens
        res = {"kind": kind, "ok": False, "error": ""}
        try:
            xfit = np.linspace(freqs.min(), freqs.max(), 1000)
            if kind == "center":
                zfs_guess = find_center(freqs, sig)["f0"]
                groups = find_groups(freqs, sig, zfs_guess, n_ensembles=n)
                popt, p0, _ = fit_odmr_signal(freqs, sig, zfs_guess, groups, n_ensembles=n)
                if popt is None:
                    raise RuntimeError("no fit converged")
                zfs, hf = float(popt[0]), float(popt[1])
                gp = popt[2:]
                # popt = [zfs, hf, (delta, amp, lw) * n, sign_left, sign_right]
                res.update(
                    zfs=zfs, hf=hf,
                    splittings=[float(v) for v in gp[0:3 * n:3]],
                    amps=[float(v) for v in gp[1:3 * n:3]],
                    linewidths=[float(v) for v in gp[2:3 * n:3]],
                    xfit=xfit,
                    yfit=center_model(xfit, zfs, hf, *gp),
                    yguess=center_model(xfit, p0[0], p0[1], *p0[2:]),
                )
            else:
                popt, _, p0 = slope_fit_odmr(freqs, sig, self.zfs, self.splittings, n_ensembles=n)
                split = np.asarray(popt[:n], dtype=float)
                amps = np.asarray(popt[n:2 * n], dtype=float)
                order = np.argsort(np.abs(amps))[::-1]            # strongest first (as device_app)
                split_sorted = split[order]
                slopes, slope_errs = fit_slopes(freqs, sig, self.zfs, split_sorted)
                res.update(
                    zfs=float(self.zfs), hf=float(popt[-2]), linewidth=float(popt[-1]),
                    splittings=[float(v) for v in split_sorted],
                    amps=[float(v) for v in amps[order]],
                    slopes=[float(v) for v in slopes], slope_errs=[float(v) for v in slope_errs],
                    xfit=xfit,
                    yfit=slope_model(xfit, self.zfs, *popt, n_ensembles=n),
                    yguess=slope_model(xfit, self.zfs, *p0, n_ensembles=n),
                )
            res["ok"] = True
        except Exception as e:                       # noqa
            res["error"] = str(e)
            self.state.log(f"{kind} fit failed: {e}")
        return res

    # ---------------------------------------------------------------
    def apply(self):
        """Blocking: write the accepted result to the device (same commands as device_app)."""
        res = self.result
        if self.busy or self.stage != "review" or not res or not res.get("ok"):
            return
        with self._lock:
            self.busy = True
        r = self.state.reader
        try:
            self._set("applying", "Writing to device…")
            if res["kind"] == "center":
                r.send(f"setCenterFreq {res['zfs']:.2f}")
                r.send(f"setHyperfineFreq {res['hf']:.3f}")
                for i, s in enumerate(res["splittings"]):
                    r.send(f"setDualResonanceFreq {i} {s}")
                self.zfs, self.hf = res["zfs"], res["hf"]
                self.splittings = list(res["splittings"])
                self.center_applied, self.slope_applied = True, False
                msg = "Center calibration applied. You can run the slope calibration now."
            else:
                zfs, hf = self.zfs, res["hf"]
                r.send(f"setCenterFreq {zfs}")
                r.send(f"storeZFS {zfs}")
                r.send(f"setHyperfineFreq {hf}")
                r.send(f"storeHF {hf}")
                for i, (s, sl) in enumerate(zip(res["splittings"], res["slopes"])):
                    k = sl * 0xFFFFFFFF / 10
                    r.send(f"setDualResonanceFreq {i} {s}")
                    r.send(f"setMagnetometerCalibrationCH{i} {k}")
                    r.send(f"storeDRFreq {i} {s}")
                    r.send(f"storeDRLockinFreq {i} {s}")
                    r.send(f"storeSlope {i} {k}")
                self.slope_applied = True
                msg = "Slope calibration stored on the device. You can start the measurement now."
            self.result = None
            self._set("idle", msg)
            self.state.log(msg)
        except Exception as e:                       # noqa
            self.state.log(f"apply error: {e}")
            self._set("error", f"Writing to device failed: {e}")
        finally:
            self.busy = False
            self._bump()

    # ---------------------------------------------------------------
    # helpers for the UI
    def chart_series(self):
        series = [{
            "name": "data", "type": "line", "showSymbol": False,
            "lineStyle": {"width": 1.2, "color": "#94a3b8"},
            "data": _pts(self.freqs, self.values),
        }]
        res = self.result
        if not (res and res.get("ok")):
            return series
        series.append({"name": "guess", "type": "line", "showSymbol": False,
                       "lineStyle": {"width": 1, "type": "dotted", "color": "#f59e0b"},
                       "data": _pts(res["xfit"], res["yguess"])})
        series.append({"name": "fit", "type": "line", "showSymbol": False,
                       "lineStyle": {"width": 2, "color": "#38bdf8"},
                       "data": _pts(res["xfit"], res["yfit"])})
        marks = [{"xAxis": (res["zfs"] - PLOT_REF_HZ) / 1e6,
                  "label": {"formatter": "ZFS", "color": "#e2e8f0", "fontSize": 10}}]
        if res["kind"] == "center":
            for s in res["splittings"]:
                for sgn in (-1, 1):
                    marks.append({"xAxis": (res["zfs"] + sgn * s - PLOT_REF_HZ) / 1e6,
                                  "label": {"show": False}})
        else:
            for i, (s, sl) in enumerate(zip(res["splittings"], res["slopes"])):
                zc = res["zfs"] + s
                xs = [zc - LINEAR_FIT_RANGE / 2, zc + LINEAR_FIT_RANGE / 2]
                series.append({"name": f"slope {i}", "type": "line", "showSymbol": False,
                               "lineStyle": {"width": 2.5, "color": "#f43f5e"},
                               "data": _pts(xs, [sl * (x - zc) for x in xs])})
                marks.append({"xAxis": (zc - PLOT_REF_HZ) / 1e6, "label": {"show": False}})
        series[0]["markLine"] = {"silent": True, "symbol": "none",
                                 "lineStyle": {"color": "#64748b", "type": "dashed", "width": 1},
                                 "data": marks}
        return series

    def result_html(self):
        res = self.result
        if not res:
            return ""
        if not res.get("ok"):
            return f'<div style="color:#f87171">Fit failed: {res.get("error", "")}</div>'
        cell = "padding:2px 18px 2px 0;text-align:left;"
        head = (f'<div style="margin-bottom:6px">ZFS <b>{res["zfs"] / 1e9:.6f} GHz</b> · '
                f'hyperfine <b>{res["hf"] / 1e6:.3f} MHz</b></div>')
        cols = ["#", "splitting [MHz]", "amplitude"]
        if res["kind"] == "center":
            cols.append("linewidth [MHz]")
        else:
            cols += ["slope [1/Hz]", "± slope"]
        rows = ""
        for i, s in enumerate(res["splittings"]):
            tds = [str(i), f"{s / 1e6:.4f}", f'{res["amps"][i]:.3e}']
            if res["kind"] == "center":
                tds.append(f'{res["linewidths"][i] / 1e6:.3f}')
            else:
                tds += [f'{res["slopes"][i]:.4e}', f'{res["slope_errs"][i]:.2e}']
            rows += "<tr>" + "".join(f'<td style="{cell}">{t}</td>' for t in tds) + "</tr>"
        th = "".join(f'<th style="{cell}">{c}</th>' for c in cols)
        return (f'<div style="font-family:monospace;font-size:0.8rem;color:#cbd5e1">{head}'
                f'<table style="border-collapse:collapse"><thead><tr style="color:#94a3b8">{th}</tr></thead>'
                f'<tbody>{rows}</tbody></table></div>')

    def _oset(self, stage=None, message=None):
        if stage is not None:
            self.odmr_stage = stage
        if message is not None:
            self.odmr_message = message
        self.odmr_version += 1

    def run_odmr(self, f_start_ghz, f_end_ghz, f_step_mhz, mspp, atten, ix, iy, iz):
        """Blocking ODMR sweep (same commands / scaling as device_app.handle_start_odmr)."""
        with self._lock:
            if self.busy:
                return
            self.busy = True
        st = self.state
        r = st.reader
        try:
            f0, f1, df = f_start_ghz * 1e9, f_end_ghz * 1e9, f_step_mhz * 1e6
            if df <= 0 or f1 <= f0:
                raise ValueError("need start < end and step > 0")
            total = len(np.arange(f0, f1 + df, df))
            self.abort_event.clear()
            self.odmr_freqs, self.odmr_values = [], []
            self.odmr_progress, self.odmr_elapsed, self.odmr_time_left = 0.0, 0.0, None
            self._oset("acquiring", "Stopping measurement…")
            st.measuring = False
            r.stop_measurement()
            st.mag_status = "calibrating"
            ser = r._ser
            if ser is None:
                raise RuntimeError("Quantum sensor port is not open.")
            st.log(f"ODMR: {f_start_ghz:g}-{f_end_ghz:g} GHz, step {f_step_mhz:g} MHz, "
                   f"{int(mspp)} ms/point, MW {atten:g} dB, {total} points")
            ser.reset_input_buffer()
            for c in (f"setIx {ix}", f"setIy {iy}", f"setIz {iz}",
                      f"setODMRSecondsPerPoint {int(mspp)}",
                      f"setODMRFrequencyStart {round(f0, 3)}",
                      f"setODMRFrequencyEnd {round(f1, 3)}",
                      f"setODMRFrequencyStep {round(df, 3)}",
                      f"setMWAttenuation {atten:g}",
                      "setDualResonanceLockinEnabled 0 0",
                      "runODMR"):
                r.send(c)
            self._oset(message="ODMR sweep running…")
            t0 = last_rx = time.monotonic()
            while len(self.odmr_freqs) < total:
                if self.abort_event.is_set():
                    r.send("stop")
                    self._oset("idle", "Aborted.")
                    return
                raw = ser.readline()
                now = time.monotonic()
                if not raw:
                    if now - last_rx > RX_TIMEOUT_S:
                        raise TimeoutError(f"no data from the device for {RX_TIMEOUT_S:.0f} s")
                    continue
                last_rx = now
                m = DATA_RE.match(raw.decode("ascii", errors="ignore"))
                if not m:
                    continue
                f_hz = float(m.group(1))
                v = -1 * (float(m.group(2)) - 0x7FFFFFFF) / 0xFFFFFFFF * 5
                self.odmr_freqs.append(f_hz)
                self.odmr_values.append(v)
                n = len(self.odmr_freqs)
                self.odmr_progress = n / total
                self.odmr_elapsed = now - t0
                if n > 3:
                    self.odmr_time_left = self.odmr_elapsed / n * (total - n)
                self.odmr_message = f"ODMR sweep: {n}/{total} points"
                print(f"[ODMR] {n:4d}/{total}  f = {f_hz / 1e9:.6f} GHz   value = {v:+.6f}")
            self.odmr_time_left = 0.0
            self._oset("done", f"ODMR finished ({total} points).")
            st.log("ODMR finished.")
        except Exception as e:                       # noqa
            st.log(f"ODMR error: {e}")
            self._oset("error", f"ODMR failed: {e}")
            if isinstance(e, OSError):
                st.mag_status = "error"
        finally:
            if st.mag_status == "calibrating":
                st.mag_status = "connected"
            self.busy = False
            self.odmr_version += 1