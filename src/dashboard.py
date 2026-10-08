# dashboard.py
#
# Live dashboard for the BLE magnetometer (LIS2/LIS3) + quantum sensor stream.
#
#   python dashboard.py --name CIRCUITPY --mag-port COM10 --out session3.csv
#   python dashboard.py --demo                       # no hardware: simulated scan
#   python dashboard.py --replay session1.csv --loop # replay a recorded session
#
# Pages (same process, same live data):
#   http://localhost:8080/engineer   detailed engineering view (also "/")
#   http://localhost:8080/visitor    big-screen, bilingual view for the exhibition
#
# What changed compared with the previous version
#   * BLE ingestion, CSV logging and rebar detection now run as background tasks,
#     independent of any browser. Closing a tab no longer stops logging, and two
#     browsers no longer fight over the same BLE lines.
#   * BLE and the quantum sensor reconnect automatically (with a data watchdog).
#   * Detection lives in rebar_engine.py (cm-based, gap-safe, sensor fusion).
#   * --demo / --replay for rehearsal and as a booth fallback.

import argparse
import asyncio
import csv
import dataclasses
import os
import re
import threading
import time as time_mod
import webbrowser
from collections import deque
from datetime import datetime
import atexit, signal, sys


import numpy as np

try:
    import serial
    import serial.tools.list_ports
except ImportError:        # demo / replay work without pyserial
    serial = None
try:
    from bleak import BleakClient, BleakScanner
except ImportError:        # demo / replay work without bleak
    BleakClient = BleakScanner = None

from nicegui import app, ui

import rebar_engine as eng
import ui_common as C
import visitor_page
from demo_source import DemoScan
from calibration import CalibrationManager

# ------------------------------------------------------------
# Quantum sensor configuration (unchanged)
# ------------------------------------------------------------
MSPP = 100
VECTOR_ENABLED = True
CONVERSION_ENABLED = True
FOURTH_AXIS_ENABLED = False
BINARY_MODE = False
LOCKIN_FREQS = (5000, 3000, 1000, 750)
LOCKIN_LOWPASS = 10
LOCKIN_MOD_DEPTH = 500e3

SCALE_FACTOR = 1.0
SCALE_FACTOR_B = 1.0
MAX_SAMPLE_AGE_S = 1.0
MAX_HISTORY = 200_000
QUANTUM_TIME_HISTORY = 5_000

# Static hard-iron offsets for the BLE magnetometers [G], ordered as Bx, By, Bz.
LIS2_OFFSET_G = (0.18, 0.4, -0.95)
LIS3_OFFSET_G = (0.15, -0.05, -0.6)

UART_SERVICE_UUID = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
UART_TX_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"

CSV_HEADER = [
    "timestamp", "Position",
    "lis2_Bx_G", "lis2_By_G", "lis2_Bz_G",
    "lis3_Bx_G", "lis3_By_G", "lis3_Bz_G",
    "quantum_Bx", "quantum_By", "quantum_Bz", "quantum_valid",
]

CSV_RE = re.compile(
    r'^DATA,'
    r'(?P<t>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),'
    r'(?P<dist>[-+]?\d+(\.\d+)?),'
    r'(?P<lis2x>[-+]?\d+(\.\d+)?),'
    r'(?P<lis2y>[-+]?\d+(\.\d+)?),'
    r'(?P<lis2z>[-+]?\d+(\.\d+)?),'
    r'(?P<lis3x>[-+]?\d+(\.\d+)?),'
    r'(?P<lis3y>[-+]?\d+(\.\d+)?),'
    r'(?P<lis3z>[-+]?\d+(\.\d+)?)\s*$'
)


def parse_channel_line(line: str) -> dict:
    out = {}
    for tok in line.strip().split(","):
        tok = tok.strip()
        if not tok:
            continue
        parts = tok.split()
        if len(parts) != 2:
            continue
        name, val = parts
        name = name.rstrip(":")
        try:
            out[name] = float(val)
        except ValueError:
            continue
    return out


def subtract_offset(values, offset):
    return tuple(value - bias for value, bias in zip(values, offset))


def ensure_csv(file_path, header, sep=","):
    if not os.path.exists(file_path):
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f, delimiter=sep).writerow(header)
        return
    with open(file_path, "r", encoding="utf-8") as f:
        first = f.readline().strip()
    existing = [c.strip() for c in first.split(sep)] if first else []
    if existing != header:
        raise ValueError(
            f"CSV header mismatch. Expected {header}, found {existing}. "
            f"Use a new --out filename."
        )


def append_row(file_path, row_dict, header, sep=",", decimal="."):
    row = []
    for key in header:
        val = row_dict.get(key, "")
        if isinstance(val, bool):
            s = "1" if val else "0"
        elif isinstance(val, float):
            s = f"{val:.9f}"
        else:
            s = str(val)
        if decimal == ",":
            s = s.replace(".", ",")
        row.append(s)
    with open(file_path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f, delimiter=sep).writerow(row)


async def find_device(name=None, address=None, timeout=10.0):
    loop = asyncio.get_running_loop()
    if address:
        return await BleakScanner.find_device_by_address(address, timeout=timeout)
    if name:
        end = loop.time() + timeout
        while loop.time() < end:
            devices = await BleakScanner.discover(timeout=2.0)
            for d in devices:
                if d.name and name.lower() in d.name.lower():
                    return d
        return None
    devices = await BleakScanner.discover(timeout=timeout)
    for d in devices:
        if d.name and ("CIRCUITPY" in d.name or "CPB" in d.name):
            return d
    return None


class BleLineBuffer:
    def __init__(self):
        self.buf = bytearray()
        self.lines = deque()
        self.last_rx = time_mod.monotonic()

    def feed(self, data: bytes):
        self.last_rx = time_mod.monotonic()
        self.buf.extend(data)
        while True:
            try:
                i = self.buf.index(0x0A)
            except ValueError:
                break
            line = self.buf[:i].decode(errors="ignore").strip()
            del self.buf[:i + 1]
            if line:
                self.lines.append(line)

    def pop_all(self):
        out = []
        while self.lines:
            out.append(self.lines.popleft())
        return out

class QuantumSensorReader:
    """Owns the serial port. Measurement (configure + reader thread) can be started
    and stopped independently of the port, so calibration can use the port."""

    def __init__(self, port, baudrate=1_000_000, timeout=0.5):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._ser = None
        self._lock = threading.Lock()
        self._latest = (0.0, 0.0, 0.0)
        self._latest_ts = 0.0
        self._have_sample = False
        self._history = deque(maxlen=QUANTUM_TIME_HISTORY)
        self._stop = threading.Event()
        self._thread = None
        self.failed = False
        self.vector = True

    def open(self):
        if serial is None:
            raise RuntimeError("pyserial is not installed (pip install pyserial)")
        self._ser = serial.Serial(self.port, baudrate=self.baudrate, timeout=self.timeout)
        return self._ser

    def close(self):
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None

    def send(self, text, delay=0.05):
        self._ser.write((text + "\n").encode("ascii"))
        time_mod.sleep(delay)

    def configure_and_run(self, vector=True):
        ser = self._ser
        assert ser is not None
        self.vector = vector
        ser.reset_input_buffer()
        cmd = self.send
        cmd("setMagnetometerBinary 0")
        cmd("setMagnetometerSlidingWindow 0")
        cmd("MagnetometerSetTriggerType 0")
        cmd("enableMagnetometerConversion" if CONVERSION_ENABLED else "disableMagnetometerConversion")
        cmd("enableVector" if vector else "disableVector")
        cmd("setMagnetometerBinary 1" if BINARY_MODE else "setMagnetometerBinary 0")
        # values stored by the calibration (same set device_app loads)
        cmd("loadHF")
        cmd("loadZFS")
        for ch in range(3):
            cmd(f"loadDRFreq {ch}")
            cmd(f"loadSlope {ch}")
            cmd(f"setDualResonanceLockinFreq {ch} {LOCKIN_FREQS[ch]}")
            cmd(f"setDualResonanceLockinLowpassFreq {ch} {LOCKIN_LOWPASS}")
            cmd(f"setDualResonanceLockinModDepth {ch} {LOCKIN_MOD_DEPTH}")
        cmd("setDualResonanceFreq 3 0")
        cmd(f"setMagnetometerMSPP {int(MSPP)}")
        cmd("setIx 0.1")
        cmd("setIy 0.2")
        cmd("setIz 0.25")
        cmd("runMagnetometer")

    def stop_measurement(self):
        """Send 'stop', end the reader thread, flush. The port stays open."""
        ser = self._ser
        if ser is not None:
            try:
                ser.write(b"stop\n")
            except Exception:
                pass
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        if ser is not None:
            try:
                time_mod.sleep(0.2)
                ser.reset_input_buffer()
            except Exception:
                pass

    def stop_mag(self):                     # used by _shutdown
        self.stop_measurement()

    def start_reading(self):
        self._stop.clear()
        self.failed = False
        with self._lock:
            self._have_sample = False
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop_reading(self):                 # used by _shutdown / supervisor: also closes the port
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        self.close()

    def _read_loop(self):
        ser = self._ser
        while not self._stop.is_set():
            try:
                raw = ser.readline()
            except Exception as e:
                if self._stop.is_set():
                    return
                print(f"[mag] serial error: {e}")
                self.failed = True
                return
            if not raw:
                continue
            try:
                line = raw.decode("ascii", errors="strict").strip()
            except UnicodeDecodeError:
                continue
            if not line:
                continue
            parsed = parse_channel_line(line)
            if "Bx" not in parsed:
                continue
            if self.vector and not all(k in parsed for k in ("By", "Bz")):
                continue
            bx = parsed["Bx"] * SCALE_FACTOR
            by = parsed.get("By", 0.0) * SCALE_FACTOR      # single-axis mode: only Bx
            bz = parsed.get("Bz", 0.0) * SCALE_FACTOR_B
            with self._lock:
                self._latest = (bx, by, bz)
                self._latest_ts = time_mod.monotonic()
                self._have_sample = True
                self._history.append((time_mod.time(), bx, by, bz))

    def get_latest(self):
        with self._lock:
            if not self._have_sample:
                return 0.0, 0.0, 0.0, False
            age = time_mod.monotonic() - self._latest_ts
            bx, by, bz = self._latest
            return bx, by, bz, age <= MAX_SAMPLE_AGE_S

    def get_history(self):
        with self._lock:
            return list(self._history)

    def clear_history(self):
        with self._lock:
            self._history.clear()

    @property
    def has_sample(self):
        with self._lock:
            return self._have_sample

# ==================================================================
# Shared live state (plain data only -- never touches a UI element)
# ==================================================================
class AppState:
    def __init__(self, args, cfg):
        self.args = args
        self.cfg = cfg
        self.cfg_version = 0
        self.mode = args.mode                      # live | demo | replay
        self.window = args.window
        self.x_vals = deque(maxlen=self.window)
        self.lis2 = {ax: deque(maxlen=self.window) for ax in "xyz"}
        self.lis3 = {ax: deque(maxlen=self.window) for ax in "xyz"}
        self.quantum = {ax: deque(maxlen=self.window) for ax in "xyz"}
        self.line_buffer = BleLineBuffer()
        self.measuring = self.mode != "live"     # live: only after "Start measurement"
        self.vector_mode = True
        self.autostart_done = False
        self.cal = CalibrationManager(self) if self.mode == "live" else None
        self.reader = (QuantumSensorReader(args.mag_port, baudrate=args.mag_baud) if self.mode == "live" else None)
        self.ble_status = "idle"        # idle|scanning|connected|not_found|error|disconnected
        self.mag_status = "init"        # init|streaming|error|simulated
        self.ble_device_name = ""
        self.ble_reconnects = 0
        self.stale_run = 0
        self.rows_written = 0
        self.bad_lines = 0
        self.csv_errors = 0
        self.t0 = None
        self.last_sample_mono = None
        self.last_quantum = (0.0, 0.0, 0.0, False)
        self._recent_sample_times = deque(maxlen=50)
        self.log_csv = self.mode == "live"
        self.log_seq = 0
        self.log_buf = deque(maxlen=500)
        # full-session history for the engine (chronological)
        self.history_position = []
        self.history_time = [] 
        self.history = {ch: {ax: [] for ax in "xyz"} for ch in ("lis2", "lis3", "quantum")}
        self.data_version = 0
        self.engine_result = None
        self.engine_error = ""
        self.scan_id = 0
        self.scan_started = time_mod.time()
        self.demo_truth = None
        self.tasks = []

    # -- logging (data only) -------------------------------------
    def log(self, msg):
        line = f'[{datetime.now().strftime("%H:%M:%S")}] {msg}'
        print(line)
        self.log_seq += 1
        self.log_buf.append((self.log_seq, line))

    # -- sample ingestion ----------------------------------------
    def add_sample(self, dist_cm, lis2, lis3, quantum, fresh):
        args = self.args
        now = time_mod.monotonic()
        if self.t0 is None:
            self.t0 = now
        x = dist_cm if args.xaxis == "dist" else (now - self.t0)
        self.x_vals.append(x)
        for a, v in zip("xyz", lis2):
            self.lis2[a].append(float(v))
        for a, v in zip("xyz", lis3):
            self.lis3[a].append(float(v))
        for a, v in zip("xyz", quantum):
            self.quantum[a].append(float(v))

        self.history_position.append(float(dist_cm))
        self.history_time.append(time_mod.time())
        for a, v in zip("xyz", lis2):
            self.history["lis2"][a].append(float(v))
        for a, v in zip("xyz", lis3):
            self.history["lis3"][a].append(float(v))
        for a, v in zip("xyz", quantum):
            self.history["quantum"][a].append(float(v) if fresh else float("nan"))
        if len(self.history_position) > MAX_HISTORY:
            cut = MAX_HISTORY // 5
            del self.history_time[:cut]
            del self.history_position[:cut]
            for ch in self.history:
                for a in "xyz":
                    del self.history[ch][a][:cut]

        if self.log_csv:
            row = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Position": float(dist_cm),
                "lis2_Bx_G": float(lis2[0]), "lis2_By_G": float(lis2[1]), "lis2_Bz_G": float(lis2[2]),
                "lis3_Bx_G": float(lis3[0]), "lis3_By_G": float(lis3[1]), "lis3_Bz_G": float(lis3[2]),
                "quantum_Bx": float(quantum[0]), "quantum_By": float(quantum[1]),
                "quantum_Bz": float(quantum[2]), "quantum_valid": bool(fresh),
            }
            try:
                append_row(args.out, row, CSV_HEADER, sep=args.sep, decimal=args.decimal)
            except Exception as e:           # disk full / file locked: keep running
                self.csv_errors += 1
                if self.csv_errors in (1, 10, 100):
                    self.log(f"CSV write error ({self.csv_errors}x): {e}")
        self.rows_written += 1
        self._recent_sample_times.append(now)
        self.last_sample_mono = now
        self.last_quantum = (float(quantum[0]), float(quantum[1]), float(quantum[2]), bool(fresh))
        self.data_version += 1

    def quantum_latest(self):
        """(bx, by, bz, fresh, has_sample)"""
        if self.reader is not None:
            bx, by, bz, fresh = self.reader.get_latest()
            return bx, by, bz, fresh, self.reader.has_sample
        bx, by, bz, fresh = self.last_quantum
        return bx, by, bz, fresh, self.last_sample_mono is not None

    def seconds_since_sample(self):
        if self.last_sample_mono is None:
            return float("inf")
        return time_mod.monotonic() - self.last_sample_mono

    def sample_rate(self):
        t = self._recent_sample_times
        if len(t) >= 2 and time_mod.monotonic() - t[-1] < 3.0:
            dt = t[-1] - t[0]
            return (len(t) - 1) / dt if dt > 0 else 0.0
        return 0.0

    def reset_view(self):
        """Clear the live charts only; CSV and the detection history stay untouched."""
        for d in (self.x_vals, *self.lis2.values(), *self.lis3.values(), *self.quantum.values()):
            d.clear()
        if self.reader is not None:
            self.reader.clear_history()
        self.log("Plots view reset.")

    # -- scans ----------------------------------------------------
    def new_scan(self):
        for d in (self.x_vals, *self.lis2.values(), *self.lis3.values(), *self.quantum.values()):
            d.clear()
        self.history_time.clear()
        self.history_position.clear()
        for ch in self.history:
            for a in "xyz":
                self.history[ch][a].clear()
        self.engine_result = None
        self.data_version += 1
        self.cfg_version += 1
        self.t0 = None
        self.scan_id += 1
        self.scan_started = time_mod.time()
        self.log(f"New scan #{self.scan_id}")

    def snapshot(self):
        return {
            "pos": np.asarray(self.history_position, dtype=float),
            "lis2": {a: np.asarray(self.history["lis2"][a], dtype=float) for a in "xyz"},
            "lis3": {a: np.asarray(self.history["lis3"][a], dtype=float) for a in "xyz"},
            "quantum": {a: np.asarray(self.history["quantum"][a], dtype=float) for a in "xyz"},
        }

    def set_cfg(self, **changes):
        self.cfg = dataclasses.replace(self.cfg, **changes)
        self.cfg_version += 1


def resize_state_windows(state, new_window):
    state.window = new_window
    state.x_vals = deque(state.x_vals, maxlen=new_window)
    state.lis2 = {ax: deque(state.lis2[ax], maxlen=new_window) for ax in "xyz"}
    state.lis3 = {ax: deque(state.lis3[ax], maxlen=new_window) for ax in "xyz"}
    state.quantum = {ax: deque(state.quantum[ax], maxlen=new_window) for ax in "xyz"}


# ==================================================================
# Background tasks (run on NiceGUI's event loop; no browser needed)
# ==================================================================
async def ingest_loop(state):
    while True:
        try:
            if not state.measuring:
                state.line_buffer.pop_all()          # discard, don't log
                await asyncio.sleep(0.05)
                continue
            for line in state.line_buffer.pop_all():
                m = CSV_RE.match(line)
                if not m:
                    state.bad_lines += 1
                    continue
                dist = float(m.group("dist"))
                if not np.isfinite(dist):
                    continue
                lis2 = tuple(float(m.group(k)) for k in ("lis2x", "lis2y", "lis2z"))
                lis3 = tuple(float(m.group(k)) for k in ("lis3x", "lis3y", "lis3z"))
                lis2 = subtract_offset(lis2, LIS2_OFFSET_G)
                lis3 = subtract_offset(lis3, LIS3_OFFSET_G)
                bx, by, bz, fresh = state.reader.get_latest()
                state.add_sample(dist, lis2, lis3, (bx, by, bz), fresh)
        except Exception as e:
            state.log(f"ingest error: {e}")
        await asyncio.sleep(0.05)


async def engine_loop(state):
    last = None
    last_err = 0.0
    while True:
        await asyncio.sleep(state.args.engine_interval)
        key = (state.data_version, state.cfg_version)
        if key == last or len(state.history_position) < 15:
            continue
        snap, cfg = state.snapshot(), state.cfg
        try:
            res = await asyncio.to_thread(eng.run_engine, snap, cfg, state.data_version * 1000 + state.cfg_version)
            state.engine_result = res
            state.engine_error = ""
            last = key
        except Exception as e:
            state.engine_error = str(e)
            if time_mod.monotonic() - last_err > 30:
                state.log(f"Engine error: {e}")
                last_err = time_mod.monotonic()
            last = key


async def ble_supervisor(state):
    args = state.args
    backoff = 2.0
    address = args.addr
    first = True
    last_scan_err = None
    while True:
        state.ble_status = "scanning"
        if first:
            state.log(f"Scanning for BLE device (name={args.name!r}, addr={args.addr!r})…")
        try:
            dev = await find_device(name=args.name, address=address, timeout=15.0)
        except Exception as e:
            state.ble_status = "error"
            if str(e) != last_scan_err:              # don't repeat the same error every retry
                state.log(f"ERROR during BLE scan: {e}  (retrying)")
                last_scan_err = str(e)
            dev = None
        if not dev:
            if state.ble_status != "error":
                state.ble_status = "not_found"
                if first:
                    state.log("BLE device not found. Check --name/--addr, that the board is powered "
                              "and advertising, and that Bluetooth is on. Retrying…")
            first = False
            await asyncio.sleep(backoff)
            backoff = min(backoff * 1.5, 10.0)
            continue
        state.ble_device_name = dev.name or dev.address
        address = dev.address                       # faster reconnects
        state.log(f"Found {state.ble_device_name}, connecting…")
        try:
            async with BleakClient(dev) as client:
                state.ble_status = "connected"
                state.line_buffer.last_rx = time_mod.monotonic()
                await client.start_notify(UART_TX_UUID, lambda _h, data: state.line_buffer.feed(data))
                state.log("Connected. Receiving data.")
                backoff = 2.0
                first = False
                while client.is_connected:
                    await asyncio.sleep(0.2)
                    # A notification pause is not necessarily a lost Bluetooth
                    # connection (for example while the board is idle).  A value
                    # of 0 disables this data watchdog and keeps the client open.
                    if (args.ble_watchdog_s > 0
                            and time_mod.monotonic() - state.line_buffer.last_rx > args.ble_watchdog_s):
                        state.log(f"No BLE data for {args.ble_watchdog_s:.0f}s - reconnecting…")
                        break
        except Exception as e:
            state.ble_status = "error"
            state.log(f"BLE error: {e}")
        state.ble_status = "disconnected"
        state.ble_reconnects += 1
        state.log("BLE disconnected. Reconnecting…")
        first = False
        await asyncio.sleep(backoff)


def begin_measurement(state):
    r = state.reader
    r.configure_and_run(state.vector_mode)
    r.start_reading()
    state.measuring = True
    state.mag_status = "streaming"


def end_measurement(state):
    state.measuring = False
    state.reader.stop_measurement()
    if state.mag_status == "streaming":
        state.mag_status = "connected"


async def quantum_supervisor(state):
    args = state.args
    last_err = -1e9
    while True:
        if state.cal.busy:                       # calibration owns the port
            await asyncio.sleep(1.0)
            continue
        r = state.reader
        if state.mag_status == "streaming" and r.failed:
            state.log("Quantum sensor connection lost - reopening…")
            await asyncio.to_thread(r.stop_reading)
            r = state.reader = QuantumSensorReader(args.mag_port, baudrate=args.mag_baud)
            state.mag_status = "error"
        if state.mag_status in ("init", "error"):
            try:
                r.close()
                await asyncio.to_thread(r.open)
                state.mag_status = "connected"
                state.log(f"Quantum sensor port {args.mag_port} @ {args.mag_baud} baud open.")
                if args.autostart and not state.autostart_done:
                    state.autostart_done = True
                    state.measuring = True
                if state.measuring:              # resume after a reconnect / autostart
                    await asyncio.to_thread(begin_measurement, state)
            except Exception as e:
                r.close()
                state.mag_status = "error"
                if time_mod.monotonic() - last_err > 30:
                    ports = ""
                    if serial is not None:
                        try:
                            ports = ", ".join(p.device for p in serial.tools.list_ports.comports()) or "none"
                        except Exception:
                            ports = "?"
                    state.log(f"Quantum sensor not available on {args.mag_port}: {e}"
                              + (f"  (serial ports found: {ports}). Retrying every 5 s." if ports else ""))
                    last_err = time_mod.monotonic()
        await asyncio.sleep(5.0)
    


async def demo_task(state):
    args = state.args
    seed = args.demo_seed
    state.mag_status = "simulated"
    while True:
        scan = DemoScan(seed=seed, length_cm=args.beam_length_cm - 20, artifacts=args.demo_artifacts,
                        quantum_online=not args.demo_no_quantum)
        state.new_scan()
        state.demo_truth = scan.truth
        state.log(f"DEMO scan seed={seed}: {len(scan.truth)} simulated rebars")
        dt = 1.0 / (scan.hz * args.demo_speed)
        for s in scan.samples():
            state.add_sample(s["pos"], s["lis2"], s["lis3"], s["quantum"], s["fresh"])
            await asyncio.sleep(dt)
        await asyncio.sleep(args.demo_pause_s)
        seed += 1


async def replay_task(state):
    args = state.args
    state.mag_status = "simulated"
    rows = []
    with open(args.replay, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter=args.sep):
            try:
                rows.append((
                    float(r["Position"]),
                    (float(r["lis2_Bx_G"]), float(r["lis2_By_G"]), float(r["lis2_Bz_G"])),
                    (float(r["lis3_Bx_G"]), float(r["lis3_By_G"]), float(r["lis3_Bz_G"])),
                    (float(r["quantum_Bx"]), float(r["quantum_By"]), float(r["quantum_Bz"])),
                    str(r.get("quantum_valid", "1")).strip() in ("1", "True", "true"),
                ))
            except (KeyError, ValueError):
                continue
    if not rows:
        state.log(f"Replay file {args.replay} has no usable rows.")
        return
    dt = 1.0 / (args.replay_hz * args.replay_speed)
    while True:
        state.new_scan()
        state.log(f"REPLAY {os.path.basename(args.replay)}: {len(rows)} rows")
        for pos, l2, l3, q, fresh in rows:
            state.add_sample(pos, l2, l3, q, fresh)
            await asyncio.sleep(dt)
        if not args.loop:
            state.log("Replay finished.")
            return
        await asyncio.sleep(args.demo_pause_s)


# ==================================================================
# Engineer page
# ==================================================================
def make_chart_options(title, unit):
    opt = {
        "backgroundColor": "transparent",
        "textStyle": {"color": "#cbd5e1", "fontFamily": C.FONT_MONO},
        "grid": {"left": 55, "right": 20, "top": 40, "bottom": 30},
        "legend": {"top": 0, "textStyle": {"color": "#94a3b8"}},
        "tooltip": {"trigger": "axis"},
        "animation": False,
        "title": {"text": title, "left": 0, "textStyle": {"color": "#e2e8f0", "fontSize": 14, "fontFamily": C.FONT_SANS}},
        "xAxis": {"type": "value", "scale": True, "axisLine": {"lineStyle": {"color": "#475569"}},
                  "splitLine": {"lineStyle": {"color": "#1e293b"}}},
        "yAxis": {"type": "value", "name": unit, "scale": True, "nameTextStyle": {"color": "#94a3b8"},
                  "axisLine": {"lineStyle": {"color": "#475569"}}, "splitLine": {"lineStyle": {"color": "#1e293b"}}},
        "series": [
            {"name": f"{title} {ax}", "type": "line", "showSymbol": False, "smooth": True,
             "lineStyle": {"width": 2, "color": C.COLORS[ax], "shadowBlur": 8, "shadowColor": C.COLORS[ax]},
             "areaStyle": {"opacity": 0.12, "color": {"type": "linear", "x": 0, "y": 0, "x2": 0, "y2": 1,
                           "colorStops": [{"offset": 0, "color": C.COLORS[ax]}, {"offset": 1, "color": "rgba(0,0,0,0)"}]}},
             "data": []}
            for ax in "xyz"
        ],
    }
    return opt


def make_quantum_time_options(unit):
    return {
        "backgroundColor": "transparent",
        "textStyle": {"color": "#cbd5e1", "fontFamily": C.FONT_MONO},
        "grid": {"left": 55, "right": 20, "top": 40, "bottom": 45},
        "legend": {"top": 0, "textStyle": {"color": "#94a3b8"}},
        "tooltip": {"trigger": "axis"},
        "animation": False,
        "title": {"text": "Quantum Sensor — real time", "left": 0,
                  "textStyle": {"color": "#e2e8f0", "fontSize": 14, "fontFamily": C.FONT_SANS}},
        "xAxis": {"type": "time", "name": "Time", "nameLocation": "middle", "nameGap": 30,
                  "axisLabel": {"formatter": "{HH}:{mm}:{ss}"},
                  "nameTextStyle": {"color": "#94a3b8"}, "axisLine": {"lineStyle": {"color": "#475569"}},
                  "splitLine": {"lineStyle": {"color": "#1e293b"}}},
        "yAxis": {"type": "value", "name": unit, "scale": True,
                  "nameTextStyle": {"color": "#94a3b8"}, "axisLine": {"lineStyle": {"color": "#475569"}},
                  "splitLine": {"lineStyle": {"color": "#1e293b"}}},
        "series": [
            {"name": f"Quantum B{ax.upper()}", "type": "line", "showSymbol": False, "smooth": False,
             "lineStyle": {"width": 1.6, "color": C.COLORS[ax]}, "data": []}
            for ax in "xyz"
        ],
    }


def make_peaks_options():
    return {
        "backgroundColor": "transparent",
        "textStyle": {"color": "#cbd5e1", "fontFamily": C.FONT_MONO},
        "grid": {"left": 75, "right": 25, "top": 50, "bottom": 40},
        "title": {"text": "Rebar detection", "left": 0,
                  "textStyle": {"color": "#e2e8f0", "fontSize": 14, "fontFamily": C.FONT_SANS}},
        "legend": {"top": 0, "right": 10, "textStyle": {"color": "#94a3b8"}},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "line"}},
        "animation": False,
        "xAxis": {"type": "value", "name": "Position [cm]", "nameLocation": "middle", "nameGap": 24, "scale": True,
                  "nameTextStyle": {"color": "#94a3b8"}, "axisLine": {"lineStyle": {"color": "#475569"}},
                  "splitLine": {"lineStyle": {"color": "#1e293b"}}},
        "yAxis": {"type": "value", "name": "", "nameLocation": "middle", "nameGap": 55, "scale": True,
                  "nameTextStyle": {"color": "#94a3b8"}, "axisLine": {"lineStyle": {"color": "#475569"}},
                  "splitLine": {"lineStyle": {"color": "#1e293b"}}},
        "series": [
            {"name": "detection signal", "type": "line", "showSymbol": False, "smooth": False,
             "lineStyle": {"width": 1.8, "color": "#e2e8f0"}, "data": []},
            {"name": "detected", "type": "scatter", "symbolSize": 11, "z": 5,
             "itemStyle": {"color": C.C_BAD, "borderColor": "#fff", "borderWidth": 1.2}, "data": []},
        ],
    }


CELL = "padding:3px 22px 3px 0;text-align:left;"


def fused_table_html(rebars):
    if not rebars:
        return '<div style="color:#94a3b8;font-size:0.8rem">No rebars detected yet.</div>'
    head = "".join(f'<th style="{CELL}">{h}</th>' for h in
                   ("#", "Position [cm]", "± [cm]", "Spacing [cm]", "Confidence", "Votes", "Status", "Sources"))
    rows = ""
    for i, r in enumerate(rebars):
        sp = f"{r['spacing_prev']:.1f}" if r["spacing_prev"] is not None else "-"
        col = C.C_OK if r["status"] == "confirmed" else C.C_WARN
        src = " ".join(k.replace("/Bx", "") for k in r["members"])
        rows += (f'<tr><td style="{CELL}">{i + 1}</td>'
                 f'<td style="{CELL}color:#fca5a5">{r["x"]:.2f}</td>'
                 f'<td style="{CELL}">{r["sigma_cm"]:.2f}</td>'
                 f'<td style="{CELL}">{sp}</td>'
                 f'<td style="{CELL}">{r["conf"] * 100:.0f}%</td>'
                 f'<td style="{CELL}">{r["votes"]}/{r["available"]}</td>'
                 f'<td style="{CELL}color:{col}">{r["status"]}</td>'
                 f'<td style="{CELL}color:#94a3b8">{src}</td></tr>')
    return (f'<table style="font-family:{C.FONT_MONO};font-size:0.75rem;color:#cbd5e1;border-collapse:collapse">'
            f'<thead><tr style="color:#94a3b8">{head}</tr></thead><tbody>{rows}</tbody></table>')


def stats_table_html(stats):
    def f(v, spec):
        return "-" if v is None else format(v, spec)
    head = "".join(f'<th style="{CELL}">{h}</th>' for h in
                   ("Channel", "Noise (raw)", "Signal SNR", "Detected", "Matched", "Recall", "Position σ [cm]"))
    rows = ""
    for ch in eng.CHANNELS:
        s = stats.get(ch, {"ok": False})
        if not s.get("ok"):
            rows += (f'<tr><td style="{CELL}">{ch}</td><td colspan="6" style="{CELL}color:#94a3b8">'
                     f'no valid data</td></tr>')
            continue
        rows += (f'<tr><td style="{CELL}">{ch}</td><td style="{CELL}">{s["noise_raw"]:.4g}</td>'
                 f'<td style="{CELL}">{f(s["snr_median"], ".1f")}×</td><td style="{CELL}">{s["n_det"]}</td>'
                 f'<td style="{CELL}">{s["matched"]}/{s["n_expected"]}</td>'
                 f'<td style="{CELL}">{f(None if s["recall"] is None else s["recall"] * 100, ".0f")}%</td>'
                 f'<td style="{CELL}">{f(s["sigma_cm"], ".2f")}</td></tr>')
    return (f'<table style="font-family:{C.FONT_MONO};font-size:0.75rem;color:#cbd5e1;border-collapse:collapse">'
            f'<thead><tr style="color:#94a3b8">{head}</tr></thead><tbody>{rows}</tbody></table>')


def export_rebars_csv(state):
    res = state.engine_result
    if res is None or not res.rebars:
        return None
    base = os.path.splitext(state.args.out)[0] if state.log_csv else "rebars"
    path = f"{base}_rebars_{datetime.now().strftime('%H%M%S')}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n", "position_cm", "sigma_cm", "spacing_prev_cm", "confidence", "votes",
                    "available_sources", "status", "sources"])
        for i, r in enumerate(res.rebars):
            w.writerow([i + 1, f"{r['x']:.3f}", f"{r['sigma_cm']:.3f}",
                        "" if r["spacing_prev"] is None else f"{r['spacing_prev']:.2f}",
                        f"{r['conf']:.3f}", r["votes"], r["available"], r["status"],
                        ";".join(f"{k}={v:.2f}" for k, v in r["members"].items())])
    return path

async def on_start(state):
    if state.cal.busy or state.measuring:
        return
    if state.mag_status != "connected":
        ui.notify("Quantum sensor is not connected.", type="warning")
        return
    try:
        await asyncio.to_thread(begin_measurement, state)
        state.log(f"Measurement started ({'vector' if state.vector_mode else 'single-axis'} mode).")
    except Exception as e:
        state.mag_status = "error"
        ui.notify(f"Could not start: {e}", type="negative")


async def on_stop(state):
    if state.cal.busy:
        return
    await asyncio.to_thread(end_measurement, state)
    state.log("Measurement stopped.")


async def on_calibrate(state, refs, kind):
    cal = state.cal
    if cal.busy:
        return
    if state.mag_status not in ("connected", "streaming"):
        ui.notify("Quantum sensor is not connected.", type="warning")
        return
    v = refs['cal_atten'].value
    atten = 20.0 if v is None else float(v)
    await asyncio.to_thread(cal.run, kind, atten, int(refs['cal_nens'].value))

async def on_odmr(state, refs):
    cal = state.cal
    if cal.busy:
        return
    if state.mag_status not in ("connected", "streaming"):
        ui.notify("Quantum sensor is not connected.", type="warning")
        return

    def val(key, default):
        v = refs[key].value
        return default if v is None else float(v)

    await asyncio.to_thread(
        cal.run_odmr,
        val('odmr_f0', 2.865), val('odmr_f1', 2.875), val('odmr_df', 0.5),
        int(val('odmr_mspp', 10)), val('odmr_atten', 20),
        val('odmr_ix', 0.1), val('odmr_iy', 0.2), val('odmr_iz', 0.25))


def make_odmr_options():
    opt = make_cal_options()
    opt["title"]["text"] = "ODMR"
    opt["xAxis"]["name"] = "Frequency [GHz]"
    return opt

async def on_accept(state):
    await asyncio.to_thread(state.cal.apply)


def make_cal_options():
    return {
        "backgroundColor": "transparent",
        "textStyle": {"color": "#cbd5e1", "fontFamily": C.FONT_MONO},
        "grid": {"left": 70, "right": 25, "top": 45, "bottom": 45},
        "legend": {"top": 0, "textStyle": {"color": "#94a3b8"}},
        "tooltip": {"trigger": "axis"},
        "animation": False,
        "title": {"text": "Calibration sweep", "left": 0,
                  "textStyle": {"color": "#e2e8f0", "fontSize": 14, "fontFamily": C.FONT_SANS}},
        "xAxis": {"type": "value", "name": "Frequency − 2.87 GHz [MHz]", "nameLocation": "middle", "nameGap": 28,
                  "scale": True, "nameTextStyle": {"color": "#94a3b8"},
                  "axisLine": {"lineStyle": {"color": "#475569"}}, "splitLine": {"lineStyle": {"color": "#1e293b"}}},
        "yAxis": {"type": "value", "name": "signal", "scale": True, "nameTextStyle": {"color": "#94a3b8"},
                  "axisLine": {"lineStyle": {"color": "#475569"}}, "splitLine": {"lineStyle": {"color": "#1e293b"}}},
        "series": [],
    }


def build_control_ui(state, refs):
    if state.mode != "live":
        return
    with ui.card().classes('dash-card w-full p-4'):
        ui.label('MEASUREMENT').classes('kpi-label')
        with ui.row().classes('items-center gap-6 flex-wrap'):
            refs['btn_start'] = ui.button('Start measurement', on_click=lambda: on_start(state)).props('color=positive')
            refs['btn_stop'] = ui.button('Stop measurement', on_click=lambda: on_stop(state)).props('outline')
            refs['vector_switch'] = ui.switch(
                'Vector mode (Bx, By, Bz)', value=state.vector_mode,
                on_change=lambda e: setattr(state, 'vector_mode', bool(e.value)))
            ui.label('Mode can only be changed while stopped.').classes('kpi-label').style('color:#64748b')

    with ui.card().classes('dash-card w-full p-4'):
        ui.label('CALIBRATION  (stops the measurement)').classes('kpi-label')
        with ui.row().classes('items-end gap-6 flex-wrap'):
            refs['cal_atten'] = ui.number('MW attenuation [dB]', value=20, step=1).classes('w-44')
            refs['cal_nens'] = ui.select([1, 2, 3, 4], value=3, label='N subensembles').classes('w-40')
            refs['btn_cal_center'] = ui.button('1 · Center calibration',
                                               on_click=lambda: on_calibrate(state, refs, 'center'))
            refs['btn_cal_slope'] = ui.button('2 · Slope calibration',
                                              on_click=lambda: on_calibrate(state, refs, 'slope'))
            refs['btn_cal_abort'] = ui.button('Abort', on_click=state.cal.abort_event.set).props('outline color=negative')
        refs['cal_progress'] = ui.linear_progress(value=0, show_value=False).classes('w-full mt-2')
        refs['cal_status'] = ui.label('Idle').classes('kpi-label').style('color:#cbd5e1')
        refs['cal_timer'] = ui.label('').classes('kpi-label').style('color:#38bdf8')
        refs['cal_flags'] = ui.label('').classes('kpi-label').style('color:#94a3b8')
        refs['cal_chart'] = ui.echart(make_cal_options()).classes('w-full').style('height:340px')
        refs['cal_result'] = C.html('').classes('w-full')
        with ui.row().classes('gap-4 mt-2'):
            refs['btn_cal_accept'] = ui.button('✔ Good - write to device', on_click=lambda: on_accept(state)).props('color=positive')
            refs['btn_cal_discard'] = ui.button('✘ Bad - discard / repeat', on_click=state.cal.discard).props('outline color=negative')
    with ui.card().classes('dash-card w-full p-4'):
        ui.label('ODMR MEASUREMENT  (stops the measurement)').classes('kpi-label')
        with ui.row().classes('items-end gap-6 flex-wrap'):
            refs['odmr_f0'] = ui.number('Start [GHz]', value=2.82, step=0.001, format='%.4f').classes('w-32')
            refs['odmr_f1'] = ui.number('End [GHz]', value=2.92, step=0.001, format='%.4f').classes('w-32')
            refs['odmr_df'] = ui.number('Step [MHz]', value=0.5, step=0.1).classes('w-28')
            refs['odmr_mspp'] = ui.number('ms / point', value=10, step=1, min=1).classes('w-28')
            refs['odmr_atten'] = ui.number('MW attenuation [dB]', value=20, step=1).classes('w-40')
            refs['odmr_ix'] = ui.number('Ix', value=0.1, step=0.05).classes('w-24')
            refs['odmr_iy'] = ui.number('Iy', value=0.2, step=0.05).classes('w-24')
            refs['odmr_iz'] = ui.number('Iz', value=0.25, step=0.05).classes('w-24')
            refs['btn_odmr'] = ui.button('Run ODMR', on_click=lambda: on_odmr(state, refs))
            refs['btn_odmr_abort'] = ui.button('Abort', on_click=state.cal.abort_event.set).props('outline color=negative')
        refs['odmr_progress'] = ui.linear_progress(value=0, show_value=False).classes('w-full mt-2')
        refs['odmr_status'] = ui.label('Idle').classes('kpi-label').style('color:#cbd5e1')
        refs['odmr_timer'] = ui.label('').classes('kpi-label').style('color:#38bdf8')
        refs['odmr_chart'] = ui.echart(make_odmr_options()).classes('w-full').style('height:320px')

def _mmss(s):
    s = int(max(0, s))
    return f"{s // 60:02d}:{s % 60:02d}"

def render_control(state, refs):
    if state.mode != "live" or 'btn_start' not in refs:
        return
    cal = state.cal
    busy = cal.busy
    refs['btn_start'].set_enabled(not busy and not state.measuring and state.mag_status == "connected")
    refs['btn_stop'].set_enabled(not busy and state.measuring)
    refs['vector_switch'].set_enabled(not busy and not state.measuring)

    reviewing = cal.stage == "review"
    refs['btn_cal_center'].set_enabled(not busy and not reviewing)
    refs['btn_cal_slope'].set_enabled(not busy and not reviewing and cal.center_applied)
    refs['btn_cal_abort'].set_enabled(busy and cal.stage == "acquiring")
    refs['btn_cal_accept'].set_enabled(reviewing and bool(cal.result and cal.result.get("ok")))
    refs['btn_cal_discard'].set_enabled(reviewing)
    refs['cal_atten'].set_enabled(not busy)
    refs['cal_nens'].set_enabled(not busy)

    refs['cal_progress'].set_value(cal.progress if cal.stage in ("acquiring", "analysing") else 0)
    refs['cal_status'].set_text(cal.message)
    if cal.stage == "acquiring" and cal.freqs:
        left = "estimating…" if cal.time_left is None else _mmss(cal.time_left)
        refs['cal_timer'].set_text(f"elapsed {_mmss(cal.elapsed)}  ·  remaining ~{left}")
    else:
        refs['cal_timer'].set_text("")
    refs['cal_flags'].set_text(
        f"center: {'✔ applied' if cal.center_applied else '– not done'}   ·   "
        f"slope: {'✔ stored' if cal.slope_applied else '– not done'}")

    key = (cal.version, len(cal.freqs), cal.stage)
    if key != refs.get('_cal_key'):
        refs['_cal_key'] = key
        chart = refs['cal_chart']
        chart.options["series"] = cal.chart_series()
        title = {"center": "Center calibration (ODMR)", "slope": "Slope calibration"}.get(cal.kind, "Calibration sweep")
        chart.options["title"]["text"] = title
        chart.update()
        refs['cal_result'].set_content(cal.result_html())
    # ---- ODMR card -----------------------------------------------------
    odmr_running = busy and cal.odmr_stage == "acquiring"
    refs['btn_odmr'].set_enabled(not busy and not reviewing)
    refs['btn_odmr_abort'].set_enabled(odmr_running)
    for k in ('odmr_f0', 'odmr_f1', 'odmr_df', 'odmr_mspp', 'odmr_atten', 'odmr_ix', 'odmr_iy', 'odmr_iz'):
        refs[k].set_enabled(not busy)
    refs['odmr_progress'].set_value(cal.odmr_progress)
    refs['odmr_status'].set_text(cal.odmr_message)
    if odmr_running and cal.odmr_freqs:
        left = "estimating…" if cal.odmr_time_left is None else _mmss(cal.odmr_time_left)
        refs['odmr_timer'].set_text(f"elapsed {_mmss(cal.odmr_elapsed)}  ·  remaining ~{left}")
    else:
        refs['odmr_timer'].set_text("")
    okey = (cal.odmr_version, len(cal.odmr_freqs), cal.odmr_stage)
    if okey != refs.get('_odmr_key'):
        refs['_odmr_key'] = okey
        chart = refs['odmr_chart']
        chart.options["series"] = [{
            "name": "ODMR", "type": "line", "showSymbol": False,
            "lineStyle": {"width": 1.6, "color": "#38bdf8"},
            "data": [[f / 1e9, v] for f, v in zip(cal.odmr_freqs, cal.odmr_values)],
        }]
        chart.update()


def build_engineer_ui(state):
    args = state.args
    ui.add_head_html(C.HEAD_HTML)
    ui.dark_mode().enable()
    refs = {}

    with ui.header().classes('items-center justify-between px-6').style(
            'background: rgba(11,15,26,0.85); backdrop-filter: blur(10px); border-bottom: 1px solid rgba(255,255,255,0.06);'):
        with ui.row().classes('items-center gap-4'):
            ui.label('QUANTUM MAG // ENGINEER').classes('glow-title text-xl')
            ui.link('Visitor view ↗', '/visitor', new_tab=True).classes('text-sm')
        with ui.row().classes('gap-3 items-center'):
            refs['mode_pill'] = C.html('').classes('status-pill')
            refs['ble_pill'] = C.html(C.status_dot(False) + 'BLE…').classes('status-pill')
            refs['mag_pill'] = C.html(C.status_dot(False) + 'Quantum…').classes('status-pill')
            refs['rate_pill'] = C.html(C.status_dot(True) + '0.0 Hz').classes('status-pill')
            refs['csv_pill'] = C.html(C.status_dot(True) + '0 rows').classes('status-pill')
            refs['eng_pill'] = C.html(C.status_dot(True) + 'engine…').classes('status-pill')

    with ui.column().classes('w-full p-6 gap-6'):
        build_control_ui(state, refs)
        with ui.row().classes('w-full gap-4'):
            for ax in "xyz":
                with ui.card().classes('dash-card flex-1 p-4'):
                    ui.label(f'QUANTUM B{ax.upper()}').classes('kpi-label')
                    refs[f'kpi_{ax}'] = ui.label('—').classes('kpi-value').style(f'color:{C.COLORS[ax]}')
            with ui.card().classes('dash-card flex-1 p-4'):
                ui.label('POSITION').classes('kpi-label')
                refs['kpi_pos'] = ui.label('—').classes('kpi-value').style('color:#e2e8f0')

        with ui.card().classes('dash-card w-full p-4'):
            with ui.row().classes('items-center gap-8 flex-wrap'):
                ui.label('Display smoothing').classes('kpi-label')
                refs['smooth_slider'] = ui.slider(min=1, max=31, step=2, value=args.smooth).classes('w-48')
                ui.label('Window').classes('kpi-label')
                refs['window_slider'] = ui.slider(min=50, max=1000, step=50, value=state.window).classes('w-48')
                refs['paused'] = ui.switch('Freeze display')
                ui.button('New scan', on_click=state.new_scan).props('outline')
                ui.button('Export rebars CSV', on_click=lambda: _export(state)).props('outline')
                ui.button('Reset plots', on_click=state.reset_view).props('outline')

        opts = {k: make_chart_options(t, u) for k, t, u in (
            ('lis2', 'LIS2', 'G'), ('lis3', 'LIS3', 'G'), ('quantum', 'Quantum Sensor', args.mag_unit))}
        for key, h in (('lis2', 220), ('lis3', 220), ('quantum', 260)):
            with ui.card().classes('dash-card w-full p-4'):
                refs[f'chart_{key}'] = ui.echart(opts[key]).classes('w-full').style(f'height:{h}px')

        with ui.card().classes('dash-card w-full p-4'):
            refs['chart_quantum_time'] = ui.echart(make_quantum_time_options(args.mag_unit)).classes('w-full').style('height:260px')

        # ---- fused result -----------------------------------------
        with ui.card().classes('dash-card w-full p-4'):
            ui.label('DETECTED REBARS (fused across sensors)').classes('kpi-label')
            refs['summary'] = ui.label('Waiting for data…').classes('kpi-label').style('color:#cbd5e1')
            refs['fused_table'] = C.html('').classes('w-full mt-2')

        # ---- per-source inspection --------------------------------
        with ui.card().classes('dash-card w-full p-4'):
            with ui.row().classes('items-center gap-6 flex-wrap'):
                ui.label('REBAR PEAKS').classes('kpi-label')
                refs['plot_channel'] = ui.select(['lis2', 'lis3', 'quantum'], value='lis2', label='Channel').classes('w-32')
                refs['plot_component'] = ui.select(['Bx', 'By', 'Bz', 'B_total'], value='Bx', label='Component').classes('w-32')
            refs['chart_peaks'] = ui.echart(make_peaks_options()).classes('w-full').style('height:300px')
            refs['peaks_info'] = ui.label('').classes('kpi-label').style('color:#94a3b8')

        with ui.card().classes('dash-card w-full p-4'):
            ui.label('SENSOR COMPARISON').classes('kpi-label')
            refs['stats_table'] = C.html('').classes('w-full mt-2')

        # ---- engine settings --------------------------------------
        cfg = state.cfg
        with ui.card().classes('dash-card w-full p-4'):
            ui.label('DETECTION SETTINGS (apply immediately)').classes('kpi-label')
            with ui.row().classes('items-end gap-8 flex-wrap').style('margin-top:28px'):
                with ui.column().classes('gap-0'):
                    ui.label('Sensitivity × (lower = more sensitive)').classes('kpi-label')
                    refs['s_sens'] = ui.slider(min=0.3, max=3.0, step=0.1, value=cfg.sensitivity).props('label-always').classes('w-56')
                with ui.column().classes('gap-0'):
                    ui.label('Smoothing [cm]').classes('kpi-label')
                    refs['s_smooth'] = ui.slider(min=5, max=31, step=2, value=int(cfg.smooth_cm)).props('label-always').classes('w-48')
                refs['s_method'] = ui.select({'original': 'original (professor)', 'robust': 'robust (cm, smoothing, centre)'},
                                             value=cfg.method, label='Method').classes('w-64')
                refs['s_spacing_n'] = ui.number('Min spacing [samples] (original)', value=cfg.min_spacing_samples,
                                                min=1, max=500, step=1).classes('w-56')
                refs['s_spacing'] = ui.number('Min spacing [cm] (robust)', value=cfg.min_spacing_cm, min=5, max=200, step=1).classes('w-44')
                refs['s_locate'] = ui.select({'center': 'bump centre', 'slope': 'steepest slope (original)'},
                                             value=cfg.locate, label='Position = ').classes('w-52')
                refs['s_votes'] = ui.select([1, 2, 3], value=cfg.min_votes, label='Sensors to confirm').classes('w-40')
                refs['s_comps'] = ui.select(list(eng.COMPONENTS), value=list(cfg.fuse_components), multiple=True,
                                            label='Fuse components').classes('w-48')
                refs['s_rf'] = ui.select([1, 2], value=cfg.rf_number, label='RF pass').classes('w-24')
                refs['s_factors'] = ui.switch('Use RF sign factors', value=cfg.use_factors)

        with ui.card().classes('dash-card w-full p-4'):
            ui.label('CONSOLE').classes('kpi-label')
            refs['log'] = ui.log(max_lines=300).classes('w-full').style(
                f'height:160px; font-family: {C.FONT_MONO}; font-size:0.75rem; background: rgba(0,0,0,0.25);')

    def apply_cfg(_=None):
        comps = tuple(refs['s_comps'].value or ()) or cfg.fuse_components
        state.set_cfg(
            sensitivity=float(refs['s_sens'].value), smooth_cm=float(refs['s_smooth'].value),
            min_spacing_cm=float(refs['s_spacing'].value or state.cfg.min_spacing_cm),
            method=refs['s_method'].value,
            min_spacing_samples=int(refs['s_spacing_n'].value or state.cfg.min_spacing_samples),
            locate=refs['s_locate'].value, min_votes=int(refs['s_votes'].value),
            fuse_components=comps, rf_number=int(refs['s_rf'].value), use_factors=bool(refs['s_factors'].value))

    for k in ('s_sens', 's_smooth', 's_method', 's_spacing_n', 's_spacing', 's_locate', 's_votes', 's_comps', 's_rf', 's_factors'):
        refs[k].on_value_change(apply_cfg)
    return refs


def _export(state):
    path = export_rebars_csv(state)
    if path:
        C.download_file(path)
        state.log(f"Exported {path}")
    else:
        ui.notify("No rebars to export yet.", type="warning")


def _overlay_lines(rebars, lo, hi):
    data = []
    for r in rebars:
        if lo <= r["x"] <= hi:
            ok = r["status"] == "confirmed"
            data.append({"xAxis": float(r["x"]),
                         "lineStyle": {"color": C.C_REBAR if ok else C.C_WARN, "type": "dashed" if ok else "dotted",
                                       "width": 1.6 if ok else 1.2},
                         "label": {"formatter": f'{r["x"]:.0f}', "color": C.C_REBAR if ok else C.C_WARN, "fontSize": 10}})
    return {"silent": True, "symbol": "none", "data": data}

def rebar_times_ms(state, rebars):
    """(time_ms, position_cm, confirmed) for each rebar: when that position was scanned."""
    n = min(len(state.history_position), len(state.history_time))
    if n == 0 or not rebars:
        return []
    pos = np.asarray(state.history_position[:n], dtype=float)
    out = []
    for r in rebars:
        i = int(np.argmin(np.abs(pos - r["x"])))
        out.append((state.history_time[i] * 1000.0, float(r["x"]), r["status"] == "confirmed"))
    return out


def _time_overlay_lines(marks, t_lo, t_hi):
    data = []
    for t, x, ok in marks:
        if t_lo <= t <= t_hi:
            col = C.C_REBAR if ok else C.C_WARN
            data.append({"xAxis": float(t),
                         "lineStyle": {"color": col, "type": "dashed" if ok else "dotted",
                                       "width": 1.6 if ok else 1.2},
                         "label": {"formatter": f"{x:.0f} cm", "color": col, "fontSize": 10}})
    return {"silent": True, "symbol": "none", "data": data}

def render_engineer(state, refs):
    args = state.args
    # console
    cursor = refs.get('_log_seq', 0)
    for seq, line in list(state.log_buf):
        if seq > cursor:
            refs['log'].push(line)
            cursor = seq
    refs['_log_seq'] = cursor

    if refs['window_slider'].value != state.window:
        resize_state_windows(state, int(refs['window_slider'].value))

    bx, by, bz, fresh, has = state.quantum_latest()
    for ax, val in zip("xyz", (bx, by, bz)):
        refs[f'kpi_{ax}'].set_text(f'{val:+.2f}')
    refs['kpi_pos'].set_text(f'{state.x_vals[-1]:.2f} cm' if state.x_vals else '—')
    update_pills(state, refs, fresh, has)
    render_control(state, refs)

    if not refs['paused'].value:
        lx_raw = list(state.x_vals)
        if args.xaxis == "dist" and len(lx_raw) > 1:
            order = np.argsort(lx_raw)
            lx = list(np.asarray(lx_raw)[order])
            reorder = lambda vals: np.asarray(vals)[order] if len(vals) == len(order) else np.asarray(vals)
        else:
            lx = lx_raw
            reorder = np.asarray
        sw = int(refs['smooth_slider'].value)
        for key, series in (('lis2', state.lis2), ('lis3', state.lis3), ('quantum', state.quantum)):
            chart = refs[f'chart_{key}']
            for i, ax in enumerate("xyz"):
                y = C.moving_average(reorder(series[ax]), sw)
                chart.options["series"][i]["data"] = list(zip(lx, [float(v) for v in y]))
            chart.update()
        if state.reader is not None:
            samples = state.reader.get_history()
            chart = refs['chart_quantum_time']
            for i in range(3):
                chart.options['series'][i]['data'] = [
                    [timestamp * 1000, values[i]] for timestamp, *values in samples
                ]
            t_lo, t_hi = ((samples[0][0] * 1000, samples[-1][0] * 1000) if samples else (1.0, 0.0))
            chart.options['series'][0]['markLine'] = _time_overlay_lines(
                refs.get('_rebar_marks', []), t_lo, t_hi)
            chart.update()

    res = state.engine_result
    ver = (res.version if res else None, refs['plot_channel'].value, refs['plot_component'].value,
           state.window, bool(refs['paused'].value))
    if ver == refs.get('_ver'):
        return
    refs['_ver'] = ver
    if res is None:
        refs['_rebar_marks'] = []
        refs['summary'].set_text('Waiting for data…')
        refs['fused_table'].set_content('')
        return

    s = res.summary
    parts = [f"{s['n_confirmed']} confirmed", f"{s['n_tentative']} tentative"]
    if s["spacing_median"] is not None:
        parts.append(f"median spacing {s['spacing_median']:.1f} cm")
    parts.append(f"{res.n_samples} samples")
    parts.append(f"engine {res.compute_ms:.0f} ms")
    refs['summary'].set_text(" · ".join(parts))
    refs['fused_table'].set_content(fused_table_html(res.rebars))
    refs['stats_table'].set_content(stats_table_html(res.stats))
    refs['_rebar_marks'] = rebar_times_ms(state, res.rebars)

    # overlay fused rebars on the three rolling charts (visible range only)
    if args.xaxis == "dist" and state.x_vals:
        lo, hi = min(state.x_vals), max(state.x_vals)
    else:
        lo, hi = 1.0, 0.0
    for key in ('lis2', 'lis3', 'quantum'):
        ch = refs[f'chart_{key}']
        ch.options['series'][0]['markLine'] = _overlay_lines(res.rebars, lo, hi)
        ch.update()

    # per-source peaks chart
    ch_name, comp = refs['plot_channel'].value, refs['plot_component'].value
    src = res.sources.get((ch_name, comp))
    chart = refs['chart_peaks']
    if src is None or not src.ok:
        chart.options["series"][0]["data"] = []
        chart.options["series"][1]["data"] = []
        chart.options["series"][0].pop("markLine", None)
        chart.options["title"]["text"] = f"Rebar detection - {ch_name} {comp}: no valid data"
        refs['peaks_info'].set_text('')
        chart.update()
        return
    unit = {0: (args.mag_unit if ch_name == "quantum" else "G"), 1: "d/dx", 2: "d²/dx²"}[src.level]
    chart.options["title"]["text"] = f"Rebar detection - {ch_name} {comp}  ({len(src.dets)} found)"
    chart.options["yAxis"]["name"] = f"{comp} [{unit}]"
    chart.options["series"][0]["data"] = C.chart_points(src.x, src.sig)
    chart.options["series"][1]["data"] = [
        {"value": [d["x"], d["y"]],
         "label": {"show": True, "position": "top", "formatter": f'{d["x"]:.1f} cm', "color": "#fca5a5",
                   "fontSize": 10, "fontFamily": C.FONT_MONO}}
        for d in src.dets]
    lines = [{"xAxis": float(r["x"]), "lineStyle": {"color": C.C_OK if r["status"] == "confirmed" else C.C_WARN,
                                                    "type": "solid" if r["status"] == "confirmed" else "dotted", "width": 1},
              "label": {"show": False}} for r in res.rebars]
    lines.append({"yAxis": float(src.thr), "lineStyle": {"color": "#64748b", "type": "dashed", "width": 1},
                  "label": {"formatter": "threshold", "color": "#64748b", "fontSize": 10, "position": "insideEndTop"}})
    chart.options["series"][0]["markLine"] = {"silent": True, "symbol": "none", "data": lines}
    chart.update()
    refs['peaks_info'].set_text(
        f"noise σ raw {src.noise_raw:.4g} · σ signal {src.noise_sig:.4g} · threshold {src.thr:.4g} · "
        f"polarity {'+' if src.polarity > 0 else '-'} · covers {src.xmin:.0f}…{src.xmax:.0f} cm · "
        f"green = fused confirmed, amber = tentative")


def update_pills(state, refs, fresh, has):
    args = state.args
    tag = {"live": "LIVE", "demo": "DEMO (simulated)", "replay": "REPLAY"}[state.mode]
    refs['mode_pill'].set_content(C.status_dot(state.mode == "live", warn=state.mode != "live") + tag)
    if state.mode == "live":
        ble_map = {
            "idle": (False, True, "BLE: starting…"),
            "scanning": (False, True, "BLE: scanning…"),
            "connected": (True, False, f"BLE: {state.ble_device_name}"),
            "not_found": (False, False, "BLE: not found - retrying"),
            "error": (False, False, "BLE: error - retrying"),
            "disconnected": (False, True, "BLE: reconnecting…"),
        }
        ok, warn, text = ble_map.get(state.ble_status, (False, True, "BLE: …"))
    else:
        ok, warn, text = True, False, "BLE: simulated"
    refs['ble_pill'].set_content(C.status_dot(ok, warn=warn) + text)

    if state.mag_status == "simulated":
        refs['mag_pill'].set_content(C.status_dot(True) + "Quantum: simulated")
    elif state.mag_status == "error":
        refs['mag_pill'].set_content(C.status_dot(False) + "Quantum: not connected")
    elif state.mag_status == "calibrating":
        refs['mag_pill'].set_content(C.status_dot(True, warn=True) + "Quantum: calibrating")
    elif state.mag_status == "connected":
        refs['mag_pill'].set_content(C.status_dot(True, warn=True) + "Quantum: connected (idle)")
    elif not has:
        refs['mag_pill'].set_content(C.status_dot(False, warn=True) + "Quantum: no data yet")
    elif fresh:
        state.stale_run = 0
        refs['mag_pill'].set_content(C.status_dot(True) + "Quantum: streaming")
    else:
        state.stale_run += 1
        refs['mag_pill'].set_content(C.status_dot(False, warn=state.stale_run < 50) + f"Quantum: {state.stale_run} stale")
    refs['rate_pill'].set_content(C.status_dot(state.sample_rate() > 0.5, warn=True) + f'{state.sample_rate():.1f} Hz')
    dest = os.path.basename(args.out) if state.log_csv else "not logging"
    refs['csv_pill'].set_content(C.status_dot(state.csv_errors == 0) + f'{state.rows_written} rows → {dest}')
    res = state.engine_result
    if state.engine_error:
        refs['eng_pill'].set_content(C.status_dot(False) + "engine error")
    else:
        refs['eng_pill'].set_content(C.status_dot(True) + (f"engine {res.compute_ms:.0f} ms" if res else "engine idle"))


# ==================================================================
# Main
# ==================================================================
def parse_kv(items, cast=float):
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        out[k.strip()] = cast(v)
    return out


def build_parser():
    p = argparse.ArgumentParser(description="Live dashboard (NiceGUI) for BLE + quantum sensor rebar scanning.")
    g = p.add_argument_group("hardware")
    g.add_argument("--name", default=None, help="BLE device name (substring)")
    g.add_argument("--addr", default=None, help="BLE device address")
    g.add_argument("--mag-port", default="COM10")
    g.add_argument("--mag-baud", type=int, default=1_000_000)
    g.add_argument("--mag-unit", default="a.u.")
    g.add_argument("--ble-watchdog-s", type=float, default=0.0,
                   help="reconnect after this many seconds without BLE data; 0 disables the watchdog")
    g = p.add_argument_group("logging / display")
    g.add_argument("--out", default=f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    g.add_argument("--sep", default=",")
    g.add_argument("--decimal", default=".")
    g.add_argument("--window", type=int, default=200)
    g.add_argument("--xaxis", choices=["t", "dist"], default="dist")
    g.add_argument("--smooth", type=int, default=5, help="display smoothing of the rolling charts")
    g.add_argument("--port", type=int, default=8080)
    g.add_argument("--open", choices=["engineer", "visitor", "none"], default="engineer",
                   help="page to open in the browser at startup")
    g = p.add_argument_group("rebar detection")
    g.add_argument("--method", choices=["original", "robust"], default="original",
                   help="original = your professor's finde_buegelpositionen exactly (default); "
                        "robust = cm-grid / smoothing / bump-centre pipeline")
    g.add_argument("--rebar-min-spacing", type=int, default=20,
                   help="original method: find_peaks `distance` in SAMPLES (Mindestbuegelabstand)")
    g.add_argument("--min-spacing-cm", type=float, default=25.0,
                   help="robust method: minimum distance between two rebars [cm]")
    g.add_argument("--rebar-sensitivity", type=float, default=1.0, help="threshold multiplier (lower = more sensitive)")
    g.add_argument("--rebar-use-factors", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--rf-number", type=int, default=1)
    g.add_argument("--smooth-cm", type=float, default=15.0, help="derivative smoothing window [cm]")
    g.add_argument("--grid-cm", type=float, default=1.0)
    g.add_argument("--locate", choices=["center", "slope"], default="center")
    g.add_argument("--min-votes", type=int, default=2, help="sensors that must agree to confirm a rebar")
    g.add_argument("--fuse-components", default="Bx", help="comma list from Bx,By,Bz,B_total")
    g.add_argument("--offset", action="append", metavar="CH=CM", help="position offset per channel, e.g. lis3=2.5")
    g.add_argument("--thr-scale", action="append", metavar="CH=X", help="fixed-threshold multiplier per channel")
    g.add_argument("--engine-interval", type=float, default=0.8)
    g = p.add_argument_group("exhibition / visitor page")
    g.add_argument("--beam-length-cm", type=float, default=500.0)
    g.add_argument("--org", default="Materialprüfungsanstalt (MPA) · Universität Stuttgart")
    g.add_argument("--logo", default=None, help="path to a logo image shown on the visitor page")
    g.add_argument("--lang", choices=["de", "en"], default="de", help="initial visitor-page language")
    g.add_argument("--idle-s", type=float, default=25.0, help="seconds without data before the attract screen")
    g = p.add_argument_group("no-hardware modes")
    g.add_argument("--demo", action="store_true", help="simulated scan, no hardware needed")
    g.add_argument("--demo-seed", type=int, default=0)
    g.add_argument("--demo-speed", type=float, default=1.0)
    g.add_argument("--demo-pause-s", type=float, default=6.0)
    g.add_argument("--demo-artifacts", action="store_true", help="add startup jump / dropout / spikes")
    g.add_argument("--demo-no-quantum", action="store_true")
    g.add_argument("--replay", default=None, help="replay a recorded CSV instead of live hardware")
    g.add_argument("--replay-hz", type=float, default=15.0)
    g.add_argument("--replay-speed", type=float, default=1.0)
    g.add_argument("--loop", action="store_true")
    g.add_argument("--autostart", action="store_true", help="start the measurement as soon as the sensor port opens")
    return p


def build_config(args):
    return eng.EngineConfig(
        method=args.method, min_spacing_samples=args.rebar_min_spacing,
        grid_cm=args.grid_cm, smooth_cm=args.smooth_cm, min_spacing_cm=args.min_spacing_cm,
        sensitivity=args.rebar_sensitivity, rf_number=args.rf_number, use_factors=args.rebar_use_factors,
        locate=args.locate, min_votes=args.min_votes,
        fuse_components=tuple(c.strip() for c in args.fuse_components.split(",") if c.strip()),
        offsets_cm=parse_kv(args.offset), thr_scale=parse_kv(args.thr_scale),
    )
def _emergency_stop(state):
    """Last-chance 'stop' on any interpreter exit (Ctrl+C, SIGTERM, exception)."""
    r = state.reader
    ser = r._ser if r is not None else None
    if ser is None:
        return
    try:
        ser.write(b"stop\n")
        ser.flush()
        time_mod.sleep(0.1)
    except Exception:
        pass

def main():
    args = build_parser().parse_args()
    args.mode = "demo" if args.demo else ("replay" if args.replay else "live")
    if args.mode == "live":
        if BleakScanner is None:
            raise SystemExit("pip install bleak nicegui pyserial numpy scipy")
        ensure_csv(args.out, CSV_HEADER, sep=args.sep)
    if args.logo and os.path.exists(args.logo):
        app.add_static_files("/assets", os.path.dirname(os.path.abspath(args.logo)))
        args.logo_url = "/assets/" + os.path.basename(args.logo)
    else:
        args.logo_url = None

    state = AppState(args, build_config(args))
    atexit.register(_emergency_stop, state)
    for sig in (signal.SIGTERM, getattr(signal, "SIGHUP", None)):
        if sig is not None:
            signal.signal(sig, lambda s, f: sys.exit(0))
    @app.on_startup
    async def _startup():
        state.log("Starting up…")
        cfg = state.cfg
        if cfg.method == "original":
            state.log(f"Engine: ORIGINAL method (finde_buegelpositionen): distance={cfg.min_spacing_samples} samples, "
                      f"sensitivity x{cfg.sensitivity:g}, fuse={'+'.join(cfg.fuse_components)}, votes>={cfg.min_votes}, "
                      f"rf={cfg.rf_number}, factors={'on' if cfg.use_factors else 'off'}")
        state.log(f"Robust-method settings: grid {cfg.grid_cm:g} cm, smoothing {cfg.smooth_cm:g} cm, min spacing "
                  f"{cfg.min_spacing_cm:g} cm, locate={cfg.locate}, fuse={'+'.join(cfg.fuse_components)}, "
                  f"votes>={cfg.min_votes}, rf={cfg.rf_number}, factors={'on' if cfg.use_factors else 'off'}")
        state.tasks.append(asyncio.create_task(engine_loop(state)))
        if state.mode == "live":
            state.tasks.append(asyncio.create_task(ingest_loop(state)))
            state.tasks.append(asyncio.create_task(ble_supervisor(state)))
            state.tasks.append(asyncio.create_task(quantum_supervisor(state)))
        elif state.mode == "demo":
            state.log("DEMO MODE - all data is simulated.")
            state.tasks.append(asyncio.create_task(demo_task(state)))
        else:
            state.log(f"REPLAY MODE - {args.replay}")
            state.tasks.append(asyncio.create_task(replay_task(state)))
        if args.open != "none":
            page = "" if args.open == "engineer" else "visitor"
            asyncio.get_running_loop().call_later(
                2.0, lambda: webbrowser.open(f"http://localhost:{args.port}/{page}"))

    @app.on_shutdown
    def _shutdown():
        if state.cal is not None:
            state.cal.abort_event.set()
        if state.reader is not None:
            state.reader.stop_mag()
            state.reader.stop_reading()

    def engineer_page():
        refs = build_engineer_ui(state)
        failed = {"n": 0}

        def tick():
            try:
                render_engineer(state, refs)
            except Exception as e:                       # never let the timer die noisily
                failed["n"] += 1
                if failed["n"] in (1, 50, 500):
                    state.log(f"UI render error: {e!r}")
        ui.timer(0.2, tick)

    ui.page('/')(engineer_page)
    ui.page('/engineer')(engineer_page)
    visitor_page.register(state)

    ui.run(title="Quantum Mag Dashboard", port=args.port, reload=False, show=False)


if __name__ in {"__main__", "__mp_main__"}:
    main()
    

# Calling the script: python dashboard.py --name CIRCUITPY --mag-port COM18 --method original --smooth 1