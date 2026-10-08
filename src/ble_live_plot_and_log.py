import asyncio
import argparse
import re
import sys
import os
import csv
from collections import deque
from datetime import datetime, time
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import serial
import time
import struct
import numpy as np

try:
    from bleak import BleakClient, BleakScanner
except ImportError:
    print("Bitte installieren: pip install bleak matplotlib")
    sys.exit(1)


# ------------------------------------------------------------
# Magnetometer configuration
# ------------------------------------------------------------
MSPP = 10
VECTOR_ENABLED = True
CONVERSION_ENABLED = True
FOURTH_AXIS_ENABLED = False
BINARY_MODE = False
#LOCKIN_FREQS = (5000, 3000, 1000, 750)
LOCKIN_LOWPASS = 10
LOCKIN_MOD_DEPTH = 500e3

# Set this to True once the firmware is sending printTripleIntBinary frames
# instead of printTriplelInt text lines for CH0/CH1/CH2.
PLOT_SAMPLES = 5000  # number of recent samples to display
SCALE_FACTOR = 1  # 2/3.59e8 #2 / 1.972e8
SCALE_FACTOR_B = 1  # 1/1e7

s = serial.Serial("COM18", baudrate=1_000_000, timeout=0.05)


def start_mag():
    s.write("getUUID\n".encode("ascii"))
    time.sleep(0.1)
    print(s.read_until(b"\n").decode("ascii"))

    s.write("stop\n".encode("ascii"))
    time.sleep(0.1)
    s.write("enableMagnetometerConversion\n".encode("ascii"))
    time.sleep(0.1)
    s.write("enableVector\n".encode("ascii"))
    time.sleep(0.05)

    # -- per-channel lock-in setup -----------
    for ch in range(3):
        s.write(f"loadDRFreq {ch}\n".encode("ascii"))
        time.sleep(0.05)
        s.write(f"loadSlope {ch}\n".encode("ascii"))
        time.sleep(0.05)
        s.write(
            f"setDualResonanceLockinLowpassFreq {ch} {LOCKIN_LOWPASS}\n".encode("ascii")
        )
        time.sleep(0.05)
        s.write(
            f"setDualResonanceLockinModDepth {ch} {LOCKIN_MOD_DEPTH}\n".encode("ascii")
        )
        time.sleep(0.05)

    s.write("setDualResonanceFreq 3 0\n".encode("ascii"))
    time.sleep(0.05)
    s.write(f"setMagnetometerMSPP {int(MSPP)}\n".encode("ascii"))
    time.sleep(0.05)
    s.write("runMagnetometer\n".encode("ascii"))
    time.sleep(0.1)


def parse_channel_line(line: str) -> dict:
    """Parse 'CH0 4004.000000, CH1 3732.000000, CH2 605.000000'
    or 'Bx 0.001, By 0.002, Bz 0.003' into {name: value}.
    """
    out = {}
    for tok in line.strip().split(","):
        tok = tok.strip()
        if not tok:
            continue
        parts = tok.split()  # split on whitespace: ["CH0", "4004.000000"]
        if len(parts) != 2:
            continue
        name, val = parts
        name = name.rstrip(":")  # "Bx:" -> "Bx", "CH0:" -> "CH0"
        out[name] = float(val)
        # try:
        #     out[name] = float(val)
        # except ValueError:
        #     pass
    return out


def pool_mag():
    a = s.read_until(b"\n")
    # print(a, end="\n")
    # if not a:
    #     continue
    try:
        a = a.decode("ascii")
    except UnicodeDecodeError:
        print("[mag] decode error")
        return None

    parsed = parse_channel_line(a)
    # for name, val in parsed.items():
    #     print name and val
    #     print(f"{name}: {val:.6f}")
    # return back only the Bx, By, Bz values not the names
    return parsed.get("Bx", 0.0), parsed.get("By", 0.0), parsed.get("Bz", 0.0)


# NUS UUIDs (Nordic UART Service)
UART_SERVICE_UUID = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
UART_RX_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"  # write
UART_TX_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"  # notify

# CSV-Header wie im USB-Skript
CSV_HEADER = [
    "timestamp",
    "Position",
    "lis2_Bx_G",
    "lis2_By_G",
    "lis2_Bz_G",
    "lis3_Bx_G",
    "lis3_By_G",
    "lis3_Bz_G",
]

# NEUES Format: "YYYY-MM-DD HH:MM:SS", Distanz in cm
CSV_RE = re.compile(
    r"^DATA,"
    r"(?P<t>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),"
    r"(?P<dist>[-+]?\d+(\.\d+)?),"
    r"(?P<lis2x>[-+]?\d+(\.\d+)?),"
    r"(?P<lis2y>[-+]?\d+(\.\d+)?),"
    r"(?P<lis2z>[-+]?\d+(\.\d+)?),"
    r"(?P<lis3x>[-+]?\d+(\.\d+)?),"
    r"(?P<lis3y>[-+]?\d+(\.\d+)?),"
    r"(?P<lis3z>[-+]?\d+(\.\d+)?)\s*$"
)


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
            f"CSV-Header passt nicht. Erwartet {header}, gefunden {existing}"
        )


def append_row(file_path, row_dict, header, sep=",", decimal="."):
    row = []
    for key in header:
        val = row_dict.get(key, "")
        s = f"{val:.9f}" if isinstance(val, float) else str(val)
        if decimal == ",":
            s = s.replace(".", ",")
        row.append(s)

    with open(file_path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f, delimiter=sep).writerow(row)


async def find_device(name=None, address=None, timeout=10.0):
    if address:
        d = await BleakScanner.find_device_by_address(address, timeout=timeout)
        return d

    if name:
        # Scanne und finde das erste Device, das den Namen enthält (case-insensitive)
        end = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < end:
            devices = await BleakScanner.discover(timeout=2.0)
            for d in devices:
                if d.name and name.lower() in d.name.lower():
                    return d
        return None

    # fallback: erstes Gerät mit NUS-Service
    devices = await BleakScanner.discover(timeout=timeout)
    for d in devices:
        # Manche Stacks liefern Services erst nach Connect; probiere trotzdem Namen
        if d.name and ("CIRCUITPY" in d.name or "CPB" in d.name):
            return d
    return None


class BleLineBuffer:
    def __init__(self):
        self.buf = bytearray()
        self.lines = deque()

    def feed(self, data: bytes):
        self.buf.extend(data)
        while True:
            try:
                i = self.buf.index(0x0A)  # \n
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


def run_plot(args):
    # Plot-Puffer
    x_vals = deque(maxlen=args.window)
    lis2x = deque(maxlen=args.window)
    lis2y = deque(maxlen=args.window)
    lis2z = deque(maxlen=args.window)
    lis3x = deque(maxlen=args.window)
    lis3y = deque(maxlen=args.window)
    lis3z = deque(maxlen=args.window)
    magx = deque(maxlen=args.window)
    magy = deque(maxlen=args.window)
    magz = deque(maxlen=args.window)

    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)

    (l2x,) = ax1.plot([], [], label="LIS2 x")
    (l2y,) = ax1.plot([], [], label="LIS2 y")
    (l2z,) = ax1.plot([], [], label="LIS2 z")
    ax1.set_ylabel("Gauss")
    ax1.legend(loc="upper right")
    ax1.grid(True)

    (l3x,) = ax2.plot([], [], label="LIS3 x")
    (l3y,) = ax2.plot([], [], label="LIS3 y")
    (l3z,) = ax2.plot([], [], label="LIS3 z")
    ax2.set_ylabel("Gauss")
    ax2.legend(loc="upper right")
    ax2.grid(True)

    (mx,) = ax3.plot([], [], label="Mag Bx")
    (my,) = ax3.plot([], [], label="Mag By")
    (mz,) = ax3.plot([], [], label="Mag Bz")
    ax3.set_xlabel("Distanz [cm]" if args.xaxis == "dist" else "Zeit")
    ax3.set_ylabel("Field")
    ax3.legend(loc="upper right")
    ax3.grid(True)

    # CSV vorbereiten
    ensure_csv(args.out, CSV_HEADER, sep=args.sep)
    print(f"[CSV] Schreibe nach: {args.out}")

    line_buffer = BleLineBuffer()
    client_holder = {"client": None}  # Zugriff im closure

    async def ble_task():
        dev = await find_device(name=args.name, address=args.addr, timeout=15.0)
        if not dev:
            print("Kein BLE-Gerät gefunden. Mit --name oder --addr einschränken.")
            return
        print(f"Verbinde zu {dev.name or dev.address} …")

        async with BleakClient(dev) as client:
            client_holder["client"] = client
            await client.start_notify(
                UART_TX_UUID, lambda _h, data: line_buffer.feed(data)
            )
            print("[BLE] Verbunden. Empfange Daten …")
            # läuft bis Plot geschlossen wird
            while client.is_connected:
                await asyncio.sleep(0.05)

    # Animation Update zieht neue Zeilen aus dem BLE-Puffer
    def update(_frame):
        lines = line_buffer.pop_all()
        for line in lines:
            m = CSV_RE.match(line)
            if not m:
                continue

            t_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            dist_cm = float(m.group("dist"))
            if args.xaxis == "dist":
                d_x = dist_cm
            else:
                d_x = mdates.date2num(datetime.now())

            x_vals.append(d_x)
            lis2x.append(float(m.group("lis2x")))
            lis2y.append(float(m.group("lis2y")))
            lis2z.append(float(m.group("lis2z")))
            lis3x.append(float(m.group("lis3x")))
            lis3y.append(float(m.group("lis3y")))
            lis3z.append(float(m.group("lis3z")))

            # pool the magnetometer on every update and push it into the plot buffers too
            mag_result = pool_mag()
            print(f"Bx, By, Bz: {mag_result}")
            if mag_result is None:
                # decode error: hold the last known value so the line doesn't break
                bx = magx[-1] if magx else 0.0
                by = magy[-1] if magy else 0.0
                bz = magz[-1] if magz else 0.0
            else:
                bx, by, bz = mag_result

            magx.append(bx)
            magy.append(by)
            magz.append(bz)

            row = {
                "timestamp": t_str,
                "Position": dist_cm,
                "lis2_Bx_G": float(m.group("lis2x")),
                "lis2_By_G": float(m.group("lis2y")),
                "lis2_Bz_G": float(m.group("lis2z")),
                "lis3_Bx_G": float(m.group("lis3x")),
                "lis3_By_G": float(m.group("lis3y")),
                "lis3_Bz_G": float(m.group("lis3z")),
            }
            append_row(
                args.out, row, CSV_HEADER, sep=args.sep, decimal=args.decimal
            )

        # Plot aktualisieren
        lx = list(x_vals)
        l2x.set_data(lx, list(lis2x))
        l2y.set_data(lx, list(lis2y))
        l2z.set_data(lx, list(lis2z))
        l3x.set_data(lx, list(lis3x))
        l3y.set_data(lx, list(lis3y))
        l3z.set_data(lx, list(lis3z))
        mx.set_data(lx, list(magx))
        my.set_data(lx, list(magy))
        mz.set_data(lx, list(magz))

        for ax in (ax1, ax2, ax3):
            if len(lx) >= 2:
                ax.set_xlim(min(lx), max(lx))

            ys = []
            for ln in ax.get_lines():
                _, yd = ln.get_data()
                yd_list = list(yd)
                if yd_list:
                    ys.extend(yd_list[-len(lx):])

            if ys:
                ymin = min(ys)
                ymax = max(ys)
                if ymin == ymax:
                    ymin -= 0.1
                    ymax += 0.1
                pad = (ymax - ymin) * 0.1
                ax.set_ylim(ymin - pad, ymax + pad)

        return l2x, l2y, l2z, l3x, l3y, l3z, mx, my, mz

    # --- Plot + Animation ---
    ani = FuncAnimation(
        fig, update, interval=30, blit=False, cache_frame_data=False
    )
    if args.xaxis == "t":
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        fig.autofmt_xdate()
    plt.tight_layout()

    # --- Async-Loop in einem Hintergrund-Thread starten ---
    loop = asyncio.new_event_loop()

    def _run_loop():
        asyncio.set_event_loop(loop)
        loop.run_forever()

    t = threading.Thread(target=_run_loop, daemon=True)
    t.start()

    # BLE-Task in diesem Loop schedulen
    ble_task_future = asyncio.run_coroutine_threadsafe(ble_task(), loop)

    try:
        plt.show()  # blockiert; BLE läuft parallel im Hintergrund-Loop
    finally:
        # BLE-Task abbrechen
        try:
            ble_task_future.cancel()
        except Exception:
            pass

        # sauber trennen, falls noch verbunden
        async def _shutdown(client_holder):
            cli = client_holder.get("client")
            if cli and cli.is_connected:
                try:
                    await cli.disconnect()
                except Exception:
                    pass

        fut = asyncio.run_coroutine_threadsafe(_shutdown(client_holder), loop)
        try:
            fut.result(timeout=2)
        except Exception:
            pass

        # Loop beenden
        loop.call_soon_threadsafe(loop.stop)
        t.join(timeout=2)


def main():
    try:
        start_mag()
        p = argparse.ArgumentParser(
            description="BLE Live-Plot + CSV-Logger (kompatibel)."
        )
        p.add_argument(
            "--name", default=None, help="Teil des Gerätenamens (z.B. CIRCUITPY)"
        )
        p.add_argument(
            "--addr", default=None, help="MAC/Adresse des Geräts (optional)"
        )
        p.add_argument(
            "--window", type=int, default=200, help="Punkte im Plot-Fenster"
        )
        p.add_argument("--xaxis", choices=["t", "dist"], default="dist")
        p.add_argument("--out", default="messung_ble_without_tail_mag_aq.csv")
        p.add_argument("--sep", default=",")
        p.add_argument("--decimal", default=".")
        args = p.parse_args()

        run_plot(args)
        print("Stopping measurement...")
        s.write(b"stop\n")
        time.sleep(0.1)
    except KeyboardInterrupt:
        print("Stopping measurement...")
        s.write(b"stop\n")
        time.sleep(0.1)


if __name__ == "__main__":
    main()