# rebar_detection.py
#
# English translation of funktionen_auswertung.py's stirrup/rebar-locating
# functions ("Buegel" = rebar stirrup -- the looped reinforcement bar that
# wraps a beam/column's main longitudinal bars at regular intervals).
#
# This file has TWO layers:
#
#   1. A faithful, line-for-line English port of your original functions
#      (find_rebar_positions_from_measurements, plot_measurements_with_rebars,
#      calculate_factor) -- same logic, same edge cases, renamed and
#      documented. Kept for fidelity with your existing offline workflow.
#
#   2. An ADAPTED core (find_rebar_positions) that works directly on
#      (positions, values) arrays instead of a "Messungen" dict keyed by
#      "RF<n>_<component>" strings -- because the live dashboard's channels
#      are named lis2 / lis3 / quantum, not RF1 / RF2 / .... This is what
#      dashboard.py actually calls.
#
# On the RF<n> / sign-factor convention (calculate_factor):
#   Re-reading plot_measurements_with_rebars, RF<n> is a SCAN PASS number,
#   not a sensor identity ("if only one sensor exists, create separate
#   plots for RF1 and RF2" -- i.e. RF1/RF2 are two runs/directions over the
#   same structure, not two different probes). That means the sign
#   convention is a property of ONE CONTINUOUS LIVE SESSION as a whole
#   (which pass/direction this run is), applied the same way to every
#   channel riding the same cart through that pass -- not something that
#   needs to be guessed per lis2/lis3/quantum. The dashboard exposes this
#   as a single --rf-number, matching your original use_factors=True usage.

import re

import numpy as np
from scipy.signal import find_peaks

# ------------------------------------------------------------
# Effective values reverse-derived from your active (uncommented) config,
# for "daten_vektor" (vector data) mode -- i.e. Bx/By/Bz all measured
# directly per channel, which matches lis2 / lis3 / quantum's data shape:
#
#   komponenten_ableitungsebene = {"Bx": 1, "By": 0, "Bz": 0, "B_total": 0}
#   fixwerte_vektor             = {"Bx": 0.02}
#   grenzwerte_basis            = {0: {"By": 0.05, "Bz": 0.1, "B_total": 0.05},
#                                   2: {"By": 0.01, "Bz": 0.01, "B_total": 0.01}}
#
# -> Bx threshold = 0.02 (from fixwerte_vektor, derivative level 1)
# -> By/Bz/B_total thresholds come from grenzwerte_basis[0], since their
#    derivative level is 0.
DEFAULT_COMPONENT_DERIVATIVE_LEVEL = {"Bx": 1, "By": 0, "Bz": 0, "B_total": 0}
DEFAULT_THRESHOLDS = {"Bx": 0.02, "By": 0.05, "Bz": 0.1, "B_total": 0.05}

# Mindestbuegelabstand (find_peaks' `distance`, in SAMPLES not cm) isn't
# given anywhere in the code you shared -- it's set earlier in your main
# script and I won't invent a number for it. Pass the same value you use
# offline via --rebar-min-spacing; there is no default.


# ==================================================================
# Layer 1 -- faithful port (same interface/behavior as the original)
# ==================================================================
def calculate_factor(component: str, channel_number: int, derivative_level: int) -> float:
    """
    1:1 English port of faktor_berechnen().

    Returns the sign (+1/-1) applied to a channel's (derivative) signal so
    a rebar always shows up as a positive peak, based on the channel's
    RF-index parity and the derivative level used.

    Preserves the ORIGINAL's exact behavior, edge case included: for
    By/Bz/B_total, only derivative_level 0 or 2 get an explicit sign --
    any other level (e.g. 1) falls through to the default +1 at the end,
    exactly as in your script. This was translated, not "fixed".
    """
    if component == "Bx":
        return -1 if channel_number % 2 == 1 else 1
    elif component == "By":
        if derivative_level == 0:
            return -1 if channel_number % 2 == 1 else 1
        elif derivative_level == 2:
            return 1 if channel_number % 2 == 1 else -1
    elif component == "Bz":
        if derivative_level == 0:
            return 1 if channel_number % 2 == 1 else -1
        if derivative_level == 2:
            return -1 if channel_number % 2 == 1 else 1
    elif component == "B_total":
        if derivative_level == 0:
            return 1 if channel_number % 2 == 1 else -1
        if derivative_level == 2:
            return -1 if channel_number % 2 == 1 else 1
    return 1


def find_rebar_positions_from_measurements(key, measurements, positions, thresholds,
                                            min_spacing, component_derivative_level,
                                            use_factors=True):
    """
    Faithful English port of finde_buegelpositionen().

    key: a measurement dict key of the form "RF<n>_<component>",
         e.g. "RF1_Bx", "RF2_B_total".
    measurements: dict of {key: ndarray}; each array is 1D (single sensor)
         or 2D with shape (n_positions, n_sensors).
    positions: 1D ndarray of scan positions aligned with each array's rows.
    thresholds: dict of {component: prominence_threshold}.
    min_spacing: minimum index distance between accepted peaks
         (find_peaks' `distance`) -- same units/value as your offline
         script's Mindestbuegelabstand.
    component_derivative_level: dict of {component: 0 | 1 | 2}.
    use_factors: True -> single-sided search using calculate_factor's sign
         convention. False -> search both polarities, keep the more
         prominent one per sensor.

    Returns: {"Sensor_<i>": ndarray of detected rebar positions}.
    """
    rebar_positions = {}

    if use_factors:
        rf_match = re.search(r"RF(\d+)", key)
        rf_number = int(rf_match.group(1))

    component = "_".join(key.split("_")[1:])
    derivative_level = component_derivative_level.get(component, 1)
    threshold = thresholds.get(component, 0.05)

    values = np.array(measurements[key], copy=True)
    num_sensors = values.shape[1] if len(values.shape) > 1 else 1

    for sensor_idx in range(num_sensors):
        sensor_key = f"Sensor_{sensor_idx + 1}"
        sensor_values = values[:, sensor_idx].copy() if values.ndim > 1 else values.flatten().copy()

        if derivative_level == 1:
            derivative_values = np.gradient(sensor_values, positions)
        elif derivative_level == 0:
            derivative_values = sensor_values
        else:
            derivative_values = np.gradient(np.gradient(sensor_values, positions), positions)

        if use_factors:
            factor = calculate_factor(component, rf_number, derivative_level)
            derivative_values = derivative_values * factor
            peaks, _ = find_peaks(derivative_values, prominence=threshold, distance=min_spacing)
        else:
            peaks_max, props_max = find_peaks(derivative_values, prominence=threshold, distance=min_spacing)
            peaks_min, props_min = find_peaks(-derivative_values, prominence=threshold, distance=min_spacing)

            max_positions = positions[peaks_max] if len(peaks_max) > 0 else []
            min_positions = positions[peaks_min] if len(peaks_min) > 0 else []

            if len(max_positions) > 0 and len(min_positions) > 0:
                max_prominence = max(props_max["prominences"]) if len(props_max["prominences"]) > 0 else -np.inf
                min_prominence = max(props_min["prominences"]) if len(props_min["prominences"]) > 0 else -np.inf
                peaks = peaks_max if max_prominence > min_prominence else peaks_min
            else:
                peaks = peaks_max if len(max_positions) > 0 else peaks_min

        peaks = peaks[peaks < len(positions)]
        rebar_positions[sensor_key] = positions[peaks]

    return rebar_positions


def save_plot_function(save_path, filename, dpi=150):
    """Default save helper for plot_measurements_with_rebars() (stands in for
    your original plot_speichern_function). Saves the current matplotlib
    figure as <save_path>/<filename>.png and closes it. Replace with your own
    helper if you have one -- same (save_path, filename) signature."""
    import os
    import matplotlib.pyplot as plt

    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"{filename}.png"), dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_measurements_with_rebars(measurements, measurement_name, detected_rebars, plot_all_rf,
                                   save_plot, save_path, rf_selection=("RF1", "RF2")):
    """
    Faithful English port of plot_messungen_mit_buegel() -- an OFFLINE,
    matplotlib-based review plot (raw signal + derivative on a secondary
    axis, detected rebar positions as vertical lines). Not used by the
    live dashboard (which overlays detections directly on its own ECharts
    instead, see render_rebar_overlay() usage in dashboard.py) -- kept here
    for your existing offline batch-analysis workflow.

    Requires a `save_plot_function(save_path, filename)` of your own (the
    original called `plot_speichern_function`, which wasn't in what you
    pasted -- wire up your existing save helper before using save_plot=True).
    """
    import matplotlib.pyplot as plt

    components = ["Bx", "By", "Bz", "B_total"]
    positions = measurements["Position"].flatten()

    example_key = next(key for key in measurements if key.startswith("RF"))
    num_sensors = measurements[example_key].shape[1] if len(measurements[example_key].shape) > 1 else 1

    if num_sensors == 1:
        for rf_group in rf_selection:
            fig, axes = plt.subplots(len(components), figsize=(10, 2.5 * len(components)), sharex=True)
            for idx, component in enumerate(components):
                ax = axes[idx]
                rf_keys = [k for k in measurements if k.startswith(rf_group) and k.endswith(f"_{component}")]

                for rf_key in rf_keys:
                    color = "blue" if "RF1" in rf_key else "red"
                    values = measurements[rf_key].flatten().copy()
                    derivative = (np.gradient(values, positions) if component == "Bx"
                                  else np.gradient(np.gradient(values, positions), positions))

                    ax.plot(positions, values, label=f"{rf_key}", color=color)
                    ax2 = ax.twinx()
                    ax2.plot(positions, derivative, label=f"{rf_key} derivative", color=color,
                             linestyle="dashed", alpha=0.6)
                    ax2.set_ylabel(f"{component} derivative", fontsize=10, color=color)

                    if rf_key in detected_rebars:
                        for pos in detected_rebars[rf_key].get("Sensor_1", []):
                            ax.axvline(x=pos, color=color, linestyle="solid", alpha=0.7, linewidth=1.5)

                    ax.set_ylabel(f"{component} (G)", fontsize=12)
                    ax.legend(fontsize=10, loc='upper left')
                    ax.grid(True)

            axes[-1].set_xlabel("Position [cm]", fontsize=12)
            fig.suptitle(f"{measurement_name}: rebar positions {rf_group}", fontsize=14)
            if save_plot:
                save_plot_function(save_path, f"{measurement_name}-4-rebar_locations_{rf_group}")
            else:
                plt.show()
    else:
        for component in components:
            rf_keys = ([k for k in measurements if k.startswith("RF") and k.endswith(f"_{component}")]
                       if plot_all_rf else
                       [k for k in measurements if k.startswith("RF1") and k.endswith(f"_{component}")])

            for rf_key in rf_keys:
                fig, axes = plt.subplots(num_sensors, figsize=(10, 2.5 * num_sensors), sharex=True)
                axes = axes if num_sensors > 1 else [axes]

                for sensor_idx in range(num_sensors):
                    ax = axes[sensor_idx]
                    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:5]
                    color = {"RF1": colors[0], "RF2": colors[1], "RF3": colors[2],
                             "RF4": colors[3], "RF5": colors[4]}.get(
                        next((g for g in ("RF1", "RF2", "RF3", "RF4", "RF5") if g in rf_key), None), "gray")

                    values = measurements[rf_key][:, sensor_idx].copy()
                    derivative = (np.gradient(values, positions) if component == "Bx"
                                  else np.gradient(np.gradient(values, positions), positions))

                    ax.plot(positions, values, label=f"{rf_key} Sensor {sensor_idx + 1}", color=color)
                    ax2 = ax.twinx()
                    ax2.plot(positions, derivative, label=f"{rf_key} derivative Sensor {sensor_idx + 1}",
                             color=color, linestyle="dashed", alpha=0.6)
                    ax2.set_ylabel(f"{component} derivative", fontsize=10, color=color)

                    if rf_key in detected_rebars:
                        for pos in detected_rebars[rf_key].get(f"Sensor_{sensor_idx + 1}", []):
                            ax.axvline(x=pos, color=color, linestyle="solid", alpha=0.7, linewidth=1.5)

                    ax.set_ylabel(f"{component} (G)", fontsize=12)
                    ax.legend(fontsize=10)
                    ax.grid(True)

                axes[-1].set_xlabel("Position [cm]", fontsize=12)
                fig.suptitle(f"{measurement_name}: rebar positions {rf_key}", fontsize=14)
                if save_plot:
                    save_plot_function(save_path, f"{measurement_name}-4-rebar_locations_{rf_key}")
                else:
                    plt.show()


# ==================================================================
# Layer 2 -- adapted core used by the live dashboard
# ==================================================================
def _refine_peak(x, y, i):
    """Sub-sample peak position: vertex of the parabola through the three
    samples around index i (works on non-uniform x). Clamped to
    [x[i-1], x[i+1]]; falls back to the sample position x[i] at the array
    edges or if the parabola is not a proper maximum."""
    if i <= 0 or i >= len(x) - 1:
        return float(x[i])
    xs, ys = x[i - 1:i + 2], y[i - 1:i + 2]
    try:
        a, b, _c = np.polyfit(xs - x[i], ys, 2)
    except Exception:
        return float(x[i])
    if not np.isfinite(a) or a >= 0:
        return float(x[i])
    xv = -b / (2 * a) + x[i]
    return float(min(max(xv, xs[0]), xs[-1]))


def analyze_rebar(positions, values, *, derivative_level=1, threshold=0.05,
                  min_spacing, sign_factor=1.0, use_factors=True, refine=True):
    """
    Identical detection to find_rebar_positions() (same derivative, same
    sign factor, same find_peaks call) but returns everything needed to DRAW
    it, so the plot shows exactly what the detector saw:

      x           positions (as passed in; sorted, unique)
      signal      the exact array find_peaks searched (after derivative +
                  sign handling) -> every detected peak is a local maximum
                  of this curve
      peak_idx    indices of accepted peaks in `signal`
      peak_x      peak positions (parabolic sub-sample refinement if
                  refine=True, else the sample position)
      peak_y      signal value at each peak sample
      prominence  prominence of each peak (>= threshold by construction)
      threshold   prominence threshold used
    """
    if min_spacing is None:
        raise ValueError("min_spacing (Mindestbuegelabstand) must be provided explicitly -- "
                          "no default is assumed.")
    positions = np.asarray(positions, dtype=float)
    values = np.asarray(values, dtype=float)

    if derivative_level == 1:
        d = np.gradient(values, positions)
    elif derivative_level == 0:
        d = values
    else:
        d = np.gradient(np.gradient(values, positions), positions)

    if use_factors:
        signal = d * sign_factor
        peaks, props = find_peaks(signal, prominence=threshold, distance=min_spacing)
    else:
        pmax, props_max = find_peaks(d, prominence=threshold, distance=min_spacing)
        pmin, props_min = find_peaks(-d, prominence=threshold, distance=min_spacing)
        if len(pmax) > 0 and len(pmin) > 0:
            use_max = max(props_max["prominences"]) > max(props_min["prominences"])
        else:
            use_max = len(pmax) > 0
        # For the min-polarity case the plotted signal is -d, so the marked
        # peaks are maxima of what is drawn.
        peaks, props, signal = (pmax, props_max, d) if use_max else (pmin, props_min, -d)

    peak_x = np.array([_refine_peak(positions, signal, int(i)) if refine else positions[i]
                       for i in peaks], dtype=float)
    return {
        "x": positions,
        "signal": signal,
        "peak_idx": peaks,
        "peak_x": peak_x,
        "peak_y": signal[peaks] if len(peaks) else np.array([]),
        "prominence": np.asarray(props.get("prominences", []), dtype=float),
        "threshold": float(threshold),
    }


def find_rebar_positions(positions, values, *, derivative_level=1, threshold=0.05,
                          min_spacing, sign_factor=1.0, use_factors=True):
    """
    Same algorithm as find_rebar_positions_from_measurements(), but
    operating directly on (positions, values) arrays -- no "RF<n>_component"
    key parsing, so it works with any channel naming (lis2, lis3, quantum).

    positions: 1D ndarray, MUST be sorted ascending and free of duplicate
        x-values (np.gradient assumes strictly increasing, non-repeating
        x) -- see dedupe_and_sort() below, which the dashboard runs first.
    values: 1D ndarray, same length as positions.
    derivative_level: 0 (raw) | 1 (first derivative) | 2 (second derivative).
    threshold: prominence threshold for find_peaks.
    min_spacing: REQUIRED. Minimum index distance between accepted peaks
        (find_peaks' `distance`). No default -- see the module docstring.
    sign_factor: used when use_factors=True. Pass calculate_factor(component,
        rf_number, derivative_level) -- rf_number identifies which scan
        pass/direction this session is (see module docstring), the same
        value for every channel since they ride the same cart through the
        same pass.
    use_factors: True (default, matches your offline script's actual usage)
        applies sign_factor and searches maxima only. False instead
        searches both polarities and keeps whichever is more prominent --
        useful as a cross-check if results look off for a given rf_number.

    Returns: 1D ndarray of detected rebar positions (same units as `positions`).
    """
    if min_spacing is None:
        raise ValueError("min_spacing (Mindestbuegelabstand) must be provided explicitly -- "
                          "no default is assumed. See rebar_detection.py module docstring.")

    if derivative_level == 1:
        derivative_values = np.gradient(values, positions)
    elif derivative_level == 0:
        derivative_values = values
    else:
        derivative_values = np.gradient(np.gradient(values, positions), positions)

    if use_factors:
        derivative_values = derivative_values * sign_factor
        peaks, _ = find_peaks(derivative_values, prominence=threshold, distance=min_spacing)
    else:
        peaks_max, props_max = find_peaks(derivative_values, prominence=threshold, distance=min_spacing)
        peaks_min, props_min = find_peaks(-derivative_values, prominence=threshold, distance=min_spacing)

        max_positions = positions[peaks_max] if len(peaks_max) > 0 else []
        min_positions = positions[peaks_min] if len(peaks_min) > 0 else []

        if len(max_positions) > 0 and len(min_positions) > 0:
            max_prominence = max(props_max["prominences"]) if len(props_max["prominences"]) > 0 else -np.inf
            min_prominence = max(props_min["prominences"]) if len(props_min["prominences"]) > 0 else -np.inf
            peaks = peaks_max if max_prominence > min_prominence else peaks_min
        else:
            peaks = peaks_max if len(max_positions) > 0 else peaks_min

    peaks = peaks[peaks < len(positions)]
    return positions[peaks]


def dedupe_and_sort(positions, values):
    """
    Live BLE data arrives in chronological (not necessarily monotonic-x)
    order, and the same physical position can be sampled more than once
    (dwell, back-and-forth motion). np.gradient requires strictly
    increasing, non-repeating x -- this sorts by position and averages
    the value(s) at any repeated position before detection runs.
    """
    positions = np.asarray(positions, dtype=float)
    values = np.asarray(values, dtype=float)
    order = np.argsort(positions)
    positions, values = positions[order], values[order]

    uniq_positions, inverse = np.unique(positions, return_inverse=True)
    if len(uniq_positions) == len(positions):
        return positions, values  # already unique, nothing to average
    uniq_values = np.zeros_like(uniq_positions)
    counts = np.zeros_like(uniq_positions)
    np.add.at(uniq_values, inverse, values)
    np.add.at(counts, inverse, 1)
    return uniq_positions, uniq_values / counts


def compute_b_total(bx, by, bz):
    """B_total = sqrt(Bx^2 + By^2 + Bz^2), as used by the original script."""
    bx, by, bz = np.asarray(bx), np.asarray(by), np.asarray(bz)
    return np.sqrt(bx**2 + by**2 + bz**2)