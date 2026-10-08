import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pickle

import sys
# Replay a saved slope calibration:  python slope_determination.py <YYYY-MM-DD_HH-MM-SS_calibration_slope.pys>
FILENAME = sys.argv[1] if len(sys.argv) > 1 else "calibration_slope.pys"


N_ENSEMBLES = 3

LINEAR_FIT_RANGE = 1.2e6


def dgaussian(f, f0, sigma):
    """First derivative of a Gaussian centred at f0 with width sigma."""
    x = (f - f0) / sigma
    return -x * np.exp(-0.5 * x**2)


def linear(x, a, c):
    """Linear function a·(x − c) with fixed zero-crossing c."""
    return a * (x - c)



def odmr_model(f, zfs, *params, n_ensembles=4):
    """
    ODMR derivative signal for the |0⟩→|-1⟩ branch only, with five hyperfine
    components (mI = -2 … +2, scaled by their relative transition strengths).

    params layout:
        centers[0..n_ensembles-1]    – offset of each ensemble from ZFS (GHz)
        amplitudes[0..n_ensembles-1] – amplitude per ensemble
        A                             – hyperfine constant (GHz)   [params[-2]]
        sigma                         – Gaussian linewidth (GHz)   [params[-1]]

    ZFS is taken from the module-level _ZFS_HZ (set before calling).
    """
    f       = np.asarray(f, dtype=float)
    centers    = params[:n_ensembles]
    amplitudes = params[n_ensembles:2 * n_ensembles]
    A, sigma   = params[-2], params[-1]
    ZFS_GHz    = zfs


    # Relative amplitudes for mI = -2, -1, 0, +1, +2
    hf_amp = {-2: -1/3, -1: -2/3, 0: -1, 1: -2/3, 2: -1/3}

    signal = np.zeros_like(f)
    for c, amp in zip(centers, amplitudes):
        f_right = zfs + c
        for hf, factor in hf_amp.items():
            signal += amp * factor * dgaussian(f, f_right + hf * A, sigma)
    return signal


# ── Fitting ───────────────────────────────────────────────────────────────────

def fit_odmr(freqs, data, center_freq_hz, splittings, n_ensembles=4):
    """
    Fit the ODMR derivative model to the measured data.

    Parameters
    ----------
    freqs          : frequency array (GHz)
    data           : signal array
    center_freq_hz : ZFS retrieved from device (Hz)
    n_ensembles    : number of NV ensemble orientations

    Returns
    -------
    popt, perr, p0
    """

    freqs = np.asarray(freqs)
    data  = np.asarray(data)

    # Hard-coded initial guesses for ensemble frequency offsets (GHz)
    #centers_guess = [
    #    guess_res1,
    #    0.0245,
    #    abs(ZFS_GHz - 2.9172),
    #    abs(ZFS_GHz - 2.9058),
    #][:n_ensembles]

    amp_guess = np.ones(n_ensembles) * (data.max() - data.min())
    p0 = (
        list(splittings) +
        list(amp_guess)     +
        [2.12e6, 500e3]   # A, sigma
    )


    def _model(f, *p):
        return odmr_model(f, center_freq_hz, *p, n_ensembles=n_ensembles)

    bounds_min = [ peak-500e3 for peak in list(splittings) ] + [ -np.inf for peak in amp_guess ] + [1.9e6 ] + [ 250e3 ]
    bounds_max = [ peak+500e3 for peak in list(splittings) ] + [ +np.inf for peak in amp_guess ] + [2.5e6 ] + [ 1.5e6 ]


    centers    = p0[:n_ensembles]
    amplitudes = p0[n_ensembles:2 * n_ensembles]
    A, sigma   = p0[-2], p0[-1]
    ZFS_GHz    = center_freq_hz
    print(f"centers: {centers}")
    print(f"amplitudes: {amplitudes}")
    print(f"A: {A}")
    print(f"sigma: {sigma}")
    print(f"zfs: {ZFS_GHz}")

    popt, pcov = curve_fit(_model, freqs, data, p0=p0, bounds=[bounds_min, bounds_max])
    return popt, np.sqrt(np.diag(pcov)), p0


def fit_slopes(freqs, data, zfs, splittings):
    """
    Fit a linear slope at each zero-crossing frequency.

    Parameters
    ----------
    freqs      : frequency array (GHz)
    data       : signal array
    splittings : iterable of splitting values (Hz) — one per ensemble

    Returns
    -------
    slopes     : list of fitted slopes (ADC / GHz)
    slope_errs : list of 1-sigma uncertainties
    """
    freqs      = np.asarray(freqs)
    data       = np.asarray(data)
    f_delta    = freqs[1] - freqs[0]
    half_range = LINEAR_FIT_RANGE / 2          # GHz
    idx_half   = int(np.round(half_range / f_delta))


    slopes, slope_errs = [], []
    for splitting_hz in splittings:
        zc  = zfs + splitting_hz            # zero-crossing (GHz)
        idx = int(np.argmin(np.abs(freqs - zc)))

        sl  = slice(idx - idx_half, idx + idx_half)

        popt, pcov = curve_fit(
            lambda x, a: linear(x, a, zc),
            freqs[sl], data[sl],
            p0=[data.max() / half_range],
        )
        slopes.append(float(popt[0]))
        slope_errs.append(float(np.sqrt(pcov[0, 0])))

    return slopes, slope_errs

def plot_data(freqs, signal):
    plt.plot(freqs, signal, label="data")
    limit_upper = np.max(signal) * 1.1
    limit_lower = np.min(signal) * 1.1
    plt.ylim(limit_lower, limit_upper)
    plt.legend()

def plot_fit(freqs, zfs, popt, p0):
    xfit = np.linspace(np.min(freqs), np.max(freqs), 1000)
    y_fit = odmr_model(xfit, zfs, *popt, n_ensembles=N_ENSEMBLES)
    y_guess = odmr_model(xfit, zfs, *p0, n_ensembles=N_ENSEMBLES)

    splittings = popt[:N_ENSEMBLES]

    for s in splittings:
        plt.axvline(s+zfs, label="FIT CENTER")

    plt.plot(xfit, y_fit, label="fit")
    plt.plot(xfit, y_guess, label="guess")
    plt.legend()

def plot_slopes(freqs, zfs, slopes, splittings):
    xfit = np.linspace(np.min(freqs), np.max(freqs), 1000)
    for i, (splitting, slope) in enumerate(zip(splittings, slopes)):
        yslope = linear(xfit, slope, splitting+zfs)
        plt.plot(xfit, yslope, label=f"Slope {i}")
    plt.legend()

if __name__ == "__main__":

    #with open(FILENAME_CENTER, "rb") as f:
    #    data_center = pickle.load(f)

    with open(FILENAME, "rb") as f:
        data = pickle.load(f)

    freqs = data["freqs"]
    signal = data["values"]

    hf_fit = data["hf_fit"]
    zfs_fit = data["zfs_fit"]
    splittings = data["splittings"]


    popt, err, p0 = fit_odmr(freqs, signal, zfs_fit, splittings, n_ensembles=N_ENSEMBLES)
    splittings_found = popt[:N_ENSEMBLES]
    amplitudes_found = popt[N_ENSEMBLES:2*N_ENSEMBLES]

    sort_index = np.argsort(amplitudes_found)[::-1]
    splittings_found = splittings_found[sort_index]

    hyperfine_found = popt[-2]
    slopes, slope_erros = fit_slopes(freqs, signal, zfs_fit, splittings_found)
    plot_data(freqs, signal)
    plot_fit(freqs, zfs_fit, popt, p0)
    plot_slopes(freqs, zfs_fit, slopes, splittings_found)
    plt.show()
