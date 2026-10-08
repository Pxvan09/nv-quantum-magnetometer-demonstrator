import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import label
import sys

# Replay a saved center calibration:  python ensemble_guesser.py <YYYY-MM-DD_HH-MM-SS_calibration_center.pys>
FILENAME = sys.argv[1] if len(sys.argv) > 1 else "calibration_center.pys"
N_ENSEMBLES = 4


def find_center(freqs, signal, smooth_pts=3.0, refine_range=2.5e6,
                step_frac=0.1, thresh_frac=0.04, **kwargs):
    """
    1) Coarse: midpoint of the region where the smoothed |signal| is above a threshold
       (the 8 groups span the 'active' region, so its midpoint ~ center).
    2) Fine: mirror correlation around f0 in coarse +/- refine_range, using a FIXED
       symmetric window (no zero padding, no partial overlap bias).
       Both even (+1) and odd (-1) symmetry are tested.
    """
    freqs = np.asarray(freqs, float)
    s = np.asarray(signal, float)
    df = np.median(np.diff(freqs))

    s = s - np.median(s)
    s_sm = gaussian_filter1d(s, smooth_pts) if smooth_pts else s

    # ---- coarse: extent of the active region ----
    env = gaussian_filter1d(np.abs(s), 8)
    idx = np.where(env > thresh_frac * env.max())[0]
    f_lo, f_hi = freqs[idx[0]], freqs[idx[-1]]
    f_coarse = 0.5 * (f_lo + f_hi)
    half_active = 0.5 * (f_hi - f_lo)

    # ---- fine: fixed symmetric window ----
    W = half_active + 3e6                       # window half-width
    W = min(W, f_coarse - freqs[0] - refine_range, freqs[-1] - f_coarse - refine_range)
    x = np.arange(0, W, df)

    f0_grid = np.arange(f_coarse - refine_range, f_coarse + refine_range, df * step_frac)
    scores = np.zeros_like(f0_grid)
    for i, f0 in enumerate(f0_grid):
        a = np.interp(f0 + x, freqs, s_sm)
        b = np.interp(f0 - x, freqs, s_sm)
        scores[i] = np.sum(a * b) / (np.sqrt(np.sum(a * a) * np.sum(b * b)) + 1e-30)

    k = np.argmax(np.abs(scores))
    sign = int(np.sign(scores[k]))

    # parabolic refinement
    f0 = f0_grid[k]
    if 0 < k < len(f0_grid) - 1:
        y0, y1, y2 = np.abs(scores[k - 1:k + 2])
        d = y0 - 2 * y1 + y2
        if d != 0:
            f0 += 0.5 * (y0 - y2) / d * (f0_grid[1] - f0_grid[0])

    return dict(f0=f0, score=scores[k], sign=sign, f0_grid=f0_grid,
                scores=scores, f_coarse=f_coarse)


def plot_center(freqs, signal, res):
    import matplotlib.pyplot as plt
    f0 = res["f0"]
    fig, ax = plt.subplots(2, 1, figsize=(9, 7))
    ax[0].plot(freqs, signal, label="data")
    ax[0].plot(2 * f0 - freqs, res["sign"] * np.asarray(signal), alpha=0.6,
               label=f"mirrored (sign {res['sign']:+d})")
    ax[0].axvline(f0, color="k", ls="--", label=f"f0 = {f0/1e9:.6f} GHz")
    ax[0].legend()
    ax[1].plot(res["f0_grid"], res["scores"])
    ax[1].axvline(f0, color="k", ls="--")
    ax[1].set_ylabel("mirror correlation")
    plt.tight_layout()

def find_peak_symmetrie(freqs, signal):
    mid = len(signal) // 2
    first_half  = signal[:mid]
    second_half = signal[mid:][::-1]  # <-- fix

    corr = correlate(first_half, second_half, mode='full')
    lags = np.arange(-(len(second_half) - 1), len(first_half))

    df = freqs[1] - freqs[0]
    max_lag_hz = 5e6 #(freqs[-1] - freqs[0]) * 0.4   # instead of fixed 2e6
    max_lag_samples = int(max_lag_hz / df)

    center = len(second_half) - 1
    window_slice = slice(center - max_lag_samples, center + max_lag_samples + 1)
    lag_index = lags[window_slice][np.argmax(corr[window_slice])]

    #print(f"Lag index: {lag_index}")

    plt.figure()
    plt.plot(corr)
    plt.axvline(center + lag_index)  # <-- fix

    true_split = mid - round(lag_index / 2)  # <-- fix
    first_half_correct    = signal[:true_split]
    second_half_corrected = signal[true_split:][::-1]

    n = min(len(first_half_correct), len(second_half_corrected))
    plt.figure()
    plt.plot(first_half_correct[-n:],    label="left")
    plt.plot(second_half_corrected[-n:], label="right flipped")
    plt.legend()

    zfs_guess = freqs[true_split]
   # print(f"ZFS GUESS: {zfs_guess}")
    return zfs_guess

def find_groups(freqs, signal, zfs_guess, n_ensembles=4, peak_width=500e3):
    # Stark glätten um Hyperfein-Substruktur zu unterdrücken
    freqs = np.array(freqs)
    df = freqs[1] - freqs[0]

    zfs_idx = np.argmin( np.abs(freqs-zfs_guess) )
    #print(zfs_idx)

    half_signal    = signal[:zfs_idx][::-1]
    half_freqs     = freqs[:zfs_idx][::-1]

    #for i in range(len(half_signal)):
    #     if np.abs(half_signal[i]) >= 0.001:
    #        half_signal[i] = 0
    #     else:
    #         half_signal[i] = half_signal[i]   


    sigma_coarse = int(1e6 / df)
    sigma_fine   = int(0.3e6 / df)  # gerade so dass 3 Peaks sichtbar

    smoothed_coarse = gaussian_filter1d(np.abs(half_signal), sigma=sigma_coarse)
    smoothed_fine   = gaussian_filter1d(np.abs(half_signal), sigma=sigma_fine)


    from scipy.signal import hilbert

    # Hilbert gibt die Einhüllende des Ableitungssignals

    #threshold = np.abs(half_signal).max() * 0.1  # anpassbar, z.B. 10% des Maximums
    #half_signal = np.where(np.abs(half_signal) > threshold, half_signal, 0)

    analytic = hilbert(half_signal)
    envelope = analytic

    threshold = np.abs(envelope).max() * 0.1  # anpassbar, z.B. 10% des Maximums
    #envelope = np.where(np.abs(envelope) > threshold, envelope, 0)

    #for i in range(len(envelope)):
    #    a = int(5e6/df)
    #    if i <= a:
    #        envelope[i] = 0

    # Jetzt find_peaks auf der Einhüllenden –
    # die ist glatt und hat genau einen Peak pro 3er-Gruppe
    from scipy.signal import find_peaks
    peaks_idx, _ = find_peaks(envelope,
                            distance=int(1.8e6 /df),
                            width=int(peak_width / df),
                            prominence=envelope.max() * 0.1)
                            #height =  4.8e-6)

    # Top 4
    top4 = peaks_idx[np.argsort(envelope[peaks_idx])[-n_ensembles*3:]]
    group_freqs = half_freqs[top4]

    group_freqs_sorted = np.sort(group_freqs) - 0.0e6 # verschiebe um ca. halbe linienbreite

     #3er-Gruppen
    groups = group_freqs_sorted.reshape(n_ensembles, 3)

    # Mittlerer Peak jeder Gruppe (Index 1) oder Schwerpunkt
    group_centers = groups[:, 1]


    #plt.figure()
    #plt.plot(half_freqs, half_signal, label="signal")
   ## plt.plot(half_freqs, smoothed_coarse, label="smoothed coarse")
   ## plt.plot(half_freqs, smoothed_fine, label="smoothed fine")
    #plt.plot(half_freqs, envelope, label="envelope")
    #plt.plot(half_freqs[peaks_idx],envelope[peaks_idx],'kx')
    #for p in group_centers:
    #    plt.axvline(p)
    #    print(p)
    #plt.legend()
   

    #plt.show()
   # sys.exit()


    return group_centers


# Analytische Ableitung eines Lorentzians:
# dL/df = A_eff * (f - f0) / (1 + ((f - f0) / gamma)^2)^2
def lorentz_deriv_old(f, A_eff, f0, gamma):
    x = (f - f0) / gamma
    return A_eff * x / (1 + x**2)**2

def model_old(f, zfs, hyperfine, linewidth, *group_params):
    # group_params: [delta_0, amp_0, delta_1, amp_1, ..., sign]
    # sign: globales Vorzeichen links vs. rechts vom ZFS
    result = np.zeros_like(f, dtype=float)
    #sign = group_params[-1]  # letzter Parameter

    n_ensembles = (len(group_params) - 2) // 2
    sign_left  = group_params[-2]
    sign_right = group_params[-1]

    sign = 1

    for i in range(n_ensembles):
        delta_i = group_params[i * 2]
        amp_i   = group_params[i * 2 + 1]

        # Drei Peaks pro Gruppe: -hyperfine, 0, +hyperfine
        for hf_offset in [-hyperfine, 0, hyperfine]:
            # Linke Seite (zfs - delta): invertiertes Vorzeichen
            f0_left = zfs - delta_i + hf_offset
            result += lorentz_deriv(f, -sign_left * amp_i, f0_left, linewidth)

            # Rechte Seite (zfs + delta)
            f0_right = zfs + delta_i + hf_offset
            result += lorentz_deriv(f, -sign_right * amp_i, f0_right, linewidth)

    return result

def get_sign_from_first_peak(sig, df):
    min_distance = int(1e6 / abs(df))
    prominence   = np.max(np.abs(sig)) * 0.1

    pos_peaks, _ = find_peaks( sig, distance=min_distance, prominence=prominence)
    neg_peaks, _ = find_peaks(-sig, distance=min_distance, prominence=prominence)

    if len(pos_peaks) == 0 and len(neg_peaks) == 0:
        return 1.0

    first_pos = pos_peaks[0] if len(pos_peaks) > 0 else np.inf
    first_neg = neg_peaks[0] if len(neg_peaks) > 0 else np.inf

    return 1.0 if first_pos < first_neg else -1.0


def fit_odmr_signal_old(freqs, signal, zfs_guess, group_center_guesses, n_ensembles=4):
    from scipy.optimize import curve_fit
    import numpy as np

    freqs = np.array(freqs)
    signal = np.array(signal)


    df = freqs[1] - freqs[0]



    zfs_idx      = np.argmin(np.abs(freqs - zfs_guess))
    left_signal  = signal[:zfs_idx]
    right_signal = signal[zfs_idx:]

    sign_left_init  = get_sign_from_first_peak(left_signal,  df)
    sign_right_init = get_sign_from_first_peak(right_signal, df)
    #print(f"Sign left: {sign_left_init}, Sign right: {sign_right_init}")

    # --- Initial Guesses ---
    zfs_init       = zfs_guess
    hyperfine_init = 2.16e6
    linewidth_init = 1e6  # 1 MHz Startbreite

    # delta_i: Abstand der Gruppe vom ZFS
    group_center_guesses = np.sort(group_center_guesses)
    delta_guesses = np.abs(group_center_guesses - zfs_guess)

    #print("Group centers:", group_center_guesses/1e9)
    #print("Deltas:", np.abs(group_center_guesses - zfs_guess)/1e6, "MHz")

    # amp_i: Peak-zu-Peak Amplitude im Signalbereich um jeden Gruppen-Center
    window = int(5e6 / np.abs(df))
    amp_guesses = []
    for gc in group_center_guesses:
        idx = np.argmin(np.abs(freqs - gc))
        sl  = slice(max(0, idx - window), min(len(signal), idx + window))
        amp_guesses.append(np.max(signal[sl]) - np.min(signal[sl]))

    #sign_init = 1.0  # links/rechts Vorzeichen

    # p0 zusammenbauen: [zfs, hyperfine, linewidth, delta_0, amp_0, ..., sign]
    p0 = [zfs_init, hyperfine_init, linewidth_init]
    for i in range(n_ensembles):
        p0.append(delta_guesses[i])
        p0.append(amp_guesses[i])

    p0.append(sign_left_init)
    p0.append(sign_right_init)

    # --- Bounds ---
    lower = [zfs_guess - 5e6,  1e6,   0.1e6]
    upper = [zfs_guess + 5e6,  5e6,   10e6]
    for i in range(n_ensembles):
        lower += [0,       0]
        upper += [50e6,    np.inf]
    lower += [-1.5, -1.5]   # sign left
    upper += [ 1.5,  1.5]   # sign right

    try:
        popt, pcov = curve_fit(model, freqs, signal,
                               p0=p0,
                               bounds=(lower, upper),
                               maxfev=10000)
        perr = np.sqrt(np.diag(pcov))
    except RuntimeError as e:
        print(f"Fit nicht konvergiert: {e}")
        return None, None

    # --- Ergebnisse ausgeben ---
    zfs_fit, hf_fit, lw_fit = popt[0], popt[1], popt[2]
    #print(f"ZFS:        {zfs_fit/1e9:.6f} GHz")
    #print(f"Hyperfine:  {hf_fit/1e6:.4f} MHz")
    #print(f"Linewidth:  {lw_fit/1e6:.4f} MHz")
    for i in range(n_ensembles):
        delta_i = popt[3 + i*2]
        amp_i   = popt[4 + i*2]
        #print(f"Gruppe {i}: delta={delta_i/1e6:.4f} MHz, amp={amp_i:.4e}")
    #print(f"Sign:       {popt[-1]:.3f}")

    # --- Plot ---
    #signal_fit = model(freqs, *popt)
    #plt.figure(figsize=(14, 5))
    #plt.plot(freqs/1e9, signal,     label='data', alpha=0.7)
    #plt.plot(freqs/1e9, signal_fit, label='fit',  linewidth=2)
    #plt.axvline(zfs_fit/1e9, color='red', linestyle='--', label='ZFS')
    #plt.legend()
    #plt.xlabel('GHz')
    #plt.tight_layout()

    return popt, p0, pcov


def lorentz_deriv(f, A_eff, f0, gamma):
    x = (f - f0) / gamma
    return A_eff * x / (1 + x**2)**2


def model(f, zfs, hyperfine, *group_params):
    # group_params: [delta_0, amp_0, lw_0, ..., delta_n, amp_n, lw_n, sign_left, sign_right]
    result      = np.zeros_like(f, dtype=float)
    n_ensembles = (len(group_params) - 2) // 3
    sign_left   = group_params[-2]
    sign_right  = group_params[-1]

    for i in range(n_ensembles):
        delta_i    = group_params[i * 3]
        amp_i      = group_params[i * 3 + 1]
        linewidth_i = group_params[i * 3 + 2]

        for hf_offset in [-hyperfine, 0, hyperfine]:
            f0_left  = zfs - delta_i + hf_offset
            f0_right = zfs + delta_i + hf_offset
            result  += lorentz_deriv(f, sign_left  * amp_i, f0_left,  linewidth_i)
            result  += lorentz_deriv(f, sign_right * amp_i, f0_right, linewidth_i)

    return result


def fit_odmr_signal(freqs, signal, zfs_guess, group_center_guesses, n_ensembles=4):
    from scipy.optimize import curve_fit

    freqs  = np.array(freqs)
    signal = np.array(signal)
    df     = freqs[1] - freqs[0]

    zfs_idx      = np.argmin(np.abs(freqs - zfs_guess))
    left_signal  = signal[:zfs_idx]
    right_signal = signal[zfs_idx:]


    sign_left_init  = get_sign_from_first_peak(left_signal,  df)
    sign_right_init = get_sign_from_first_peak(right_signal, df)
    # print(f"Sign left init: {sign_left_init}, Sign right init: {sign_right_init}")

    group_center_guesses = np.sort(group_center_guesses)
    delta_guesses        = np.abs(group_center_guesses - zfs_guess)

    window      = int(5e6 / np.abs(df))
    amp_guesses = []
    for gc in group_center_guesses:
        idx = np.argmin(np.abs(freqs - gc))
        sl  = slice(max(0, idx - window), min(len(signal), idx + window))
        amp_guesses.append(np.max(np.abs(signal[sl])))

    # Basis p0 ohne sign: [zfs, hyperfine, delta_0, amp_0, lw_0, ...]
    p0_base = [zfs_guess, 2.16e6]
    for i in range(n_ensembles):
        p0_base.append(delta_guesses[i])
        p0_base.append(amp_guesses[i])
        p0_base.append(1e6)  # linewidth guess 1 MHz pro Gruppe

    # Basis bounds ohne sign
    lower_base = [zfs_guess - 1000e6, 1e6]
    upper_base = [zfs_guess + 1000e6, 5e6]
    for i in range(n_ensembles):
        lower_base += [0,    5e-6,    0.1e6]
        upper_base += [55e6, 2*np.max(np.abs(signal)), 15e6]

    best_result   = None
    best_residual = np.inf

    for sl, sr in [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)]:
        p0    = p0_base + [sl, sr]
        lower = lower_base + [sl - 0.1, sr - 0.1]
        upper = upper_base + [sl + 0.1, sr + 0.1]

        try:
            popt_try, pcov_try = curve_fit(model, freqs, signal,
                                            p0=p0,
                                           # bounds=(lower, upper),
                                            maxfev=10000)
            residual = np.sum((signal - model(freqs, *popt_try))**2)
            # print(f"  sign=({sl:+.0f},{sr:+.0f}) residual={residual:.4e}")
            if residual < best_residual:
                best_residual = residual
                best_result   = (popt_try, p0, pcov_try)
        except RuntimeError:
            print(f"  sign=({sl:+.0f},{sr:+.0f}) nicht konvergiert")
            continue

    if best_result is None:
        print("Kein Fit konvergiert.")
        return None, None, None

    popt, p0_best, pcov = best_result

    zfs_fit, hf_fit = popt[0], popt[1]
    #print(f"\nBester Fit:")
    #print(f"ZFS:        {zfs_fit/1e9:.6f} GHz")
    #print(f"Hyperfine:  {hf_fit/1e6:.4f} MHz")
    #print(f"Sign left:  {popt[-2]:.3f}")
    #print(f"Sign right: {popt[-1]:.3f}")
    for i in range(n_ensembles):
        delta_i    = popt[2 + i * 3]
        amp_i      = popt[3 + i * 3]
        lw_i       = popt[4 + i * 3]
        #print(f"Gruppe {i}: delta={delta_i/1e6:.4f} MHz, amp={amp_i:.4e}, lw={lw_i/1e6:.4f} MHz")

    return popt, p0_best, pcov



def plot_data(freqs, signal):
    plt.figure()
    plt.plot(freqs, signal, label="data")
    plt.legend()

def plot_fit(freqs, popt, p0):
    xfit = np.linspace(np.min(freqs), np.max(freqs), 1000)
    zfs_fit = popt[0]
    hyperfine_fit = popt[1]
    linewidth_fit = popt[2]
    group_params_fit = popt[3:]

    zfs_guess = p0[0]
    hyperfine_guess = p0[1]
    linewidth_guess = p0[2]
    group_params_guess = p0[3:]
    yfit = model(xfit, zfs_fit, hyperfine_fit, linewidth_fit, *group_params_fit)
    yguess = model(xfit, zfs_guess, hyperfine_guess, linewidth_guess, *group_params_guess)
    plt.plot(xfit, yfit, label="FIT")
    plt.plot(xfit, yguess, label="INITIAL GUESS")

if __name__ == '__main__':
    with open(FILENAME, "rb") as f:
        data = pickle.load(f)


    freqs = data["freqs"]
    signal = data["values"]
    #print signal freqs
    #for i in range(len(freqs)):
    #    print(f"{freqs[i]/1e9:.6f} GHz: {signal[i]:.4e}")

    signal_filtered = savgol_filter(signal,11,3)
    #zfs_guess = find_peak_symmetrie(freqs, signal_filtered)
    res=find_center(freqs, signal)
    zfs_guess = res["f0"]                # try mode="envelope" as a cross-check
    #print(f"Center: {res['f0']/1e9:.6f} GHz  (corr={res['score']:.3f}, sign={res['sign']:+d})")
    plot_center(freqs, signal, res)
    #print res
    #print(f"Result: {res}")
    
    plt.show()
    group_center_guesses = find_groups(freqs, signal, zfs_guess, n_ensembles=N_ENSEMBLES)
    popt, p0, pcov = fit_odmr_signal(freqs, signal, zfs_guess, group_center_guesses, n_ensembles=N_ENSEMBLES)

    zfs_fit = popt[0]



    plot_data(freqs, signal)
    plot_fit(freqs, popt, p0)
    plt.axvline(zfs_guess, color='r', label="ZFS guess")
    plt.axvline(zfs_fit, color='purple', label="ZFS fit")
    for p in group_center_guesses:
        plt.axvline(p, color='orange', label="Group guess")
    
    plt.legend()
    plt.show()

