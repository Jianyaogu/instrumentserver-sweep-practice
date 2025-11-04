from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from labcore.data.datadict_storage import datadict_from_hdf5

# ---------- paste your fitter(s) here ----------
# If you have _fit_double_hanger or _fit_double_hanger_improved already,
# import them; otherwise comment out the "FIT & OVERLAY" section below.
# from your_module import _fit_double_hanger_improved as fit_fn

# ---------- loader that works for your ddh5 ----------
def load_ddh5_res_spec(path_str):
    """Return flux (N,), freqs (M,), S2 (N,M) complex from a ddh5 resonator-vs-flux file."""
    dd = datadict_from_hdf5(Path(path_str))

    S      = np.asarray(dd["signal"]["values"])         # (Ncur, Nrep, Nfreq) complex
    axes   = list(dd["signal"]["axes"])                 # ['current','repetition','ssb_frequency']
    shape  = tuple(dd["signal"]["__shape__"])           # (Ncur, Nrep, Nfreq)

    # Helper: get a 1D axis vector even if file stored a broadcast grid
    def axis_vector(name):
        vals = np.asarray(dd[name]["values"])
        if vals.size == shape[axes.index(name)]:        # already 1D
            return vals.astype(float).ravel()
        if vals.size == np.prod(shape):                 # full grid
            arr = vals.reshape(shape).astype(float)
            idx = [0]*len(shape)
            idx[axes.index(name)] = slice(None)         # take a line along that axis
            return arr[tuple(idx)]
        u, ix = np.unique(vals, return_index=True)      # fallback
        return u[np.argsort(ix)].astype(float)

    # Axis names in your file
    flux  = axis_vector("current")                      # (Ncur,)
    freqs = axis_vector("ssb_frequency")                # (Nfreq,)

    # Average repetition → (Ncur, Nfreq)
    if "repetition" in axes:
        S = S.mean(axis=axes.index("repetition"))
        axes = [a for a in axes if a != "repetition"]   # now ['current','ssb_frequency']

    # Ensure order (flux, freq)
    cur_i  = axes.index("current")
    freq_i = axes.index("ssb_frequency")
    if (cur_i, freq_i) != (0, 1):
        S = np.transpose(S, (cur_i, freq_i))

    # Sort by flux just in case
    order = np.argsort(flux)
    flux  = flux[order]
    S     = S[order]

    # Sanity
    assert S.ndim == 2, S.shape
    assert freqs.ndim == 1, freqs.shape
    assert S.shape[1] == freqs.size, f"{S.shape} vs {freqs.shape}"

    return flux, freqs, S

# ---------- simple unwrap/delay removal (optional for fitter) ----------
def unwrap_remove_linear_phase(freq_Hz, s):
    ph = np.unwrap(np.angle(s))
    b1, b0 = np.polyfit(freq_Hz, ph, 1)
    return s * np.exp(-1j*(b0 + b1*freq_Hz)), b1/(2*np.pi)

# ---------- RUN: change this path ----------
PATH = r"C:\Users\yoyo1\SQcircuit-examples\examples\data.ddh5"

flux, freqs, S2 = load_ddh5_res_spec(PATH)  # S2: (Nflux, Nfreq) complex

# ---------- PLOT |S12| map ----------
#plt.imshow(np.abs(S2.T), origin='upper', aspect='auto',
#           extent=[flux.min(), flux.max(), freqs.min(), freqs.max()],
#           cmap='inferno')
#plt.colorbar(label='|S12| (Mag)')
#plt.xlabel('current (A) / flux')
#plt.ylabel('frequency (Hz)')
#plt.title('|S12| vs frequency and flux')
#plt.tight_layout()
#plt.show()

import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, Optional, Callable, Sequence, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
@dataclass
class FluxList:
    name: str
    value: Iterable[float]

def _unwrap_and_remove_linear_phase(freq: np.ndarray, sig: np.ndarray) -> Tuple[np.ndarray, float]:
    """Unwrap phase and remove best-fit linear slope (cable delay)."""
    # unwrap and linear fit: angle ≈ φ0 + 2π τ f, so it becomes continuous
    ph = np.unwrap(np.angle(sig))
    slope = np.polyfit(freq, ph, 1)[0]
    sig_unw = sig * np.exp(-1j * slope * freq)
    return sig_unw, slope/(2*np.pi)  # return τ estimate in seconds if freq in Hz

def _hanger_single(f: np.ndarray, f0, Ql, Qc, theta, a, phi0, tau) -> np.ndarray:
    x = (f - f0) / f0
    notch = 1.0 - (Ql/Qc) * np.exp(1j*theta) / (1.0 + 2j*Ql*x)
    return a * np.exp(1j*(phi0 + 2*np.pi*f*tau)) * notch

def _hanger_double_mix(f: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    """Mixture of two nearby resonances with shared line params, weighted by p_g."""
    f0g, f0e = params["f0g"], params["f0e"]
    Ql, Qc   = params["Ql"],  params["Qc"]
    theta    = params["theta"]
    a        = params["a"]
    phi0     = params["phi0"]
    tau      = params["tau"]
    pg       = params["pg"]  # in [0,1]
    s_g = _hanger_single(f, f0g, Ql, Qc, theta, a, phi0, tau)
    s_e = _hanger_single(f, f0e, Ql, Qc, theta, a, phi0, tau)
    return pg*s_g + (1.0 - pg)*s_e

def _fit_double_hanger(freq: np.ndarray, sig_cplx: np.ndarray):
    f = np.asarray(freq, float)
    yC = np.asarray(sig_cplx, complex)

    # -------- de-embed linear phase (done in your pipeline, but safe here) --------
    ph = np.unwrap(np.angle(yC))
    b1, b0 = np.polyfit(f, ph, 1)         # ph ≈ b0 + b1 f
    y = yC * np.exp(-1j * (b0 + b1*f))    # unwrapped/de-sloped
    # NOTE: we fit in this frame; we won't refit tau here.

    # -------- initial guesses via local minima on smoothed |S| --------
    mag = np.abs(y)
    win = max(5, len(f)//80)
    mag_s = np.convolve(mag, np.ones(win)/win, mode="same")

    # local minima: indices i with mag_s[i] < neighbors
    locmins = np.where((mag_s[1:-1] < mag_s[:-2]) & (mag_s[1:-1] < mag_s[2:]))[0] + 1
    if len(locmins) == 0:
        locmins = np.array([int(np.argmin(mag_s))])
    # sort by depth
    locmins = locmins[np.argsort(mag_s[locmins])]

    # pick the deepest as ground; the closest *distinct* other min as excited
    i0 = locmins[0]
    # distinct = at least a few bins away
    min_sep_bins = max(5, len(f)//200)
    candidates = [j for j in locmins[1:] if abs(j - i0) >= min_sep_bins]
    j0 = candidates[0] if candidates else (i0 + min_sep_bins if i0 + min_sep_bins < len(f) else i0 - min_sep_bins)

    f0g0, f0e0 = np.sort([f[i0], f[j0]])

    # -------- line-term seeds (gentle) --------
    a0 = float(np.median(mag))
    theta0 = 0.0
    phi00 = 0.0
    tau0 = 0.0
    Ql0, Qc0 = 1.5e4, 3.0e4
    pg0 = 0.7

    # -------- bounds with a narrow f0e window around its seed --------
    # estimate natural linewidth ~ f0 / Ql0; allow a wide multiple
    linewidth = f0g0 / Ql0
    df_win = max(10.0*linewidth, 1.5e6)   # e.g. +/- 1.5 MHz minimum window
    lb = dict(f0g=f.min(),  f0e=f0e0 - 4*df_win, Ql=500,  Qc=500,
              theta=-np.pi, a=0.0, phi0=-np.pi, tau=-1e-6, pg=0.0)
    ub = dict(f0g=f.max(),  f0e=f0e0 + 4*df_win, Ql=5e5, Qc=5e6,
              theta= np.pi, a=10*np.max(mag), phi0= np.pi, tau= 1e-6, pg=1.0)

    params0 = dict(f0g=f0g0, f0e=f0e0, Ql=Ql0, Qc=Qc0, theta=theta0, a=a0, phi0=phi00, tau=tau0, pg=pg0)

    # -------- model & packing --------
    def _hanger_single(ff, f0, Ql, Qc, theta, a, phi0, tau):
        x = (ff - f0)/f0
        notch = 1.0 - (Ql/Qc)*np.exp(1j*theta)/(1.0 + 2j*Ql*x)
        return a*np.exp(1j*(phi0 + 2*np.pi*ff*tau))*notch

    def _hanger_double_mix(ff, p):
        sg = _hanger_single(ff, p["f0g"], p["Ql"], p["Qc"], p["theta"], p["a"], p["phi0"], p["tau"])
        se = _hanger_single(ff, p["f0e"], p["Ql"], p["Qc"], p["theta"], p["a"], p["phi0"], p["tau"])
        return p["pg"]*sg + (1.0 - p["pg"])*se

    keys = ["f0g","f0e","Ql","Qc","theta","a","phi0","tau","pg"]
    def pack(p):   return np.array([p[k] for k in keys], float)
    def unpack(v): return {k: float(x) for k,x in zip(keys, v)}

    v0  = pack(params0)
    vlb = pack(lb)
    vub = pack(ub)

    # -------- residual with weights (emphasize dips) --------
    # weight ~ 1/(mag_s^2) but bounded to avoid extremes
    w = 1.0/np.maximum(mag_s**2, 1e-4)
    w = w/np.max(w)
    def resid(v):
        p = unpack(v)
        s = _hanger_double_mix(f, p)
        r = np.stack([np.real(s) - np.real(y), np.imag(s) - np.imag(y)], axis=1)
        r = (r.T * w).T     # apply weights pointwise to both Re/Im
        return r.ravel()

    # -------- optimize (SciPy if present, else projected gradient) --------
    try:
        from scipy.optimize import least_squares
        res = least_squares(resid, v0, bounds=(vlb, vub), xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=4000)
        vf = res.x
    except Exception:
        vf = v0.copy()
        lr = 2e-6
        eps = np.array([5e3, 5e3, 50.0, 50.0, 1e-3, 1e-3, 1e-3, 1e-9, 1e-2])
        for _ in range(800):
            r0 = resid(vf); base = np.dot(r0, r0)
            g = np.zeros_like(vf)
            for k in range(len(vf)):
                vtmp = vf.copy(); vtmp[k] = np.clip(vtmp[k] + eps[k], vlb[k], vub[k])
                r2 = resid(vtmp); g[k] = (np.dot(r2,r2) - base)/eps[k]
            vf -= lr*g
            vf = np.minimum(np.maximum(vf, vlb), vub)

    p_fit = unpack(vf)
    # enforce ordering convention
    if p_fit["f0g"] > p_fit["f0e"]:
        p_fit["f0g"], p_fit["f0e"] = p_fit["f0e"], p_fit["f0g"]
        p_fit["pg"] = 1.0 - p_fit["pg"]
    

    # derived Qi and a simple SNR proxy
    Qi = (p_fit["Ql"]*p_fit["Qc"])/max(p_fit["Qc"] - p_fit["Ql"], 1e-6)
    rr = resid(vf).reshape(-1,2)
    noise = np.std(np.hypot(rr[:,0], rr[:,1]))
    amp = np.max(mag) - np.min(mag)
    snr = float(abs(amp)/(4*max(noise, 1e-12)))
    return p_fit, {"Qi": Qi}, snr


def fit_all_rows(ddh5_path):
    flux, freqs, S2 = load_ddh5_res_spec(ddh5_path)

    fr_g = np.full(len(flux), np.nan)
    fr_e = np.full(len(flux), np.nan)
    Ql   = np.full(len(flux), np.nan)
    Qc   = np.full(len(flux), np.nan)
    Qi   = np.full(len(flux), np.nan)
    pg   = np.full(len(flux), np.nan)
    snr  = np.full(len(flux), np.nan)

    for i in range(len(flux)):
        s_unw, _ = unwrap_remove_linear_phase(freqs, S2[i])
        p, derived, s = _fit_double_hanger(freqs, s_unw)
        fr_g[i], fr_e[i] = p["f0g"], p["f0e"]
        Ql[i], Qc[i], Qi[i] = p["Ql"], p["Qc"], derived["Qi"]
        pg[i], snr[i] = p["pg"], s
        

    # Plot overlays (correct axes: freq vertical, flux horizontal)
    plt.figure(figsize=(8,5))
    plt.imshow(np.abs(S2).T, origin="upper", aspect="auto",
               extent=[float(flux.min()), float(flux.max()),
                       float(freqs.min()), float(freqs.max())],
               cmap="inferno")
    plt.plot(flux, fr_g, 'c', lw=2, label='f_r,g')
    plt.plot(flux, fr_e, 'm', lw=2, label='f_r,e')
    plt.colorbar(label="|S12| (Mag)")
    plt.xlabel("current (A) / flux")
    plt.ylabel("frequency (Hz)")
    plt.title("Double-hanger fits over |S12| map")
    plt.legend(); plt.tight_layout(); plt.show()

    out = dict(flux=flux, freqs=freqs, fr_g=fr_g, fr_e=fr_e,
               Ql=Ql, Qc=Qc, Qi=Qi, pg=pg, snr=snr)
    np.savez("results_res_spec_vs_flux.npz", **out)
    print("Saved: results_res_spec_vs_flux.npz")
    return out

if __name__ == "__main__":
    DDH5 = r"C:\Users\yoyo1\SQcircuit-examples\examples\data.ddh5"  # <- set your path
    fit_all_rows(DDH5)
data = np.load("results_res_spec_vs_flux.npz")
flux = data["flux"]
fr_g = data["fr_g"]
fr_e = data["fr_e"]


# ---- Plot only f_r_g vs flux ----
plt.figure(figsize=(6,4))
plt.plot(flux, fr_e/1e9, 'c.', label=r"$f_{r,g}$")   # in GHz for readability
plt.xlabel("Current (A) / Flux")
plt.ylabel("Resonator Frequency (GHz)")
plt.title(r"$f_{r,e}$ vs Flux")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

def _local_linear_slope(x, y, x0, halfwidth):
    """Fit y ≈ a*(x-x0) + b on [x0-halfwidth, x0+halfwidth]; return slope a."""
    lo, hi = x0 - halfwidth, x0 + halfwidth
    mask = (x >= lo) & (x <= hi)
    xx, yy = x[mask], y[mask]
    if len(xx) < 3:
        return np.nan
    u = xx - x0
    a, b = np.polyfit(u, yy, 1)   # slope, intercept
    return float(a)

def _symmetry_score(x, y, c, halfwidth, n_pairs, eps=1e-12):
    """
    Symmetry score at center c:
        score = RMS_{j} [ y(c+Δ_j) - y(c-Δ_j) ] / scale
    where Δ_j are n_pairs offsets in (0, halfwidth], and y is linearly interpolated.
    """
    # build offsets that are available on both sides within data range
    Δmax = min(c - x[0], x[-1] - c, halfwidth)
    if Δmax <= 0:
        return np.inf
    Δ = np.linspace(Δmax/n_pairs, Δmax, n_pairs)

    yp = np.interp(c + Δ, x, y)
    ym = np.interp(c - Δ, x, y)

    r = yp - ym
    rms = np.sqrt(np.mean(r*r))

    # robust scale for normalization (MAD over data)
    med = np.median(y)
    mad = np.median(np.abs(y - med)) + eps
    return float(rms / mad)

def find_even_centers_with_flatness(
        I, fr_g,
        halfwidth_sym=None,    # window half-width for symmetry (same units as I)
        n_pairs=25,
        halfwidth_slope=None,  # window half-width for slope fit
        slope_tol=None,        # |slope| threshold; if None => data-driven
        score_rel_drop=0.25,   # how deep a local min must be vs neighbors
        max_candidates=8
    ):
    """
    Return centers where the curve is most even AND locally flat (|slope| small).

    Parameters
    ----------
    I : (N,) array_like
        Monotone current/flux samples (unknown offset/period fine).
    fr_g : (N,) array_like
        Measured resonator feature vs I (use magnitude branch).
    halfwidth_sym : float, optional
        Half-width around candidate centers for symmetry check.
        Default: 10% of span(I).
    n_pairs : int
        Number of mirrored pairs for the symmetry score.
    halfwidth_slope : float, optional
        Half-width for linear slope fit. Default: 5% of span(I).
    slope_tol : float, optional
        Absolute slope threshold. If None, set to 0.25 * median|dy/dx|.
    score_rel_drop : float
        Candidate local minima must be at least this fraction lower than a
        simple 5-point neighborhood average.
    max_candidates : int
        Keep at most this many top (lowest-score) candidates before slope filter.

    Returns
    -------
    result : dict with keys
        - 'picked'      : list of dicts {center, score, slope}
        - 'candidates'  : (before slope filter) list of dicts {center, score, slope}
        - 'status'      : 'ok' or message
        - 'notes'       : list of strings describing thresholds used
    """
    x = np.asarray(I, float)
    y = np.asarray(fr_g, float)
    N = len(x)
    if N < 7 or np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        return dict(picked=[], candidates=[], status="bad_input", notes=["non-finite or too short"])

    # defaults for windows
    span = x[-1] - x[0]
    if halfwidth_sym   is None: halfwidth_sym   = 0.10 * span
    if halfwidth_slope is None: halfwidth_slope = 0.05 * span

    # data-driven slope tolerance if not provided
    # estimate derivative with central differences
    dx = np.gradient(x)
    dy = np.gradient(y, x)
    med_abs_slope = np.median(np.abs(dy[np.isfinite(dy)]))
    if slope_tol is None:
        slope_tol = 0.25 * (med_abs_slope if med_abs_slope > 0 else 1.0)

    # compute symmetry score on a coarse grid of candidate centers
    # (use every point but we’ll smooth to avoid noise)
    scores = np.empty(N, float)
    for k in range(N):
        scores[k] = _symmetry_score(x, y, x[k], halfwidth_sym, n_pairs)

    # find local minima of the score (simple 5-point neighborhood)
    winsize = 5
    halfw = winsize // 2
    candidates = []
    for k in range(halfw, N - halfw):
        local_avg = np.mean(scores[k - halfw:k + halfw + 1])
        if scores[k] <= (1.0 - score_rel_drop) * local_avg:
            c = x[k]
            slope = _local_linear_slope(x, y, c, halfwidth_slope)
            candidates.append(dict(center=c, score=scores[k], slope=slope))

    # keep best few by score before flatness filter
    candidates = sorted(candidates, key=lambda d: d["score"])[:max_candidates]

    # apply flatness (|slope| <= tol)
    picked = [c for c in candidates if np.isfinite(c["slope"]) and abs(c["slope"]) <= slope_tol]

    notes = [
        f"halfwidth_sym={halfwidth_sym:.3g}",
        f"halfwidth_slope={halfwidth_slope:.3g}",
        f"slope_tol={slope_tol:.3g} (data-driven 0.25*median|dy/dx|)",
        f"score_rel_drop={score_rel_drop}",
        f"n_pairs={n_pairs}",
        f"span={span:.3g}, N={N}",
    ]
    status = "ok" if picked else "no_flat_even_center_found"

    return dict(picked=picked, candidates=candidates, status=status, notes=notes)
res = find_even_centers_with_flatness(flux, fr_g)

print(res["status"])
for c in res["picked"]:
    print(f"center ≈ {c['center']:.6g}, score={c['score']:.3e}, slope={c['slope']:.3e}")
