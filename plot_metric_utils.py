"""Shared metric helpers for v2 plotting (Fig 3/4/5 main + percentile supp + histograms).

All summaries are computed over the flattened ``seed × memory`` axis after
moving lag to the front, i.e. ``Aflat`` has shape ``(T, n_seeds * n_last)``.

Math contract:
    SNR_std(t) = (mean(angle_t) - mean(baseline)) / (std(angle_t) + std(baseline))
    SNR_pct(t) = (median(angle_t) - median(baseline))
                 / (sigma_pct(angle_t) + sigma_pct(baseline))
    sigma_pct  = (p80 - p20) / 2
    AUC        = mean_t max(log(SNR(t)), 0)

mean / std / median / percentile are all over pooled ``seed × memory`` samples;
no per-seed reduction.
"""
from pathlib import Path
import numpy as np

EPS = 1e-3


def seed_sibling_dirs(root_dir, seeds=(1, 2, 3)):
    """Return the base seed directory plus any existing ``_seedN`` siblings."""
    root = Path(root_dir)
    dirs = [root]
    for seed in seeds:
        sib = root.with_name(f"{root.name}_seed{seed}")
        if sib.exists():
            dirs.append(sib)
    return dirs


def load_pooled_summary(*dirs):
    """Concatenate summary.npz arrays from multiple seed dirs along axis 0.

    Each summary.npz is expected to hold per-key arrays of shape
    (n_seeds_in_file, T, n_last). Pooling stacks them so downstream
    ``stats_pool`` sees ``(sum_of_n_seeds, T, n_last)``.

    Returns ``None`` if no input dir exists, mirroring the previous ``load``
    behavior so callers can ``if d is None: skip``.
    """
    parts = []
    for d in dirs:
        p = Path(d) / "summary.npz"
        if p.exists():
            parts.append(dict(np.load(p)))
    if not parts:
        return None
    keys = parts[0].keys()
    out = {}
    for k in keys:
        out[k] = np.concatenate([p[k] for p in parts if k in p], axis=0)
    return out


def angle_deg_from_cos(cos_angles):
    return np.degrees(np.arccos(np.clip(cos_angles, -1.0, 1.0)))


def stats_pool(metric):
    """metric: (n_seeds, T, n_last); row 0 along T = lag=-1 baseline.

    Returns baseline scalars (mu_B, median_B, sigma_B, p20_B, p80_B) and
    per-lag arrays (mean, median, sigma_std, p20, p80) for t = 0..T-2,
    flattened across seeds * memories.
    """
    B = metric[:, 0, :].ravel()
    A = metric[:, 1:, :]
    # Move lag axis to the front so reshape preserves it. With shape
    # (n_seeds, T-1, n_last), naive A.reshape(T-1, -1) interleaves adjacent
    # lags from seed 0 — silently corrupts the per-lag pool. Transpose first.
    Aflat = np.transpose(A, (1, 0, 2)).reshape(A.shape[1], -1)
    return {
        "mu_B":      float(B.mean()),
        "median_B":  float(np.median(B)),
        "sigma_B":   float(B.std()),
        "p20_B":     float(np.percentile(B, 20)),
        "p80_B":     float(np.percentile(B, 80)),
        "mean":      Aflat.mean(axis=1),
        "median":    np.median(Aflat, axis=1),
        "sigma_std": Aflat.std(axis=1),
        "p20":       np.percentile(Aflat, 20, axis=1),
        "p80":       np.percentile(Aflat, 80, axis=1),
        "n_pool":    int(Aflat.shape[1]),
    }


def snr_std_curve(s):
    """SNR with mean/std-noise: (mean - mu_B) / (sigma_B + sigma_std(t))."""
    noise = s["sigma_B"] + s["sigma_std"]
    noise = np.where(noise < 1e-12, 1e-12, noise)
    return (s["mean"] - s["mu_B"]) / noise


def snr_pct_curve(s):
    """SNR with median/percentile-noise:
        (median - median_B) / (sigma_B_pct + sigma_pct(t))
    where sigma_X_pct = (p80_X - p20_X) / 2 for X in {baseline, signal-at-t}."""
    sb = (s["p80_B"] - s["p20_B"]) / 2
    ss = (s["p80"]   - s["p20"])   / 2
    noise = sb + ss
    noise = np.where(noise < 1e-12, 1e-12, noise)
    return (s["median"] - s["median_B"]) / noise


def logsnr(snr):
    return np.log(np.clip(snr, EPS, None))


def auc_positive_logsnr(curve):
    """AUC = mean_t max(curve_t, 0). curve is a 1D log(SNR) array."""
    return float(np.maximum(curve, 0.0).mean())
