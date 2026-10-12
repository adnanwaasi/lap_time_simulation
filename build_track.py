"""Build a track CSV (ds, curvature, gradient) from real F1 position telemetry.

Usage: python build_track.py [year] [event] [session] [out.csv]
       python build_track.py 2025 Silverstone Q tracks/silverstone_real.csv

The racing line comes from averaging every quick lap in the session, so the result is
the line the cars actually drive, not the track centreline.

1. Fit a smooth closed curve through the pole lap to get a distance coordinate s.
2. Project every quick lap's X/Y/Z samples onto that curve and average them in 5 m bins.
   Position noise is random, so it shrinks roughly as 1/sqrt(number of laps).
3. Fit a smooth closed curve through the averaged line and take
   curvature = (x' y'' - y' x'') / (x'^2 + y'^2)^1.5 and gradient = dz/ds.
4. Smooth curvature over ~25 m: a car can't change its turning radius instantly,
   so faster wiggles are measurement noise, not track shape.
"""

import logging
import sys
from pathlib import Path

import fastf1
import numpy as np
from scipy.interpolate import splev, splprep
from scipy.spatial import cKDTree

DS = 5.0               # Output segment length (m), same as main.cpp
CURV_SMOOTH_M = 25.0   # Curvature smoothing window (m)
ELEV_SMOOTH_M = 100.0  # Elevation smoothing window (m); Z is noisier than X/Y


def positions(tel):
    """X/Y/Z in metres (FastF1 uses 1/10 m), dropping repeated samples."""
    p = tel[["X", "Y", "Z"]].to_numpy() / 10.0
    keep = np.r_[True, np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1) > 0.5]
    return p[keep]


def closed_spline(p, smoothing):
    """Periodic smoothing spline through points p, parameterised by arc length."""
    closed = np.vstack([p, p[:1]])
    u = np.r_[0, np.cumsum(np.linalg.norm(np.diff(closed[:, :2], axis=0), axis=1))]
    tck, _ = splprep([closed[:, 0], closed[:, 1]], u=u, s=smoothing, per=1)
    return tck, u[-1]


def circular_mean(values, window):
    """Centred moving average that wraps around the lap."""
    w = max(1, int(round(window)))
    padded = np.r_[values[-w:], values, values[:w]]
    kernel = np.ones(w) / w
    return np.convolve(padded, kernel, mode="same")[w:-w]


def main():
    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
    event = sys.argv[2] if len(sys.argv) > 2 else "Silverstone"
    kind = sys.argv[3] if len(sys.argv) > 3 else "Q"
    out = Path(sys.argv[4] if len(sys.argv) > 4 else f"tracks/{event.lower().replace(' ', '_')}_real.csv")

    logging.getLogger("fastf1").setLevel(logging.WARNING)
    Path("data/fastf1_cache").mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache("data/fastf1_cache")
    session = fastf1.get_session(year, event, kind)
    session.load(weather=False, messages=False)

    # 1. Reference line from the pole lap
    pole = session.laps.pick_fastest()
    ref_tck, ref_len = closed_spline(positions(pole.get_pos_data()), smoothing=2000)
    ref_s = np.arange(0.0, ref_len, 0.5)
    tree = cKDTree(np.array(splev(ref_s, ref_tck)).T)

    # 2. Project every quick lap onto it and average in DS bins
    samples = []
    laps = session.laps.pick_quicklaps(1.03).pick_wo_box()
    for _, lap in laps.iterlaps():
        try:
            p = positions(lap.get_pos_data())
        except Exception:
            continue
        dist, idx = tree.query(p[:, :2])
        ok = dist < 15.0  # ignore samples far off the line (bad data)
        samples.append(np.c_[ref_s[idx[ok]], p[ok]])
    samples = np.vstack(samples)
    bins = np.floor(samples[:, 0] / DS).astype(int)
    line = np.array([samples[bins == b, 1:].mean(axis=0) for b in np.unique(bins)])
    print(f"Averaged {len(laps)} laps, {len(samples)} samples, {len(line)} bins")

    # 3. Smooth closed curve through the averaged line, sampled every DS metres
    tck, length = closed_spline(line, smoothing=200)
    s = np.arange(0.0, length, DS)
    dx, dy = splev(s, tck, der=1)
    ddx, ddy = splev(s, tck, der=2)
    curvature = (dx * ddy - dy * ddx) / (dx * dx + dy * dy) ** 1.5

    # Elevation: nearest averaged point for each s, then smoothed; gradient = dz/ds
    line_s = cKDTree(np.array(splev(np.arange(0, length, 0.5), tck)).T).query(line[:, :2])[1] * 0.5
    order = np.argsort(line_s)
    z = np.interp(s, line_s[order], line[order, 2], period=length)
    z = circular_mean(z, ELEV_SMOOTH_M / DS)
    gradient = (np.roll(z, -1) - np.roll(z, 1)) / (2 * DS)

    # 4. Smooth curvature (sign kept: + left, - right; the sim uses |curvature|)
    curvature = circular_mean(curvature, CURV_SMOOTH_M / DS)

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("ds,curvature,gradient\n")
        for k, g in zip(curvature, gradient):
            f.write(f"{DS},{k:.6f},{g:.5f}\n")

    # Sanity check: lateral g the pole lap would need on this line (F1 peaks ~5-6 g)
    tel = pole.get_telemetry()
    _, j = cKDTree(np.array(splev(s, tck)).T).query(tel[["X", "Y"]].to_numpy() / 10.0)
    lat_g = (tel["Speed"].to_numpy() / 3.6) ** 2 * np.abs(curvature[j]) / 9.81
    print(f"Length {length:.0f} m, min radius {1 / np.abs(curvature).max():.0f} m, "
          f"elevation range {z.max() - z.min():.1f} m, max gradient {100 * np.abs(gradient).max():.1f}%")
    print(f"Pole lap ({pole['Driver']}) lateral g: median {np.median(lat_g):.2f}, "
          f"95th pct {np.percentile(lat_g, 95):.2f}, max {lat_g.max():.2f}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
