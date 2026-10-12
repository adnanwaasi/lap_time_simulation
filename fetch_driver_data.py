"""Export two teammates' fastest laps from real F1 telemetry and compare them.

Usage: python fetch_driver_data.py [year] [event] [session] [driver_a] [driver_b]
       python fetch_driver_data.py 2025 Silverstone Q NOR PIA

Writes data/<year>_<event>_<session>_<driver>.csv with distance (m), speed (m/s),
throttle (%), brake (0/1), gear and X/Y position, then prints an estimate of how much
of the car's grip each driver used relative to their teammate.
Teammates share a car, so most of the gap between them is the driver.
"""

import sys
from pathlib import Path

import fastf1
import numpy as np

CACHE = Path("data/fastf1_cache")
OUT = Path("data")


def fastest_lap_telemetry(session, driver):
    lap = session.laps.pick_drivers(driver).pick_fastest()
    tel = lap.get_telemetry()
    tel = tel[["Distance", "Speed", "Throttle", "Brake", "nGear", "X", "Y"]].copy()
    tel["Speed"] = tel["Speed"] / 3.6  # km/h -> m/s, same units as main.cpp
    tel["Brake"] = tel["Brake"].astype(int)
    return lap["LapTime"].total_seconds(), tel


def corner_min_speeds(tel, min_gap_m=150.0):
    """Minimum speed at each braking zone: the lowest speed before the throttle comes back."""
    d, v = tel["Distance"].to_numpy(), tel["Speed"].to_numpy()
    mins, last_d = [], -np.inf
    for i in range(1, len(v) - 1):
        if v[i] <= v[i - 1] and v[i] < v[i + 1] and d[i] - last_d > min_gap_m:
            mins.append((d[i], v[i]))
            last_d = d[i]
    return mins


def main():
    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
    event = sys.argv[2] if len(sys.argv) > 2 else "Silverstone"
    kind = sys.argv[3] if len(sys.argv) > 3 else "Q"
    a = sys.argv[4] if len(sys.argv) > 4 else "NOR"
    b = sys.argv[5] if len(sys.argv) > 5 else "PIA"

    CACHE.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE))
    session = fastf1.get_session(year, event, kind)
    session.load(laps=True, telemetry=True, weather=False, messages=False)

    laps = {}
    for drv in (a, b):
        lap_time, tel = fastest_lap_telemetry(session, drv)
        path = OUT / f"{year}_{event.replace(' ', '_')}_{kind}_{drv}.csv"
        tel.to_csv(path, index=False)
        laps[drv] = (lap_time, tel)
        print(f"{drv}: {lap_time:.3f}s, top {tel['Speed'].max() * 3.6:.0f} km/h -> {path}")

    # Match each of A's corners to B's slowest point within +-50 m.
    # Low-speed cornering: v^2 ~ mu*g*r, so (v_a / v_b)^2 ~ grip used by A relative to B.
    # (Downforce breaks this in fast corners, so only corners under 200 km/h are used.)
    tel_b = laps[b][1]
    print(f"\nCorner minimum speeds (km/h), {a} vs {b}:")
    ratios = []
    for dist, v_a in corner_min_speeds(laps[a][1]):
        near = tel_b[(tel_b["Distance"] - dist).abs() < 50]
        if near.empty or v_a * 3.6 > 200:
            continue
        v_b = near["Speed"].min()
        ratios.append((v_a / v_b) ** 2)
        print(f"  {dist:6.0f} m  {v_a * 3.6:5.1f}  {v_b * 3.6:5.1f}")

    if ratios:
        r = float(np.median(ratios))
        print(f"\n{a} grip use relative to {b}: {r:.3f} (median of {len(ratios)} corners)")


if __name__ == "__main__":
    main()
