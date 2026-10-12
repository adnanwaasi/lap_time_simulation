"""Compare the simulator's speed trace with a driver's real fastest lap.

Usage: ./laptimesim tracks/silverstone_real.csv
       python validate.py [year] [event] [session] [driver]
       python validate.py 2025 Silverstone Q VER

Reads data/sim_trace.csv (written by main.cpp), prints lap time and speed errors,
and saves a speed-vs-distance plot to data/validation_<driver>.png.
"""

import logging
import sys
from pathlib import Path

import fastf1
import matplotlib

matplotlib.use("Agg")  # write files, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
    event = sys.argv[2] if len(sys.argv) > 2 else "Silverstone"
    kind = sys.argv[3] if len(sys.argv) > 3 else "Q"
    driver = sys.argv[4] if len(sys.argv) > 4 else "VER"

    sim = pd.read_csv("data/sim_trace.csv")
    if driver not in sim.columns:
        sys.exit(f"{driver} not in data/sim_trace.csv (columns: {', '.join(sim.columns[1:])})")

    logging.getLogger("fastf1").setLevel(logging.WARNING)
    Path("data/fastf1_cache").mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache("data/fastf1_cache")
    session = fastf1.get_session(year, event, kind)
    session.load(weather=False, messages=False)
    lap = session.laps.pick_drivers(driver).pick_fastest()
    tel = lap.get_car_data().add_distance()

    # FastF1's distance comes from integrating speed, so it doesn't exactly match the
    # geometric length of our racing line; stretch it to the same lap length.
    sim_len = sim["distance"].iloc[-1] + (sim["distance"].iloc[1] - sim["distance"].iloc[0])
    real_d = tel["Distance"].to_numpy() * sim_len / tel["Distance"].iloc[-1]
    real_v = tel["Speed"].to_numpy()  # km/h
    sim_v = np.interp(real_d, sim["distance"], sim[driver] * 3.6)

    # Lap time from the sim trace: t = sum ds / v, same as main.cpp
    v = sim[driver].to_numpy()
    ds = np.diff(np.r_[sim["distance"].to_numpy(), sim_len])
    sim_time = np.sum(2 * ds / (v + np.roll(v, -1)))
    real_time = lap["LapTime"].total_seconds()

    err = sim_v - real_v
    print(f"{driver} {year} {event} {kind}")
    print(f"  lap time   real {real_time:.3f}s   sim {sim_time:.3f}s   "
          f"({100 * (sim_time - real_time) / real_time:+.1f}%)")
    print(f"  top speed  real {real_v.max():.0f}   sim {sim_v.max():.0f} km/h")
    print(f"  speed error: mean {err.mean():+.1f} km/h, RMS {np.sqrt(np.mean(err ** 2)):.1f} km/h")

    # Corner minimum speeds: real local minima under 250 km/h vs the sim at the same place
    print("  corner minimum speeds (km/h):")
    last = -1e9
    for i in range(1, len(real_v) - 1):
        if real_v[i] <= real_v[i - 1] and real_v[i] < real_v[i + 1] and real_v[i] < 250 \
                and real_d[i] - last > 150:
            window = np.abs(real_d - real_d[i]) < 60
            print(f"    {real_d[i]:6.0f} m  real {real_v[i]:5.0f}  sim {sim_v[window].min():5.0f}")
            last = real_d[i]

    fig, (ax, ax_err) = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1]})
    ax.plot(real_d, real_v, label=f"{driver} real ({real_time:.3f}s)")
    ax.plot(sim["distance"], sim[driver] * 3.6, label=f"sim ({sim_time:.3f}s)")
    ax.set_ylabel("speed (km/h)")
    ax.legend()
    ax_err.plot(real_d, err, color="gray")
    ax_err.axhline(0, color="black", linewidth=0.5)
    ax_err.set_ylabel("sim - real")
    ax_err.set_xlabel("distance (m)")
    fig.tight_layout()
    out = Path(f"data/validation_{driver}.png")
    fig.savefig(out, dpi=110)
    print(f"  plot: {out}")


if __name__ == "__main__":
    main()
