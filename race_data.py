"""Turn real F1 data (FastF1) into parameters for the strategy agent and race simulator.

    python race_data.py track 2025 "United States Grand Prix" --track tracks/cota_real.csv
        -> data/race_<event>.json: laps, pit loss, track temperature, per-compound pace and
           wear (fitted from the race laps), and safety-car rates from that event's history.

    python race_data.py form 2026 --last 6
        -> data/form_<year>.json: each driver's race pace and qualifying pace relative to the
           field over the last N races, each team's car characteristics measured from
           qualifying telemetry, and the season's DNF rate.

Needs: pip install fastf1 scipy  (a .venv with them: .venv/bin/python race_data.py ...)
"""

import argparse
import json
import logging
from pathlib import Path

import fastf1
import numpy as np

import strategy

CACHE = Path("data/fastf1_cache")
FUEL_S_PER_KG = 0.03  # typical F1 rule of thumb
SLICKS = {"SOFT": "S", "MEDIUM": "M", "HARD": "H"}


def load(year, event, kind, telemetry=False):
    CACHE.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE))
    logging.getLogger("fastf1").setLevel(logging.ERROR)
    session = fastf1.get_session(year, event, kind)
    session.load(laps=True, telemetry=telemetry, weather=True, messages=False)
    return session


def clean_laps(session):
    """Green-flag racing laps on slicks: no pit in/out laps, no lap 1, no obvious traffic or
    mistakes (slower than 103% of that driver's median)."""
    laps = session.laps
    laps = laps[(laps["TrackStatus"] == "1") & laps["IsAccurate"].astype(bool)
                & laps["PitInTime"].isna() & laps["PitOutTime"].isna()
                & (laps["LapNumber"] > 1) & laps["Compound"].isin(list(SLICKS))]
    laps = laps.dropna(subset=["LapTime", "TyreLife"]).copy()
    laps["t"] = laps["LapTime"].dt.total_seconds()
    med = laps.groupby("Driver")["t"].transform("median")
    return laps[laps["t"] < 1.03 * med]


def fit_race(laps, fuel_per_lap):
    """Least-squares fit of every clean lap:

        t = driver_d + offset_compound + deg_compound * tyre_age + fuel_per_lap * (laps left)

    Fuel is fixed from the physics (lighter car -> faster), not fitted, because within a stint
    "more laps done" means both less fuel and older tyres: the data alone can't tell them apart
    (the two columns are nearly collinear). Fixing one makes the other identifiable.
    Driver terms soak up car + driver pace, so compound offsets and wear are measured within
    each car, not by comparing a fast car on softs with a slow car on hards.
    """
    drivers = sorted(laps["Driver"].unique())
    comps = [c for c in ("S", "M", "H") if (laps["Compound"].map(SLICKS) == c).sum() >= 15]
    t = laps["t"].to_numpy() + fuel_per_lap * laps["LapNumber"].to_numpy()  # remove fuel burn-off
    code = laps["Compound"].map(SLICKS).to_numpy()
    age = laps["TyreLife"].to_numpy()
    drv = laps["Driver"].to_numpy()

    cols = [(drv == d).astype(float) for d in drivers]                    # one intercept per driver
    cols += [(code == c).astype(float) for c in comps if c != "M"]        # offsets vs medium
    cols += [(code == c) * age for c in comps]                            # wear per compound
    X = np.column_stack(cols)
    keep = np.isin(code, comps)
    for _ in range(2):  # refit once without >3 sigma outliers
        beta, *_ = np.linalg.lstsq(X[keep], t[keep], rcond=None)
        res = t - X @ beta
        keep = np.isin(code, comps) & (np.abs(res) < 3 * res[keep].std())

    nd = len(drivers)
    offsets = dict(zip([c for c in comps if c != "M"], beta[nd:nd + len(comps) - 1]))
    offsets["M"] = 0.0
    deg = dict(zip(comps, beta[nd + len(comps) - 1:]))
    driver_t = dict(zip(drivers, beta[:nd]))
    return {
        # Oldest tyres seen still lapping cleanly (95th percentile): no cliff before that age
        "compounds": {c: {"offset": float(offsets[c]), "deg": float(deg[c]),
                          "laps": int((code == c).sum()),
                          "max_age": int(np.percentile(age[code == c], 95))} for c in comps},
        "drivers": {d: float(v) for d, v in driver_t.items()},
        "residual_s": float(res[keep].std()),
        "n_laps": int(keep.sum()),
    }


def pit_loss(session):
    """Median time a green-flag stop costs: (in-lap + out-lap) minus two normal laps."""
    laps = session.laps
    ref = clean_laps(session).groupby("Driver")["t"].median()
    losses = []
    for _, lap in laps[laps["PitInTime"].notna()].iterrows():
        out = laps[(laps["Driver"] == lap["Driver"]) & (laps["LapNumber"] == lap["LapNumber"] + 1)]
        if out.empty or lap["Driver"] not in ref or lap["TrackStatus"] != "1" \
                or out.iloc[0]["TrackStatus"] != "1" or pd_isna(lap["LapTime"]) \
                or pd_isna(out.iloc[0]["LapTime"]):
            continue
        t = lap["LapTime"].total_seconds() + out.iloc[0]["LapTime"].total_seconds()
        losses.append(t - 2 * ref[lap["Driver"]])
    return float(np.median(losses)) if losses else None


def pd_isna(x):
    return x is None or (isinstance(x, float) and np.isnan(x)) or str(x) == "NaT"


def neutralisations(session):
    """Number of safety car ('4') and virtual safety car ('6') deployments in a race."""
    status = session.track_status["Status"].astype(str).to_numpy()
    starts = np.r_[True, status[1:] != status[:-1]]
    return int(((status == "4") & starts).sum()), int(((status == "6") & starts).sum())


def cmd_track(args):
    track = strategy.load_track(args.track)
    car, driver = strategy.TEAMS["Red Bull"][0], strategy.TEAMS["Red Bull"][1][0]
    pace = strategy.PaceModel(track, car, driver)

    race = load(args.year, args.event, "R")
    n_laps = int(race.laps["LapNumber"].max())
    fuel_per_lap = pace.fuel_sens * strategy.RaceConfig.fuel_start / n_laps
    fit = fit_race(clean_laps(race), fuel_per_lap)
    loss = pit_loss(race)
    temp = float(race.weather_data["TrackTemp"].mean())
    print(f"{args.year} {race.event.EventName}: {n_laps} laps, track {temp:.0f} C, "
          f"pit loss {loss:.1f} s, {fit['n_laps']} clean laps, fit residual {fit['residual_s']:.2f} s")
    for c, v in fit["compounds"].items():
        print(f"  {c}: {v['offset']:+.2f} s vs medium, wear {v['deg']:.3f} s/lap ({v['laps']} laps)")

    # Safety-car history: every edition of this race in the last few years
    sc, vsc, n = 0, 0, 0
    for y in range(args.year - args.history + 1, args.year + 1):
        try:
            s, v = neutralisations(race if y == args.year else load(y, args.event, "R"))
        except Exception as e:  # race not held / no data that year
            print(f"  {y}: no data ({type(e).__name__})")
            continue
        sc, vsc, n = sc + s, vsc + v, n + 1
        print(f"  {y}: {s} safety car(s), {v} VSC(s)")

    out = {
        "event": race.event.EventName, "year": args.year, "track": args.track,
        "laps": n_laps, "track_temp": temp, "pit_loss": loss, "fuel_per_lap_s": fuel_per_lap,
        "compounds": fit["compounds"],
        "sc_per_race": sc / max(n, 1), "vsc_per_race": vsc / max(n, 1), "history_races": n,
    }
    path = Path("data") / f"race_{race.event.EventName.lower().replace(' ', '_')}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {path}")


# --- car characteristics from qualifying telemetry ----------------------------------------------


def corner_minima(tel, min_gap_m=150.0):
    """(distance, speed km/h) at each local speed minimum: one per corner."""
    d, v = tel["Distance"].to_numpy(), tel["Speed"].to_numpy()
    out, last = [], -np.inf
    for i in range(1, len(v) - 1):
        if v[i] <= v[i - 1] and v[i] < v[i + 1] and d[i] - last > min_gap_m:
            out.append((d[i], v[i]))
            last = d[i]
    return out


def car_metrics(quali):
    """Per team, relative to the field median at this track:
    slow-corner speed (< 150 km/h: mechanical grip), fast-corner speed (> 200 km/h: downforce),
    and top speed (power vs drag). Each team's best lap of the session is used."""
    laps = quali.laps.dropna(subset=["LapTime"])
    best = laps.loc[laps.groupby("Team")["LapTime"].idxmin()]
    pole = laps.loc[laps["LapTime"].idxmin()]
    corners = corner_minima(pole.get_car_data().add_distance())
    team_tel = {}
    for _, lap in best.iterrows():
        try:
            team_tel[lap["Team"]] = lap.get_car_data().add_distance()
        except Exception:
            continue

    def speed_near(tel, dist):
        near = tel[(tel["Distance"] - dist).abs() < 50]
        return near["Speed"].min() if not near.empty else np.nan

    table = {t: [speed_near(tel, d) for d, _ in corners] for t, tel in team_tel.items()}
    speeds = np.array(list(table.values()))
    field = np.nanmedian(speeds, axis=0)
    slow, fast = field < 150, field > 200
    top = {t: tel["Speed"].max() for t, tel in team_tel.items()}
    top_med = np.median(list(top.values()))
    out = {}
    for (team, row) in table.items():
        r = np.array(row) / field
        out[team] = {"slow": float(np.nanmedian(r[slow])) if slow.any() else 1.0,
                     "fast": float(np.nanmedian(r[fast])) if fast.any() else 1.0,
                     "top": float(top[team] / top_med),
                     "slow_kmh": float(np.nanmedian(field[slow])) if slow.any() else np.nan,
                     "fast_kmh": float(np.nanmedian(field[fast])) if fast.any() else np.nan}
    return out


def cmd_form(args):
    schedule = fastf1.get_event_schedule(args.year, include_testing=False)
    done = schedule[schedule["EventDate"] < np.datetime64("today")]
    rounds = list(done["RoundNumber"])[-args.last:]

    race_pace, quali_pace, cars, teams = {}, {}, {}, {}
    starts = dnfs = 0
    for w, rnd in enumerate(rounds, start=1):
        weight = w  # later races count more: form changes as teams upgrade
        try:
            race = load(args.year, rnd, "R")
            quali = load(args.year, rnd, "Q", telemetry=True)
        except Exception as e:
            print(f"Round {rnd}: skipped ({type(e).__name__})")
            continue
        name = race.event.EventName
        laps = clean_laps(race)
        if laps.empty:
            print(f"Round {rnd} {name}: no clean slick laps, skipped")
            continue
        n_laps = int(race.laps["LapNumber"].max())
        # Generic fuel effect (no track model for every venue); it only shifts every driver's
        # intercept by the same amount, so the relative pace barely depends on it.
        fit = fit_race(laps, FUEL_S_PER_KG * strategy.RaceConfig.fuel_start / n_laps)
        med = np.median(list(fit["drivers"].values()))
        ref_lap = float(laps["t"].median())
        for d, v in fit["drivers"].items():
            race_pace.setdefault(d, []).append(((v - med) / ref_lap, weight))

        qbest = quali.laps.dropna(subset=["LapTime"]).groupby("Driver")["LapTime"].min().dt.total_seconds()
        for d, v in qbest.items():
            if v < 1.07 * qbest.min():  # ignore crashes / no-time sessions
                quali_pace.setdefault(d, []).append((v / qbest.min() - 1.0, weight))

        entry = list(race.results["Abbreviation"])  # the latest race's line-up is the next race's
        for _, r in race.results.iterrows():
            teams[r["Abbreviation"]] = r["TeamName"]
            starts += 1
            dnfs += str(r["Status"]) not in ("Finished", "Lapped") and not str(r["Status"]).startswith("+")
        try:
            for team, m in car_metrics(quali).items():
                cars.setdefault(team, []).append((m, weight))
        except Exception as e:
            print(f"  car metrics failed: {type(e).__name__}: {e}")
        print(f"Round {rnd} {name}: {fit['n_laps']} clean laps, residual {fit['residual_s']:.2f} s")

    def wmedian(pairs):
        """Weighted median: one race ruined by damage or a penalty barely moves it, where it
        would drag a mean a long way."""
        pairs = sorted(pairs)
        ws = np.cumsum([w for _, w in pairs], dtype=float)
        return float(pairs[int(np.searchsorted(ws, ws[-1] / 2))][0])

    drivers = {}
    for d in entry:
        team = teams[d]
        if d not in race_pace:
            continue
        drivers[d] = {"team": team, "race_pace": wmedian(race_pace[d]),
                      "quali_pace": wmedian(quali_pace[d]) if d in quali_pace else 0.02,
                      "races": len(race_pace[d])}
    car_out = {t: {k: wmedian([(m[k], w) for m, w in v]) for k in ("slow", "fast", "top")}
               for t, v in cars.items()}
    out = {"year": args.year, "rounds": [int(r) for r in rounds], "dnf_rate": dnfs / max(starts, 1),
           "drivers": drivers, "cars": car_out}
    path = Path("data") / f"form_{args.year}.json"
    path.write_text(json.dumps(out, indent=2))

    print(f"\nDNF rate {out['dnf_rate']:.1%} over {starts} starts")
    print(f"{'':4} {'team':<18} {'race':>7} {'quali':>7}")
    for d, v in sorted(drivers.items(), key=lambda kv: kv[1]["race_pace"]):
        print(f"{d:4} {v['team']:<18} {100 * v['race_pace']:+6.2f}% {100 * v['quali_pace']:6.2f}%")
    print(f"\n{'car':<18} {'slow':>6} {'fast':>6} {'top':>6}  (speed vs field median)")
    for t, m in sorted(car_out.items()):
        print(f"{t:<18} {m['slow']:6.3f} {m['fast']:6.3f} {m['top']:6.3f}")
    print(f"Wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("track")
    t.add_argument("year", type=int)
    t.add_argument("event")
    t.add_argument("--track", required=True, help="track CSV from build_track.py")
    t.add_argument("--history", type=int, default=4, help="past editions for safety-car rates")
    f = sub.add_parser("form")
    f.add_argument("year", type=int)
    f.add_argument("--last", type=int, default=6, help="number of most recent races")
    args = ap.parse_args()
    {"track": cmd_track, "form": cmd_form}[args.cmd](args)


if __name__ == "__main__":
    main()
