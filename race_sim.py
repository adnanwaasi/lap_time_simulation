"""Monte Carlo race prediction: simulate the whole field many times and count the outcomes.

    .venv/bin/python race_sim.py --race data/race_united_states_grand_prix.json \\
                                 --form data/form_2026.json [--sims 2000] [--grid VER,LEC,...]

Inputs (from race_data.py):
  race JSON  the circuit: laps, pit loss, track temperature, fitted tyres, safety-car rates
  form JSON  the season: each driver's race/quali pace, each car's speed profile, DNF rate

Each simulated race:
  1. Pace. Every driver's pace is their recent form, nudged by how their car should suit this
     track: each car's grip, downforce and drag are fitted to its telemetry (slow-corner,
     fast-corner and top speed vs the field), and the physics simulates a lap of this circuit.
  2. Grid from qualifying pace (+ noise), unless the real grid is passed with --grid.
  3. Lap by lap: tyre wear, fuel burn, lap-time noise; a car that catches the one ahead must be
     clearly faster to get past, otherwise it is held up (dirty air).
  4. Strategy: every car starts on a plan from the strategy agent (teams pick among the
     near-optimal ones). When a safety car or VSC comes out, the agent re-plans from that car's
     current tyre and decides whether to take the cheap stop.
  5. Random safety cars (at this circuit's historical rate) and retirements (season DNF rate).
"""

import argparse
import json
from collections import Counter
from dataclasses import replace

import numpy as np

import strategy as st

POINTS = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]

# Rough 2026-regulation car: lighter, ~30% less downforce and markedly less drag than 2025.
# Only differences between teams are used, so these set the scale, not the result.
REF_CAR = st.Vehicle(1.60, 770.0, 700e3, 3.6, 1.10, 0.55, 0.012)
REF_DRIVER = st.Driver("REF", 1.0, 1.0)


# --- car data -> track-specific pace ------------------------------------------------------------


def team_car(m, fast_kmh=230.0):
    """Fit a team's car to its measured speed ratios vs the field median.

    - Slow corners (< 150 km/h) are limited by tyre grip: v^2 = mu g r, so mu ~ (speed ratio)^2.
    - Fast corners add downforce: v^2 (kappa - mu k) = mu g. Using a typical fast corner
      (the reference car takes it at fast_kmh), solve that equation for the team's k = ClA term.
    - At top speed, power = drag * v = (1/2 rho CdA v^2) v, so CdA ~ 1 / (top speed ratio)^3,
      assuming every power unit makes about the same power.
    """
    mu = REF_CAR.mu * m["slow"] ** 2
    k_ref = 0.5 * st.RHO * REF_CAR.cl_a / REF_CAR.mass
    v_ref = fast_kmh / 3.6
    kappa = REF_CAR.mu * st.G / v_ref ** 2 + REF_CAR.mu * k_ref
    v = v_ref * m["fast"]
    k = max(0.0, (kappa - mu * st.G / v ** 2) / mu)
    return replace(REF_CAR, mu=mu, cl_a=k * 2 * REF_CAR.mass / st.RHO, cd_a=REF_CAR.cd_a / m["top"] ** 3)


def car_track_pace(track, cars):
    """Each team's simulated lap here, as a fraction vs the median team."""
    times = {t: st.simulate_lap(track, team_car(m), REF_DRIVER) for t, m in cars.items()}
    med = np.median(list(times.values()))
    return {t: v / med - 1.0 for t, v in times.items()}


def driver_paces(form, car_pace, weight):
    """Race and quali pace (fraction vs field) per driver.

    Form already contains the car, averaged over recent tracks. The car-physics estimate is
    blended in at the team level: team pace = (1-w) * form team average + w * car sim here.
    Each driver keeps their gap to their team-mate from form.
    """
    drivers = form["drivers"]
    teams = {}
    for d, v in drivers.items():
        teams.setdefault(v["team"], []).append(v["race_pace"])
    out = {}
    for d, v in drivers.items():
        team_form = float(np.mean(teams[v["team"]]))
        shift = weight * (car_pace.get(v["team"], team_form) - team_form) if car_pace else 0.0
        out[d] = (v["race_pace"] + shift, v["quali_pace"] + shift)
    return out


# --- the race ------------------------------------------------------------------------------------


class RaceSim:
    def __init__(self, agent, race, form, paces, rng, pass_margin=0.5, noise=0.3):
        self.agent, self.race, self.rng = agent, race, rng
        self.N = agent.cfg.laps
        self.start_plans = self._start_plans()   # also solves the DP, which fills agent.lt
        self.ref_lap = float(agent.lt[1, 1, 1])   # medium, early race: scale for % pace
        self.paces = paces
        self.dnf_rate = form["dnf_rate"]
        self.pass_margin = pass_margin  # pace advantage (s/lap) needed for a 50% pass chance
        self.noise = noise              # lap-to-lap scatter (s)
        self.plan_cache = {}
        self._solved_for = None

    def _start_plans(self):
        """Near-optimal opening strategies (every start tyre x number of stops), with a weight
        that favours the faster ones: real teams split strategies when they are close.
        Planned before the race, so no safety car is assumed."""
        for c in self.agent.cond:
            c.safety_car = False
        self.agent.solve()
        self._solved_for = None
        plans = []
        for c in st.SLICKS:
            self.agent.state = st.RaceState(0, c, 0, st.mask_bit(c))
            for k in range(1, self.agent.cfg.max_stops + 1):
                p = self.agent.best_plan(k)
                if p is not None:
                    plans.append(p)
        best = min(p.time for p in plans)
        plans = [p for p in plans if p.time - best < 8.0]
        w = np.array([np.exp(-(p.time - best) / 2.0) for p in plans])
        return plans, w / w.sum()

    def replan(self, lap, car):
        """Pit wall call under a safety car/VSC on lap index `lap`: re-plan from this car's state
        with a cheap stop available at the end of this lap. Results are cached, and the DP is
        only re-solved when the neutralised lap changes."""
        key = (lap, car["tyre"], min(car["age"], self.N), car["used"])
        if key not in self.plan_cache:
            if self._solved_for != lap:
                for c in self.agent.cond:
                    c.safety_car = False
                self.agent.cond[lap].safety_car = True
                self.agent.solve()
                self._solved_for = lap
            self.agent.state = st.RaceState(lap, car["tyre"], min(car["age"], self.N), car["used"])
            plans = [p for p in (self.agent.best_plan(k) for k in range(self.agent.cfg.max_stops + 1)) if p]
            self.plan_cache[key] = min(plans, key=lambda p: p.time) if plans else None
        return self.plan_cache[key]

    def neutralisations(self):
        """Per-lap flag: None, 'SC' or 'VSC'. Counts drawn from this circuit's history."""
        flags = [None] * self.N
        for kind, rate, length in (("SC", self.race["sc_per_race"], (3, 5)),
                                   ("VSC", self.race["vsc_per_race"], (1, 2))):
            for _ in range(self.rng.poisson(rate)):
                start = self.rng.integers(1, self.N - 3)
                for L in range(start, min(self.N - 1, start + self.rng.integers(length[0], length[1] + 1))):
                    flags[L] = flags[L] or kind
        return flags

    def run(self, grid=None):
        rng, N, lt = self.rng, self.N, self.agent.lt
        plans, plan_w = self.start_plans
        drivers = list(self.paces)

        # Race-day form and tyre wear vary a little from the averages
        day = {d: self.paces[d][0] + rng.normal(0, 0.0015) for d in drivers}
        if grid is None:
            grid = sorted(drivers, key=lambda d: self.paces[d][1] + rng.normal(0, 0.0025))
        cars = {}
        for slot, d in enumerate(grid):
            plan = plans[rng.choice(len(plans), p=plan_w)]
            cars[d] = {"tyre": plan.stints[0].tyre, "age": 0, "used": st.mask_bit(plan.stints[0].tyre),
                       "stints": list(plan.stints[1:]), "pit_next": None,
                       "time": 0.3 * slot + abs(rng.normal(0, 0.4)) * (slot > 0),  # grid spacing + launch
                       "delta": day[d] * self.ref_lap, "wear": rng.normal(1.0, 0.08),
                       "out_lap": rng.integers(1, N) if rng.random() < self.dnf_rate else None,
                       "laps": 0, "strategy": st.COMPOUNDS[plan.stints[0].tyre].code + "1"}

        flags = self.neutralisations()
        for L in range(N):
            order = sorted((d for d in cars if cars[d]["out_lap"] is None or cars[d]["out_lap"] > L),
                           key=lambda d: cars[d]["time"])
            flag = flags[L]
            if flag:  # every pit wall re-plans: is this the moment for the cheap stop?
                for d in order:
                    c = cars[d]
                    if c["stints"] and c["stints"][0].first == L + 1:
                        continue  # already boxing on this lap: don't re-decide and postpone it
                    p = self.replan(L, c)
                    if p is not None:
                        c["stints"] = list(p.stints[1:])

            free = {}
            for d in order:
                c = cars[d]
                pitting = bool(c["stints"]) and c["stints"][0].first == L + 1
                if pitting:  # this lap is run on the new set (same convention as the agent's DP)
                    nxt = c["stints"].pop(0)
                    under = flags[L - 1] if L > 0 and flags[L - 1] else ""
                    c["strategy"] += f" {st.COMPOUNDS[nxt.tyre].code}{L + 1}{'(' + under + ')' if under else ''}"
                    c["tyre"], c["age"] = nxt.tyre, 0
                    c["used"] |= st.mask_bit(nxt.tyre)
                age = min(N, int(round(c["age"] * c["wear"])))
                t = lt[L, c["tyre"], age] + c["delta"] + rng.normal(0, self.noise)
                if flag == "SC":
                    t = max(t, 1.4 * self.ref_lap)
                elif flag == "VSC":
                    t = max(t, 1.3 * self.ref_lap)
                if pitting:  # the stop happens at the end of the previous lap (the in-lap)
                    in_flag = flags[L - 1] if L > 0 else None
                    t += self.agent.cfg.pit_loss * (0.5 if in_flag == "SC" else 0.6 if in_flag == "VSC" else 1.0)
                free[d] = (t, pitting)

            # Resolve the road order front to back. A car that would catch the one ahead only
            # gets past if it is clearly faster this lap; otherwise it's stuck in dirty air.
            prev_end = None
            prev_d = None
            for d in order:
                c = cars[d]
                t, pitting = free[d]
                end = c["time"] + t
                if prev_end is not None and not flag and not pitting and not free[prev_d][1] \
                        and end < prev_end + 0.3:
                    advantage = free[prev_d][0] - t
                    p_pass = 1.0 / (1.0 + np.exp(-(advantage - self.pass_margin) / 0.2))
                    if rng.random() < p_pass:
                        cars[prev_d]["time"] += 0.2  # the defender loses a bit fighting
                        end = min(end, cars[prev_d]["time"] - 0.1)
                    else:
                        end = prev_end + 0.3 + 0.1   # held up: sits in dirty air behind
                c["time"] = end
                c["age"] += 1
                c["laps"] = L + 1
                prev_end, prev_d = cars[d]["time"], d

            if flag == "SC" and (L + 1 >= N or flags[L + 1] != "SC"):  # restart: field bunched up
                order = sorted(order, key=lambda d: cars[d]["time"])
                lead = cars[order[0]]["time"]
                for i, d in enumerate(order):
                    cars[d]["time"] = lead + 0.8 * i

        running = sorted((d for d in cars if cars[d]["out_lap"] is None), key=lambda d: cars[d]["time"])
        out = sorted((d for d in cars if cars[d]["out_lap"] is not None), key=lambda d: -cars[d]["out_lap"])
        return running + out, {d: cars[d]["out_lap"] is not None for d in cars}, \
            {d: cars[d]["strategy"] for d in cars}, grid, flags


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--race", default="data/race_united_states_grand_prix.json")
    ap.add_argument("--form", default="data/form_2026.json")
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--grid", help="real starting grid, comma separated, e.g. VER,LEC,NOR,...")
    ap.add_argument("--car-weight", type=float, default=0.5,
                    help="0 = recent form only, 1 = car physics only, for team pace")
    ap.add_argument("--temp", type=float, help="race-day track temperature (C)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    race = json.loads(open(args.race).read())
    form = json.loads(open(args.form).read())
    if args.temp is not None:
        race["track_temp"] = args.temp
    rng = np.random.default_rng(args.seed)

    track = st.load_track(race["track"])
    print(f"{race['event']} prediction: {race['laps']} laps, track {race['track_temp']:.0f} C, "
          f"pit loss {race['pit_loss']:.1f} s, {race['sc_per_race']:.2f} SC + "
          f"{race['vsc_per_race']:.2f} VSC per race, DNF rate {form['dnf_rate']:.0%}")

    car_pace = car_track_pace(track, form["cars"]) if form.get("cars") else {}
    print("\nCar physics at this track (fitted from 2026 telemetry; - = faster):")
    for t, v in sorted(car_pace.items(), key=lambda kv: kv[1]):
        m = form["cars"][t]
        print(f"  {t:<18} {100 * v:+6.2f}%   slow {m['slow']:.3f}  fast {m['fast']:.3f}  top {m['top']:.3f}")
    paces = driver_paces(form, car_pace, args.car_weight)

    pace = st.PaceModel(track, REF_CAR, REF_DRIVER)
    cfg = st.RaceConfig(laps=race["laps"], pit_loss=race["pit_loss"])
    agent = st.StrategyAgent(pace, cfg, st.compounds_from_fit(pace, race))
    for c in agent.cond:
        c.track_temp = race["track_temp"]
    sim = RaceSim(agent, race, form, paces, rng)

    plans, w = sim.start_plans
    print("\nStrategy agent's opening plans (teams split between them by this weight):")
    best = min(p.time for p in plans)
    for p, wt in sorted(zip(plans, w), key=lambda x: x[0].time):
        print(f"  {p.describe():<28} {p.time - best:+5.1f} s   {wt:4.0%}")

    grid = args.grid.upper().split(",") if args.grid else None
    if grid:
        missing = set(grid) - set(paces)
        if missing:
            raise SystemExit(f"Unknown drivers in --grid: {', '.join(sorted(missing))}")
        grid += [d for d in paces if d not in grid]

    stats = {d: Counter() for d in paces}
    win_strats = Counter()
    for _ in range(args.sims):
        result, dnf, strats, g, _flags = sim.run(grid)
        for pos, d in enumerate(result):
            s = stats[d]
            s["pos_sum"] += pos + 1
            s["grid_sum"] += g.index(d) + 1
            s["win"] += pos == 0
            s["podium"] += pos < 3
            s["points_finish"] += pos < 10 and not dnf[d]
            s["points"] += POINTS[pos] if pos < 10 and not dnf[d] else 0
            s["dnf"] += dnf[d]
        win_strats[strats[result[0]]] += 1

    n = args.sims
    rows = sorted(paces, key=lambda d: -stats[d]["points"])
    print(f"\nPredicted result over {n} simulated races:")
    print(f"{'':3} {'driver':<6} {'team':<17} {'grid':>5} {'win':>6} {'podium':>7} "
          f"{'points':>7} {'avg pos':>8} {'exp pts':>8}")
    for i, d in enumerate(rows, 1):
        s = stats[d]
        print(f"{i:>2}. {d:<6} {form['drivers'][d]['team'][:17]:<17} {s['grid_sum'] / n:5.1f} "
              f"{s['win'] / n:6.1%} {s['podium'] / n:7.1%} {s['points_finish'] / n:7.1%} "
              f"{s['pos_sum'] / n:8.1f} {s['points'] / n:8.1f}")
    print("\nWinners' tyre sequences (compound + first lap on it; (SC)/(VSC) = stop under neutralisation):")
    for strat, k in win_strats.most_common(8):
        print(f"  {strat:<28} {k / n:5.1%}")


if __name__ == "__main__":
    main()
