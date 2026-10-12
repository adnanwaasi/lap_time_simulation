"""Pit wall strategy agent.

Plans the tyre strategy that minimises total race time, then re-plans as the race unfolds:
you feed it lap times, pit stops, weather and safety cars, and it says when to box and for what.

    python3 strategy.py [--driver VER] [--track tracks/silverstone.csv] [--laps 52]

Type `help` at the prompt. Commands can also be piped in:

    printf 'rain 30 40 0.6\\nplan\\n' | python3 strategy.py

How it works
------------
1. Pace model: the physics from main.cpp (ported below) is run at many tyre grip levels to get
   lap time as a function of grip. Fuel is a separate linear term (seconds per kg).
2. Tyre model: each compound has a peak grip, a wear rate, a "cliff" beyond which wear
   accelerates, and an out-lap warm-up cost. Water, track temperature and rubbering-in modify it.
3. Optimiser: dynamic programming over (lap, compound, tyre age, compounds used, stops left).
   It is exact for the model and forecast it is given, so it never "misses" an undercut or a
   safety-car stop that the model can see.
4. Agent loop: every observed lap time is compared with the model; the tyre wear rate is
   re-fitted to the real laps and the plan is recomputed.
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass, field, replace

import numpy as np

G = 9.81     # Gravity (m/s^2)
RHO = 1.225  # Air density (kg/m^3)

# ---------------------------------------------------------------------------------------------
# Physics: a port of simulate_lap() from main.cpp. main.cpp stays the source of truth; if the
# physics there changes, mirror it here.
# ---------------------------------------------------------------------------------------------


@dataclass
class Vehicle:
    mu: float         # Tyre friction coefficient (fresh soft, dry track)
    mass: float       # Car + driver, no race fuel (kg)
    power: float      # Power at the wheels (W)
    cl_a: float       # Downforce coefficient * frontal area (m^2)
    cd_a: float       # Drag coefficient * frontal area (m^2)
    rear_load: float  # Fraction of tyre load on the driven wheels
    crr: float        # Rolling resistance coefficient


@dataclass
class Driver:
    name: str
    grip_use: float
    brake_use: float


TEAMS = {
    "Mercedes": (Vehicle(1.60, 800.0, 700e3, 4.7, 1.40, 0.55, 0.012),
                 [Driver("RUS", 0.99, 0.99), Driver("ANT", 0.97, 0.97)]),
    "Red Bull": (Vehicle(1.60, 800.0, 705e3, 4.8, 1.38, 0.55, 0.012),
                 [Driver("VER", 1.00, 1.00), Driver("TSU", 0.96, 0.97)]),
    "McLaren": (Vehicle(1.62, 800.0, 700e3, 5.0, 1.42, 0.55, 0.012),
                [Driver("NOR", 0.986, 1.00), Driver("PIA", 1.00, 1.00)]),
}


def load_track(path):
    """Rows of (ds, curvature, gradient); gradient is optional in the CSV."""
    track = []
    with open(path) as f:
        next(f)
        for line in f:
            cols = line.strip().split(",")
            if len(cols) < 2:
                continue
            ds, curv = float(cols[0]), float(cols[1])
            grad = float(cols[2]) if len(cols) > 2 else 0.0
            if ds > 0:
                track.append((ds, curv, grad))
    return track


def simulate_lap(track, car, driver):
    """Flying-lap time (s): forward pass for traction/power, backward pass for braking."""
    k = 0.5 * RHO * car.cl_a / car.mass
    mu = car.mu * driver.grip_use
    n = len(track)

    def normal(grad, v):
        return G / math.sqrt(1 + grad * grad) + k * v * v

    def resistance(grad, v):
        drag = 0.5 * RHO * car.cd_a * v * v / car.mass
        return drag + car.crr * normal(grad, v) + G * grad / math.sqrt(1 + grad * grad)

    def long_grip(curv, grad, v):
        total = mu * normal(grad, v)
        lat = v * v * abs(curv)
        return math.sqrt(max(0.0, total * total - lat * lat))

    def corner_limit(curv, grad):
        denom = abs(curv) - mu * k
        return math.inf if denom <= 0 else math.sqrt(mu * normal(grad, 0.0) / denom)

    limits = [corner_limit(c, g) for _, c, g in track]
    speeds = [0.0] * n
    v = 0.0
    for step in range(2 * n):
        i = step % n
        ds, curv, grad = track[i]
        v = min(v, limits[i])
        speeds[i] = v
        traction = car.rear_load * long_grip(curv, grad, v)
        power = car.power / (car.mass * v) if v > 0 else traction
        a = min(traction, power) - resistance(grad, v)
        v = math.sqrt(max(0.0, v * v + 2 * a * ds))

    for step in range(2 * n - 1, -1, -1):
        i = step % n
        ds, curv, grad = track[i]
        v_next = speeds[(i + 1) % n]
        decel = driver.brake_use * long_grip(curv, grad, v_next) + resistance(grad, v_next)
        speeds[i] = min(speeds[i], math.sqrt(max(0.0, v_next * v_next + 2 * decel * ds)))

    return sum(2 * track[i][0] / (speeds[i] + speeds[(i + 1) % n]) for i in range(n))


class PaceModel:
    """Lap time as a function of tyre grip and fuel load, built from the physics.

    Grip enters the physics non-linearly (corner speed ~ sqrt(mu)), and a wet track can cut
    grip by a third, so grip gets a lookup table. Fuel only changes mass by a few percent, so a
    straight-line seconds-per-kg slope is accurate enough and needs just two simulations.
    """

    def __init__(self, track, car, driver, mu_lo=0.4, mu_hi=2.0, step=0.02):
        self.mus = np.arange(mu_lo, mu_hi + 1e-9, step)
        self.times = np.array([simulate_lap(track, _with(car, mu=m), driver) for m in self.mus])
        self.base_mu = car.mu
        t0 = simulate_lap(track, car, driver)
        t1 = simulate_lap(track, _with(car, mass=car.mass + 10.0), driver)
        self.fuel_sens = (t1 - t0) / 10.0  # s per kg

    def lap_time(self, mu, fuel_kg):
        return np.interp(mu, self.mus, self.times) + self.fuel_sens * fuel_kg


def _with(car, **changes):
    return Vehicle(**{**car.__dict__, **changes})


# ---------------------------------------------------------------------------------------------
# Tyres and track conditions
# ---------------------------------------------------------------------------------------------


@dataclass
class Compound:
    code: str
    name: str
    grip: float        # Peak grip relative to a fresh soft on a dry track
    wear: float        # Fraction of grip lost per lap at 35 C track temperature
    cliff: int         # Tyre age (laps) after which wear accelerates
    cliff_wear: float  # Extra grip fraction lost per (lap past the cliff)^2
    warmup: float      # Seconds lost on the out-lap bringing the tyre into its window


# Tuned so that a fresh soft is ~0.7% of a lap faster than a medium, which is ~0.6% faster than
# a hard, and degradation is ~0.11 / 0.07 / 0.04 s per lap (on this track's ~70 s lap): roughly
# the Pirelli picture at Silverstone, scaled to the shorter simulated lap. Inters and wets have grip 1.0 here; wet_grip() sets their real level.
COMPOUNDS = [
    Compound("S", "Soft",   1.000, 0.0026, 20, 0.0004, 0.3),
    Compound("M", "Medium", 0.986, 0.0016, 32, 0.0003, 0.7),
    Compound("H", "Hard",   0.973, 0.0010, 45, 0.0002, 1.3),
    Compound("I", "Inter",  1.000, 0.0012, 30, 0.0003, 0.6),
    Compound("W", "Wet",    1.000, 0.0009, 40, 0.0002, 0.6),
]
CODES = {c.code: i for i, c in enumerate(COMPOUNDS)}
SLICKS = (0, 1, 2)
INTER, WET = 3, 4


def wet_grip(c, wetness):
    """Grip multiplier for compound index c at wetness 0 (dry) .. 1 (standing water).

    Slicks lose grip quickly once there is water. Inters work best on a damp track (~0.45).
    Full wets clear the most water but are slow on a drying track. The crossovers land near
    wetness 0.15 (slick -> inter) and 0.85 (inter -> wet).
    """
    if c == INTER:
        return 0.90 - 0.25 * abs(wetness - 0.45)
    if c == WET:
        return 0.67 + 0.15 * wetness
    return 1.0 - 0.65 * wetness


def wear_factor(c, wetness, track_temp):
    """How much faster than nominal a tyre wears in these conditions."""
    # Hotter track -> tyres overheat and wear faster; colder -> slower (graining not modelled).
    f = min(2.5, max(0.5, 1.0 + 0.04 * (track_temp - 35.0)))
    # Rain tyres on a drying track overheat their soft tread blocks and destroy themselves.
    if c == INTER:
        f *= 1.0 + 4.0 * max(0.0, 0.25 - wetness) / 0.25
    elif c == WET:
        f *= 1.0 + 4.0 * max(0.0, 0.6 - wetness) / 0.6
    return f


def compounds_from_fit(pace, race):
    """Compound table from a fitted race (race_data.py), which measured each slick's pace
    offset vs the medium (s) and its wear (s per lap of age) on real laps.

    The tyre model works in grip, so convert through the physics: `per_frac` is the lap time
    lost per unit fraction of grip at this track (e.g. 1% grip ~ 0.4 s). Wear is also divided
    by the temperature factor of the race it was measured at, so that a hotter or cooler race
    day scales it again through wear_factor().
    """
    mu = pace.base_mu
    per_frac = (np.interp(mu * 0.99, pace.mus, pace.times) - np.interp(mu, pace.mus, pace.times)) / 0.01
    temp = wear_factor(0, 0.0, race["track_temp"])
    fits = race["compounds"]
    fastest = min(f["offset"] for f in fits.values())
    table = []
    for comp in COMPOUNDS:
        f = fits.get(comp.code)
        if f is None:
            table.append(comp)
        else:
            table.append(replace(comp, grip=1.0 - (f["offset"] - fastest) / per_frac,
                                 wear=max(1e-5, f["deg"] / per_frac / temp),
                                 cliff=max(comp.cliff, f.get("max_age", 0))))
    return table


@dataclass
class LapConditions:
    wetness: float = 0.0     # 0 dry, 1 standing water
    track_temp: float = 40.0  # C
    safety_car: bool = False


@dataclass
class RaceConfig:
    laps: int = 52              # Silverstone race distance
    pit_loss: float = 20.0      # Time lost driving through the pit lane + stationary time (s)
    sc_pit_factor: float = 0.5  # Under a safety car the field is slow, so a stop costs about half
    fuel_start: float = 100.0   # kg
    track_evo: float = 0.01     # Grip gained as rubber is laid down (fraction, approached over ~10 laps)
    max_stops: int = 3


# ---------------------------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------------------------


@dataclass
class RaceState:
    lap: int = 0             # Laps completed
    tyre: int | None = None  # None before the start: the agent picks the start tyre
    age: int = 0             # Laps completed on the current set
    used: int = 0            # Bitmask of compound groups used (see mask_bit)


def mask_bit(c):
    """Rules only care which dry compounds were used and whether any rain tyre was, so inters
    and wets share a bit. 4 bits -> 16 masks instead of 32, which halves the DP."""
    return 1 << min(c, 3)


def rules_ok(mask):
    """F1 rule: in a dry race, use at least two different dry compounds. Using inters or wets
    lifts the requirement."""
    return bool(mask & 0b1000) or bin(mask & 0b0111).count("1") >= 2


@dataclass
class Stint:
    tyre: int
    first: int  # 1-based, inclusive
    last: int


@dataclass
class Plan:
    time: float
    stints: list = field(default_factory=list)

    def describe(self):
        return "  ".join(f"{COMPOUNDS[s.tyre].code}{s.first}-{s.last}" for s in self.stints)

    def stops(self):
        return len(self.stints) - 1


class StrategyAgent:
    def __init__(self, pace, cfg, compounds=None):
        self.pace = pace
        self.cfg = cfg
        self.compounds = compounds or COMPOUNDS
        self.cond = [LapConditions() for _ in range(cfg.laps)]  # index = lap number - 1
        self.state = RaceState()
        self.deg_scale = 1.0     # Fitted multiplier on every compound's wear rate
        self.pace_offset = 0.0   # Fitted constant: real car vs model (s/lap); doesn't change strategy
        self.observations = []   # (lap index, tyre, age, time) of clean racing laps

    # --- model -------------------------------------------------------------------------------

    def lap_times(self, lap, c, ages):
        """Predicted time of lap index `lap` (0-based) on compound c, for an array of tyre ages."""
        cond = self.cond[lap]
        comp = self.compounds[c]
        rate = comp.wear * self.deg_scale * wear_factor(c, cond.wetness, cond.track_temp)
        past_cliff = np.maximum(0, ages - comp.cliff)
        worn = rate * ages + comp.cliff_wear * past_cliff ** 2
        evo = 1.0 + self.cfg.track_evo * (1.0 - math.exp(-lap / 10.0))
        mu = (self.pace.base_mu * comp.grip * wet_grip(c, cond.wetness) * evo
              * np.maximum(0.3, 1.0 - worn))
        fuel = self.cfg.fuel_start * (1.0 - (lap + 0.5) / self.cfg.laps)
        t = self.pace.lap_time(mu, fuel)
        if lap > 0:  # the race start is warmed up by the formation lap
            t = t + np.where(ages == 0, comp.warmup, 0.0)
        return t

    def pit_loss(self, lap):
        """Cost of a stop at the end of lap index `lap`."""
        sc = self.cond[lap].safety_car
        return self.cfg.pit_loss * (self.cfg.sc_pit_factor if sc else 1.0)

    # --- optimiser ---------------------------------------------------------------------------

    def solve(self):
        """Backward dynamic programming.

        V[L, c, a, m, r] = least time to finish the race from the start of lap index L, on
        compound c aged a laps, having used compound mask m, with exactly r stops still to make.
        Each lap, either stay out (age + 1) or box at the end of the previous lap: pay the pit
        loss and run this lap on a fresh set. Working backwards from the flag means every
        state's best future is already known when we need it, so the whole strategy space
        (millions of combinations) is searched in one sweep.
        """
        N, C, R = self.cfg.laps, len(self.compounds), self.cfg.max_stops + 1
        A, M = N + 1, 16
        ages = np.arange(A)
        self.lt = np.array([[self.lap_times(L, c, ages) for c in range(C)] for L in range(N)])

        V = np.full((N + 1, C, A, M, R), np.inf)
        for m in range(M):
            if rules_ok(m):
                V[N, :, :, m, 0] = 0.0

        next_age = np.minimum(ages + 1, N)
        new_mask = np.array([[m | mask_bit(c) for c in range(C)] for m in range(M)])
        for L in range(N - 1, -1, -1):
            stay = self.lt[L][:, :, None, None] + V[L + 1][:, next_age]
            V[L] = stay
            if L == 0:
                continue  # no stop before the first lap
            pit = np.full((M, R), np.inf)
            for c2 in range(C):
                # A fresh set of c2 costs the same whatever we were on; only the mask changes.
                cand = self.pit_loss(L - 1) + self.lt[L, c2, 0] + V[L + 1, c2, 1, new_mask[:, c2], :-1]
                pit[:, 1:] = np.minimum(pit[:, 1:], cand)
            V[L] = np.minimum(stay, pit[None, None])
        self.V = V

    def best_plan(self, stops):
        """Best plan from the current state with exactly `stops` more stops (None if impossible).

        The current lap is already under way, so the first decision is forced to "stay out":
        the earliest possible stop is at the end of the current lap.
        """
        s, N = self.state, self.cfg.laps
        if s.lap >= N:
            return None
        starts = range(len(self.compounds)) if s.tyre is None else [s.tyre]
        best = None
        for c in starts:
            age, used = (0, mask_bit(c)) if s.tyre is None else (s.age, s.used)
            t = self.lt[s.lap, c, min(age, N)] + self.V[s.lap + 1, c, min(age + 1, N), used, stops]
            if best is None or t < best[0]:
                best = (t, c, age, used)
        if not np.isfinite(best[0]):
            return None

        total, c, age, used = best
        plan = Plan(total)
        first, r = s.lap + 1, stops
        age = min(age + 1, N)
        for L in range(s.lap + 1, N):
            stay = self.lt[L, c, age] + self.V[L + 1, c, min(age + 1, N), used, r]
            choice, choice_t = None, stay
            if r > 0:
                for c2 in range(len(self.compounds)):
                    nm = used | mask_bit(c2)
                    t = self.pit_loss(L - 1) + self.lt[L, c2, 0] + self.V[L + 1, c2, 1, nm, r - 1]
                    if t < choice_t - 1e-9:
                        choice, choice_t = c2, t
            if choice is None:
                age = min(age + 1, N)
            else:
                plan.stints.append(Stint(c, first, L))
                c, age, used, r, first = choice, 1, used | mask_bit(choice), r - 1, L + 1
        plan.stints.append(Stint(c, first, N))
        return plan

    def plans(self):
        self.solve()
        return {k: self.best_plan(k) for k in range(self.cfg.max_stops + 1)}

    # --- learning from real laps -------------------------------------------------------------

    def calibrate(self):
        """Fit the wear multiplier to observed laps by grid search.

        For each candidate multiplier, predict every observed lap; the best constant offset is
        then the mean residual, and we keep the multiplier with the smallest remaining squared
        error. The offset absorbs "the car is just faster/slower than the model"; only the
        *slope* of lap time against tyre age informs the wear rate, so a few laps with
        increasing tyre age are needed before the fit means anything.
        """
        if len(self.observations) < 4:
            return
        best = None
        for scale in np.arange(0.3, 3.0001, 0.02):
            self.deg_scale = scale
            pred = np.array([self.lap_times(L, c, np.array([a]))[0] for L, c, a, _ in self.observations])
            res = np.array([t for *_, t in self.observations]) - pred
            sse = float(((res - res.mean()) ** 2).sum())
            if best is None or sse < best[0]:
                best = (sse, scale, res.mean())
        _, self.deg_scale, self.pace_offset = best

    # --- commands ----------------------------------------------------------------------------

    def report(self, full=False):
        plans = self.plans()
        legal = {k: p for k, p in plans.items() if p is not None}
        if not legal:
            print("No legal strategy from this state (check tyre rules / stops).")
            return
        k_best = min(legal, key=lambda k: legal[k].time)
        best = legal[k_best]
        s = self.state
        nxt = s.lap + 1

        if len(best.stints) > 1 and best.stints[0].last == nxt:
            new = self.compounds[best.stints[1].tyre].name.upper()
            print(f">>> BOX BOX, end of lap {nxt}, for {new}")
        elif len(best.stints) > 1:
            print(f">>> Stay out. Next stop end of lap {best.stints[0].last} "
                  f"for {self.compounds[best.stints[1].tyre].name}")
        else:
            print(">>> Stay out to the flag")

        if full:
            on = "start tyre: agent's choice" if s.tyre is None else \
                f"on {self.compounds[s.tyre].name}, {s.age} laps old"
            print(f"\nLap {s.lap}/{self.cfg.laps} done, {on}")
            print(f"Best: {k_best} stop(s), {best.time:.1f} s to the flag")
            for st in best.stints:
                print(f"  {self.compounds[st.tyre].name:<6} laps {st.first:>2}-{st.last:<2} "
                      f"({st.last - st.first + 1} laps)")
            print("Alternatives:")
            for k, p in plans.items():
                txt = "not legal / not possible" if p is None else \
                    f"{p.time - best.time:+6.1f} s   {p.describe()}"
                print(f"  {k} stop(s): {txt}")
        if len(self.observations) >= 4:
            print(f"(fitted: tyres wearing {self.deg_scale:.2f}x model rate, "
                  f"car {self.pace_offset:+.2f} s/lap vs model)")

    def tyre_table(self):
        lap = min(self.state.lap, self.cfg.laps - 1)
        ages = np.array([1, 10, 20, 30])
        c = self.cond[lap]
        print(f"Predicted lap {lap + 1} times (wetness {c.wetness:.2f}, track {c.track_temp:.0f} C)")
        print("        " + "".join(f"{'age ' + str(a):>9}" for a in ages))
        for i, comp in enumerate(self.compounds):
            t = self.lap_times(lap, i, ages)
            print(f"  {comp.name:<6}" + "".join(f"{x:9.2f}" for x in t))

    def handle(self, line):
        parts = line.split()
        if not parts:
            return True
        cmd, args = parts[0].lower(), parts[1:]
        s, N = self.state, self.cfg.laps
        try:
            if cmd in ("quit", "exit"):
                return False
            elif cmd == "help":
                print(HELP)
            elif cmd == "plan":
                self.report(full=True)
            elif cmd == "tyres":
                self.tyre_table()
            elif cmd == "status":
                s.lap, s.tyre, s.age = int(args[0]), CODES[args[1].upper()], int(args[2])
                used = args[3].upper() if len(args) > 3 else args[1].upper()
                s.used = 0
                for ch in used:
                    s.used |= mask_bit(CODES[ch])
                self.report(full=True)
            elif cmd == "lap":
                if s.tyre is None:
                    print("Set the start tyre first: status 0 <S|M|H|I|W> 0")
                    return True
                t = float(args[0])
                if not self.cond[s.lap].safety_car and s.age > 0:
                    self.observations.append((s.lap, s.tyre, s.age, t))
                self.solve()
                pred = self.lt[s.lap, s.tyre, min(s.age, N)] + self.pace_offset
                print(f"Lap {s.lap + 1}: {t:.3f} s (model {pred:.3f}, {t - pred:+.3f})")
                s.lap, s.age = s.lap + 1, s.age + 1
                self.calibrate()
                self.report()
            elif cmd == "pit":
                c = CODES[args[0].upper()]
                s.lap, s.tyre, s.age = s.lap + 1, c, 1
                s.used |= mask_bit(c)
                print(f"Lap {s.lap}: pitted for {self.compounds[c].name}")
                self.report()
            elif cmd == "rain":
                a, b, w = int(args[0]), int(args[1]), float(args[2])
                for L in range(max(1, a) - 1, min(N, b)):
                    self.cond[L].wetness = min(1.0, max(0.0, w))
                print(f"Forecast: wetness {w:.2f} on laps {a}-{b}")
                self.report(full=True)
            elif cmd == "temp":
                t = float(args[0])
                start = int(args[1]) if len(args) > 1 else s.lap + 1
                for L in range(start - 1, N):
                    self.cond[L].track_temp = t
                print(f"Track temperature {t:.0f} C from lap {start}")
                self.report(full=True)
            elif cmd == "sc":
                a = int(args[0])
                b = int(args[1]) if len(args) > 1 else a
                for L in range(a - 1, min(N, b)):
                    self.cond[L].safety_car = True
                print(f"Safety car laps {a}-{b}: stops cost {self.cfg.pit_loss * self.cfg.sc_pit_factor:.0f} s")
                self.report(full=True)
            else:
                print(f"Unknown command '{cmd}'. Type help.")
        except (IndexError, KeyError, ValueError):
            print(f"Bad arguments for '{cmd}'. Type help.")
        return True


HELP = """Commands (laps are 1-based; tyres S M H I W):
  plan                         full recommended strategy and alternatives
  tyres                        predicted lap times per compound and tyre age
  status <lap> <tyre> <age> [used]
                               set race state: laps completed, current tyre, its age, compounds
                               used so far (e.g. SM). Before the start, the agent picks the tyre.
  lap <seconds>                report a clean lap just completed; the agent re-fits tyre wear
  pit <tyre>                   we just completed a lap with a stop onto <tyre>
  rain <from> <to> <wetness>   forecast/observed water on track, 0 = dry .. 1 = standing water
  temp <celsius> [from lap]    track temperature
  sc <lap> [to lap]            safety car: stops cost half
  quit"""


def main():
    ap = argparse.ArgumentParser(description="Pit wall strategy agent")
    ap.add_argument("--driver", default="VER")
    ap.add_argument("--track", default="tracks/silverstone.csv")
    ap.add_argument("--laps", type=int, default=52)
    ap.add_argument("--pit-loss", type=float, default=20.0)
    ap.add_argument("--temp", type=float, default=40.0, help="track temperature (C)")
    ap.add_argument("--race", help="fitted race JSON from race_data.py (sets track, laps, "
                                   "pit loss, temperature and tyre data)")
    args = ap.parse_args()
    race = json.loads(open(args.race).read()) if args.race else None
    if race:
        args.track, args.laps, args.temp = race["track"], race["laps"], race["track_temp"]
        args.pit_loss = race["pit_loss"]

    found = [(car, d) for car, drivers in TEAMS.values() for d in drivers if d.name == args.driver.upper()]
    if not found:
        names = [d.name for _, ds in TEAMS.values() for d in ds]
        sys.exit(f"Unknown driver {args.driver}; choose from {', '.join(names)}")
    car, driver = found[0]

    print(f"Building pace model for {driver.name} on {args.track} ...")
    pace = PaceModel(load_track(args.track), car, driver)
    print(f"Dry soft lap ~{np.interp(car.mu, pace.mus, pace.times):.2f} s with no fuel, "
          f"fuel costs {pace.fuel_sens:.3f} s/kg\n")

    compounds = compounds_from_fit(pace, race) if race else None
    agent = StrategyAgent(pace, RaceConfig(laps=args.laps, pit_loss=args.pit_loss), compounds)
    for c in agent.cond:
        c.track_temp = args.temp
    agent.report(full=True)

    interactive = sys.stdin.isatty()
    while True:
        if interactive:
            print("\n> ", end="", flush=True)
        line = sys.stdin.readline()
        if not line:
            break
        if not interactive:
            print(f"\n> {line.strip()}")
        if not agent.handle(line):
            break


if __name__ == "__main__":
    main()
