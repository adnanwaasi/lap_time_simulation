# Lap Time Simulation

A Formula 1 lap time simulator in C++, with Python tools that use real F1 data (via
[FastF1](https://docs.fastf1.dev/)) to build tracks, check the simulator against real laps,
plan pit strategy and predict races.

## Overview

| Part | What it does |
|---|---|
| `main.cpp` | Simulates one flying lap per driver on a track and prints qualifying standings |
| `build_track.py` | Builds a track CSV from the real F1 racing line |
| `validate.py` | Compares the simulated speed trace with a driver's real fastest lap |
| `strategy.py` | Pit wall strategy agent (tyres, fuel, weather, safety cars) |
| `race_data.py` | Fits model parameters from real race and qualifying data |
| `race_sim.py` | Monte Carlo race prediction for the whole field |
| `fetch_driver_data.py` | Estimates relative driver grip use from teammates' fastest laps |

## Lap simulator (`main.cpp`)

### Build and run

```bash
g++ -std=c++17 -O2 -Wall -Wextra -o laptimesim main.cpp
./laptimesim                              # regenerates tracks/silverstone.csv and simulates it
./laptimesim tracks/silverstone_real.csv  # simulates any track CSV
```

Run it from the repo root. Each run also writes every driver's speed vs distance to
`data/sim_trace.csv`.

### Physics model

The car is a point mass, and each 5 m segment is treated as steady state.

- **Grip:** the tyres give `mu * N`. The normal load `N` is the car's weight (`g·cosθ` on a
  slope) plus downforce, and downforce grows with `v²`.
- **Cornering:** the top speed in a corner is the speed where the grip needed (`v²·curvature`)
  equals the grip available. If downforce grows faster than the grip a corner needs, the corner
  is taken flat out.
- **Friction circle:** grip used for cornering can't be used for braking or accelerating.
- **Acceleration:** the smaller of rear-wheel traction and engine power (`P = F·v`), minus
  aero drag, rolling resistance and gravity on slopes.
- **Braking:** all four tyres brake, and the same resistances help slow the car.
- **Flying lap:** an acceleration pass and a braking pass each go around the circuit twice. The
  car crosses the start/finish line at the speed it carried from the previous lap.
- **Drivers:** each driver uses a fraction of the car's limit (`grip_use`, `brake_use`).

Not modelled: tyre temperature and wear (`strategy.py` adds wear separately), weight transfer,
energy deployment limits (peak power is constant), DRS and slipstream, and the drag/downforce
tradeoff between wing levels.

The car and driver numbers in `TEAMS` are rough placeholders, not calibrated values.

### Example output

```
QUALIFYING STANDINGS:
------------------------------------------
 1. PIA  McLaren   69.039s  327 km/h
 2. NOR  McLaren   69.582s  327 km/h
 3. VER  Red Bull  69.987s  331 km/h
 4. RUS  Mercedes  70.914s  329 km/h
 5. TSU  Red Bull  71.581s  331 km/h
 6. ANT  Mercedes  71.711s  329 km/h
```

These times are much faster than a real Silverstone lap because the hand-built `silverstone.csv`
is only 4.6 km long, against about 5.8 km for the real circuit. On `silverstone_real.csv` the
same cars lap in about 89–90 s, a few seconds slower than real pole times.

## Track data

Tracks are CSVs with one row per segment:

```
ds,curvature,gradient
5.0,0.000602,0.00648
```

- `ds`: segment length (m)
- `curvature`: 1/radius (1/m); the sign gives the turn direction, 0 = straight
- `gradient`: rise over run (dz/ds), positive uphill. This column is optional and defaults to 0.

Files in `tracks/`:

- `silverstone.csv` is a hand-built approximation. It is **regenerated** from the
  `add_section(...)` calls in `main.cpp` on every run, so edit the code, not the CSV.
- `silverstone_real.csv` and `cota_real.csv` come from real telemetry via `build_track.py`.

## Python tools

They need a virtual environment with FastF1:

```bash
uv venv .venv && uv pip install --python .venv/bin/python fastf1 numpy scipy matplotlib pandas
```

FastF1 downloads are cached in `data/fastf1_cache`. All outputs go to `data/`, which is not
committed.

```bash
# Real racing line -> track CSV (averages every quick lap in the session)
.venv/bin/python build_track.py 2025 Silverstone Q tracks/silverstone_real.csv

# Check the simulator against a real lap: lap time, speed error, plot
./laptimesim tracks/silverstone_real.csv
.venv/bin/python validate.py 2025 Silverstone Q VER

# Interactive strategy agent (type `help` once it's running)
python3 strategy.py

# Race prediction
.venv/bin/python race_data.py track 2025 "United States Grand Prix" --track tracks/cota_real.csv
.venv/bin/python race_data.py form 2026 --last 6
.venv/bin/python race_sim.py --race data/race_united_states_grand_prix.json \
                             --form data/form_2026.json [--grid VER,LEC,...] [--car-weight 0.5]
```

### What the race prediction is based on

- **Circuit (`race_data.py track`):** fitted from one race at this circuit:
  - laps, pit-lane time loss and track temperature
  - per-compound pace and tyre wear (fuel effect fixed from the physics)
  - safety-car and VSC rates from that event's last `--history` editions (default 4)
- **Form (`race_data.py form`):** fitted from the last `--last` completed races of the season
  (default 6). Later races count more.
  - Each driver's race pace and qualifying pace relative to the field (a weighted median, so
    one bad race doesn't dominate)
  - Each team's slow-corner, fast-corner and top speeds from qualifying telemetry
  - The season's DNF rate over those races
- **Simulation (`race_sim.py`):**
  - Each team's speeds are turned into a car (grip, downforce, drag), and that car is simulated
    on this circuit.
  - Team pace is a blend of recent form and this circuit simulation (`--car-weight`: 0 = form
    only, 1 = physics only). Drivers keep their gap to their teammate from form.
  - Each simulated race includes lap-time noise, overtaking difficulty, random safety cars and
    retirements. Every car's strategy agent re-plans when a safety car comes out.

## Author

Developed by Adnan Waasi.
