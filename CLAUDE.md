# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & run

No build system — single translation unit:

```bash
g++ -std=c++17 -O2 -Wall -Wextra -o laptimesim main.cpp && ./laptimesim
```

Run from the repo root: paths like `tracks/silverstone.csv` are relative to the working directory.
`./laptimesim <track.csv>` simulates that track instead of regenerating Silverstone. Every run writes per-driver speed traces to `data/sim_trace.csv`.
C++ style: `.clang-format` (Google base, 4-space indent, 100 columns).
Lint: `clang-tidy main.cpp -- -std=c++17` (`.clang-tidy`) and `.venv/bin/ruff check .` (`ruff.toml`).

## Architecture

- `main.cpp` is the source of truth for the simulator. The notebooks (`physics_mai.ipynb`, `visuals.ipynb`) are earlier Python exploration; don't port changes back to them unless asked.
- `tracks/silverstone.csv` is a **generated** artifact: `main()` rewrites it from the `add_section(...)` calls on every run (the first cell of `physics_mai.ipynb` does too). To change the track, edit the section list in code, not the CSV.
- Track format: `ds,curvature[,gradient]` per row, `ds` in m, curvature = 1/radius in 1/m (sign = direction, 0 = straight), gradient = dz/ds (optional, defaults to 0).
- Physics model: quasi-steady-state point mass, flying lap (both passes wrap around the circuit twice so start/finish speed matches the end of the previous lap). Normal load N = g·cosθ + k·v², k = ½ρ·ClA/m. Cornering limit `v² = mu·g·cosθ / (|curvature| − mu·k)` (flat out when ≤ 0). Friction circle: longitudinal grip = √((mu·N)² − (v²·curvature)²). Forward pass: min(rear_load·long grip, P/(m·v)) − resistance; backward pass: brake_use·long grip + resistance. Resistance = drag + crr·N + g·sinθ. Explicit Euler step in v² per segment. Lap time = Σ 2ds/(v_in+v_out).
- Drivers scale the car's limit: `grip_use` (multiplies mu everywhere) and `brake_use` (braking). Values in `TEAMS` are placeholders.
- `strategy.py`: pit wall strategy agent (`python3 strategy.py`, then `help`). Contains a Python port of `simulate_lap` that must be kept in sync with `main.cpp`; it builds a lap-time-vs-grip table from it, adds a tyre/fuel/weather model, and finds the optimal plan by dynamic programming. Re-fits tyre wear from lap times fed in during the race.
- Python tools that need FastF1/scipy run from the project venv: `.venv/bin/python` (gitignored; `uv venv .venv && uv pip install --python .venv/bin/python fastf1 numpy scipy ruff`).
- `race_data.py`: FastF1 -> model parameters. `track <year> <event> --track <csv>` fits per-compound pace/wear (least squares, fuel slope fixed from physics), pit loss and safety-car rates -> `data/race_<event>.json`. `form <year> --last N` gives driver race/quali pace, per-team car speed ratios from quali telemetry, DNF rate -> `data/form_<year>.json`.
- `race_sim.py`: Monte Carlo race prediction using those JSONs; team cars fitted from telemetry and run through the physics on the circuit, blended with form (`--car-weight`); `strategy.StrategyAgent` is each car's pit wall and re-plans under SC/VSC. `--grid` takes the real grid once qualifying is done.
- `build_track.py <year> <event> <session> <out.csv>` (venv): real racing line from averaged FastF1 position telemetry -> `ds,curvature,gradient`. Produced `tracks/silverstone_real.csv` and `tracks/cota_real.csv` (`2025 "United States" Q`).
- `validate.py <year> <event> <session> <driver>` (venv): compares `data/sim_trace.csv` with that driver's real fastest lap. Run `./laptimesim tracks/silverstone_real.csv` first.
- `fetch_driver_data.py` (needs `pip install fastf1`) exports teammates' fastest laps to `data/` (gitignored) and estimates relative grip use from slow-corner minimum speeds.
- Not modelled (in `main.cpp`): tyre temperature/wear, weight transfer, energy deployment limits (constant peak power), DRS/slipstream, the drag/downforce tradeoff of different wing levels.

## Conventions

- Work on the `waasi` branch; keep commits small and scoped to one change.
- Don't commit build output (`a.out`, `laptimesim`).
