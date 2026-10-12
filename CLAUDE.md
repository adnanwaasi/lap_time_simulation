# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & run

No build system — single translation unit:

```bash
g++ -std=c++17 -O2 -Wall -Wextra -o laptimesim main.cpp && ./laptimesim
```

Run from the repo root: paths like `tracks/silverstone.csv` are relative to the working directory.

## Architecture

- `main.cpp` is the source of truth for the simulator. The notebooks (`physics_mai.ipynb`, `visuals.ipynb`) are earlier Python exploration; don't port changes back to them unless asked.
- `tracks/silverstone.csv` is a **generated** artifact: `main()` rewrites it from the `add_section(...)` calls on every run (the first cell of `physics_mai.ipynb` does too). To change the track, edit the section list in code, not the CSV.
- Track format: `ds,curvature` per row, `ds` in m, curvature = 1/radius in 1/m, 0 = straight.
- Physics model: quasi-steady-state point mass, flying lap (both passes wrap around the circuit twice so start/finish speed matches the end of the previous lap). Tyre force = `mu * (m*g + downforce)`, downforce and drag ∝ v². Cornering limit `v² = mu*g / (curvature - mu*k)`, k = ½ρ·ClA/m (flat out when curvature ≤ mu*k). Forward pass: min(grip, power/(m*v)) − drag. Backward pass: grip braking + drag. Lap time = Σ 2ds/(v_in+v_out).
- Drivers scale the car's limit: `grip_use` (cornering mu) and `brake_use` (braking). Values in `TEAMS` are placeholders.
- `strategy.py`: pit wall strategy agent (`python3 strategy.py`, then `help`). Contains a Python port of `simulate_lap` that must be kept in sync with `main.cpp`; it builds a lap-time-vs-grip table from it, adds a tyre/fuel/weather model, and finds the optimal plan by dynamic programming. Re-fits tyre wear from lap times fed in during the race.
- Python tools that need FastF1/scipy run from the project venv: `.venv/bin/python` (gitignored; `uv venv .venv && uv pip install --python .venv/bin/python fastf1 numpy scipy`).
- `race_data.py`: FastF1 -> model parameters. `track <year> <event> --track <csv>` fits per-compound pace/wear (least squares, fuel slope fixed from physics), pit loss and safety-car rates -> `data/race_<event>.json`. `form <year> --last N` gives driver race/quali pace, per-team car speed ratios from quali telemetry, DNF rate -> `data/form_<year>.json`.
- `race_sim.py`: Monte Carlo race prediction using those JSONs; team cars fitted from telemetry and run through the physics on the circuit, blended with form (`--car-weight`); `strategy.StrategyAgent` is each car's pit wall and re-plans under SC/VSC. `--grid` takes the real grid once qualifying is done.
- `tracks/cota_real.csv`: from `build_track.py 2025 "United States" Q`.
- `fetch_driver_data.py` (needs `pip install fastf1`) exports teammates' fastest laps to `data/` (gitignored) and estimates relative grip use from slow-corner minimum speeds.
- Not modelled: elevation, tyre temperature/wear, combined grip (no friction circle: braking and cornering don't share grip), traction limited to the driven wheels, energy deployment limits (constant peak power), DRS/slipstream, the drag/downforce tradeoff of different wing levels.

## Conventions

- Work on the `waasi` branch; keep commits small and scoped to one change.
- Don't commit build output (`a.out`, `laptimesim`).
