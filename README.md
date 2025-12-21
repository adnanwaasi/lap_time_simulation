# Lap Time Simulation

## Overview
Lap Time Simulation is a program designed to simulate lap times for Formula 1 cars on a given track. The simulation takes into account various factors such as tire grip, vehicle acceleration, braking capabilities, and track curvature. The program supports multiple teams with unique vehicle setups and calculates lap times for each team.

## Features
- **Track Simulation**: Load track data from a CSV file, including segment lengths and curvatures.
- **Vehicle Dynamics**: Simulate vehicle behavior based on tire grip, acceleration, and braking.
- **Team Configurations**: Predefined setups for teams like Mercedes, Red Bull, and McLaren.
- **Lap Time Calculation**: Forward and backward passes to compute realistic lap times.
- **Qualifying Standings**: Compare lap times across teams and display standings.

## How It Works
1. **Track Data**: The program reads track data from a CSV file (e.g., `tracks/silverstone.csv`). Each segment of the track is defined by its length and curvature.
2. **Vehicle Properties**: Each team has a unique vehicle setup, including:
   - Tire friction coefficient (`mu`)
   - Acceleration (`m/s^2`)
   - Braking (`m/s^2`)
3. **Simulation**: The program calculates the maximum speed for each track segment based on lateral and longitudinal forces. It then integrates the lap time over the entire track.
4. **Results**: Lap times for each team are calculated and displayed in qualifying order.

## Usage
### Prerequisites
- A C++ compiler (e.g., `g++`)
- A CSV file with track data (e.g., `tracks/silverstone.csv`)

### Compilation
To compile the program, run:
```bash
g++ -o laptimesim main.cpp
```

### Running the Program
To run the program, execute:
```bash
./laptimesim
```

### Example Output
```
QUALIFYING STANDINGS:
----------------------------
1. Mercedes - 85.32s
2. Redbull  - 86.45s
3. Mclaren  - 87.12s
```

## File Structure
```
.
├── main.cpp               # Main simulation code
├── tracks/
│   └── silverstone.csv    # Example track data
├── README.md              # Project documentation
```

## Track Data Format
The track data CSV file should have the following format:
```
ds,curvature
10,0.0
10,0.1
...
```
- `ds`: Segment length in meters
- `curvature`: Curvature (1/radius) in 1/m

## Future Improvements
- Add support for dynamic weather conditions.
- Include more detailed vehicle dynamics (e.g., downforce, drag).
- Visualize lap simulations with graphical tools.

## License
This project is open-source and available under the [MIT License](LICENSE).

## Author
Developed by Adnan Waasi.
