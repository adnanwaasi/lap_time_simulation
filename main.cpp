#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <sstream>

const double DS = 5.0;     // Generated segment length (m)
const double G = 9.81;     // Gravity (m/s^2)
const double RHO = 1.225;  // Air density (kg/m^3), sea level at 15 C

struct TrackSegment {
    double ds;         // Segment length (m)
    double curvature;  // 1/radius (1/m), sign = turn direction
    double gradient;   // Rise over run (dz/ds), + uphill; 0 if the CSV has no gradient column
};

struct Vehicle {
    double mu;         // Tyre friction coefficient
    double mass;       // Car + driver + fuel (kg)
    double power;      // Power unit output at the wheels (W)
    double cl_a;       // Downforce coefficient * frontal area (m^2)
    double cd_a;       // Drag coefficient * frontal area (m^2)
    double rear_load;  // Fraction of the tyre load on the driven (rear) wheels
    double crr;        // Rolling resistance coefficient
};

// How much of the car's limit a driver actually uses (1.0 = perfect).
// Placeholders until calibrated from telemetry (see fetch_driver_data.py). McLaren grip_use
// is from 2025 Silverstone Q, relative to PIA = 1.0 (NOR/PIA slow-corner speed ratio squared).
struct Driver {
    std::string name;
    double grip_use;   // Fraction of tyre grip used
    double brake_use;  // Fraction of the available braking used
};

struct Team {
    std::string name;
    Vehicle vehicle;
    std::vector<Driver> drivers;
};

// Rough 2025-era figures: ~800 kg qualifying mass, ~700 kW, ClA ~4.8, CdA ~1.4.
const std::vector<Team> TEAMS = {
    {"Mercedes", {1.60, 800.0, 700e3, 4.7, 1.40, 0.55, 0.012}, {{"RUS", 0.99, 0.99}, {"ANT", 0.97, 0.97}}},
    {"Red Bull", {1.60, 800.0, 705e3, 4.8, 1.38, 0.55, 0.012}, {{"VER", 1.00, 1.00}, {"TSU", 0.96, 0.97}}},
    {"McLaren", {1.62, 800.0, 700e3, 5.0, 1.42, 0.55, 0.012}, {{"NOR", 0.986, 1.00}, {"PIA", 1.00, 1.00}}},
};

// All forces below are per unit mass (m/s^2), so the car's mass cancels wherever it can.

// Downforce per unit mass per unit v^2: F_down / m = k * v^2 (1/m)
double downforce_k(const Vehicle &car) { return 0.5 * RHO * car.cl_a / car.mass; }

// Load pressing the tyres into the road: the slope-normal part of gravity plus downforce.
// On a slope of angle theta, gradient = tan(theta), so cos(theta) = 1 / sqrt(1 + gradient^2).
double normal_accel(const Vehicle &car, const TrackSegment &seg, double v) {
    return G / std::sqrt(1.0 + seg.gradient * seg.gradient) + downforce_k(car) * v * v;
}

// Forces that always slow the car: aero drag, rolling resistance, and gravity along the slope
// (sin(theta) = gradient / sqrt(1 + gradient^2)); going downhill this term is negative and helps.
double resistance_accel(const Vehicle &car, const TrackSegment &seg, double v) {
    const double drag = 0.5 * RHO * car.cd_a * v * v / car.mass;
    const double rolling = car.crr * normal_accel(car, seg, v);
    const double slope = G * seg.gradient / std::sqrt(1.0 + seg.gradient * seg.gradient);
    return drag + rolling + slope;
}

// Cornering: v^2 * curvature = mu * (g cos(theta) + k v^2)  =>  v^2 = mu g cos / (curvature - mu k).
// If curvature <= mu k, downforce grows faster than the grip needed: the corner is flat out.
double lateral_speed_limit(const Vehicle &car, double mu, const TrackSegment &seg) {
    const double denom = std::abs(seg.curvature) - mu * downforce_k(car);
    if (denom <= 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    return std::sqrt(mu * normal_accel(car, seg, 0.0) / denom);
}

// Friction circle: the tyre has mu * N of grip in total, shared between cornering and
// accelerating/braking. Whatever cornering uses is gone: long = sqrt(total^2 - lat^2).
double longitudinal_grip(const Vehicle &car, double mu, const TrackSegment &seg, double v) {
    const double total = mu * normal_accel(car, seg, v);
    const double lateral = v * v * std::abs(seg.curvature);
    return std::sqrt(std::max(0.0, total * total - lateral * lateral));
}

// Net forward acceleration: the smaller of what the rear tyres can transmit (traction) and
// what the power unit can deliver (P = F v), minus resistances.
double drive_accel(const Vehicle &car, double mu, const TrackSegment &seg, double v) {
    const double traction = car.rear_load * longitudinal_grip(car, mu, seg, v);
    const double power = v > 0.0 ? car.power / (car.mass * v) : traction;
    return std::min(traction, power) - resistance_accel(car, seg, v);
}

// Deceleration while braking: all four tyres brake, and resistances help slow the car.
double brake_decel(const Vehicle &car, const Driver &driver, double mu, const TrackSegment &seg,
                   double v) {
    return driver.brake_use * longitudinal_grip(car, mu, seg, v) + resistance_accel(car, seg, v);
}

// Flying-lap simulation on a closed circuit. speeds[i] is the speed at the start of segment i.
// Both passes run two laps around the loop so the speed carried across the start/finish line
// is consistent with the end of the previous lap, instead of starting from standstill.
// Accelerations depend on speed, so each segment steps v^2 using the speed at its entry
// (explicit Euler in distance); with 5 m segments the error is small.
double simulate_lap(const std::vector<TrackSegment> &track, const Vehicle &car,
                    const Driver &driver, std::vector<double> &speeds) {
    const size_t n = track.size();
    speeds.assign(n, 0.0);
    if (n == 0) {
        return 0.0;
    }

    const double mu = car.mu * driver.grip_use;

    // Forward pass (acceleration), capped by the cornering limit of each segment
    double v = 0.0;
    for (size_t k = 0; k < 2 * n; ++k) {
        const size_t i = k % n;
        v = std::min(v, lateral_speed_limit(car, mu, track[i]));
        speeds[i] = v;
        v = std::sqrt(std::max(0.0, v * v + 2 * drive_accel(car, mu, track[i], v) * track[i].ds));
    }

    // Backward pass (braking): arrive at each segment no faster than we can brake from
    for (size_t k = 2 * n; k-- > 0;) {
        const size_t i = k % n;
        const double v_next = speeds[(i + 1) % n];
        const double decel = brake_decel(car, driver, mu, track[i], v_next);
        const double v_allowed = std::sqrt(std::max(0.0, v_next * v_next + 2 * decel * track[i].ds));
        speeds[i] = std::min(speeds[i], v_allowed);
    }

    // Lap time: exact for constant acceleration within a segment, t = 2 ds / (v_in + v_out)
    double lap_time = 0.0;
    for (size_t i = 0; i < n; ++i) {
        lap_time += 2 * track[i].ds / (speeds[i] + speeds[(i + 1) % n]);
    }

    return lap_time;
}

void add_section(std::ofstream &file, double length, double curvature) {
    int n = static_cast<int>(length / DS);
    for (int i = 0; i < n; ++i) {
        file << DS << "," << curvature << "\n";
    }
}

bool write_silverstone(const std::string &path) {
    std::ofstream file(path);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << path << std::endl;
        return false;
    }

    file << "ds,curvature\n";

    // Hamilton Straight
    add_section(file, 770, 0.000);

    // Abbey
    add_section(file, 150, 0.0067);

    // Farm Straight
    add_section(file, 300, 0.000);

    // Village
    add_section(file, 120, 0.0167);

    // Loop
    add_section(file, 90, 0.0286);

    // Wellington Straight
    add_section(file, 700, 0.000);

    // Brooklands
    add_section(file, 180, 0.0060);

    // Luffield
    add_section(file, 300, 0.0100);

    // Copse
    add_section(file, 180, 0.0056);

    // Maggotts–Becketts–Chapel
    add_section(file, 450, 0.0111);

    // Hangar Straight
    add_section(file, 770, 0.000);

    // Stowe
    add_section(file, 200, 0.0050);

    // Vale + Club
    add_section(file, 400, 0.0120);

    std::cout << "Track data written to " << path << std::endl;
    return true;
}

std::vector<TrackSegment> load_track_from_csv(const std::string &path) {
    std::vector<TrackSegment> track;
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << path << std::endl;
        return track;
    }

    std::string header;
    std::getline(file, header);  // Skip header

    // Each row is "ds,curvature" or "ds,curvature,gradient". Reading a whole line first and
    // then parsing it with a stringstream lets the optional third column be missing.
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream row(line);
        double ds = 0.0, curvature = 0.0, gradient = 0.0;
        char comma;
        if (!(row >> ds >> comma >> curvature)) {
            continue;
        }
        row >> comma >> gradient;  // leaves gradient = 0 if absent
        if (ds > 0.0) {
            track.push_back({ds, curvature, gradient});
        }
    }

    return track;
}

// Usage: laptimesim [track.csv]
// With no argument, regenerates tracks/silverstone.csv and simulates it.
// tracks/silverstone_real.csv (from build_track.py) is the real racing line with elevation.
int main(int argc, char **argv) {
    std::string path = "tracks/silverstone.csv";
    if (argc > 1) {
        path = argv[1];
    } else if (!write_silverstone(path)) {
        return 1;
    }

    std::vector<TrackSegment> track = load_track_from_csv(path);
    if (track.empty()) {
        std::cerr << "Error: No track segments in " << path << std::endl;
        return 1;
    }

    struct Result {
        std::string driver;
        std::string team;
        double lap_time;
        double top_speed;  // m/s
    };
    std::vector<Result> results;
    std::vector<std::vector<double>> traces;  // one speed trace per driver, in results order
    std::vector<double> speeds;

    for (const auto &team : TEAMS) {
        for (const auto &driver : team.drivers) {
            double lap_time = simulate_lap(track, team.vehicle, driver, speeds);
            double top = *std::max_element(speeds.begin(), speeds.end());
            results.push_back({driver.name, team.name, lap_time, top});
            traces.push_back(speeds);
        }
    }

    // Speed vs distance for every driver, for comparing against real telemetry (validate.py).
    // Written before sorting so traces[j] still matches results[j].
    const std::string trace_path = "data/sim_trace.csv";
    std::filesystem::create_directories("data");
    std::ofstream trace(trace_path);
    trace << "distance";
    for (const auto &r : results) {
        trace << "," << r.driver;
    }
    trace << "\n";
    double distance = 0.0;
    for (size_t i = 0; i < track.size(); ++i) {
        trace << distance;
        for (const auto &t : traces) {
            trace << "," << t[i];
        }
        trace << "\n";
        distance += track[i].ds;
    }

    std::sort(results.begin(), results.end(),
              [](const Result &a, const Result &b) { return a.lap_time < b.lap_time; });

    std::cout << "QUALIFYING STANDINGS:\n";
    std::cout << "------------------------------------------\n";
    std::cout << std::fixed << std::setprecision(3);
    for (size_t i = 0; i < results.size(); ++i) {
        std::cout << std::setw(2) << i + 1 << ". " << std::left << std::setw(4) << results[i].driver
                  << " " << std::setw(9) << results[i].team << std::right << " "
                  << results[i].lap_time << "s  " << std::setprecision(0)
                  << results[i].top_speed * 3.6 << " km/h\n"
                  << std::setprecision(3);
    }
    std::cout << "Speed traces written to " << trace_path << "\n";

    return 0;
}
