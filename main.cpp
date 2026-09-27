#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>

const double DS = 5.0;  // Generated segment length (m)
const double G = 9.81;  // Gravity (m/s^2)

struct TrackSegment {
    double ds;         // Segment length (m)
    double curvature;  // 1/radius (1/m)
};

struct Vehicle {
    double mu;     // Tire friction coefficient
    double accel;  // Acceleration (m/s^2)
    double brake;  // Braking (m/s^2)
};

struct Team {
    std::string name;
    Vehicle vehicle;
};

const std::vector<Team> TEAMS = {
    {"Mercedes", {1.35, 4.5, 9.0}},
    {"Red Bull", {1.30, 4.6, 8.8}},
    {"McLaren", {1.28, 4.3, 8.5}},
};

double lateral_speed_limit(double mu, double curvature) {
    if (curvature == 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    return std::sqrt(mu * G / std::abs(curvature));
}

// Flying-lap simulation on a closed circuit. speeds[i] is the speed at the start of segment i.
// Both passes run two laps around the loop so the speed carried across the start/finish line
// is consistent with the end of the previous lap, instead of starting from standstill.
double simulate_lap(const std::vector<TrackSegment> &track, const Vehicle &vehicle,
                    std::vector<double> &speeds) {
    const size_t n = track.size();
    speeds.assign(n, 0.0);
    if (n == 0) {
        return 0.0;
    }

    // Forward pass (acceleration), capped by the cornering limit of each segment
    double v = 0.0;
    for (size_t k = 0; k < 2 * n; ++k) {
        const size_t i = k % n;
        v = std::min(v, lateral_speed_limit(vehicle.mu, track[i].curvature));
        speeds[i] = v;
        v = std::sqrt(v * v + 2 * vehicle.accel * track[i].ds);
    }

    // Backward pass (braking): arrive at each segment no faster than we can brake from
    for (size_t k = 2 * n; k-- > 0;) {
        const size_t i = k % n;
        const double v_next = speeds[(i + 1) % n];
        const double v_allowed = std::sqrt(v_next * v_next + 2 * vehicle.brake * track[i].ds);
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

    double ds, curvature;
    char comma;
    while (file >> ds >> comma >> curvature) {
        if (ds > 0.0) {
            track.push_back({ds, curvature});
        }
    }

    return track;
}

// Usage: laptimesim [track.csv]
// With no argument, regenerates tracks/silverstone.csv and simulates it.
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

    std::vector<std::pair<std::string, double>> results;
    std::vector<double> speeds;

    for (const auto &team : TEAMS) {
        double lap_time = simulate_lap(track, team.vehicle, speeds);
        results.push_back({team.name, lap_time});
    }

    std::sort(results.begin(), results.end(),
              [](const auto &a, const auto &b) { return a.second < b.second; });

    std::cout << "QUALIFYING STANDINGS:\n";
    std::cout << "----------------------------\n";
    std::cout << std::fixed << std::setprecision(3);
    for (size_t i = 0; i < results.size(); ++i) {
        std::cout << i + 1 << ". " << std::left << std::setw(10) << results[i].first << " "
                  << results[i].second << "s\n";
    }

    return 0;
}
