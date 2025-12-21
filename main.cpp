#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>

const double DS = 5.0;

struct TrackSegment {
    double ds;         // Segment length (m)
    double curvature;  // 1/radius (1/m)
};

struct Vehicle {
    double mu;     // Tire friction coefficient
    double g;      // Gravity (m/s^2)
    double accel;  // Acceleration (m/s^2)
    double brake;  // Braking (m/s^2)
};

Vehicle get_team_vehicle(const std::string &team_name) {
    if (team_name == "mercedes") {
        return {1.35, 9.81, 4.5, 9.0};
    } else if (team_name == "redbull") {
        return {1.30, 9.81, 4.6, 8.8};
    } else if (team_name == "mclaren") {
        return {1.28, 9.81, 4.3, 8.5};
    } else {
        return {1.2, 9.81, 4.0, 8.0}; // Default vehicle
    }
}

double lateral_speed_limit(double mu, double g, double curvature) {
    if (curvature == 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    return std::sqrt(mu * g / curvature);
}

double simulate_lap(const std::vector<TrackSegment> &track, const Vehicle &vehicle, std::vector<double> &speeds) {
    size_t n = track.size();
    speeds.assign(n, 0.0);

    // Forward pass (acceleration)
    for (size_t i = 0; i < n - 1; ++i) {
        double ds = track[i].ds;
        double v_possible = std::sqrt(speeds[i] * speeds[i] + 2 * vehicle.accel * ds);
        double v_lat = lateral_speed_limit(vehicle.mu, vehicle.g, track[i + 1].curvature);
        speeds[i + 1] = std::min(v_possible, v_lat);
    }

    // Backward pass (braking)
    for (size_t i = n - 1; i > 0; --i) {
        double ds = track[i - 1].ds;
        double v_allowed = std::sqrt(speeds[i] * speeds[i] + 2 * vehicle.brake * ds);
        speeds[i - 1] = std::min(speeds[i - 1], v_allowed);
    }

    // Lap time integration
    double lap_time = 0.0;
    for (size_t i = 1; i < n; ++i) {
        lap_time += track[i - 1].ds / speeds[i];
    }

    return lap_time;
}

void add_section(std::ofstream &file, double length, double curvature) {
    int n = static_cast<int>(length / DS);
    for (int i = 0; i < n; ++i) {
        file << DS << "," << curvature << "\n";
    }
}

std::vector<TrackSegment> load_track_from_csv(const std::string &path) {
    std::vector<TrackSegment> track;
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << path << std::endl;
        return track;
    }

    std::string header;
    std::getline(file, header); // Skip header

    double ds, curvature;
    char comma;
    while (file >> ds >> comma >> curvature) {
        track.push_back({ds, curvature});
    }

    return track;
}

int main() {
    std::ofstream file("tracks/silverstone.csv");
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file." << std::endl;
        return 1;
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

    file.close();
    std::cout << "Track data written to tracks/silverstone.csv" << std::endl;

    std::vector<TrackSegment> track = load_track_from_csv("tracks/silverstone.csv");
    if (track.empty()) {
        return 1;
    }

    std::vector<std::pair<std::string, double>> results;
    std::vector<double> speeds;

    std::vector<std::string> teams = {"mercedes", "redbull", "mclaren"};
    for (const auto &team : teams) {
        Vehicle vehicle = get_team_vehicle(team);
        double lap_time = simulate_lap(track, vehicle, speeds);
        results.push_back({team, lap_time});
    }

    std::sort(results.begin(), results.end(), [](const auto &a, const auto &b) {
        return a.second < b.second;
    });

    std::cout << "QUALIFYING STANDINGS:\n";
    std::cout << "----------------------------\n";
    for (size_t i = 0; i < results.size(); ++i) {
        std::cout << i + 1 << ". " << results[i].first << " - " << results[i].second << "s\n";
    }

    return 0;
}