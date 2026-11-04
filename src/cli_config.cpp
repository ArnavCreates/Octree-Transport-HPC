#include "transport/cli_config.hpp"
#include <iostream>
#include <stdexcept>
#include <cstdlib>

namespace transport {

void TransportCliConfig::printUsage(const char* prog_name) {
    std::cout << "=================================================================\n"
              << " Octree-Transport-HPC: Parallel Barnes-Hut O(N log N) Engine\n"
              << "=================================================================\n"
              << "Usage: " << prog_name << " [options]\n\n"
              << "Options:\n"
              << "  -v, --version <1..7>       Solver version:\n"
              << "                               1: CPU Direct Sequential (O(N^2))\n"
              << "                               2: CPU Direct OpenMP (O(N^2))\n"
              << "                               3: GPU Direct Naive GLSL (O(N^2))\n"
              << "                               4: GPU Direct Tiled GLSL (O(N^2))\n"
              << "                               5: CPU Barnes-Hut Sequential (O(N log N))\n"
              << "                               6: CPU Barnes-Hut OpenMP Disjoint (O(N log N))\n"
              << "                               7: GPU Barnes-Hut GLSL (O(N log N))\n"
              << "  -i, --init <1..8>          Phase-space initial conditions:\n"
              << "                               1: Solid Cube\n"
              << "                               2: Rotating Disk\n"
              << "                               3: Lagrange 3-Body Triangle\n"
              << "                               4: Spherical Shell\n"
              << "                               5: Plummer Ball\n"
              << "                               6: Cubic Shell\n"
              << "                               7: Bound Kepler Orbit System\n"
              << "                               8: Extreme Close-Approach Stress\n"
              << "  -f, --file <path>          Load phase-space coordinates from text file\n"
              << "  -n, --num-particles <N>    Number of particles (default: 1024)\n"
              << "  -t, --dt <float>           Integration time step dt (default: 0.005)\n"
              << "  -s, --softening <float>    Gravitational softening epsilon (default: 0.05)\n"
              << "  --theta <float>            Barnes-Hut opening angle (default: 0.5)\n"
              << "  --integrator <euler|verlet> Numerical integrator (default: verlet)\n"
              << "  --headless                 Run headlessly without interactive GUI window\n"
              << "  --compare-direct           Measure force error and speedup against direct O(N^2) forces\n"
              << "  --reset-after-warmup       Restore the initial state after the untimed warmup step\n"
              << "  --steps <count>            Number of simulation steps in headless mode (default: 100)\n"
              << "  --json-out <path>          Write benchmark telemetry JSON to file\n"
              << "  -h, --help                 Display this help message and exit\n"
              << "=================================================================\n";
}

TransportCliConfig TransportCliConfig::parse(int argc, char** argv) {
    TransportCliConfig cfg;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            std::exit(0);
        } else if ((arg == "-v" || arg == "--version") && i + 1 < argc) {
            int v = std::stoi(argv[++i]);
            if (v < 1 || v > 7) {
                throw std::runtime_error("Invalid solver version: " + std::to_string(v) + " (must be 1..7)");
            }
            cfg.version = static_cast<SolverVersion>(v);
        } else if ((arg == "-i" || arg == "--init") && i + 1 < argc) {
            int init = std::stoi(argv[++i]);
            if (init < 1 || init > 8) {
                throw std::runtime_error("Invalid initializer: " + std::to_string(init) + " (must be 1..8)");
            }
            cfg.init_type = static_cast<InitializerType>(init);
        } else if ((arg == "-f" || arg == "--file") && i + 1 < argc) {
            cfg.custom_file_path = argv[++i];
            cfg.init_type = InitializerType::CustomFile;
        } else if ((arg == "-n" || arg == "--num-particles") && i + 1 < argc) {
            long long n = std::stoll(argv[++i]);
            if (n <= 0) {
                throw std::runtime_error("Particle count must be positive");
            }
            cfg.num_particles = static_cast<size_t>(n);
        } else if ((arg == "-t" || arg == "--dt") && i + 1 < argc) {
            cfg.delta_time = std::stof(argv[++i]);
        } else if ((arg == "-s" || arg == "--softening") && i + 1 < argc) {
            cfg.softening = std::stof(argv[++i]);
        } else if (arg == "--theta" && i + 1 < argc) {
            cfg.theta = std::stof(argv[++i]);
            if (cfg.theta < 0.0f) {
                throw std::runtime_error("Theta must be non-negative");
            }
        } else if (arg == "--integrator" && i + 1 < argc) {
            std::string integ = argv[++i];
            if (integ == "euler") {
                cfg.integrator = IntegratorType::ForwardEuler;
            } else if (integ == "verlet") {
                cfg.integrator = IntegratorType::VelocityVerlet;
            } else {
                throw std::runtime_error("Unknown integrator: " + integ + " (must be 'euler' or 'verlet')");
            }
        } else if (arg == "--headless") {
            cfg.headless = true;
        } else if (arg == "--compare-direct") {
            cfg.compare_direct = true;
        } else if (arg == "--reset-after-warmup") {
            cfg.reset_after_warmup = true;
        } else if (arg == "--steps" && i + 1 < argc) {
            cfg.steps = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--json-out" && i + 1 < argc) {
            cfg.json_output_path = argv[++i];
        } else {
            std::cerr << "Warning: Unrecognized CLI argument: " << arg << "\n";
        }
    }

    return cfg;
}

} // namespace transport
