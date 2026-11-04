#pragma once

#include "types.hpp"
#include <string>

namespace transport {

struct TransportCliConfig {
    SolverVersion version{SolverVersion::CPU_Direct_OMP};
    InitializerType init_type{InitializerType::SolidCube};
    std::string custom_file_path{};
    size_t num_particles{1024};
    float delta_time{0.005f};
    float softening{0.05f};
    float theta{0.5f};
    IntegratorType integrator{IntegratorType::VelocityVerlet};
    bool headless{false};
    bool compare_direct{false};
    bool reset_after_warmup{false};
    uint32_t steps{100};
    std::string json_output_path{};

    static TransportCliConfig parse(int argc, char** argv);
    static void printUsage(const char* prog_name);
};

} // namespace transport
