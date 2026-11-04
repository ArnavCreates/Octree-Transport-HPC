#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <glm/glm.hpp>

namespace transport {

constexpr float G_CONSTANT = 1.0f;
constexpr uint32_t MAX_OCTREE_DEPTH = 20;
constexpr float MIN_CELL_HALF_EXTENT = 1.0e-5f;

enum class SolverVersion : int {
    CPU_Direct_Seq = 1,
    CPU_Direct_OMP = 2,
    GPU_Direct_Naive = 3,
    GPU_Direct_Tiled = 4,
    CPU_BarnesHut_Seq = 5,
    CPU_BarnesHut_OMP = 6,
    GPU_BarnesHut_GLSL = 7
};

inline const char* solverVersionToString(SolverVersion v) {
    switch (v) {
        case SolverVersion::CPU_Direct_Seq: return "CPU_Direct_Seq (v1)";
        case SolverVersion::CPU_Direct_OMP: return "CPU_Direct_OMP (v2)";
        case SolverVersion::GPU_Direct_Naive: return "GPU_Direct_Naive (v3)";
        case SolverVersion::GPU_Direct_Tiled: return "GPU_Direct_Tiled (v4)";
        case SolverVersion::CPU_BarnesHut_Seq: return "CPU_BarnesHut_Seq (v5)";
        case SolverVersion::CPU_BarnesHut_OMP: return "CPU_BarnesHut_OMP (v6)";
        case SolverVersion::GPU_BarnesHut_GLSL: return "GPU_BarnesHut_GLSL (v7)";
        default: return "Unknown";
    }
}

enum class IntegratorType : int {
    ForwardEuler = 0,
    VelocityVerlet = 1
};

inline const char* integratorTypeToString(IntegratorType t) {
    switch (t) {
        case IntegratorType::ForwardEuler: return "ForwardEuler";
        case IntegratorType::VelocityVerlet: return "VelocityVerlet";
        default: return "Unknown";
    }
}

enum class InitializerType : int {
    SolidCube = 1,
    RotatingDisk = 2,
    LagrangeTriangle = 3,
    SphericalShell = 4,
    PlummerBall = 5,
    CubicShell = 6,
    BoundKeplerOrbit = 7,
    CloseApproachStress = 8,
    CustomFile = 9
};

inline const char* initializerTypeToString(InitializerType t) {
    switch (t) {
        case InitializerType::SolidCube: return "SolidCube (i1)";
        case InitializerType::RotatingDisk: return "RotatingDisk (i2)";
        case InitializerType::LagrangeTriangle: return "LagrangeTriangle (i3)";
        case InitializerType::SphericalShell: return "SphericalShell (i4)";
        case InitializerType::PlummerBall: return "PlummerBall (i5)";
        case InitializerType::CubicShell: return "CubicShell (i6)";
        case InitializerType::BoundKeplerOrbit: return "BoundKeplerOrbit (i7)";
        case InitializerType::CloseApproachStress: return "CloseApproachStress (i8)";
        case InitializerType::CustomFile: return "CustomFile (-f)";
        default: return "Unknown";
    }
}

// Memory layout aligned to 16 bytes for seamless std430 SSBO interoperability
struct alignas(16) FlatOctreeNode {
    glm::vec4 com_mass{0.0f, 0.0f, 0.0f, 0.0f};          // xyz = center of mass, w = total mass M
    glm::vec4 bbox_center_half{0.0f, 0.0f, 0.0f, 0.0f};  // xyz = cell center, w = cell half-width s/2
    int32_t first_child_idx{-1};                         // -1 if leaf
    int32_t next_node_idx{-1};                           // stackless skip-rope index (-1 = traversal end)
    int32_t particle_idx{-1};                            // >= 0 single particle, -2 coincident/aggregated, -1 internal
    int32_t depth{0};                                    // Tree depth
};
static_assert(sizeof(FlatOctreeNode) == 48, "FlatOctreeNode must match std430 48-byte layout");

struct ParticleSystem {
    std::vector<glm::vec4> pos_mass; // xyz = position, w = mass
    std::vector<glm::vec4> vel_pad;  // xyz = velocity, w = 0.0f
    std::vector<glm::vec4> acc_pad;  // xyz = acceleration, w = 0.0f

    size_t size() const { return pos_mass.size(); }
    void resize(size_t n) {
        pos_mass.resize(n, glm::vec4(0.0f));
        vel_pad.resize(n, glm::vec4(0.0f));
        acc_pad.resize(n, glm::vec4(0.0f));
    }
    void clear() {
        pos_mass.clear();
        vel_pad.clear();
        acc_pad.clear();
    }
};

struct EnergyMetrics {
    double kinetic_energy{0.0};
    double potential_energy{0.0};
    double total_energy{0.0};
    glm::dvec3 total_linear_momentum{0.0};
    glm::dvec3 total_angular_momentum{0.0};
};

struct StepTelemetry {
    double step_wall_ms{0.0};
    double tree_build_ms{0.0};
    double force_eval_ms{0.0};
    double integration_ms{0.0};
    EnergyMetrics energy;
    size_t octree_nodes_count{0};
    uint32_t octree_max_depth{0};
};

} // namespace transport
