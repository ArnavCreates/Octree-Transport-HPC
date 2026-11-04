#include <gtest/gtest.h>
#include "transport/types.hpp"
#include "transport/cli_config.hpp"
#include "transport/phase_space_seeder.hpp"
#include "transport/octree_arena.hpp"
#include "transport/solvers.hpp"
#include "transport/gpu_compute_engine.hpp"
#include "transport/physics_diagnostics.hpp"

#include <cmath>
#include <vector>
#include <iostream>
#include <memory>
#include <string>

using namespace transport;

#define CREATE_GPU_OR_SKIP(gpu_name) \
    std::unique_ptr<GpuComputeEngine> gpu_name; \
    try { \
        gpu_name = std::make_unique<GpuComputeEngine>(true); \
    } catch (const std::exception& ex) { \
        const std::string message = ex.what(); \
        if (message.find("OpenGL context unavailable") != std::string::npos) { \
            GTEST_SKIP() << "Skipping GPU compute shader assertions: no OpenGL 4.3+ context available on this headless host"; \
        } \
        throw; \
    } \
    if (!gpu_name->isInitialized()) \
        GTEST_SKIP() << "Skipping GPU compute shader assertions: no OpenGL 4.3+ context available on this headless host"

// 1. CliConfigParsingTest
TEST(OctreeTransportTest, CliConfigParsingTest) {
    const char* argv[] = {
        "octree_transport_hpc",
        "-v", "6",
        "-i", "7",
        "-n", "2048",
        "-t", "0.002",
        "-s", "0.025",
        "--theta", "0.6",
        "--integrator", "verlet",
        "--headless",
        "--steps", "250",
        "--json-out", "bench_test.json",
        "--compare-direct",
        "--reset-after-warmup"
    };
    int argc = sizeof(argv) / sizeof(argv[0]);

    TransportCliConfig cfg = TransportCliConfig::parse(argc, const_cast<char**>(argv));

    EXPECT_EQ(cfg.version, SolverVersion::CPU_BarnesHut_OMP);
    EXPECT_EQ(cfg.init_type, InitializerType::BoundKeplerOrbit);
    EXPECT_EQ(cfg.num_particles, 2048u);
    EXPECT_NEAR(cfg.delta_time, 0.002f, 1e-6f);
    EXPECT_NEAR(cfg.softening, 0.025f, 1e-6f);
    EXPECT_NEAR(cfg.theta, 0.6f, 1e-6f);
    EXPECT_EQ(cfg.integrator, IntegratorType::VelocityVerlet);
    EXPECT_TRUE(cfg.headless);
    EXPECT_EQ(cfg.steps, 250u);
    EXPECT_EQ(cfg.json_output_path, "bench_test.json");
    EXPECT_TRUE(cfg.compare_direct);
    EXPECT_TRUE(cfg.reset_after_warmup);

    // Test Euler integrator and short flags
    const char* argv2[] = {
        "octree_transport_hpc",
        "--version", "2",
        "--init", "1",
        "--integrator", "euler"
    };
    int argc2 = sizeof(argv2) / sizeof(argv2[0]);
    TransportCliConfig cfg2 = TransportCliConfig::parse(argc2, const_cast<char**>(argv2));
    EXPECT_EQ(cfg2.version, SolverVersion::CPU_Direct_OMP);
    EXPECT_EQ(cfg2.init_type, InitializerType::SolidCube);
    EXPECT_EQ(cfg2.integrator, IntegratorType::ForwardEuler);
}

// 2. DirectSolversExactParityTest
TEST(OctreeTransportTest, DirectSolversExactParityTest) {
    const size_t N = 512;
    const float softening = 0.05f;

    ParticleSystem ps_ref = PhaseSpaceSeeder::seedSolidCube(N, 2.0f);
    ParticleSystem ps_omp = ps_ref;

    // -v 1: CPU Sequential
    CpuSolvers::computeDirectForcesSeq(ps_ref, softening);

    // -v 2: CPU OpenMP
    CpuSolvers::computeDirectForcesOMP(ps_omp, softening);

    // Verify CPU OpenMP parity against -v 1.
    for (size_t i = 0; i < N; ++i) {
        glm::vec3 a_ref(ps_ref.acc_pad[i]);
        float ref_mag = glm::length(a_ref);
        float norm_denom = std::max(ref_mag, 1.0e-5f);

        // CPU OMP parity
        glm::vec3 a_omp(ps_omp.acc_pad[i]);
        float diff_omp = glm::length(a_omp - a_ref) / norm_denom;
        EXPECT_LT(diff_omp, 1.0e-4f) << "OMP parity mismatch at index " << i;
    }
}

TEST(OctreeTransportTest, GpuDirectSolversParityTest) {
    const size_t N = 512;
    const float softening = 0.05f;
    ParticleSystem ps_ref = PhaseSpaceSeeder::seedSolidCube(N, 2.0f);
    ParticleSystem ps_gpu_naive = ps_ref;
    ParticleSystem ps_gpu_tiled = ps_ref;
    CpuSolvers::computeDirectForcesSeq(ps_ref, softening);

    CREATE_GPU_OR_SKIP(gpu);
    gpu->computeDirectForcesNaiveV3(ps_gpu_naive, softening);
    gpu->computeDirectForcesTiledV4(ps_gpu_tiled, softening);

    for (size_t i = 0; i < N; ++i) {
        glm::vec3 a_ref(ps_ref.acc_pad[i]);
        float norm_denom = std::max(glm::length(a_ref), 1.0e-5f);
        glm::vec3 a_gpu_naive(ps_gpu_naive.acc_pad[i]);
        float diff_gpu_naive = glm::length(a_gpu_naive - a_ref) / norm_denom;
        EXPECT_LT(diff_gpu_naive, 1.0e-4f) << "GPU Naive parity mismatch at index " << i;

        glm::vec3 a_gpu_tiled(ps_gpu_tiled.acc_pad[i]);
        float diff_gpu_tiled = glm::length(a_gpu_tiled - a_ref) / norm_denom;
        EXPECT_LT(diff_gpu_tiled, 1.0e-4f) << "GPU Tiled parity mismatch at index " << i;
    }
}

// 3. BarnesHutMultipoleConvergenceTest
TEST(OctreeTransportTest, BarnesHutMultipoleConvergenceTest) {
    const size_t N = 512;
    const float softening = 0.05f;

    ParticleSystem ps_exact = PhaseSpaceSeeder::seedRotatingDisk(N, 0.5f, 3.0f);
    CpuSolvers::computeDirectForcesSeq(ps_exact, softening);

    // Case A: Theta -> 0.0 (exact multipole limit)
    OctreeArena arena_zero(N);
    arena_zero.buildSequential(ps_exact);
    ParticleSystem ps_theta_zero = ps_exact;
    CpuSolvers::computeBarnesHutForcesSeq(ps_theta_zero, arena_zero, softening, 0.0f);

    auto metrics_zero = PhysicsDiagnostics::compareForces(ps_exact, ps_theta_zero);
    EXPECT_LT(metrics_zero.relative_l2_error, 1.0e-4)
        << "Barnes-Hut with theta=0.0 must converge to exact O(N^2) force";

    // Case B: Standard opening angle theta = 0.5
    OctreeArena arena_half(N);
    arena_half.buildSequential(ps_exact);
    ParticleSystem ps_theta_half = ps_exact;
    CpuSolvers::computeBarnesHutForcesSeq(ps_theta_half, arena_half, softening, 0.5f);

    auto metrics_half = PhysicsDiagnostics::compareForces(ps_exact, ps_theta_half);
    EXPECT_LT(metrics_half.relative_l2_error, 0.015)
        << "Barnes-Hut with theta=0.5 must achieve < 1.5% relative L2 error against exact force";
    EXPECT_GT(metrics_half.relative_l2_error, 0.0)
        << "Multipole approximation introduces small controlled non-zero truncation error";
}

// 4. ParallelOctreeDeterministicParityTest
TEST(OctreeTransportTest, ParallelOctreeDeterministicParityTest) {
    const size_t N = 1024;
    const float softening = 0.05f;
    const float theta = 0.5f;

    ParticleSystem ps_base = PhaseSpaceSeeder::seedPlummerBall(N, 1.5f);

    // -v 5: Sequential CPU Barnes-Hut
    OctreeArena arena_seq(N);
    arena_seq.buildSequential(ps_base);
    ParticleSystem ps_seq = ps_base;
    CpuSolvers::computeBarnesHutForcesSeq(ps_seq, arena_seq, softening, theta);

    // -v 6: OpenMP Disjoint Octant Barnes-Hut
    OctreeArena arena_omp(N);
    arena_omp.buildParallelOMP(ps_base);
    EXPECT_EQ(arena_omp.getNodeCount(), arena_omp.getNodes().size());
    EXPECT_LT(arena_omp.getNodeCount(), 8 * N + 4096);
    EXPECT_EQ(arena_omp.computeDepthHistogram().size(), MAX_OCTREE_DEPTH + 1);
    ParticleSystem ps_omp = ps_base;
    CpuSolvers::computeBarnesHutForcesOMP(ps_omp, arena_omp, softening, theta);

    auto metrics_omp = PhysicsDiagnostics::compareForces(ps_seq, ps_omp);
    EXPECT_LT(metrics_omp.relative_l2_error, 1.0e-4)
        << "Parallel OpenMP Barnes-Hut (-v 6) must match Sequential (-v 5) within 1e-4 relative L2 error";

}

TEST(OctreeTransportTest, GpuBarnesHutParityTest) {
    const size_t N = 1024;
    const float softening = 0.05f;
    const float theta = 0.5f;
    ParticleSystem ps_base = PhaseSpaceSeeder::seedPlummerBall(N, 1.5f);

    OctreeArena arena_seq(N);
    arena_seq.buildSequential(ps_base);
    ParticleSystem ps_seq = ps_base;
    CpuSolvers::computeBarnesHutForcesSeq(ps_seq, arena_seq, softening, theta);

    OctreeArena arena_omp(N);
    arena_omp.buildParallelOMP(ps_base);
    CREATE_GPU_OR_SKIP(gpu);

    ParticleSystem ps_gpu = ps_base;
    gpu->computeBarnesHutForcesV7(ps_gpu, arena_omp, softening, theta);
    auto metrics_gpu = PhysicsDiagnostics::compareForces(ps_seq, ps_gpu);
    EXPECT_LT(metrics_gpu.relative_l2_error, 1.0e-4)
        << "GPU Stackless Barnes-Hut (-v 7) must match CPU reference (-v 5) within 1e-4 relative L2 error";
}

// 5. CloseApproachSubdivisionGuardTest
TEST(OctreeTransportTest, CloseApproachSubdivisionGuardTest) {
    const size_t N = 2000;
    const float softening = 0.05f;
    const float theta = 0.5f;

    // Seeds 50 exact coincident pairs (dr = 0) and 50 ultra-close pairs (dr = 1e-9)
    ParticleSystem ps = PhaseSpaceSeeder::seedCloseApproachStress(N);
    EXPECT_EQ(glm::length(glm::vec3(ps.pos_mass[0]) - glm::vec3(ps.pos_mass[1])), 0.0f);
    const float injected_pair_separation = glm::length(
        glm::vec3(ps.pos_mass[101]) - glm::vec3(ps.pos_mass[100]));
    EXPECT_NEAR(injected_pair_separation, 1.0e-9f, 1.0e-10f);

    float expected_total_mass = 0.0f;
    for (size_t i = 0; i < N; ++i) {
        expected_total_mass += ps.pos_mass[i].w;
    }

    // Build Sequential (-v 5)
    OctreeArena arena_seq(N);
    EXPECT_NO_THROW(arena_seq.buildSequential(ps));
    EXPECT_LE(arena_seq.getMaxDepth(), MAX_OCTREE_DEPTH);
    EXPECT_GT(arena_seq.getMaxDepth(), 1u);
    EXPECT_NEAR(arena_seq.getTotalMass(), expected_total_mass, 1.0e-4f);
    EXPECT_GT(arena_seq.getCoincidentMerges(), 0u);

    // Build Parallel OpenMP (-v 6)
    OctreeArena arena_omp(N);
    EXPECT_NO_THROW(arena_omp.buildParallelOMP(ps));
    EXPECT_LE(arena_omp.getMaxDepth(), MAX_OCTREE_DEPTH);
    EXPECT_GT(arena_omp.getMaxDepth(), 1u);
    EXPECT_NEAR(arena_omp.getTotalMass(), expected_total_mass, 1.0e-4f);
    EXPECT_GT(arena_omp.getCoincidentMerges(), 0u);
    EXPECT_EQ(arena_omp.getNodeCount(), arena_omp.getNodes().size());
    const auto histogram = arena_omp.computeDepthHistogram();
    EXPECT_GT(histogram[0], 0u);
    EXPECT_GT(histogram[arena_omp.getMaxDepth()], 0u);
    for (int octant = 0; octant < 8; ++octant) {
        const auto& seq_root = arena_seq.getNodes()[static_cast<size_t>(octant + 1)];
        const auto& omp_root = arena_omp.getNodes()[static_cast<size_t>(octant + 1)];
        EXPECT_NEAR(seq_root.com_mass.w, omp_root.com_mass.w, 1.0e-5f);
        EXPECT_LT(glm::length(glm::vec3(seq_root.com_mass) - glm::vec3(omp_root.com_mass)), 1.0e-5f);
    }

    ParticleSystem ps_seq_forces = ps;
    ParticleSystem ps_omp_forces = ps;
    CpuSolvers::computeBarnesHutForcesSeq(ps_seq_forces, arena_seq, softening, 0.0f);
    CpuSolvers::computeBarnesHutForcesOMP(ps_omp_forces, arena_omp, softening, 0.0f);
    const auto stress_force_parity = PhysicsDiagnostics::compareForces(ps_seq_forces, ps_omp_forces);
    ParticleSystem ps_direct = ps;
    CpuSolvers::computeDirectForcesOMP(ps_direct, softening);
    const auto parallel_force_accuracy = PhysicsDiagnostics::compareForces(ps_direct, ps_omp_forces);
    EXPECT_LT(stress_force_parity.relative_l2_error, 1.0e-4);
    EXPECT_LT(parallel_force_accuracy.relative_l2_error, 1.0e-3);

    // Evaluate forces and step simulation - must produce finite numbers without NaN/Inf
    ParticleSystem ps_test = ps;
    EXPECT_NO_THROW(CpuSolvers::computeBarnesHutForcesOMP(ps_test, arena_omp, softening, theta));

    for (size_t i = 0; i < N; ++i) {
        EXPECT_FALSE(std::isnan(ps_test.acc_pad[i].x));
        EXPECT_FALSE(std::isnan(ps_test.acc_pad[i].y));
        EXPECT_FALSE(std::isnan(ps_test.acc_pad[i].z));
        EXPECT_FALSE(std::isinf(ps_test.acc_pad[i].x));
        EXPECT_FALSE(std::isinf(ps_test.acc_pad[i].y));
        EXPECT_FALSE(std::isinf(ps_test.acc_pad[i].z));
    }

}

TEST(OctreeTransportTest, GpuBarnesHutCloseApproachStressTest) {
    const size_t N = 2000;
    const float softening = 0.05f;
    const float theta = 0.5f;
    ParticleSystem ps = PhaseSpaceSeeder::seedCloseApproachStress(N);
    OctreeArena arena_omp(N);
    arena_omp.buildParallelOMP(ps);

    CREATE_GPU_OR_SKIP(gpu);
    ParticleSystem ps_gpu = ps;
    EXPECT_NO_THROW(gpu->computeBarnesHutForcesV7(ps_gpu, arena_omp, softening, theta));

    for (size_t i = 0; i < N; ++i) {
        EXPECT_FALSE(std::isnan(ps_gpu.acc_pad[i].x));
        EXPECT_FALSE(std::isnan(ps_gpu.acc_pad[i].y));
        EXPECT_FALSE(std::isnan(ps_gpu.acc_pad[i].z));
    }
}

// 6. SymplecticVelocityVerletVsEulerEnergyTest
TEST(OctreeTransportTest, SymplecticVelocityVerletVsEulerEnergyTest) {
    const size_t N = 256;
    const float dt = 0.002f;
    const float softening = 0.05f;
    const uint32_t STEPS = 500;

    ParticleSystem ps_initial = PhaseSpaceSeeder::seedBoundKeplerSystem(N);

    // Energy at t = 0
    EnergyMetrics em0 = PhysicsDiagnostics::computeMetrics(ps_initial, softening);
    double E0 = em0.total_energy;
    EXPECT_LT(E0, 0.0) << "Keplerian bound system must have strictly negative total energy E0 < 0";

    // Run Forward Euler
    ParticleSystem ps_euler = ps_initial;
    CpuSolvers::computeDirectForcesOMP(ps_euler, softening);
    auto compute_forces_euler = [&](ParticleSystem& p) {
        CpuSolvers::computeDirectForcesOMP(p, softening);
    };

    for (uint32_t step = 0; step < STEPS; ++step) {
        CpuSolvers::stepForwardEuler(ps_euler, dt, compute_forces_euler);
    }
    EnergyMetrics em_euler = PhysicsDiagnostics::computeMetrics(ps_euler, softening);
    double drift_euler = std::abs((em_euler.total_energy - E0) / E0);

    // Run Velocity Verlet
    ParticleSystem ps_verlet = ps_initial;
    CpuSolvers::computeDirectForcesOMP(ps_verlet, softening);
    auto compute_forces_verlet = [&](ParticleSystem& p) {
        CpuSolvers::computeDirectForcesOMP(p, softening);
    };

    for (uint32_t step = 0; step < STEPS; ++step) {
        CpuSolvers::stepVelocityVerlet(ps_verlet, dt, compute_forces_verlet);
    }
    EnergyMetrics em_verlet = PhysicsDiagnostics::computeMetrics(ps_verlet, softening);
    double drift_verlet = std::abs((em_verlet.total_energy - E0) / E0);

    std::cout << "[Symplectic Verification] Initial Energy E0:  " << E0 << "\n"
              << "                          Euler Drift:       " << drift_euler << "\n"
              << "                          Verlet Drift:      " << drift_verlet << "\n"
              << "                          Drift Ratio (E/V): " << (drift_euler / std::max(drift_verlet, 1.0e-12)) << "x\n";

    // Assert that Velocity Verlet reduces energy drift by at least 20x compared to Forward Euler
    EXPECT_LT(drift_verlet, 1.0e-3) << "Velocity Verlet energy drift must be tightly bounded (< 1e-3)";
    EXPECT_GT(drift_euler, 20.0 * drift_verlet)
        << "Velocity Verlet must outperform Forward Euler energy conservation by at least 20x";
}
