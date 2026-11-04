#include "transport/types.hpp"
#include "transport/cli_config.hpp"
#include "transport/phase_space_seeder.hpp"
#include "transport/octree_arena.hpp"
#include "transport/solvers.hpp"
#include "transport/gpu_compute_engine.hpp"
#include "transport/gl_renderer.hpp"
#include "transport/physics_diagnostics.hpp"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <numeric>
#include <cmath>
#include <optional>
#include <omp.h>

using namespace transport;

int main(int argc, char** argv) {
    try {
        TransportCliConfig cfg = TransportCliConfig::parse(argc, argv);

        std::cout << "=================================================================\n"
                  << " Octree-Transport-HPC Engine Starting\n"
                  << "=================================================================\n"
                  << " Solver Regime:  " << solverVersionToString(cfg.version) << "\n"
                  << " Integrator:     " << integratorTypeToString(cfg.integrator) << "\n"
                  << " Initializer:    " << initializerTypeToString(cfg.init_type) << "\n"
                  << " Particles (N):  " << cfg.num_particles << "\n"
                  << " Time Step (dt): " << cfg.delta_time << "\n"
                  << " Softening (s):  " << cfg.softening << "\n"
                  << " Opening Angle:  " << cfg.theta << "\n"
                  << " Execution Mode: " << (cfg.headless ? "HEADLESS BENCHMARK" : "INTERACTIVE OPENGL GUI") << "\n"
                  << " OpenMP Threads: " << omp_get_max_threads() << "\n"
                  << "=================================================================\n";

        // 1. Seed phase-space particles
        ParticleSystem ps = PhaseSpaceSeeder::seed(cfg.init_type, cfg.num_particles, cfg.custom_file_path);
        size_t n = ps.size();

        // 2. Initialize GPU engine if needed
        bool needs_gpu = (cfg.version == SolverVersion::GPU_Direct_Naive ||
                          cfg.version == SolverVersion::GPU_Direct_Tiled ||
                          cfg.version == SolverVersion::GPU_BarnesHut_GLSL);

        std::unique_ptr<GpuComputeEngine> gpu_engine;
        if (needs_gpu) {
            gpu_engine = std::make_unique<GpuComputeEngine>(true);
            std::cout << "[GPU Engine] Active Hardware: " << gpu_engine->getRendererInfo() << "\n";
        }

        // 3. Shared Octree Arena for Barnes-Hut solvers
        OctreeArena octree(n);

        double last_tree_build_ms = 0.0;
        double last_force_eval_ms = 0.0;

        auto compute_forces = [&](ParticleSystem& p_sys) {
            auto t0 = std::chrono::high_resolution_clock::now();

            switch (cfg.version) {
                case SolverVersion::CPU_Direct_Seq: {
                    CpuSolvers::computeDirectForcesSeq(p_sys, cfg.softening);
                    auto t1 = std::chrono::high_resolution_clock::now();
                    last_tree_build_ms = 0.0;
                    last_force_eval_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                    break;
                }
                case SolverVersion::CPU_Direct_OMP: {
                    CpuSolvers::computeDirectForcesOMP(p_sys, cfg.softening);
                    auto t1 = std::chrono::high_resolution_clock::now();
                    last_tree_build_ms = 0.0;
                    last_force_eval_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                    break;
                }
                case SolverVersion::GPU_Direct_Naive: {
                    gpu_engine->computeDirectForcesNaiveV3(p_sys, cfg.softening);
                    auto t1 = std::chrono::high_resolution_clock::now();
                    last_tree_build_ms = 0.0;
                    last_force_eval_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                    break;
                }
                case SolverVersion::GPU_Direct_Tiled: {
                    gpu_engine->computeDirectForcesTiledV4(p_sys, cfg.softening);
                    auto t1 = std::chrono::high_resolution_clock::now();
                    last_tree_build_ms = 0.0;
                    last_force_eval_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                    break;
                }
                case SolverVersion::CPU_BarnesHut_Seq: {
                    auto tb0 = std::chrono::high_resolution_clock::now();
                    octree.buildSequential(p_sys);
                    auto tb1 = std::chrono::high_resolution_clock::now();
                    last_tree_build_ms = std::chrono::duration<double, std::milli>(tb1 - tb0).count();

                    auto tf0 = std::chrono::high_resolution_clock::now();
                    CpuSolvers::computeBarnesHutForcesSeq(p_sys, octree, cfg.softening, cfg.theta);
                    auto tf1 = std::chrono::high_resolution_clock::now();
                    last_force_eval_ms = std::chrono::duration<double, std::milli>(tf1 - tf0).count();
                    break;
                }
                case SolverVersion::CPU_BarnesHut_OMP: {
                    auto tb0 = std::chrono::high_resolution_clock::now();
                    octree.buildParallelOMP(p_sys);
                    auto tb1 = std::chrono::high_resolution_clock::now();
                    last_tree_build_ms = std::chrono::duration<double, std::milli>(tb1 - tb0).count();

                    auto tf0 = std::chrono::high_resolution_clock::now();
                    CpuSolvers::computeBarnesHutForcesOMP(p_sys, octree, cfg.softening, cfg.theta);
                    auto tf1 = std::chrono::high_resolution_clock::now();
                    last_force_eval_ms = std::chrono::duration<double, std::milli>(tf1 - tf0).count();
                    break;
                }
                case SolverVersion::GPU_BarnesHut_GLSL: {
                    auto tb0 = std::chrono::high_resolution_clock::now();
                    octree.buildParallelOMP(p_sys);
                    auto tb1 = std::chrono::high_resolution_clock::now();
                    last_tree_build_ms = std::chrono::duration<double, std::milli>(tb1 - tb0).count();

                    auto tf0 = std::chrono::high_resolution_clock::now();
                    gpu_engine->computeBarnesHutForcesV7(p_sys, octree, cfg.softening, cfg.theta);
                    auto tf1 = std::chrono::high_resolution_clock::now();
                    last_force_eval_ms = std::chrono::duration<double, std::milli>(tf1 - tf0).count();
                    break;
                }
            }
        };

        // Step function dispatching chosen integrator
        auto step_simulation = [&]() {
            if (cfg.integrator == IntegratorType::ForwardEuler) {
                CpuSolvers::stepForwardEuler(ps, cfg.delta_time, compute_forces);
            } else {
                CpuSolvers::stepVelocityVerlet(ps, cfg.delta_time, compute_forces);
            }
        };

        // 4. Initial acceleration calculation & initial energy
        compute_forces(ps);
        EnergyMetrics init_metrics = PhysicsDiagnostics::computeMetrics(ps, cfg.softening);
        double e0 = init_metrics.total_energy;
        glm::dvec3 p0 = init_metrics.total_linear_momentum;
        glm::dvec3 l0 = init_metrics.total_angular_momentum;

        std::optional<double> relative_l2_force_error;
        double direct_force_ms = 0.0;
        if (cfg.compare_direct) {
            ParticleSystem ps_direct = ps;
            const auto direct_start = std::chrono::high_resolution_clock::now();
            CpuSolvers::computeDirectForcesOMP(ps_direct, cfg.softening);
            const auto direct_end = std::chrono::high_resolution_clock::now();
            direct_force_ms = std::chrono::duration<double, std::milli>(direct_end - direct_start).count();
            relative_l2_force_error = PhysicsDiagnostics::compareForces(ps_direct, ps).relative_l2_error;
        }

        const ParticleSystem initial_state = ps;

        std::cout << std::fixed << std::setprecision(6)
                  << "[Initial State] E0 = " << e0
                  << " (K0 = " << init_metrics.kinetic_energy
                  << ", U0 = " << init_metrics.potential_energy << ")\n"
                  << "                |P0| = " << glm::length(p0) << "\n";

        // 5. Execution branch
        if (cfg.headless) {
            std::cout << "[Headless] Executing " << cfg.steps << " benchmark steps...\n";

            std::vector<double> step_times_ms;
            step_times_ms.reserve(cfg.steps);
            std::vector<double> tree_times_ms;
            tree_times_ms.reserve(cfg.steps);
            std::vector<double> force_times_ms;
            force_times_ms.reserve(cfg.steps);
            std::vector<double> energy_history;
            energy_history.reserve(cfg.steps);

            double max_rel_drift = 0.0;
            double max_momentum_drift = 0.0;
            double max_angular_momentum_drift = 0.0;
            double final_energy = e0;
            double last_sampled_energy_drift = 0.0;

            std::cout << "[Headless] Running one untimed warmup step...\n";
            step_simulation();
            if (cfg.reset_after_warmup) {
                ps = initial_state;
                compute_forces(ps);
            }

            for (uint32_t step = 0; step < cfg.steps; ++step) {
                auto t_start = std::chrono::high_resolution_clock::now();
                step_simulation();
                auto t_end = std::chrono::high_resolution_clock::now();

                double dt_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
                step_times_ms.push_back(dt_ms);
                tree_times_ms.push_back(last_tree_build_ms);
                force_times_ms.push_back(last_force_eval_ms);

                const bool sample_energy = n <= 16384 || step == 0 || step == cfg.steps - 1;
                double rel_drift = last_sampled_energy_drift;
                if (sample_energy) {
                    EnergyMetrics em = PhysicsDiagnostics::computeMetrics(ps, cfg.softening);
                    final_energy = em.total_energy;
                    energy_history.push_back(em.total_energy);
                    rel_drift = std::abs((em.total_energy - e0) / (std::abs(e0) > 1.0e-7 ? e0 : 1.0));
                    last_sampled_energy_drift = rel_drift;
                    max_rel_drift = std::max(max_rel_drift, rel_drift);
                }

                glm::dvec3 momentum = PhysicsDiagnostics::computeTotalLinearMomentum(ps);
                glm::dvec3 angular_momentum = PhysicsDiagnostics::computeTotalAngularMomentum(ps);
                double p_drift = glm::length(momentum - p0);
                max_momentum_drift = std::max(max_momentum_drift, p_drift);
                double l_drift = glm::length(angular_momentum - l0);
                max_angular_momentum_drift = std::max(max_angular_momentum_drift, l_drift);

                if (cfg.steps <= 10 || (step + 1) % (cfg.steps / 10) == 0 || step == cfg.steps - 1) {
                    std::cout << "Step " << std::setw(5) << (step + 1) << " / " << cfg.steps
                              << " | Latency: " << std::setw(7) << std::setprecision(2) << dt_ms << " ms"
                              << " (Tree: " << last_tree_build_ms << " ms, Force: " << last_force_eval_ms << " ms)"
                              << " | |dE/E0|: " << std::scientific << std::setprecision(3) << rel_drift
                              << std::fixed << "\n";
                }
            }

            // Statistical calculations
            double sum_ms = std::accumulate(step_times_ms.begin(), step_times_ms.end(), 0.0);
            double mean_ms = sum_ms / static_cast<double>(cfg.steps);

            std::vector<double> sorted_times = step_times_ms;
            std::sort(sorted_times.begin(), sorted_times.end());
            double median_ms = sorted_times[sorted_times.size() / 2];
            double p95_ms = sorted_times[static_cast<size_t>(sorted_times.size() * 0.95)];

            double variance = 0.0;
            for (double t : step_times_ms) variance += (t - mean_ms) * (t - mean_ms);
            double std_ms = std::sqrt(variance / static_cast<double>(cfg.steps));

            double mean_tree_ms = std::accumulate(tree_times_ms.begin(), tree_times_ms.end(), 0.0) / cfg.steps;
            double mean_force_ms = std::accumulate(force_times_ms.begin(), force_times_ms.end(), 0.0) / cfg.steps;

            double m_interactions_per_sec = (static_cast<double>(n) * static_cast<double>(n)) / (mean_ms * 1.0e3);
            double k_particles_per_sec = (static_cast<double>(n)) / (mean_ms * 1.0e-3);

            std::cout << "\n=================================================================\n"
                      << " Headless Benchmark Summary\n"
                      << "=================================================================\n"
                      << " Total Steps:        " << cfg.steps << "\n"
                      << " Mean Step Latency:  " << mean_ms << " ms (Std: " << std_ms << " ms)\n"
                      << " Median Step:        " << median_ms << " ms | P95: " << p95_ms << " ms\n"
                      << " Breakdown:          Tree: " << mean_tree_ms << " ms | Force: " << mean_force_ms << " ms\n"
                      << " Throughput:         " << k_particles_per_sec << " particles/s ("
                      << m_interactions_per_sec << " M interactions/s eq.)\n"
                      << " Initial Energy E0:  " << e0 << "\n"
                      << " Final Energy ET:    " << final_energy << "\n"
                      << " Max |dE/E0| Drift:  " << std::scientific << max_rel_drift << std::fixed << "\n"
                      << " Max |dP| Drift:     " << std::scientific << max_momentum_drift << std::fixed << "\n"
                      << " Max |dL| Drift:     " << std::scientific << max_angular_momentum_drift << std::fixed << "\n"
                      << " Octree Nodes Count: " << octree.getNodeCount() << "\n"
                      << " Octree Max Depth:   " << octree.getMaxDepth() << "\n"
                      << "=================================================================\n";

            // Optional JSON output
            if (!cfg.json_output_path.empty()) {
                std::ofstream jout(cfg.json_output_path);
                if (jout.is_open()) {
                    jout << std::setprecision(17)
                         << "{\n"
                         << "  \"solver_version\": \"" << solverVersionToString(cfg.version) << "\",\n"
                         << "  \"integrator\": \"" << integratorTypeToString(cfg.integrator) << "\",\n"
                         << "  \"num_particles\": " << n << ",\n"
                         << "  \"dt\": " << cfg.delta_time << ",\n"
                         << "  \"softening\": " << cfg.softening << ",\n"
                         << "  \"theta\": " << cfg.theta << ",\n"
                         << "  \"steps\": " << cfg.steps << ",\n"
                         << "  \"mean_step_ms\": " << mean_ms << ",\n"
                         << "  \"median_step_ms\": " << median_ms << ",\n"
                         << "  \"p95_step_ms\": " << p95_ms << ",\n"
                         << "  \"std_step_ms\": " << std_ms << ",\n"
                         << "  \"mean_tree_ms\": " << mean_tree_ms << ",\n"
                         << "  \"mean_force_ms\": " << mean_force_ms << ",\n"
                         << "  \"particles_per_sec\": " << k_particles_per_sec << ",\n"
                         << "  \"openmp_max_threads\": " << omp_get_max_threads() << ",\n"
                         << "  \"energy_drift_sampling\": \""
                         << (n <= 16384 ? "every_timed_step" : "first_and_final_timed_steps") << "\",\n"
                         << "  \"initial_energy\": " << e0 << ",\n"
                         << "  \"final_energy\": " << final_energy << ",\n"
                         << "  \"max_energy_drift\": " << max_rel_drift << ",\n"
                         << "  \"max_relative_energy_drift\": " << max_rel_drift << ",\n"
                         << "  \"max_momentum_drift\": " << max_momentum_drift << ",\n"
                         << "  \"max_angular_momentum_drift\": " << max_angular_momentum_drift << ",\n"
                         << "  \"octree_nodes\": " << octree.getNodeCount() << ",\n"
                         << "  \"octree_max_depth\": " << octree.getMaxDepth() << ",\n"
                         << "  \"renderer\": \"" << (needs_gpu ? gpu_engine->getRendererInfo() : "CPU OpenMP") << "\",\n"
                         << "  \"relative_l2_force_error\": ";
                    if (relative_l2_force_error.has_value()) jout << *relative_l2_force_error;
                    else jout << "null";
                    jout << ",\n"
                         << "  \"direct_force_ms\": " << direct_force_ms << ",\n"
                         << "  \"speedup_vs_direct\": ";
                    if (direct_force_ms > 0.0 && mean_force_ms > 0.0) jout << direct_force_ms / mean_force_ms;
                    else jout << "null";
                    jout << ",\n"
                         << "  \"coincident_merges\": " << octree.getCoincidentMerges() << ",\n"
                         << "  \"octree_depth_histogram\": [";
                    const auto depth_histogram = octree.computeDepthHistogram();
                    for (size_t i = 0; i < depth_histogram.size(); ++i) {
                        jout << depth_histogram[i] << (i + 1 < depth_histogram.size() ? ", " : "");
                    }
                    jout << "],\n"
                         << "  \"energy_history\": [";
                    for (size_t i = 0; i < energy_history.size(); ++i) {
                        jout << energy_history[i] << (i + 1 < energy_history.size() ? ", " : "");
                    }
                    jout << "]\n}\n";
                    std::cout << "[Output] Telemetry JSON written to " << cfg.json_output_path << "\n";
                }
            }
        } else {
            // Interactive GUI Mode
            GlRenderer renderer(1280, 720, "Octree-Transport-HPC | Barnes-Hut Engine");
            std::cout << "[Interactive Mode] Running. Controls: Space = Pause, Q = Point Size, Esc = Quit, Drag = Orbit\n";

            while (!renderer.shouldClose()) {
                renderer.pollEvents();

                if (!renderer.isPaused()) {
                    step_simulation();
                }

                renderer.renderFrame(ps);
                renderer.swapBuffers();
            }
        }

    } catch (const std::exception& ex) {
        std::cerr << "[FATAL ERROR] " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
