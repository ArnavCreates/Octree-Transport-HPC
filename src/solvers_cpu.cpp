#include "transport/solvers.hpp"
#include <cmath>
#include <omp.h>

namespace transport {

void CpuSolvers::computeDirectForcesSeq(ParticleSystem& ps, float softening) {
    size_t n = ps.size();
    float eps_sq = softening * softening;

    for (size_t i = 0; i < n; ++i) {
        glm::vec3 pi(ps.pos_mass[i]);
        glm::vec3 acc(0.0f);

        for (size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            glm::vec3 r = glm::vec3(ps.pos_mass[j]) - pi;
            float dist_sq = glm::dot(r, r) + eps_sq;
            float inv_dist = 1.0f / (dist_sq * std::sqrt(dist_sq));
            acc += r * (G_CONSTANT * ps.pos_mass[j].w * inv_dist);
        }

        ps.acc_pad[i] = glm::vec4(acc, 0.0f);
    }
}

void CpuSolvers::computeDirectForcesOMP(ParticleSystem& ps, float softening) {
    size_t n = ps.size();
    float eps_sq = softening * softening;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        glm::vec3 pi(ps.pos_mass[i]);
        glm::vec3 acc(0.0f);

        for (size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            glm::vec3 r = glm::vec3(ps.pos_mass[j]) - pi;
            float dist_sq = glm::dot(r, r) + eps_sq;
            float inv_dist = 1.0f / (dist_sq * std::sqrt(dist_sq));
            acc += r * (G_CONSTANT * ps.pos_mass[j].w * inv_dist);
        }

        ps.acc_pad[i] = glm::vec4(acc, 0.0f);
    }
}

void CpuSolvers::computeBarnesHutForcesSeq(ParticleSystem& ps, const OctreeArena& arena, float softening, float theta) {
    size_t n = ps.size();
    float eps_sq = softening * softening;
    float theta_sq = theta * theta;

    for (size_t i = 0; i < n; ++i) {
        glm::vec3 pi(ps.pos_mass[i]);
        glm::vec3 acc = arena.computeForce(pi, eps_sq, theta_sq);
        ps.acc_pad[i] = glm::vec4(acc, 0.0f);
    }
}

void CpuSolvers::computeBarnesHutForcesOMP(ParticleSystem& ps, const OctreeArena& arena, float softening, float theta) {
    size_t n = ps.size();
    float eps_sq = softening * softening;
    float theta_sq = theta * theta;

    #pragma omp parallel for schedule(guided)
    for (size_t i = 0; i < n; ++i) {
        glm::vec3 pi(ps.pos_mass[i]);
        glm::vec3 acc = arena.computeForce(pi, eps_sq, theta_sq);
        ps.acc_pad[i] = glm::vec4(acc, 0.0f);
    }
}

void CpuSolvers::stepForwardEuler(ParticleSystem& ps, float dt, const ForceComputeFn& compute_forces) {
    size_t n = ps.size();

    // 1. Compute forces at current positions
    compute_forces(ps);

    // 2. Update positions and velocities
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        glm::vec3 r(ps.pos_mass[i]);
        glm::vec3 v(ps.vel_pad[i]);
        glm::vec3 a(ps.acc_pad[i]);

        r += v * dt;
        v += a * dt;

        ps.pos_mass[i] = glm::vec4(r, ps.pos_mass[i].w);
        ps.vel_pad[i] = glm::vec4(v, 0.0f);
    }
}

void CpuSolvers::stepVelocityVerlet(ParticleSystem& ps, float dt, const ForceComputeFn& compute_forces) {
    size_t n = ps.size();
    float half_dt = 0.5f * dt;

    // 1. Kick & Drift: v(t + dt/2) = v(t) + 0.5 * a(t) * dt; r(t + dt) = r(t) + v(t + dt/2) * dt
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        glm::vec3 r(ps.pos_mass[i]);
        glm::vec3 v(ps.vel_pad[i]);
        glm::vec3 a(ps.acc_pad[i]);

        v += a * half_dt;
        r += v * dt;

        ps.pos_mass[i] = glm::vec4(r, ps.pos_mass[i].w);
        ps.vel_pad[i] = glm::vec4(v, 0.0f);
    }

    // 2. Recompute accelerations at new positions: a(t + dt)
    compute_forces(ps);

    // 3. Final Kick: v(t + dt) = v(t + dt/2) + 0.5 * a(t + dt) * dt
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        glm::vec3 v(ps.vel_pad[i]);
        glm::vec3 a(ps.acc_pad[i]);

        v += a * half_dt;

        ps.vel_pad[i] = glm::vec4(v, 0.0f);
    }
}

} // namespace transport
