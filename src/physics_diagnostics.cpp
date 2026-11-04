#include "transport/physics_diagnostics.hpp"
#include <cmath>
#include <algorithm>
#include <omp.h>

namespace transport {

EnergyMetrics PhysicsDiagnostics::computeMetrics(const ParticleSystem& ps, float softening) {
    EnergyMetrics em;
    em.kinetic_energy = computeKineticEnergy(ps);
    em.potential_energy = computePotentialEnergy(ps, softening);
    em.total_energy = em.kinetic_energy + em.potential_energy;
    em.total_linear_momentum = computeTotalLinearMomentum(ps);
    em.total_angular_momentum = computeTotalAngularMomentum(ps);
    return em;
}

double PhysicsDiagnostics::computeKineticEnergy(const ParticleSystem& ps) {
    size_t n = ps.size();
    double total_ke = 0.0;

    #pragma omp parallel for reduction(+: total_ke) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        double m = ps.pos_mass[i].w;
        glm::dvec3 v(ps.vel_pad[i]);
        total_ke += 0.5 * m * glm::dot(v, v);
    }

    return total_ke;
}

double PhysicsDiagnostics::computePotentialEnergy(const ParticleSystem& ps, float softening) {
    size_t n = ps.size();
    if (n < 2) return 0.0;

    double eps_sq = static_cast<double>(softening) * softening;
    double total_pe = 0.0;

    #pragma omp parallel for reduction(+: total_pe) schedule(guided)
    for (size_t i = 0; i < n; ++i) {
        double mi = ps.pos_mass[i].w;
        glm::dvec3 pi(ps.pos_mass[i]);
        double local_sum = 0.0;

        for (size_t j = i + 1; j < n; ++j) {
            double mj = ps.pos_mass[j].w;
            glm::dvec3 pj(ps.pos_mass[j]);
            glm::dvec3 dr = pj - pi;
            double dist = std::sqrt(glm::dot(dr, dr) + eps_sq);
            local_sum -= (mi * mj) / dist;
        }

        total_pe += local_sum;
    }

    return G_CONSTANT * total_pe;
}

glm::dvec3 PhysicsDiagnostics::computeTotalLinearMomentum(const ParticleSystem& ps) {
    size_t n = ps.size();
    double px = 0.0, py = 0.0, pz = 0.0;

    #pragma omp parallel for reduction(+: px, py, pz) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        double m = ps.pos_mass[i].w;
        px += m * ps.vel_pad[i].x;
        py += m * ps.vel_pad[i].y;
        pz += m * ps.vel_pad[i].z;
    }

    return glm::dvec3(px, py, pz);
}

glm::dvec3 PhysicsDiagnostics::computeTotalAngularMomentum(const ParticleSystem& ps) {
    size_t n = ps.size();
    double lx = 0.0, ly = 0.0, lz = 0.0;

    #pragma omp parallel for reduction(+: lx, ly, lz) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        double m = ps.pos_mass[i].w;
        glm::dvec3 r(ps.pos_mass[i]);
        glm::dvec3 v(ps.vel_pad[i]);
        glm::dvec3 ang = glm::cross(r, m * v);
        lx += ang.x;
        ly += ang.y;
        lz += ang.z;
    }

    return glm::dvec3(lx, ly, lz);
}

PhysicsDiagnostics::ForceAccuracyMetrics PhysicsDiagnostics::compareForces(const ParticleSystem& ref_exact,
                                                                          const ParticleSystem& test_solver) {
    ForceAccuracyMetrics fam;
    size_t n = ref_exact.size();
    if (n == 0 || test_solver.size() != n) return fam;

    double diff_sq_sum = 0.0;
    double ref_sq_sum = 0.0;
    double max_rel = 0.0;
    double rel_sum = 0.0;

    for (size_t i = 0; i < n; ++i) {
        glm::dvec3 a_ref(ref_exact.acc_pad[i]);
        glm::dvec3 a_test(test_solver.acc_pad[i]);
        glm::dvec3 diff = a_test - a_ref;

        double d_sq = glm::dot(diff, diff);
        double r_sq = glm::dot(a_ref, a_ref);

        diff_sq_sum += d_sq;
        ref_sq_sum += r_sq;

        double d_norm = std::sqrt(d_sq);
        double r_norm = std::sqrt(r_sq);
        double rel = (r_norm > 1.0e-7) ? (d_norm / r_norm) : 0.0;

        max_rel = std::max(max_rel, rel);
        rel_sum += rel;
    }

    fam.relative_l2_error = (ref_sq_sum > 0.0) ? std::sqrt(diff_sq_sum / ref_sq_sum) : 0.0;
    fam.max_relative_error = max_rel;
    fam.mean_relative_error = rel_sum / static_cast<double>(n);

    return fam;
}

} // namespace transport
