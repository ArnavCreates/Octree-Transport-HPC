#pragma once

#include "types.hpp"
#include <vector>

namespace transport {

class PhysicsDiagnostics {
public:
    static EnergyMetrics computeMetrics(const ParticleSystem& ps, float softening);
    static double computeKineticEnergy(const ParticleSystem& ps);
    static double computePotentialEnergy(const ParticleSystem& ps, float softening);
    static glm::dvec3 computeTotalLinearMomentum(const ParticleSystem& ps);
    static glm::dvec3 computeTotalAngularMomentum(const ParticleSystem& ps);

    struct ForceAccuracyMetrics {
        double relative_l2_error{0.0};
        double max_relative_error{0.0};
        double mean_relative_error{0.0};
    };

    static ForceAccuracyMetrics compareForces(const ParticleSystem& reference_exact,
                                              const ParticleSystem& test_solver);
};

} // namespace transport
