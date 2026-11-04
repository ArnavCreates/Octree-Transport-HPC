#pragma once

#include "types.hpp"
#include "octree_arena.hpp"
#include <functional>

namespace transport {

using ForceComputeFn = std::function<void(ParticleSystem&)>;

class CpuSolvers {
public:
    static void computeDirectForcesSeq(ParticleSystem& ps, float softening);
    static void computeDirectForcesOMP(ParticleSystem& ps, float softening);

    static void computeBarnesHutForcesSeq(ParticleSystem& ps, const OctreeArena& arena, float softening, float theta);
    static void computeBarnesHutForcesOMP(ParticleSystem& ps, const OctreeArena& arena, float softening, float theta);

    static void stepForwardEuler(ParticleSystem& ps, float dt, const ForceComputeFn& compute_forces);
    static void stepVelocityVerlet(ParticleSystem& ps, float dt, const ForceComputeFn& compute_forces);
};

} // namespace transport
