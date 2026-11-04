#pragma once

#include "types.hpp"
#include <string>

namespace transport {

class PhaseSpaceSeeder {
public:
    static ParticleSystem seed(InitializerType type, size_t n, const std::string& custom_file = "");

    static ParticleSystem seedSolidCube(size_t n, float half_side = 2.0f);
    static ParticleSystem seedRotatingDisk(size_t n, float inner_r = 0.5f, float outer_r = 3.5f);
    static ParticleSystem seedLagrangeTriangle(size_t n);
    static ParticleSystem seedSphericalShell(size_t n, float radius = 2.5f);
    static ParticleSystem seedPlummerBall(size_t n, float scale_radius = 1.5f);
    static ParticleSystem seedCubicShell(size_t n, float half_side = 2.0f);
    static ParticleSystem seedBoundKeplerSystem(size_t n);
    static ParticleSystem seedCloseApproachStress(size_t n);
    static ParticleSystem loadFromFile(const std::string& filepath);
};

} // namespace transport
