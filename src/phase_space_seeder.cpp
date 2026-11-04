#include "transport/phase_space_seeder.hpp"
#include <random>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <numbers>

namespace transport {

ParticleSystem PhaseSpaceSeeder::seed(InitializerType type, size_t n, const std::string& custom_file) {
    switch (type) {
        case InitializerType::SolidCube: return seedSolidCube(n);
        case InitializerType::RotatingDisk: return seedRotatingDisk(n);
        case InitializerType::LagrangeTriangle: return seedLagrangeTriangle(n);
        case InitializerType::SphericalShell: return seedSphericalShell(n);
        case InitializerType::PlummerBall: return seedPlummerBall(n);
        case InitializerType::CubicShell: return seedCubicShell(n);
        case InitializerType::BoundKeplerOrbit: return seedBoundKeplerSystem(n);
        case InitializerType::CloseApproachStress: return seedCloseApproachStress(n);
        case InitializerType::CustomFile: return loadFromFile(custom_file);
        default: return seedSolidCube(n);
    }
}

ParticleSystem PhaseSpaceSeeder::seedSolidCube(size_t n, float half_side) {
    ParticleSystem ps;
    ps.resize(n);
    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> dist_pos(-half_side, half_side);
    std::uniform_real_distribution<float> dist_vel(-0.05f, 0.05f);

    float m = 1.0f / static_cast<float>(n);
    for (size_t i = 0; i < n; ++i) {
        ps.pos_mass[i] = glm::vec4(dist_pos(rng), dist_pos(rng), dist_pos(rng), m);
        ps.vel_pad[i] = glm::vec4(dist_vel(rng), dist_vel(rng), dist_vel(rng), 0.0f);
        ps.acc_pad[i] = glm::vec4(0.0f);
    }
    return ps;
}

ParticleSystem PhaseSpaceSeeder::seedRotatingDisk(size_t n, float inner_r, float outer_r) {
    ParticleSystem ps;
    ps.resize(n);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist_u(0.0f, 1.0f);
    std::uniform_real_distribution<float> dist_theta(0.0f, 2.0f * std::numbers::pi_v<float>);
    std::uniform_real_distribution<float> dist_z(-0.05f, 0.05f);

    float total_mass = 10.0f;
    float m = total_mass / static_cast<float>(n);

    for (size_t i = 0; i < n; ++i) {
        float r = std::sqrt(dist_u(rng) * (outer_r * outer_r - inner_r * inner_r) + inner_r * inner_r);
        float theta = dist_theta(rng);
        float z = dist_z(rng);

        float x = r * std::cos(theta);
        float y = r * std::sin(theta);

        // Circular orbital speed: v = sqrt(G * M(r) / r)
        float enclosed_mass = total_mass * ((r * r - inner_r * inner_r) / (outer_r * outer_r - inner_r * inner_r));
        float v_mag = std::sqrt(std::max(0.01f, G_CONSTANT * enclosed_mass / (r + 0.1f)));

        float vx = -v_mag * std::sin(theta);
        float vy = v_mag * std::cos(theta);
        float vz = 0.01f * (dist_u(rng) - 0.5f);

        ps.pos_mass[i] = glm::vec4(x, y, z, m);
        ps.vel_pad[i] = glm::vec4(vx, vy, vz, 0.0f);
        ps.acc_pad[i] = glm::vec4(0.0f);
    }
    return ps;
}

ParticleSystem PhaseSpaceSeeder::seedLagrangeTriangle(size_t n) {
    ParticleSystem ps;
    ps.resize(n);
    std::mt19937 rng(2026);

    // 3 primary bodies in equilateral configuration with circular velocities
    float R = 2.0f;
    float primary_mass = 20.0f;
    float omega = std::sqrt(G_CONSTANT * primary_mass / (R * R * R * std::sqrt(3.0f)));

    for (size_t i = 0; i < n; ++i) {
        size_t cluster = i % 3;
        float base_angle = static_cast<float>(cluster) * 2.0f * std::numbers::pi_v<float> / 3.0f;
        float cx = R * std::cos(base_angle);
        float cy = R * std::sin(base_angle);
        float cz = 0.0f;

        float vx = -omega * cy;
        float vy = omega * cx;
        float vz = 0.0f;

        if (i >= 3) {
            std::normal_distribution<float> jitter(0.0f, 0.15f);
            cx += jitter(rng);
            cy += jitter(rng);
            cz += jitter(rng) * 0.3f;
            vx += jitter(rng) * 0.05f;
            vy += jitter(rng) * 0.05f;
        }

        float m = (i < 3) ? primary_mass : (1.0f / static_cast<float>(n));
        ps.pos_mass[i] = glm::vec4(cx, cy, cz, m);
        ps.vel_pad[i] = glm::vec4(vx, vy, vz, 0.0f);
        ps.acc_pad[i] = glm::vec4(0.0f);
    }
    return ps;
}

ParticleSystem PhaseSpaceSeeder::seedSphericalShell(size_t n, float radius) {
    ParticleSystem ps;
    ps.resize(n);
    std::mt19937 rng(777);
    std::uniform_real_distribution<float> dist_u(-1.0f, 1.0f);
    std::uniform_real_distribution<float> dist_phi(0.0f, 2.0f * std::numbers::pi_v<float>);

    float m = 1.0f / static_cast<float>(n);
    for (size_t i = 0; i < n; ++i) {
        float cos_theta = dist_u(rng);
        float sin_theta = std::sqrt(std::max(0.0f, 1.0f - cos_theta * cos_theta));
        float phi = dist_phi(rng);

        float x = radius * sin_theta * std::cos(phi);
        float y = radius * sin_theta * std::sin(phi);
        float z = radius * cos_theta;

        ps.pos_mass[i] = glm::vec4(x, y, z, m);
        ps.vel_pad[i] = glm::vec4(-0.02f * y, 0.02f * x, 0.0f, 0.0f);
        ps.acc_pad[i] = glm::vec4(0.0f);
    }
    return ps;
}

ParticleSystem PhaseSpaceSeeder::seedPlummerBall(size_t n, float scale_radius) {
    ParticleSystem ps;
    ps.resize(n);
    std::mt19937 rng(888);
    std::uniform_real_distribution<float> dist_u(0.001f, 0.999f);
    std::uniform_real_distribution<float> dist_costheta(-1.0f, 1.0f);
    std::uniform_real_distribution<float> dist_phi(0.0f, 2.0f * std::numbers::pi_v<float>);

    float total_mass = 15.0f;
    float m = total_mass / static_cast<float>(n);

    for (size_t i = 0; i < n; ++i) {
        // Plummer radius sampling: r = a / sqrt(u^(-2/3) - 1)
        float u = dist_u(rng);
        float r = scale_radius / std::sqrt(std::pow(u, -2.0f / 3.0f) - 1.0f);
        r = std::min(r, scale_radius * 6.0f);

        float cos_t = dist_costheta(rng);
        float sin_t = std::sqrt(std::max(0.0f, 1.0f - cos_t * cos_t));
        float phi = dist_phi(rng);

        float x = r * sin_t * std::cos(phi);
        float y = r * sin_t * std::sin(phi);
        float z = r * cos_t;

        // Escape velocity v_esc = sqrt(2 * G * M / sqrt(r^2 + a^2))
        float v_esc = std::sqrt(2.0f * G_CONSTANT * total_mass / std::sqrt(r * r + scale_radius * scale_radius));
        float v_mag = 0.5f * v_esc * dist_u(rng);

        float v_phi = dist_phi(rng);
        float v_cost = dist_costheta(rng);
        float v_sint = std::sqrt(std::max(0.0f, 1.0f - v_cost * v_cost));

        ps.pos_mass[i] = glm::vec4(x, y, z, m);
        ps.vel_pad[i] = glm::vec4(v_mag * v_sint * std::cos(v_phi),
                                   v_mag * v_sint * std::sin(v_phi),
                                   v_mag * v_cost, 0.0f);
        ps.acc_pad[i] = glm::vec4(0.0f);
    }
    return ps;
}

ParticleSystem PhaseSpaceSeeder::seedCubicShell(size_t n, float half_side) {
    ParticleSystem ps;
    ps.resize(n);
    std::mt19937 rng(999);
    std::uniform_int_distribution<int> face_dist(0, 5);
    std::uniform_real_distribution<float> coord_dist(-half_side, half_side);

    float m = 1.0f / static_cast<float>(n);
    for (size_t i = 0; i < n; ++i) {
        int face = face_dist(rng);
        float u1 = coord_dist(rng);
        float u2 = coord_dist(rng);
        glm::vec3 pos(0.0f);

        switch (face) {
            case 0: pos = glm::vec3( half_side, u1, u2); break;
            case 1: pos = glm::vec3(-half_side, u1, u2); break;
            case 2: pos = glm::vec3(u1,  half_side, u2); break;
            case 3: pos = glm::vec3(u1, -half_side, u2); break;
            case 4: pos = glm::vec3(u1, u2,  half_side); break;
            case 5: pos = glm::vec3(u1, u2, -half_side); break;
        }

        ps.pos_mass[i] = glm::vec4(pos, m);
        ps.vel_pad[i] = glm::vec4(-0.01f * pos.y, 0.01f * pos.x, 0.0f, 0.0f);
        ps.acc_pad[i] = glm::vec4(0.0f);
    }
    return ps;
}

ParticleSystem PhaseSpaceSeeder::seedBoundKeplerSystem(size_t n) {
    if (n < 2) n = 2;
    ParticleSystem ps;
    ps.resize(n);
    std::mt19937 rng(4242);

    // Central massive star
    float central_mass = 100.0f;
    ps.pos_mass[0] = glm::vec4(0.0f, 0.0f, 0.0f, central_mass);
    ps.vel_pad[0] = glm::vec4(0.0f);
    ps.acc_pad[0] = glm::vec4(0.0f);

    float companion_mass = 1.0f / static_cast<float>(n - 1);
    std::uniform_real_distribution<float> r_dist(0.8f, 3.8f);
    std::uniform_real_distribution<float> z_dist(-0.05f, 0.05f);

    for (size_t i = 1; i < n; ++i) {
        float r = r_dist(rng);
        float theta = static_cast<float>(i - 1) * (2.0f * std::numbers::pi_v<float> / static_cast<float>(n - 1));
        float x = r * std::cos(theta);
        float y = r * std::sin(theta);
        float z = z_dist(rng);

        // Circular orbit velocity: v = sqrt(G * M_core / r)
        float v_circ = std::sqrt(G_CONSTANT * central_mass / r);
        float vx = -v_circ * std::sin(theta);
        float vy = v_circ * std::cos(theta);
        float vz = 0.0f;

        ps.pos_mass[i] = glm::vec4(x, y, z, companion_mass);
        ps.vel_pad[i] = glm::vec4(vx, vy, vz, 0.0f);
        ps.acc_pad[i] = glm::vec4(0.0f);
    }
    return ps;
}

ParticleSystem PhaseSpaceSeeder::seedCloseApproachStress(size_t n) {
    ParticleSystem ps;
    ps.resize(n);
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> pos_dist(-2.0f, 2.0f);
    std::uniform_real_distribution<float> vel_dist(-0.1f, 0.1f);

    float m = 1.0f / static_cast<float>(n);

    // Seed normal background particles first
    for (size_t i = 0; i < n; ++i) {
        ps.pos_mass[i] = glm::vec4(pos_dist(rng), pos_dist(rng), pos_dist(rng), m);
        ps.vel_pad[i] = glm::vec4(vel_dist(rng), vel_dist(rng), vel_dist(rng), 0.0f);
        ps.acc_pad[i] = glm::vec4(0.0f);
    }

    size_t curr = 0;

    // 1. 50 pairs of EXACT coincident particles (dr = 0.0)
    size_t coincident_pairs = std::min<size_t>(50, n / 4);
    for (size_t k = 0; k < coincident_pairs && curr + 1 < n; ++k) {
        glm::vec3 p(pos_dist(rng), pos_dist(rng), pos_dist(rng));
        p *= 1.0e-3f;
        ps.pos_mass[curr]     = glm::vec4(p, m);
        ps.pos_mass[curr + 1] = glm::vec4(p, m); // Exact match!
        curr += 2;
    }

    // 2. 50 pairs of ultra-close particles (dr = 1e-9)
    size_t close_pairs = std::min<size_t>(50, (n - curr) / 4);
    for (size_t k = 0; k < close_pairs && curr + 1 < n; ++k) {
        glm::vec3 p(pos_dist(rng), pos_dist(rng), pos_dist(rng));
        p *= 1.0e-3f;
        ps.pos_mass[curr]     = glm::vec4(p, m);
        ps.pos_mass[curr + 1] = glm::vec4(p + glm::vec3(1.0e-9f, 0.0f, 0.0f), m);
        curr += 2;
    }

    // 3. Dense coincident cluster of particles within 1e-8
    size_t cluster_size = std::min<size_t>(64, n - curr);
    glm::vec3 cluster_center(pos_dist(rng), pos_dist(rng), pos_dist(rng));
    cluster_center *= 1.0e-3f;
    for (size_t k = 0; k < cluster_size; ++k) {
        float offset = static_cast<float>(k) * 1.0e-9f;
        ps.pos_mass[curr] = glm::vec4(cluster_center + glm::vec3(offset, 0.0f, 0.0f), m);
        curr++;
    }

    return ps;
}

ParticleSystem PhaseSpaceSeeder::loadFromFile(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open custom particle state file: " + filepath);
    }

    ParticleSystem ps;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        float x, y, z, vx, vy, vz, m;
        if (iss >> x >> y >> z >> vx >> vy >> vz >> m) {
            ps.pos_mass.emplace_back(x, y, z, m);
            ps.vel_pad.emplace_back(vx, vy, vz, 0.0f);
            ps.acc_pad.emplace_back(0.0f);
        }
    }
    if (ps.pos_mass.empty()) {
        throw std::runtime_error("Empty or invalid particle file: " + filepath);
    }
    return ps;
}

} // namespace transport
