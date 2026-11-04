#include "transport/gpu_compute_engine.hpp"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cmath>

namespace transport {

namespace {

const char* EMBEDDED_DIRECT_V3 = R"glsl(
#version 430 core
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

uniform int numParticles;
uniform float squaredSoftening;
uniform float G;

layout(std430, binding = 0) readonly buffer PositionsBuffer {
    vec4 pos_mass[];
};
layout(std430, binding = 1) writeonly buffer AccelerationsBuffer {
    vec4 acc_pad[];
};

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(numParticles)) return;

    vec3 pi = pos_mass[i].xyz;
    vec3 acc = vec3(0.0);

    for (uint j = 0; j < uint(numParticles); ++j) {
        if (i == j) continue;
        vec3 r = pos_mass[j].xyz - pi;
        float dist_sq = dot(r, r) + squaredSoftening;
        float inv_dist = inversesqrt(dist_sq);
        float inv_dist3 = inv_dist * inv_dist * inv_dist;
        acc += r * (G * pos_mass[j].w * inv_dist3);
    }

    acc_pad[i] = vec4(acc, 0.0);
}
)glsl";

const char* EMBEDDED_TILED_V4 = R"glsl(
#version 430 core
#define TILE_SIZE 256
layout(local_size_x = TILE_SIZE, local_size_y = 1, local_size_z = 1) in;

uniform int numParticles;
uniform float squaredSoftening;
uniform float G;

layout(std430, binding = 0) readonly buffer PositionsBuffer {
    vec4 pos_mass[];
};
layout(std430, binding = 1) writeonly buffer AccelerationsBuffer {
    vec4 acc_pad[];
};

shared vec4 tile_pos_mass[TILE_SIZE];

void main() {
    uint i = gl_GlobalInvocationID.x;
    vec3 pi = (i < uint(numParticles)) ? pos_mass[i].xyz : vec3(0.0);
    vec3 acc = vec3(0.0);

    uint numTiles = (uint(numParticles) + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < numTiles; ++t) {
        uint loadIdx = t * TILE_SIZE + gl_LocalInvocationID.x;
        if (loadIdx < uint(numParticles)) {
            tile_pos_mass[gl_LocalInvocationID.x] = pos_mass[loadIdx];
        } else {
            tile_pos_mass[gl_LocalInvocationID.x] = vec4(0.0);
        }

        barrier();

        if (i < uint(numParticles)) {
            for (uint k = 0; k < TILE_SIZE; ++k) {
                uint global_j = t * TILE_SIZE + k;
                if (global_j == i || global_j >= uint(numParticles)) continue;

                vec3 r = tile_pos_mass[k].xyz - pi;
                float dist_sq = dot(r, r) + squaredSoftening;
                float inv_dist = inversesqrt(dist_sq);
                float inv_dist3 = inv_dist * inv_dist * inv_dist;
                acc += r * (G * tile_pos_mass[k].w * inv_dist3);
            }
        }

        barrier();
    }

    if (i < uint(numParticles)) {
        acc_pad[i] = vec4(acc, 0.0);
    }
}
)glsl";

const char* EMBEDDED_OCTREE_V7 = R"glsl(
#version 430 core
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

uniform int numParticles;
uniform float squaredSoftening;
uniform float G;
uniform float theta;

struct FlatOctreeNode {
    vec4 com_mass;          // xyz = center of mass, w = total mass M
    vec4 bbox_center_half;  // xyz = cell center, w = cell half-width s/2
    int first_child_idx;    // -1 if leaf
    int next_node_idx;      // stackless skip-rope index (-1 = traversal end)
    int particle_idx;       // >= 0 single particle, -2 coincident/aggregated, -1 internal
    int depth;
};

layout(std430, binding = 0) readonly buffer PositionsBuffer {
    vec4 pos_mass[];
};
layout(std430, binding = 1) writeonly buffer AccelerationsBuffer {
    vec4 acc_pad[];
};
layout(std430, binding = 2) readonly buffer OctreeBuffer {
    FlatOctreeNode octreeNodes[];
};

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(numParticles)) return;

    vec3 pi = pos_mass[i].xyz;
    vec3 acc = vec3(0.0);
    float theta_sq = theta * theta;

    int curr = 0; // Root node
    while (curr >= 0) {
        FlatOctreeNode node = octreeNodes[curr];
        if (node.com_mass.w <= 0.0) {
            curr = node.next_node_idx;
            continue;
        }

        vec3 dr = node.com_mass.xyz - pi;
        float dist_sq = dot(dr, dr);
        float s = node.bbox_center_half.w * 2.0;
        bool is_leaf = (node.first_child_idx < 0);

        if (is_leaf || (s * s < theta_sq * dist_sq)) {
            if (dist_sq > 0.0) {
                float eff_dist_sq = dist_sq + squaredSoftening;
                float inv_dist = inversesqrt(eff_dist_sq);
                float inv_dist3 = inv_dist * inv_dist * inv_dist;
                acc += dr * (G * node.com_mass.w * inv_dist3);
            }
            curr = node.next_node_idx;
        } else {
            curr = node.first_child_idx;
        }
    }

    acc_pad[i] = vec4(acc, 0.0);
}
)glsl";

} // namespace

GpuComputeEngine::GpuComputeEngine(bool headless) : headless_(headless) {
    initOpenGL();
    compileComputeShaders();
    initialized_ = true;
}

GpuComputeEngine::~GpuComputeEngine() {
    if (ssbo_pos_mass_) glDeleteBuffers(1, &ssbo_pos_mass_);
    if (ssbo_acc_pad_) glDeleteBuffers(1, &ssbo_acc_pad_);
    if (ssbo_octree_) glDeleteBuffers(1, &ssbo_octree_);

    if (prog_direct_v3_) glDeleteProgram(prog_direct_v3_);
    if (prog_tiled_v4_) glDeleteProgram(prog_tiled_v4_);
    if (prog_octree_v7_) glDeleteProgram(prog_octree_v7_);

    if (window_) {
        glfwDestroyWindow(window_);
        window_ = nullptr;
    }
    glfwTerminate();
}

void GpuComputeEngine::initOpenGL() {
    if (!glfwInit()) {
        throw std::runtime_error("GpuComputeEngine: OpenGL context unavailable: failed to initialize GLFW");
    }

    glfwWindowHint(GLFW_VISIBLE, headless_ ? GLFW_FALSE : GLFW_TRUE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window_ = glfwCreateWindow(640, 480, "Octree-Transport-HPC GPU Engine", nullptr, nullptr);
    if (!window_) {
        // Fallback to 4.3 if 4.6 profile creation fails
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        window_ = glfwCreateWindow(640, 480, "Octree-Transport-HPC GPU Engine", nullptr, nullptr);
    }

    if (!window_) {
        glfwTerminate();
        throw std::runtime_error("GpuComputeEngine: OpenGL context unavailable: failed to create a GLFW window with OpenGL 4.3+ core context");
    }

    glfwMakeContextCurrent(window_);

    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        glfwDestroyWindow(window_);
        window_ = nullptr;
        glfwTerminate();
        throw std::runtime_error(std::string("GpuComputeEngine: OpenGL context unavailable: GLEW initialization failed: ") +
                                 reinterpret_cast<const char*>(glewGetErrorString(err)));
    }

    GLint major_version = 0;
    GLint minor_version = 0;
    glGetIntegerv(GL_MAJOR_VERSION, &major_version);
    glGetIntegerv(GL_MINOR_VERSION, &minor_version);
    if (major_version < 4 || (major_version == 4 && minor_version < 3)) {
        glfwDestroyWindow(window_);
        window_ = nullptr;
        glfwTerminate();
        throw std::runtime_error("GpuComputeEngine: OpenGL context unavailable: OpenGL 4.3+ is required, but the host provided " +
                                 std::to_string(major_version) + "." + std::to_string(minor_version));
    }
}

std::string GpuComputeEngine::getRendererInfo() const {
    const char* rend = reinterpret_cast<const char*>(glGetString(GL_RENDERER));
    const char* vend = reinterpret_cast<const char*>(glGetString(GL_VENDOR));
    const char* vers = reinterpret_cast<const char*>(glGetString(GL_VERSION));
    return std::string(rend ? rend : "Unknown") + " | " +
           (vend ? vend : "Unknown") + " | GL " +
           (vers ? vers : "Unknown");
}

uint32_t GpuComputeEngine::createComputeProgram(const std::string& shader_path, const char* embedded_src) {
    std::string source;
    std::ifstream file(shader_path);
    if (file.is_open()) {
        std::stringstream ss;
        ss << file.rdbuf();
        source = ss.str();
    } else {
        source = embedded_src;
    }

    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
    const char* src_ptr = source.c_str();
    glShaderSource(shader, 1, &src_ptr, nullptr);
    glCompileShader(shader);

    GLint success = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[1024];
        glGetShaderInfoLog(shader, sizeof(infoLog), nullptr, infoLog);
        glDeleteShader(shader);
        throw std::runtime_error("Compute shader compilation failed for " + shader_path + ": " + infoLog);
    }

    GLuint program = glCreateProgram();
    glAttachShader(program, shader);
    glLinkProgram(program);

    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[1024];
        glGetProgramInfoLog(program, sizeof(infoLog), nullptr, infoLog);
        glDeleteShader(shader);
        glDeleteProgram(program);
        throw std::runtime_error("Compute program link failed for " + shader_path + ": " + infoLog);
    }

    glDeleteShader(shader);
    return program;
}

void GpuComputeEngine::compileComputeShaders() {
    prog_direct_v3_ = createComputeProgram("shaders/direct_n2_v3.comp", EMBEDDED_DIRECT_V3);
    prog_tiled_v4_  = createComputeProgram("shaders/tiled_n2_v4.comp", EMBEDDED_TILED_V4);
    prog_octree_v7_ = createComputeProgram("shaders/octree_bh_v7.comp", EMBEDDED_OCTREE_V7);

    glGenBuffers(1, &ssbo_pos_mass_);
    glGenBuffers(1, &ssbo_acc_pad_);
    glGenBuffers(1, &ssbo_octree_);
}

void GpuComputeEngine::ensureParticleBuffers(size_t n) {
    if (n > allocated_particles_cap_) {
        allocated_particles_cap_ = n + 1024;
        size_t bytes = allocated_particles_cap_ * sizeof(glm::vec4);

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_pos_mass_);
        glBufferData(GL_SHADER_STORAGE_BUFFER, bytes, nullptr, GL_DYNAMIC_DRAW);

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_acc_pad_);
        glBufferData(GL_SHADER_STORAGE_BUFFER, bytes, nullptr, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }
}

void GpuComputeEngine::ensureOctreeBuffer(size_t node_count) {
    if (node_count > allocated_octree_cap_) {
        allocated_octree_cap_ = node_count + 2048;
        size_t bytes = allocated_octree_cap_ * sizeof(FlatOctreeNode);

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_octree_);
        glBufferData(GL_SHADER_STORAGE_BUFFER, bytes, nullptr, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }
}

void GpuComputeEngine::computeDirectForcesNaiveV3(ParticleSystem& ps, float softening) {
    size_t n = ps.size();
    if (n == 0) return;

    ensureParticleBuffers(n);

    // Upload positions and masses
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_pos_mass_);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, n * sizeof(glm::vec4), ps.pos_mass.data());

    glUseProgram(prog_direct_v3_);
    glUniform1i(glGetUniformLocation(prog_direct_v3_, "numParticles"), static_cast<int>(n));
    glUniform1f(glGetUniformLocation(prog_direct_v3_, "squaredSoftening"), softening * softening);
    glUniform1f(glGetUniformLocation(prog_direct_v3_, "G"), G_CONSTANT);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo_pos_mass_);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo_acc_pad_);

    GLuint groups = (static_cast<GLuint>(n) + 255) / 256;
    glDispatchCompute(groups, 1, 1);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    glFinish();

    // Read back computed accelerations
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_acc_pad_);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, n * sizeof(glm::vec4), ps.acc_pad.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void GpuComputeEngine::computeDirectForcesTiledV4(ParticleSystem& ps, float softening) {
    size_t n = ps.size();
    if (n == 0) return;

    ensureParticleBuffers(n);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_pos_mass_);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, n * sizeof(glm::vec4), ps.pos_mass.data());

    glUseProgram(prog_tiled_v4_);
    glUniform1i(glGetUniformLocation(prog_tiled_v4_, "numParticles"), static_cast<int>(n));
    glUniform1f(glGetUniformLocation(prog_tiled_v4_, "squaredSoftening"), softening * softening);
    glUniform1f(glGetUniformLocation(prog_tiled_v4_, "G"), G_CONSTANT);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo_pos_mass_);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo_acc_pad_);

    GLuint groups = (static_cast<GLuint>(n) + 255) / 256;
    glDispatchCompute(groups, 1, 1);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    glFinish();

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_acc_pad_);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, n * sizeof(glm::vec4), ps.acc_pad.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void GpuComputeEngine::computeBarnesHutForcesV7(ParticleSystem& ps, const OctreeArena& arena, float softening, float theta) {
    size_t n = ps.size();
    size_t node_count = arena.getNodeCount();
    if (n == 0 || node_count == 0) return;

    ensureParticleBuffers(n);
    ensureOctreeBuffer(node_count);

    // Upload particle positions and tree nodes
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_pos_mass_);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, n * sizeof(glm::vec4), ps.pos_mass.data());

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_octree_);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, node_count * sizeof(FlatOctreeNode), arena.getNodes().data());

    glUseProgram(prog_octree_v7_);
    glUniform1i(glGetUniformLocation(prog_octree_v7_, "numParticles"), static_cast<int>(n));
    glUniform1f(glGetUniformLocation(prog_octree_v7_, "squaredSoftening"), softening * softening);
    glUniform1f(glGetUniformLocation(prog_octree_v7_, "G"), G_CONSTANT);
    glUniform1f(glGetUniformLocation(prog_octree_v7_, "theta"), theta);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo_pos_mass_);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo_acc_pad_);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssbo_octree_);

    GLuint groups = (static_cast<GLuint>(n) + 255) / 256;
    glDispatchCompute(groups, 1, 1);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    glFinish();

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_acc_pad_);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, n * sizeof(glm::vec4), ps.acc_pad.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

} // namespace transport
