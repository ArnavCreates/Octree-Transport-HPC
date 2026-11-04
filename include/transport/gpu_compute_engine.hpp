#pragma once

#include "types.hpp"
#include "octree_arena.hpp"
#include <string>

struct GLFWwindow;

namespace transport {

class GpuComputeEngine {
public:
    explicit GpuComputeEngine(bool headless = true);
    ~GpuComputeEngine();

    GpuComputeEngine(const GpuComputeEngine&) = delete;
    GpuComputeEngine& operator=(const GpuComputeEngine&) = delete;

    bool isInitialized() const { return initialized_; }
    std::string getRendererInfo() const;

    void computeDirectForcesNaiveV3(ParticleSystem& ps, float softening);
    void computeDirectForcesTiledV4(ParticleSystem& ps, float softening);
    void computeBarnesHutForcesV7(ParticleSystem& ps, const OctreeArena& arena, float softening, float theta);

    GLFWwindow* getWindow() const { return window_; }

private:
    bool headless_{true};
    bool initialized_{false};
    GLFWwindow* window_{nullptr};

    uint32_t prog_direct_v3_{0};
    uint32_t prog_tiled_v4_{0};
    uint32_t prog_octree_v7_{0};

    uint32_t ssbo_pos_mass_{0};
    uint32_t ssbo_acc_pad_{0};
    uint32_t ssbo_octree_{0};

    size_t allocated_particles_cap_{0};
    size_t allocated_octree_cap_{0};

    void initOpenGL();
    void compileComputeShaders();
    uint32_t createComputeProgram(const std::string& shader_path, const char* embedded_src);
    void ensureParticleBuffers(size_t n);
    void ensureOctreeBuffer(size_t node_count);
};

} // namespace transport
