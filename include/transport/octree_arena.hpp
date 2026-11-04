#pragma once

#include "types.hpp"
#include <array>
#include <vector>
#include <cstdint>
#include <glm/glm.hpp>

namespace transport {

class OctreeArena {
public:
    OctreeArena() = default;
    explicit OctreeArena(size_t initial_particle_count);

    void reset(size_t particle_count);
    void buildSequential(const ParticleSystem& ps);
    void buildParallelOMP(const ParticleSystem& ps);

    glm::vec3 computeForce(const glm::vec3& p_pos, float softening_sq, float theta_sq) const;

    const std::vector<FlatOctreeNode>& getNodes() const { return nodes_; }
    size_t getNodeCount() const { return node_count_; }
    uint32_t getMaxDepth() const { return max_depth_reached_; }
    uint64_t getCoincidentMerges() const { return coincident_merges_; }
    std::array<uint32_t, MAX_OCTREE_DEPTH + 1> computeDepthHistogram() const;
    float getTotalMass() const { return node_count_ > 0 ? nodes_[0].com_mass.w : 0.0f; }

private:
    std::vector<FlatOctreeNode> nodes_;
    size_t node_count_{0};
    uint32_t max_depth_reached_{0};
    uint64_t coincident_merges_{0};

    int32_t allocateNodes(size_t count);
    static int getOctant(const glm::vec3& p, const glm::vec3& center);
    static glm::vec3 getOctantCenter(const glm::vec3& parent_center, float half_extent, int octant);

    void insertParticleSeq(size_t p_idx, const glm::vec3& p_pos, float p_mass, int32_t node_idx);
    void insertParticleSubtree(size_t p_idx, const glm::vec3& p_pos, float p_mass, int32_t node_idx,
                               std::vector<FlatOctreeNode>& subtree_nodes,
                               uint64_t& coincident_merges);
    void propagateComSeq();
};

} // namespace transport
