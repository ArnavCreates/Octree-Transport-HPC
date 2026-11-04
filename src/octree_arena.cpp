#include "transport/octree_arena.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <omp.h>

namespace transport {

OctreeArena::OctreeArena(size_t initial_particle_count) {
    reset(initial_particle_count);
}

void OctreeArena::reset(size_t particle_count) {
    size_t capacity = std::max<size_t>(64000, 8 * particle_count + 2048);
    nodes_.assign(capacity, FlatOctreeNode{});
    node_count_ = 0;
    max_depth_reached_ = 0;
    coincident_merges_ = 0;
}

int32_t OctreeArena::allocateNodes(size_t count) {
    if (node_count_ + count > nodes_.size()) {
        size_t new_cap = std::max(nodes_.size() * 2, node_count_ + count + 1024);
        nodes_.resize(new_cap, FlatOctreeNode{});
    }
    int32_t start_idx = static_cast<int32_t>(node_count_);
    node_count_ += count;
    return start_idx;
}

int OctreeArena::getOctant(const glm::vec3& p, const glm::vec3& center) {
    int oct = 0;
    if (p.x >= center.x) oct |= 1;
    if (p.y >= center.y) oct |= 2;
    if (p.z >= center.z) oct |= 4;
    return oct;
}

glm::vec3 OctreeArena::getOctantCenter(const glm::vec3& parent_center, float half_extent, int octant) {
    float quarter = 0.5f * half_extent;
    return parent_center + glm::vec3(
        (octant & 1) ? quarter : -quarter,
        (octant & 2) ? quarter : -quarter,
        (octant & 4) ? quarter : -quarter
    );
}

void OctreeArena::buildSequential(const ParticleSystem& ps) {
    size_t n = ps.size();
    if (n == 0) {
        node_count_ = 0;
        max_depth_reached_ = 0;
        coincident_merges_ = 0;
        return;
    }

    reset(n);

    // 1. Compute global bounding box
    glm::vec3 min_p(std::numeric_limits<float>::max());
    glm::vec3 max_p(std::numeric_limits<float>::lowest());

    for (size_t i = 0; i < n; ++i) {
        glm::vec3 p(ps.pos_mass[i]);
        min_p = glm::min(min_p, p);
        max_p = glm::max(max_p, p);
    }

    glm::vec3 center = 0.5f * (min_p + max_p);
    glm::vec3 extents = max_p - min_p;
    float half_extent = 0.5f * std::max({extents.x, extents.y, extents.z, 2.0f});

    // 2. Initialize root node at index 0
    node_count_ = 1;
    nodes_[0].bbox_center_half = glm::vec4(center, half_extent);
    nodes_[0].first_child_idx = -1;
    nodes_[0].next_node_idx = -1;
    nodes_[0].particle_idx = -1;
    nodes_[0].depth = 0;
    nodes_[0].com_mass = glm::vec4(0.0f);
    max_depth_reached_ = 0;

    // 3. Insert particles sequentially
    for (size_t i = 0; i < n; ++i) {
        insertParticleSeq(i, glm::vec3(ps.pos_mass[i]), ps.pos_mass[i].w, 0);
    }

    // 4. Bottom-up mass and center-of-mass propagation
    propagateComSeq();
}

void OctreeArena::insertParticleSeq(size_t p_idx, const glm::vec3& p_pos, float p_mass, int32_t node_idx) {
    int32_t curr = node_idx;

    while (curr >= 0) {
        FlatOctreeNode& node = nodes_[curr];
        max_depth_reached_ = std::max<uint32_t>(max_depth_reached_, static_cast<uint32_t>(node.depth));

        // Case 1: Empty leaf
        if (node.first_child_idx < 0 && node.particle_idx == -1 && node.com_mass.w == 0.0f) {
            node.particle_idx = static_cast<int32_t>(p_idx);
            node.com_mass = glm::vec4(p_pos, p_mass);
            return;
        }

        // Case 2: Coincident leaf already aggregated
        if (node.first_child_idx < 0 && node.particle_idx == -2) {
            ++coincident_merges_;
            float old_m = node.com_mass.w;
            float new_m = old_m + p_mass;
            glm::vec3 old_com(node.com_mass);
            glm::vec3 new_com = (old_com * old_m + p_pos * p_mass) / (new_m > 0.0f ? new_m : 1.0f);
            node.com_mass = glm::vec4(new_com, new_m);
            return;
        }

        // Case 3: Leaf containing an existing single particle
        if (node.first_child_idx < 0 && node.particle_idx >= 0) {
            glm::vec3 exist_pos(node.com_mass);
            float exist_m = node.com_mass.w;
            int32_t exist_idx = node.particle_idx;

            float half_w = node.bbox_center_half.w;

            // Close-approach / depth guard
            if (node.depth >= static_cast<int32_t>(MAX_OCTREE_DEPTH) ||
                half_w <= MIN_CELL_HALF_EXTENT) {
                node.particle_idx = -2; // convert to aggregated leaf
                ++coincident_merges_;
                float total_m = exist_m + p_mass;
                glm::vec3 agg_com = (exist_pos * exist_m + p_pos * p_mass) / (total_m > 0.0f ? total_m : 1.0f);
                node.com_mass = glm::vec4(agg_com, total_m);
                return;
            }

            // Subdivide node into 8 children
            int32_t child_base = allocateNodes(8);
            // Re-fetch reference in case vector resized
            FlatOctreeNode& parent_ref = nodes_[curr];
            parent_ref.first_child_idx = child_base;
            parent_ref.particle_idx = -1; // Now internal
            int32_t parent_next = parent_ref.next_node_idx;
            glm::vec3 p_center(parent_ref.bbox_center_half);
            float child_half_w = 0.5f * half_w;
            int32_t child_depth = parent_ref.depth + 1;

            for (int j = 0; j < 8; ++j) {
                FlatOctreeNode& ch = nodes_[child_base + j];
                ch.bbox_center_half = glm::vec4(getOctantCenter(p_center, half_w, j), child_half_w);
                ch.first_child_idx = -1;
                ch.particle_idx = -1;
                ch.com_mass = glm::vec4(0.0f);
                ch.depth = child_depth;
                ch.next_node_idx = (j < 7) ? (child_base + j + 1) : parent_next;
            }

            // Re-insert existing particle into appropriate child
            int oct_exist = getOctant(exist_pos, p_center);
            nodes_[child_base + oct_exist].particle_idx = exist_idx;
            nodes_[child_base + oct_exist].com_mass = glm::vec4(exist_pos, exist_m);

            // Now insert new particle into child
            int oct_new = getOctant(p_pos, p_center);
            if (oct_new != oct_exist) {
                nodes_[child_base + oct_new].particle_idx = static_cast<int32_t>(p_idx);
                nodes_[child_base + oct_new].com_mass = glm::vec4(p_pos, p_mass);
                return;
            } else {
                // Both fell into the same child, continue iterative descent
                curr = child_base + oct_new;
                continue;
            }
        }

        // Case 4: Internal node, descend to child
        glm::vec3 center(node.bbox_center_half);
        int oct = getOctant(p_pos, center);
        curr = node.first_child_idx + oct;
    }
}

void OctreeArena::propagateComSeq() {
    if (node_count_ == 0) return;

    for (int i = static_cast<int>(node_count_) - 1; i >= 0; --i) {
        if (nodes_[i].first_child_idx >= 0) {
            int base = nodes_[i].first_child_idx;
            glm::vec3 weighted_com(0.0f);
            float total_m = 0.0f;

            for (int c = 0; c < 8; ++c) {
                float cm = nodes_[base + c].com_mass.w;
                if (cm > 0.0f) {
                    total_m += cm;
                    weighted_com += glm::vec3(nodes_[base + c].com_mass) * cm;
                }
            }

            if (total_m > 0.0f) {
                nodes_[i].com_mass = glm::vec4(weighted_com / total_m, total_m);
            } else {
                nodes_[i].com_mass = glm::vec4(0.0f);
            }
        }
    }
}

void OctreeArena::buildParallelOMP(const ParticleSystem& ps) {
    size_t n = ps.size();
    if (n == 0) {
        node_count_ = 0;
        coincident_merges_ = 0;
        max_depth_reached_ = 0;
        return;
    }

    // 1. Parallel reduction to find global bounding box
    float min_x = std::numeric_limits<float>::max();
    float min_y = min_x;
    float min_z = min_x;
    float max_x = std::numeric_limits<float>::lowest();
    float max_y = max_x;
    float max_z = max_x;

    #pragma omp parallel for reduction(min: min_x, min_y, min_z) reduction(max: max_x, max_y, max_z)
    for (size_t i = 0; i < n; ++i) {
        float x = ps.pos_mass[i].x;
        float y = ps.pos_mass[i].y;
        float z = ps.pos_mass[i].z;
        min_x = std::min(min_x, x);
        min_y = std::min(min_y, y);
        min_z = std::min(min_z, z);
        max_x = std::max(max_x, x);
        max_y = std::max(max_y, y);
        max_z = std::max(max_z, z);
    }

    glm::vec3 min_p(min_x, min_y, min_z);
    glm::vec3 max_p(max_x, max_y, max_z);
    glm::vec3 root_center = 0.5f * (min_p + max_p);
    glm::vec3 extents = max_p - min_p;
    float root_half_extent = 0.5f * std::max({extents.x, extents.y, extents.z, 2.0f});

    // 2. Count and scatter particles into deterministic, disjoint octant ranges.
    const int max_threads = std::max(1, omp_get_max_threads());
    std::vector<size_t> local_counts(static_cast<size_t>(max_threads) * 8, 0);
    std::vector<size_t> write_offsets(static_cast<size_t>(max_threads) * 8, 0);
    std::array<size_t, 9> octant_offsets{};
    std::vector<size_t> partitioned_particles(n);
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const int thread_count = omp_get_num_threads();
        const size_t begin = n * static_cast<size_t>(tid) / static_cast<size_t>(thread_count);
        const size_t end = n * static_cast<size_t>(tid + 1) / static_cast<size_t>(thread_count);
        size_t* counts = local_counts.data() + static_cast<size_t>(tid) * 8;
        for (size_t i = begin; i < end; ++i) {
            glm::vec3 p(ps.pos_mass[i]);
            int oct = getOctant(p, root_center);
            ++counts[oct];
        }

        #pragma omp barrier
        #pragma omp single
        {
            size_t cursor = 0;
            for (int oct = 0; oct < 8; ++oct) {
                for (int thread = 0; thread < thread_count; ++thread) {
                    size_t slot = static_cast<size_t>(thread) * 8 + static_cast<size_t>(oct);
                    write_offsets[slot] = cursor;
                    cursor += local_counts[slot];
                }
                octant_offsets[oct + 1] = cursor;
            }
        }

        size_t cursors[8];
        for (int oct = 0; oct < 8; ++oct) {
            cursors[oct] = write_offsets[static_cast<size_t>(tid) * 8 + static_cast<size_t>(oct)];
        }
        for (size_t i = begin; i < end; ++i) {
            const int oct = getOctant(glm::vec3(ps.pos_mass[i]), root_center);
            partitioned_particles[cursors[oct]++] = i;
        }
    }

    // 3. Initialize the global root and its eight fixed root-octant entries.
    nodes_.assign(9, FlatOctreeNode{});
    node_count_ = 0;

    // Initialize root node (node 0)
    nodes_[0].bbox_center_half = glm::vec4(root_center, root_half_extent);
    nodes_[0].first_child_idx = 1;
    nodes_[0].next_node_idx = -1;
    nodes_[0].particle_idx = -1;
    nodes_[0].depth = 0;
    nodes_[0].com_mass = glm::vec4(0.0f);

    float child_half_w = 0.5f * root_half_extent;
    for (int oct = 0; oct < 8; ++oct) {
        int32_t c_idx = 1 + oct;
        nodes_[c_idx].bbox_center_half = glm::vec4(getOctantCenter(root_center, root_half_extent, oct), child_half_w);
        nodes_[c_idx].first_child_idx = -1;
        nodes_[c_idx].particle_idx = -1;
        nodes_[c_idx].depth = 1;
        nodes_[c_idx].com_mass = glm::vec4(0.0f);
        // Stackless skip-rope: octant k skips to octant k + 1, octant 7 skips to -1 (traversal end)
        nodes_[c_idx].next_node_idx = (oct < 7) ? (1 + oct + 1) : -1;
    }

    uint32_t global_max_depth = 1;
    std::array<uint64_t, 8> merge_counts{};
    std::array<std::vector<FlatOctreeNode>, 8> subtree_nodes;

    // 4. Build each root octant in its own growable buffer without shared writes.
    #pragma omp parallel for schedule(dynamic, 1) reduction(max: global_max_depth)
    for (int oct = 0; oct < 8; ++oct) {
        const size_t p_begin = octant_offsets[oct];
        const size_t p_end = octant_offsets[oct + 1];
        const size_t p_count = p_end - p_begin;
        int32_t root_child_idx = 1 + oct;

        if (p_count == 0) {
            continue;
        }

        if (p_count == 1) {
            size_t p_idx = partitioned_particles[p_begin];
            nodes_[root_child_idx].particle_idx = static_cast<int32_t>(p_idx);
            nodes_[root_child_idx].com_mass = ps.pos_mass[p_idx];
            continue;
        }

        auto& subtree = subtree_nodes[oct];
        subtree.reserve(std::max<size_t>(64, 8 * p_count + 512));
        subtree.push_back(nodes_[root_child_idx]);
        subtree[0].next_node_idx = -1;
        uint64_t local_merges = 0;
        for (size_t partition_idx = p_begin; partition_idx < p_end; ++partition_idx) {
            size_t p_idx = partitioned_particles[partition_idx];
            insertParticleSubtree(p_idx, glm::vec3(ps.pos_mass[p_idx]), ps.pos_mass[p_idx].w,
                                  0, subtree, local_merges);
        }
        merge_counts[oct] = local_merges;

        // Bottom-up COM propagation within this octant's private buffer.
        for (int i = static_cast<int>(subtree.size()) - 1; i >= 0; --i) {
            if (subtree[static_cast<size_t>(i)].first_child_idx >= 0) {
                int base = subtree[static_cast<size_t>(i)].first_child_idx;
                glm::vec3 wcom(0.0f);
                float tm = 0.0f;
                for (int c = 0; c < 8; ++c) {
                    float cm = subtree[static_cast<size_t>(base + c)].com_mass.w;
                    if (cm > 0.0f) {
                        tm += cm;
                        wcom += glm::vec3(subtree[static_cast<size_t>(base + c)].com_mass) * cm;
                    }
                }
                if (tm > 0.0f) {
                    subtree[static_cast<size_t>(i)].com_mass = glm::vec4(wcom / tm, tm);
                }
            }
        }

        const int32_t root_next = (oct < 7) ? root_child_idx + 1 : -1;
        subtree[0].next_node_idx = root_next;
        nodes_[root_child_idx] = subtree[0];
        for (const FlatOctreeNode& node : subtree) {
            global_max_depth = std::max(global_max_depth, static_cast<uint32_t>(node.depth));
        }
    }

    max_depth_reached_ = global_max_depth;
    coincident_merges_ = 0;
    for (uint64_t count : merge_counts) coincident_merges_ += count;

    // 5. O(1) Root stitching & COM calculation
    glm::vec3 root_wcom(0.0f);
    float root_total_mass = 0.0f;
    for (int oct = 0; oct < 8; ++oct) {
        float cm = nodes_[1 + oct].com_mass.w;
        if (cm > 0.0f) {
            root_total_mass += cm;
            root_wcom += glm::vec3(nodes_[1 + oct].com_mass) * cm;
        }
    }

    if (root_total_mass > 0.0f) {
        nodes_[0].com_mass = glm::vec4(root_wcom / root_total_mass, root_total_mass);
    } else {
        nodes_[0].com_mass = glm::vec4(0.0f);
    }

    // Compact each independently built subtree and remap its internal skip ropes.
    size_t compacted_count = 9;
    for (const auto& subtree : subtree_nodes) if (!subtree.empty()) compacted_count += subtree.size() - 1;
    std::vector<FlatOctreeNode> compacted(compacted_count);
    std::copy_n(nodes_.begin(), 9, compacted.begin());
    size_t compact_cursor = 9;
    for (int oct = 0; oct < 8; ++oct) {
        const auto& subtree = subtree_nodes[oct];
        if (subtree.empty()) continue;
        const size_t new_start = compact_cursor;
        const int32_t root_next = (oct < 7) ? 2 + oct : -1;
        auto remap = [new_start](int32_t old_index) {
            return static_cast<int32_t>(new_start + static_cast<size_t>(old_index - 1));
        };
        FlatOctreeNode& root_child = compacted[1 + oct];
        if (root_child.first_child_idx >= 1) root_child.first_child_idx = remap(root_child.first_child_idx);
        root_child.next_node_idx = root_next;
        for (size_t local_index = 1; local_index < subtree.size(); ++local_index) {
            FlatOctreeNode node = subtree[local_index];
            if (node.first_child_idx >= 1) node.first_child_idx = remap(node.first_child_idx);
            if (node.next_node_idx >= 1) {
                node.next_node_idx = remap(node.next_node_idx);
            } else if (root_next >= 0) {
                node.next_node_idx = root_next;
            }
            compacted[compact_cursor++] = node;
        }
    }
    nodes_.swap(compacted);
    node_count_ = compact_cursor;
}

std::array<uint32_t, MAX_OCTREE_DEPTH + 1> OctreeArena::computeDepthHistogram() const {
    std::array<uint32_t, MAX_OCTREE_DEPTH + 1> histogram{};
    if (node_count_ == 0) return histogram;

    std::vector<int32_t> pending{0};
    while (!pending.empty()) {
        const int32_t idx = pending.back();
        pending.pop_back();
        const FlatOctreeNode& node = nodes_[static_cast<size_t>(idx)];
        if (node.com_mass.w <= 0.0f || node.depth < 0 || node.depth > static_cast<int32_t>(MAX_OCTREE_DEPTH)) {
            continue;
        }
        ++histogram[static_cast<size_t>(node.depth)];
        if (node.first_child_idx >= 0) {
            for (int child = 0; child < 8; ++child) {
                pending.push_back(node.first_child_idx + child);
            }
        }
    }
    return histogram;
}

void OctreeArena::insertParticleSubtree(size_t p_idx, const glm::vec3& p_pos, float p_mass, int32_t node_idx,
                                        std::vector<FlatOctreeNode>& subtree_nodes,
                                        uint64_t& coincident_merges) {
    int32_t curr = node_idx;

    while (curr >= 0) {
        FlatOctreeNode& node = subtree_nodes[static_cast<size_t>(curr)];

        // Case 1: Empty leaf
        if (node.first_child_idx < 0 && node.particle_idx == -1 && node.com_mass.w == 0.0f) {
            node.particle_idx = static_cast<int32_t>(p_idx);
            node.com_mass = glm::vec4(p_pos, p_mass);
            return;
        }

        // Case 2: Coincident leaf already aggregated
        if (node.first_child_idx < 0 && node.particle_idx == -2) {
            ++coincident_merges;
            float old_m = node.com_mass.w;
            float new_m = old_m + p_mass;
            glm::vec3 old_com(node.com_mass);
            glm::vec3 new_com = (old_com * old_m + p_pos * p_mass) / (new_m > 0.0f ? new_m : 1.0f);
            node.com_mass = glm::vec4(new_com, new_m);
            return;
        }

        // Case 3: Leaf containing an existing single particle
        if (node.first_child_idx < 0 && node.particle_idx >= 0) {
            glm::vec3 exist_pos(node.com_mass);
            float exist_m = node.com_mass.w;
            int32_t exist_idx = node.particle_idx;

            float half_w = node.bbox_center_half.w;

            // Close-approach / depth guard
            if (node.depth >= static_cast<int32_t>(MAX_OCTREE_DEPTH) ||
                half_w <= MIN_CELL_HALF_EXTENT) {
                node.particle_idx = -2; // convert to aggregated leaf
                ++coincident_merges;
                float total_m = exist_m + p_mass;
                glm::vec3 agg_com = (exist_pos * exist_m + p_pos * p_mass) / (total_m > 0.0f ? total_m : 1.0f);
                node.com_mass = glm::vec4(agg_com, total_m);
                return;
            }

            // Grow this octant's private buffer by one complete child block.
            int32_t child_base = static_cast<int32_t>(subtree_nodes.size());
            subtree_nodes.resize(subtree_nodes.size() + 8, FlatOctreeNode{});

            FlatOctreeNode& parent_ref = subtree_nodes[static_cast<size_t>(curr)];
            parent_ref.first_child_idx = child_base;
            parent_ref.particle_idx = -1;
            int32_t parent_next = parent_ref.next_node_idx;
            glm::vec3 p_center(parent_ref.bbox_center_half);
            float child_half_w = 0.5f * half_w;
            int32_t child_depth = parent_ref.depth + 1;

            for (int j = 0; j < 8; ++j) {
                FlatOctreeNode& ch = subtree_nodes[static_cast<size_t>(child_base + j)];
                ch.bbox_center_half = glm::vec4(getOctantCenter(p_center, half_w, j), child_half_w);
                ch.first_child_idx = -1;
                ch.particle_idx = -1;
                ch.com_mass = glm::vec4(0.0f);
                ch.depth = child_depth;
                ch.next_node_idx = (j < 7) ? (child_base + j + 1) : parent_next;
            }

            // Re-insert existing particle
            int oct_exist = getOctant(exist_pos, p_center);
            subtree_nodes[static_cast<size_t>(child_base + oct_exist)].particle_idx = exist_idx;
            subtree_nodes[static_cast<size_t>(child_base + oct_exist)].com_mass = glm::vec4(exist_pos, exist_m);

            // Insert new particle
            int oct_new = getOctant(p_pos, p_center);
            if (oct_new != oct_exist) {
                subtree_nodes[static_cast<size_t>(child_base + oct_new)].particle_idx = static_cast<int32_t>(p_idx);
                subtree_nodes[static_cast<size_t>(child_base + oct_new)].com_mass = glm::vec4(p_pos, p_mass);
                return;
            } else {
                curr = child_base + oct_new;
                continue;
            }
        }

        // Case 4: Internal node, descend
        glm::vec3 center(node.bbox_center_half);
        int oct = getOctant(p_pos, center);
        curr = subtree_nodes[static_cast<size_t>(curr)].first_child_idx + oct;
    }
}

glm::vec3 OctreeArena::computeForce(const glm::vec3& p_pos, float softening_sq, float theta_sq) const {
    glm::vec3 force(0.0f);
    if (node_count_ == 0 || nodes_[0].com_mass.w <= 0.0f) {
        return force;
    }

    int32_t curr = 0;
    while (curr >= 0) {
        const auto& node = nodes_[curr];

        // Skip empty nodes immediately
        if (node.com_mass.w <= 0.0f) {
            curr = node.next_node_idx;
            continue;
        }

        glm::vec3 dr = glm::vec3(node.com_mass) - p_pos;
        float dist_sq = glm::dot(dr, dr);
        float s = node.bbox_center_half.w * 2.0f; // cell size
        bool is_leaf = (node.first_child_idx < 0);

        // Multipole acceptance criterion: s^2 < theta^2 * dist_sq
        if (is_leaf || (s * s < theta_sq * dist_sq)) {
            if (dist_sq > 0.0f) {
                float eff_dist_sq = dist_sq + softening_sq;
                float inv_dist = 1.0f / (eff_dist_sq * std::sqrt(eff_dist_sq));
                force += dr * (G_CONSTANT * node.com_mass.w * inv_dist);
            }
            curr = node.next_node_idx;
        } else {
            curr = node.first_child_idx;
        }
    }

    return force;
}

} // namespace transport
