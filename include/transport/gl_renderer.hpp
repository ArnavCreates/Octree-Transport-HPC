#pragma once

#include "types.hpp"
#include <string>
#include <glm/glm.hpp>

struct GLFWwindow;

namespace transport {

class GlRenderer {
public:
    GlRenderer(int width = 1280, int height = 720, const std::string& title = "Octree-Transport-HPC");
    ~GlRenderer();

    GlRenderer(const GlRenderer&) = delete;
    GlRenderer& operator=(const GlRenderer&) = delete;

    bool shouldClose() const;
    void pollEvents();
    void renderFrame(const ParticleSystem& ps);
    void swapBuffers();

    bool isPaused() const { return paused_; }
    void togglePause() { paused_ = !paused_; }

    GLFWwindow* getWindow() const { return window_; }

private:
    int width_{1280};
    int height_{720};
    GLFWwindow* window_{nullptr};

    uint32_t render_prog_{0};
    uint32_t vao_{0};
    uint32_t vbo_pos_{0};
    uint32_t vbo_vel_{0};

    // Camera orbit parameters
    float camera_yaw_{0.0f};
    float camera_pitch_{0.3f};
    float camera_distance_{12.0f};
    float point_scale_{45.0f};
    bool paused_{false};

    double last_mouse_x_{0.0};
    double last_mouse_y_{0.0};
    bool mouse_pressed_{false};

    void initWindow(const std::string& title);
    void initShaders();
    void updateCamera();

    static void keyCallback(GLFWwindow* w, int key, int scancode, int action, int mods);
    static void mouseButtonCallback(GLFWwindow* w, int button, int action, int mods);
    static void cursorPosCallback(GLFWwindow* w, double xpos, double ypos);
    static void scrollCallback(GLFWwindow* w, double xoffset, double yoffset);
};

} // namespace transport
