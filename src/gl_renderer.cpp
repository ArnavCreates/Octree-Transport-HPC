#include "transport/gl_renderer.hpp"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cmath>

namespace transport {

namespace {

const char* EMBEDDED_VERT_SRC = R"glsl(
#version 430 core
layout(location = 0) in vec4 in_pos_mass;
layout(location = 1) in vec4 in_vel;

uniform mat4 u_view_proj;
uniform float u_point_scale;

out vec3 v_color;

void main() {
    gl_Position = u_view_proj * vec4(in_pos_mass.xyz, 1.0);

    float dist = gl_Position.w;
    gl_PointSize = clamp(u_point_scale / max(dist, 0.1), 1.5, 32.0);

    float speed = length(in_vel.xyz);
    float t = clamp(speed * 0.4, 0.0, 1.0);

    vec3 c_cold = vec3(0.15, 0.4, 1.0);
    vec3 c_mid  = vec3(0.1, 0.85, 0.85);
    vec3 c_hot  = vec3(1.0, 0.6, 0.1);
    vec3 c_peak = vec3(1.0, 0.98, 0.95);

    if (t < 0.33) {
        v_color = mix(c_cold, c_mid, t / 0.33);
    } else if (t < 0.75) {
        v_color = mix(c_mid, c_hot, (t - 0.33) / 0.42);
    } else {
        v_color = mix(c_hot, c_peak, (t - 0.75) / 0.25);
    }
}
)glsl";

const char* EMBEDDED_FRAG_SRC = R"glsl(
#version 430 core
in vec3 v_color;
out vec4 frag_color;

void main() {
    vec2 coord = gl_PointCoord - vec2(0.5);
    float r2 = dot(coord, coord);
    if (r2 > 0.25) discard;

    float alpha = exp(-r2 * 10.0);
    frag_color = vec4(v_color, alpha);
}
)glsl";

} // namespace

GlRenderer::GlRenderer(int width, int height, const std::string& title)
    : width_(width), height_(height) {
    initWindow(title);
    initShaders();
}

GlRenderer::~GlRenderer() {
    if (vao_) glDeleteVertexArrays(1, &vao_);
    if (vbo_pos_) glDeleteBuffers(1, &vbo_pos_);
    if (vbo_vel_) glDeleteBuffers(1, &vbo_vel_);
    if (render_prog_) glDeleteProgram(render_prog_);
    if (window_) {
        glfwDestroyWindow(window_);
        window_ = nullptr;
    }
}

void GlRenderer::initWindow(const std::string& title) {
    if (!glfwInit()) {
        throw std::runtime_error("GlRenderer: Failed to initialize GLFW");
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);

    window_ = glfwCreateWindow(width_, height_, title.c_str(), nullptr, nullptr);
    if (!window_) {
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        window_ = glfwCreateWindow(width_, height_, title.c_str(), nullptr, nullptr);
    }

    if (!window_) {
        throw std::runtime_error("GlRenderer: Failed to create GLFW window");
    }

    glfwMakeContextCurrent(window_);
    glfwSetWindowUserPointer(window_, this);

    glfwSetKeyCallback(window_, keyCallback);
    glfwSetMouseButtonCallback(window_, mouseButtonCallback);
    glfwSetCursorPosCallback(window_, cursorPosCallback);
    glfwSetScrollCallback(window_, scrollCallback);

    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        throw std::runtime_error(std::string("GlRenderer: glewInit failed: ") +
                                 reinterpret_cast<const char*>(glewGetErrorString(err)));
    }

    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE); // Additive blending for luminous cosmic glow
    glDisable(GL_DEPTH_TEST);
}

void GlRenderer::initShaders() {
    std::string v_src = EMBEDDED_VERT_SRC;
    std::string f_src = EMBEDDED_FRAG_SRC;

    std::ifstream v_file("shaders/particle.vert");
    if (v_file.is_open()) {
        std::stringstream ss; ss << v_file.rdbuf(); v_src = ss.str();
    }
    std::ifstream f_file("shaders/particle.frag");
    if (f_file.is_open()) {
        std::stringstream ss; ss << f_file.rdbuf(); f_src = ss.str();
    }

    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    const char* v_ptr = v_src.c_str();
    glShaderSource(vs, 1, &v_ptr, nullptr);
    glCompileShader(vs);

    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    const char* f_ptr = f_src.c_str();
    glShaderSource(fs, 1, &f_ptr, nullptr);
    glCompileShader(fs);

    render_prog_ = glCreateProgram();
    glAttachShader(render_prog_, vs);
    glAttachShader(render_prog_, fs);
    glLinkProgram(render_prog_);

    glDeleteShader(vs);
    glDeleteShader(fs);

    glGenVertexArrays(1, &vao_);
    glGenBuffers(1, &vbo_pos_);
    glGenBuffers(1, &vbo_vel_);

    glBindVertexArray(vao_);

    glBindBuffer(GL_ARRAY_BUFFER, vbo_pos_);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), (void*)0);

    glBindBuffer(GL_ARRAY_BUFFER, vbo_vel_);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), (void*)0);

    glBindVertexArray(0);
}

bool GlRenderer::shouldClose() const {
    return window_ ? glfwWindowShouldClose(window_) : true;
}

void GlRenderer::pollEvents() {
    glfwPollEvents();
}

void GlRenderer::renderFrame(const ParticleSystem& ps) {
    if (!window_) return;

    int w, h;
    glfwGetFramebufferSize(window_, &w, &h);
    glViewport(0, 0, w, h);
    glClearColor(0.01f, 0.015f, 0.03f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    size_t n = ps.size();
    if (n == 0) return;

    // Upload particle positions and velocities to VBOs
    glBindBuffer(GL_ARRAY_BUFFER, vbo_pos_);
    glBufferData(GL_ARRAY_BUFFER, n * sizeof(glm::vec4), ps.pos_mass.data(), GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, vbo_vel_);
    glBufferData(GL_ARRAY_BUFFER, n * sizeof(glm::vec4), ps.vel_pad.data(), GL_DYNAMIC_DRAW);

    // Compute View-Projection Matrix
    float aspect = (h > 0) ? (static_cast<float>(w) / static_cast<float>(h)) : 1.0f;
    glm::mat4 proj = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 1000.0f);

    float cx = camera_distance_ * std::cos(camera_pitch_) * std::sin(camera_yaw_);
    float cy = camera_distance_ * std::sin(camera_pitch_);
    float cz = camera_distance_ * std::cos(camera_pitch_) * std::cos(camera_yaw_);
    glm::vec3 eye(cx, cy, cz);
    glm::mat4 view = glm::lookAt(eye, glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4 vp = proj * view;

    glUseProgram(render_prog_);
    glUniformMatrix4fv(glGetUniformLocation(render_prog_, "u_view_proj"), 1, GL_FALSE, glm::value_ptr(vp));
    glUniform1f(glGetUniformLocation(render_prog_, "u_point_scale"), point_scale_);

    glBindVertexArray(vao_);
    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(n));
    glBindVertexArray(0);
}

void GlRenderer::swapBuffers() {
    if (window_) {
        glfwSwapBuffers(window_);
    }
}

void GlRenderer::keyCallback(GLFWwindow* w, int key, int scancode, int action, int mods) {
    auto* self = static_cast<GlRenderer*>(glfwGetWindowUserPointer(w));
    if (!self) return;

    if (action == GLFW_PRESS) {
        if (key == GLFW_KEY_ESCAPE) {
            glfwSetWindowShouldClose(w, GLFW_TRUE);
        } else if (key == GLFW_KEY_SPACE) {
            self->togglePause();
        } else if (key == GLFW_KEY_Q) {
            self->point_scale_ = (self->point_scale_ < 60.0f) ? 80.0f : 35.0f;
        }
    }
}

void GlRenderer::mouseButtonCallback(GLFWwindow* w, int button, int action, int mods) {
    auto* self = static_cast<GlRenderer*>(glfwGetWindowUserPointer(w));
    if (!self) return;

    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            self->mouse_pressed_ = true;
            glfwGetCursorPos(w, &self->last_mouse_x_, &self->last_mouse_y_);
        } else if (action == GLFW_RELEASE) {
            self->mouse_pressed_ = false;
        }
    }
}

void GlRenderer::cursorPosCallback(GLFWwindow* w, double xpos, double ypos) {
    auto* self = static_cast<GlRenderer*>(glfwGetWindowUserPointer(w));
    if (!self || !self->mouse_pressed_) return;

    double dx = xpos - self->last_mouse_x_;
    double dy = ypos - self->last_mouse_y_;
    self->last_mouse_x_ = xpos;
    self->last_mouse_y_ = ypos;

    self->camera_yaw_ += static_cast<float>(dx) * 0.005f;
    self->camera_pitch_ += static_cast<float>(dy) * 0.005f;
    self->camera_pitch_ = glm::clamp(self->camera_pitch_, -1.5f, 1.5f);
}

void GlRenderer::scrollCallback(GLFWwindow* w, double xoffset, double yoffset) {
    auto* self = static_cast<GlRenderer*>(glfwGetWindowUserPointer(w));
    if (!self) return;

    self->camera_distance_ -= static_cast<float>(yoffset) * 1.0f;
    self->camera_distance_ = glm::clamp(self->camera_distance_, 2.0f, 200.0f);
}

} // namespace transport
