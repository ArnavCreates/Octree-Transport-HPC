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

    // Color gradient based on velocity magnitude
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
