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
