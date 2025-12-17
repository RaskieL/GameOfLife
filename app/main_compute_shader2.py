import pygame
import moderngl
import numpy as np
import sys
import struct
from typing import Tuple

WIDTH, HEIGHT = 1280, 720
GROUP_SIZE_X, GROUP_SIZE_Y = 16, 16

# SHADERS
SRC_COMPUTE_LIFE = f"""
#version 430
layout(local_size_x={GROUP_SIZE_X}, local_size_y={GROUP_SIZE_Y}) in;

layout(rgba32f, binding=0) uniform image2D input_grid;
layout(rgba32f, binding=1) uniform image2D temp_grid;

void main() {{
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(input_grid);
    if (pos.x >= size.x || pos.y >= size.y) return;

    vec4 cell = imageLoad(input_grid, pos);
    float alive = cell.r;

    int neighbors = 0;
    for (int dy = -1; dy <= 1; dy++) {{
        for (int dx = -1; dx <= 1; dx++) {{
            if (dx == 0 && dy == 0) continue;
            ivec2 n_pos = (pos + ivec2(dx, dy) + size) % size;
            if (imageLoad(input_grid, n_pos).r > 0.5) neighbors++;
        }}
    }}

    float next_alive = 0.0;
    if (alive > 0.5) {{ 
        if (neighbors == 2 || neighbors == 3) next_alive = 1.0;
    }} else {{
        if (neighbors == 3) next_alive = 1.0;
    }}

    imageStore(temp_grid, pos, vec4(next_alive, cell.g, cell.b, 1.0));
}}
"""

# 2. Physics Shader (Newton): Moves surviving cells.
SRC_COMPUTE_PHYSICS = f"""
#version 430
layout(local_size_x={GROUP_SIZE_X}, local_size_y={GROUP_SIZE_Y}) in;

layout(rgba32f, binding=1) uniform image2D temp_grid;
layout(rgba32f, binding=2) uniform image2D output_grid;

struct Star {{
    vec2 pos;
    float mass;
}};
uniform int num_stars;
uniform Star stars[10]; 

void main() {{
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(temp_grid);
    if (pos.x >= size.x || pos.y >= size.y) return;

    vec4 cell = imageLoad(temp_grid, pos);
    
    if (cell.r < 0.5) return; 

    vec2 current_pos = vec2(pos);
    vec2 vel = cell.gb; 

    vec2 acc = vec2(0.0);
    for (int i=0; i<num_stars; i++) {{
        vec2 d = stars[i].pos - current_pos;
        float dist_sq = dot(d, d);
        float dist = sqrt(dist_sq);
        if (dist < 5.0) dist = 5.0; 
        
        float force = stars[i].mass / (dist_sq + 1.0);
        acc += (d / dist) * force;
    }}
    
    vel += acc;
    vel *= 0.98; // Friction
    vel = clamp(vel, vec2(-2.0), vec2(2.0));

    vec2 next_pos_f = current_pos + vel;
    
    if (next_pos_f.x < 0.0 || next_pos_f.x >= size.x) {{ vel.x *= -0.9; next_pos_f.x = clamp(next_pos_f.x, 0.0, size.x-1.0); }}
    if (next_pos_f.y < 0.0 || next_pos_f.y >= size.y) {{ vel.y *= -0.9; next_pos_f.y = clamp(next_pos_f.y, 0.0, size.y-1.0); }}

    ivec2 target = ivec2(round(next_pos_f));
    imageStore(output_grid, target, vec4(1.0, vel.x, vel.y, 1.0));
}}
"""

SRC_VERT = """
#version 330
in vec2 in_vert;
in vec2 in_texcoord;
out vec2 v_texcoord;
void main() { gl_Position = vec4(in_vert, 0.0, 1.0); v_texcoord = in_texcoord; }
"""

SRC_FRAG = """
#version 330
uniform sampler2D tex;
uniform vec2 resolution;
uniform vec2 star_pos[10];
uniform int num_stars; 

in vec2 v_texcoord;
out vec4 f_color;

void main() {
    // 1. Draw stars (on top)
    vec2 pixel_pos = v_texcoord * resolution;
    
    for (int i=0; i<num_stars; i++) {
        float d = distance(pixel_pos, star_pos[i]);
        if (d < 15.0) { // Visual radius of 15 pixels
            f_color = vec4(1.0, 0.8, 0.0, 1.0); // Golden yellow
            return; // Already drawn, exit
        }
    }

    // 2. Draw particles (if not star)
    vec4 data = texture(tex, v_texcoord);
    if (data.r > 0.5) f_color = vec4(0.8, 0.8 + abs(data.g), 0.8 + abs(data.b), 1.0);
    else f_color = vec4(0.0, 0.0, 0.0, 1.0);
}
"""

class GpuSimulation:
    def __init__(self, ctx, width, height):
        self.ctx = ctx
        self.width, self.height = width, height
        
        self.tex_current = self.create_texture()
        self.tex_temp = self.create_texture()
        self.tex_next = self.create_texture()
        
        self.randomize_grid()
        
        self.prog_life = self.ctx.compute_shader(SRC_COMPUTE_LIFE)
        self.prog_physics = self.ctx.compute_shader(SRC_COMPUTE_PHYSICS)
        
        self.stars = [
            {'pos': (width*0.5, height*0.5), 'mass': 100.0},
            {'pos': (width*0.2, height*0.2), 'mass': 50.0},
        ]
        self.update_uniforms()

    def create_texture(self):
        tex = self.ctx.texture((self.width, self.height), 4, dtype='f4')
        tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        return tex

    def randomize_grid(self):
        total = self.width * self.height
        alive = (np.random.rand(total) < 0.2).astype(np.float32)
        vel = (np.random.rand(total * 2) - 0.5).astype(np.float32) * 0.0
        zeros = np.zeros(total, dtype=np.float32)
        data = np.dstack((alive, vel[::2], vel[1::2], zeros)).flatten().tobytes()
        self.tex_current.write(data)

    def update_uniforms(self):
        self.prog_physics['num_stars'] = len(self.stars)
        for i, s in enumerate(self.stars):
            self.prog_physics[f'stars[{i}].pos'] = s['pos']
            self.prog_physics[f'stars[{i}].mass'] = s['mass']

    def step(self):
        gw = int(np.ceil(self.width / GROUP_SIZE_X))
        gh = int(np.ceil(self.height / GROUP_SIZE_Y))

        fbo = self.ctx.framebuffer(color_attachments=[self.tex_next])
        fbo.clear(0.0, 0.0, 0.0, 0.0)

        self.tex_current.bind_to_image(0, read=True, write=False)
        self.tex_temp.bind_to_image(1, read=False, write=True)
        self.prog_life.run(gw, gh)

        self.tex_temp.bind_to_image(1, read=True, write=False)
        self.tex_next.bind_to_image(2, read=False, write=True)
        self.prog_physics.run(gw, gh)

        self.tex_current, self.tex_next = self.tex_next, self.tex_current

    def render(self, quad_vao):
        self.tex_current.use(0)
        quad_vao.render(moderngl.TRIANGLE_STRIP)

# MAIN

def main():
    pygame.init()
    pygame.display.set_caption("GPU Conway + Gravity")
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF)
    ctx = moderngl.create_context()
    ctx.enable(moderngl.BLEND)

    # Quad for rendering
    prog_render = ctx.program(vertex_shader=SRC_VERT, fragment_shader=SRC_FRAG)
    
    # CHANGE: Send resolution to shader once
    prog_render['resolution'].value = (WIDTH, HEIGHT)
    
    vertices = np.array([-1, -1, 0, 0,  1, -1, 1, 0,  -1, 1, 0, 1,  1, 1, 1, 1], dtype='f4')
    vbo = ctx.buffer(vertices.tobytes())
    vao = ctx.vertex_array(prog_render, [(vbo, '2f 2f', 'in_vert', 'in_texcoord')])

    sim = GpuSimulation(ctx, WIDTH, HEIGHT)
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.MOUSEMOTION:
                x, y = event.pos
                sim.stars[0]['pos'] = (x, HEIGHT - y)
                sim.update_uniforms()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                sim.randomize_grid()
            

        # Update visual star positions
        prog_render['num_stars'].value = len(sim.stars)
        
        # 1. Create a list of all positions
        # The shader expects an array of size 10 (vec2 star_pos[10])
        # We must provide exactly 10 values, so we pad the list with (0.0, 0.0)
        positions = [s['pos'] for s in sim.stars]
        while len(positions) < 10:
            positions.append((0.0, 0.0))
            
        # 2. Send the whole array at once using the base name
        # We use .get() or check existence to be safe, but usually the key is 'star_pos'
        if 'star_pos' in prog_render:
            prog_render['star_pos'].value = positions
        elif 'star_pos[0]' in prog_render:
            # Fallback for drivers that force indexed naming
            prog_render['star_pos[0]'].value = positions

        sim.step()

        ctx.clear(0.1, 0.1, 0.1)
        sim.render(vao)
        pygame.display.flip()
        
        pygame.display.set_caption(f"FPS: {clock.get_fps():.1f}")
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()