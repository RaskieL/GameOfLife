import pygame
import moderngl
import numpy as np
import sys
import math

# CONFIGURATION
WIDTH, HEIGHT = 1280, 720
GROUP_SIZE_X, GROUP_SIZE_Y = 16, 16

# SHADERS

# 1. LIFE SHADER (Conway)
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

    // Keep the velocity (gb) for the next step
    imageStore(temp_grid, pos, vec4(next_alive, cell.g, cell.b, 1.0));
}}
"""

# 2. PHYSICS SHADER (STABILIZED)
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
    
    // If the particle is dead, we clear its velocity and exit
    if (cell.r < 0.5) {{
        imageStore(output_grid, pos, vec4(0.0, 0.0, 0.0, 1.0));
        return; 
    }}

    vec2 current_pos = vec2(pos);
    vec2 vel = cell.gb; 

    vec2 acc = vec2(0.0);

    // Soft Gravity
    for (int i=0; i<num_stars; i++) {{
        vec2 d = stars[i].pos - current_pos;
        float dist_sq = dot(d, d);
        float dist = sqrt(dist_sq);
        
        // STABILIZATION 1: Large safety radius
        // If it gets closer than 20 pixels, gravity stops increasing.
        // This avoids the infinite 'catapult' effect.
        if (dist < 20.0) dist = 20.0; 
        
        // STABILIZATION 2: Softened gravity
        // The +200.0 softens the force curve
        float force = stars[i].mass / (dist_sq + 200.0);
        
        acc += (d / dist) * force;
    }}
    
    // Acceleration clamp (avoids sudden jerks)
    acc = clamp(acc, vec2(-0.5), vec2(0.5));

    vel += acc;

    // STABILIZATION 3: High Friction
    // We slow down particles quickly (0.90) so they don't accumulate infinite inertia
    vel *= 0.90; 

    // STABILIZATION 4: Strict Speed Limit
    // Maximum 1.5 pixels per frame. This prevents them from 'breaking' the Game of Life.
    vel = clamp(vel, vec2(-1.5), vec2(1.5));

    vec2 next_pos_f = current_pos + vel;
    
    // Bounces
    if (next_pos_f.x < 0.0 || next_pos_f.x >= size.x) {{ vel.x *= -0.8; next_pos_f.x = clamp(next_pos_f.x, 0.0, size.x-1.0); }}
    if (next_pos_f.y < 0.0 || next_pos_f.y >= size.y) {{ vel.y *= -0.8; next_pos_f.y = clamp(next_pos_f.y, 0.0, size.y-1.0); }}

    ivec2 target = ivec2(round(next_pos_f));
    
    // We write to the new position
    // Important: We write 1.0 (alive) in red
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
    // 1. Draw Stars (Golden Circles)
    vec2 pixel_pos = v_texcoord * resolution;
    for (int i=0; i<num_stars; i++) {
        float d = distance(pixel_pos, star_pos[i]);
        if (d < 12.0) { 
            f_color = vec4(1.0, 0.8, 0.0, 1.0); 
            return; 
        }
    }

    // 2. Draw Particles
    vec4 data = texture(tex, v_texcoord);
    if (data.r > 0.5) {
        // Color based on speed (Green still, Blue fast)
        float speed = length(data.gb);
        f_color = vec4(0.2, 0.5 + speed, 0.8 + speed, 1.0);
    }
    else {
        f_color = vec4(0.0, 0.0, 0.0, 1.0);
    }
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
        
        # Stars with masses adjusted for the new physics
        self.stars = [
            {'pos': [width*0.5, height*0.5], 'vel': [0.0, 0.0], 'mass': 400.0}, # Central Sun
            {'pos': [width*0.5 + 200, height*0.5], 'vel': [0.0, 2.0], 'mass': 200.0}, # Planet
        ]
        self.update_physics_uniforms()

    def create_texture(self):
        tex = self.ctx.texture((self.width, self.height), 4, dtype='f4')
        tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        return tex

    def randomize_grid(self):
        total = self.width * self.height
        # Fewer initial particles to avoid chaos (0.1 instead of 0.2)
        alive = (np.random.rand(total) < 0.1).astype(np.float32)
        vel = (np.random.rand(total * 2) - 0.5).astype(np.float32) * 0.0
        zeros = np.zeros(total, dtype=np.float32)
        data = np.dstack((alive, vel[::2], vel[1::2], zeros)).flatten().tobytes()
        self.tex_current.write(data)

    def update_physics_uniforms(self):
        self.prog_physics['num_stars'] = len(self.stars)
        for i, s in enumerate(self.stars):
            self.prog_physics[f'stars[{i}].pos'] = tuple(s['pos'])
            self.prog_physics[f'stars[{i}].mass'] = s['mass']

    # STAR PHYSICS (CPU)
    def update_star_physics(self, mouse_pos):
        # Mouse attracts the Sun
        dx = mouse_pos[0] - self.stars[0]['pos'][0]
        dy = mouse_pos[1] - self.stars[0]['pos'][1]
        dist = math.sqrt(dx*dx + dy*dy)
        if dist > 1.0:
            force_mouse = 0.01 * dist # Very soft mouse attraction
            self.stars[0]['vel'][0] += (dx / dist) * force_mouse * 0.05
            self.stars[0]['vel'][1] += (dy / dist) * force_mouse * 0.05

        # Gravity between stars
        for i in range(len(self.stars)):
            ax, ay = 0.0, 0.0
            for j in range(len(self.stars)):
                if i == j: continue
                dx = self.stars[j]['pos'][0] - self.stars[i]['pos'][0]
                dy = self.stars[j]['pos'][1] - self.stars[i]['pos'][1]
                dist_sq = dx*dx + dy*dy
                dist = math.sqrt(dist_sq)
                if dist < 20.0: dist = 20.0 # Seguridad
                
                force = (self.stars[j]['mass'] * 0.5) / (dist_sq + 100.0)
                ax += (dx / dist) * force
                ay += (dy / dist) * force

            self.stars[i]['vel'][0] += ax
            self.stars[i]['vel'][1] += ay

        # Move stars
        for s in self.stars:
            s['vel'][0] *= 0.99
            s['vel'][1] *= 0.99
            s['pos'][0] += s['vel'][0]
            s['pos'][1] += s['vel'][1]

        # Bounces
            if s['pos'][0] < 0: s['pos'][0]=0; s['vel'][0]*=-0.7
            if s['pos'][0] >= self.width: s['pos'][0]=self.width-1; s['vel'][0]*=-0.7
            if s['pos'][1] < 0: s['pos'][1]=0; s['vel'][1]*=-0.7
            if s['pos'][1] >= self.height: s['pos'][1]=self.height-1; s['vel'][1]*=-0.7

    def step(self):
        gw = int(np.ceil(self.width / GROUP_SIZE_X))
        gh = int(np.ceil(self.height / GROUP_SIZE_Y))

        # 1. Calculate Life
        self.tex_current.bind_to_image(0, read=True, write=False)
        self.tex_temp.bind_to_image(1, read=False, write=True)
        self.prog_life.run(gw, gh)

        # 2. Calculate Physics (Movement)
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
    pygame.display.set_caption("GPU Stable Gravity")
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF)
    
    ctx = moderngl.create_context()
    ctx.enable(moderngl.BLEND)

    prog_render = ctx.program(vertex_shader=SRC_VERT, fragment_shader=SRC_FRAG)
    prog_render['resolution'].value = (WIDTH, HEIGHT)
    
    vertices = np.array([-1, -1, 0, 0,  1, -1, 1, 0,  -1, 1, 0, 1,  1, 1, 1, 1], dtype='f4')
    vbo = ctx.buffer(vertices.tobytes())
    vao = ctx.vertex_array(prog_render, [(vbo, '2f 2f', 'in_vert', 'in_texcoord')])

    sim = GpuSimulation(ctx, WIDTH, HEIGHT)
    clock = pygame.time.Clock()

    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()
        mouse_target = (mouse_pos[0], HEIGHT - mouse_pos[1])

        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                sim.randomize_grid()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False

        # Star Physics
        sim.update_star_physics(mouse_target)
        sim.update_physics_uniforms()

        # Visual Uniforms
        positions = [tuple(s['pos']) for s in sim.stars]
        while len(positions) < 10: positions.append((0.0, 0.0))
        
        prog_render['num_stars'].value = len(sim.stars)
        if 'star_pos' in prog_render:
            prog_render['star_pos'].value = positions
        elif 'star_pos[0]' in prog_render:
            prog_render['star_pos[0]'].value = positions

        # GPU Step
        sim.step()

        # Render
        ctx.clear(0.1, 0.1, 0.1)
        sim.render(vao)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()