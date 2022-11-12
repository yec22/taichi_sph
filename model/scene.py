import taichi as ti
import numpy as np
from .settings import *


@ti.data_oriented
class Scene:
    def __init__(self):
        self.N = ti.field(dtype=int, shape=()) # total particle numbers
        self.fuild_N = ti.field(dtype=int, shape=()) # fluid particle numbers
        self.boundary_N = ti.field(dtype=int, shape=()) # boundary particle numbers
        self.boundary_x = ti.Vector.field(Dim, dtype=ti.float32)

        # time varying properties of each particle
        self.density = ti.field(dtype=ti.float32)
        self.v = ti.Vector.field(Dim, dtype=ti.float32)
        self.x = ti.Vector.field(Dim, dtype=ti.float32) # range: [0, 1024 / Scale_Ratio]
        self.pressure = ti.field(dtype=ti.float32) # fluid pressure
        self.type = ti.field(dtype=int)

        # neignbors inside the support radius (for SPH computation)
        self.neighbor_idx = ti.field(dtype=int)
        self.neighbor_N = ti.field(dtype=int)

        # data layout
        self.particles = ti.root.dense(ti.i, MAX_Particle_Number)
        self.particles.place(self.density, self.v, self.x, self.pressure, self.type, self.neighbor_N)
        self.particles.dense(ti.j, MAX_Neighbor_Number).place(self.neighbor_idx)
        ti.root.dense(ti.i, MAX_Particle_Number).place(self.boundary_x)

        # grid for neighbor search
        self.N_per_grid = ti.field(dtype=int)
        self.idx_per_grid = ti.field(dtype=int)
        self.gridSize = np.ceil(np.array(GUI_Resolution) / Support_Radius).astype(int)
        self.grids = ti.root.dense(ti.ij if Dim == 2 else ti.ijk, self.gridSize)
        self.grids.place(self.N_per_grid)
        self.grids.dense(ti.k if Dim == 2 else ti.l, MAX_Particle_Per_Grid).place(self.idx_per_grid)

        self.init_scene()

    def add_stuff(self, stuff_type, pos_dim, v0):
        pos_dim[0][0] = min(max(pos_dim[0][0], Support_Radius), GUI_Resolution[0]/Scale_Ratio-Support_Radius)
        pos_dim[1][0] = min(max(pos_dim[1][0], Support_Radius), GUI_Resolution[0]/Scale_Ratio-Support_Radius)
        pos_dim[0][1] = min(max(pos_dim[0][1], Support_Radius), GUI_Resolution[1]/Scale_Ratio-Support_Radius)
        pos_dim[1][1] = min(max(pos_dim[1][1], Support_Radius), GUI_Resolution[1]/Scale_Ratio-Support_Radius)

        particle_pos_dim = [np.arange(pos_dim[0][0], pos_dim[0][1], Particle_Radius),
                            np.arange(pos_dim[1][0], pos_dim[1][1], Particle_Radius)]
        particle_pos = np.array(np.meshgrid(*particle_pos_dim, indexing='ij'), dtype=np.float32)
        particle_pos = particle_pos.reshape(Dim, -1).transpose(1, 0)
        particle_num = particle_pos.shape[0]

        if self.N[None] + particle_num >= MAX_Particle_Number:
            print("warning: max particle number!! cannot add anymore!!")
            return 0

        particle_velocity = np.full_like(particle_pos, v0).astype(np.float32)
        particle_density = np.full_like(np.zeros(particle_num), Density0).astype(np.float32)
        particle_pressure = np.full_like(np.zeros(particle_num), 0.0).astype(np.float32)
        particle_type = np.full_like(np.zeros(particle_num), stuff_type).astype(np.int32)
        self.add_particle(particle_num, particle_velocity, particle_pos, particle_density, particle_pressure, particle_type)
        return particle_num

    def init_scene(self):
        # init the scene
        self.N[None] = 0

        # add fluid_1 particles
        particle_num = self.add_stuff(FLUID, [[2.0, 6.0], [4.0, 6.0]], [0.0, -20.0])
        print('fluid1_particle_num', particle_num)

        # add fluid_2 particles
        particle_num = self.add_stuff(FLUID, [[7.5, 9.5], [6.0, 9.0]], [0.0, -15.0])
        print('fluid2_particle_num', particle_num)

        # add boundary particles
        # down
        particle_num = self.add_stuff(BOUNDARY, [[0, GUI_Resolution[0] / Scale_Ratio], [0, Support_Radius + Particle_Radius]], [0.0, 0.0])
        print('down_boundary_particle_num', particle_num)

        # up
        particle_num = self.add_stuff(BOUNDARY, [[0, GUI_Resolution[0] / Scale_Ratio], [GUI_Resolution[1] / Scale_Ratio - Support_Radius, GUI_Resolution[1] / Scale_Ratio + Particle_Radius]], [0.0, 0.0])
        print('up_boundary_particle_num', particle_num)

        # left
        particle_num = self.add_stuff(BOUNDARY, [[0, Support_Radius], [Support_Radius, GUI_Resolution[1] / Scale_Ratio - Support_Radius]], [0.0, 0.0])
        print('left_boundary_particle_num', particle_num)

        # right
        particle_num = self.add_stuff(BOUNDARY, [[GUI_Resolution[0] / Scale_Ratio - Support_Radius, GUI_Resolution[0] / Scale_Ratio + Particle_Radius], [Support_Radius, GUI_Resolution[1] / Scale_Ratio - Support_Radius]], [0.0, 0.0])
        print('right_boundary_particle_num', particle_num)

        # add two columns
        particle_num = self.add_stuff(BOUNDARY, COLUMN1, [0.0, 0.0])
        print('column1_particle_num', particle_num)

        particle_num = self.add_stuff(BOUNDARY, COLUMN2, [0.0, 0.0])
        print('column2_particle_num', particle_num)


    @ti.kernel
    def add_particle(self, particle_num: int, particle_velocity: ti.types.ndarray(), 
                     particle_pos: ti.types.ndarray(), particle_density: ti.types.ndarray(),
                     particle_pressure: ti.types.ndarray(), particle_type: ti.types.ndarray()):
        for p in range(particle_num):
            v = ti.Vector.zero(ti.float32, Dim)
            x = ti.Vector.zero(ti.float32, Dim)
            for i in ti.static(range(Dim)):
                v[i] = particle_velocity[p, i]
                x[i] = particle_pos[p, i]
            self.x[self.N[None]+p] = x
            self.v[self.N[None]+p] = v
            self.density[self.N[None]+p] = particle_density[p]
            self.pressure[self.N[None]+p] = particle_pressure[p]
            self.type[self.N[None]+p] = particle_type[p]

            if particle_type[p] == BOUNDARY:
                self.boundary_x[self.boundary_N[None]+p] = x
        
        self.N[None] += particle_num
        if particle_type[0] == FLUID:
            self.fuild_N[None] += particle_num
        if particle_type[1] == BOUNDARY:
            self.boundary_N[None] += particle_num
        
    
    @ti.kernel
    def allocate(self):
        for i in range(self.N[None]):
            grid_idx = (self.x[i] / Support_Radius).cast(int)
            grid_offset = ti.atomic_add(self.N_per_grid[grid_idx], 1)
            self.idx_per_grid[grid_idx, grid_offset] = i

    @ti.kernel
    def local_search(self):
        for p_i in range(self.N[None]):
            center_grid_idx = (self.x[p_i] / Support_Radius).cast(int)
            n_neighbors = 0
            for offset in ti.grouped(ti.ndrange(*((-1, 2),) * Dim)):
                grid_idx = center_grid_idx + offset
                flag = True
                jump_out = False
                for d in ti.static(range(Dim)):
                    flag = flag and (0 <= grid_idx[d] < self.gridSize[d])
                if flag:
                    for j in range(self.N_per_grid[grid_idx]):
                        p_j = self.idx_per_grid[grid_idx, j]
                        dist = (self.x[p_i] - self.x[p_j]).norm()
                        if p_i != p_j and dist < Support_Radius:
                            self.neighbor_idx[p_i, n_neighbors] = p_j
                            n_neighbors += 1
                            if n_neighbors >= MAX_Neighbor_Number:
                                jump_out = True
                                break
                if jump_out:
                    break
            self.neighbor_N[p_i] = n_neighbors
    
    @ti.kernel
    def reset(self):
        for i in range(self.N[None]):
            self.neighbor_N[i] = 0
        self.N_per_grid.fill(0)

    def search_neighbors(self):
        # reset
        self.reset()

        # allocate each particle to different grids
        self.allocate()

        # search only 3**Dim grids (locally)
        self.local_search()

    def get_particle_pos(self):
        fluid_pos = np.ndarray((self.N[None], Dim), dtype=np.float32)
        self.copy2numpy(self.N[None], self.x, fluid_pos)

        boundary_pos = np.ndarray((self.boundary_N[None], Dim), dtype=np.float32)
        self.copy2numpy(self.boundary_N[None], self.boundary_x, boundary_pos)
        return fluid_pos, boundary_pos
    
    @ti.kernel
    def copy2numpy(self, num: ti.int32, src_arr: ti.template(), np_arr: ti.types.ndarray()):
        for i in range(num):
            for j in ti.static(range(Dim)):
                np_arr[i, j] = src_arr[i][j]
                
    

# unit_test
if __name__ == "__main__":
    import os
    ti.init(arch=ti.gpu, device_memory_GB=4, packed=True)
    scene = Scene()

    gui = ti.GUI(background_color=Background_Color, show_gui=False, res=GUI_Resolution)
    cnt = 0
    fluid_pos, boundary_pos = scene.get_particle_pos()
    while gui.running:
        gui.circles(pos=fluid_pos * Scale_Ratio / GUI_Resolution[0], # range: [0, 1]
                    radius=Particle_Radius * Visualize_Ratio * Scale_Ratio,
                    color=Particle_Color)
        gui.circles(pos=boundary_pos * Scale_Ratio / GUI_Resolution[0], # range: [0, 1]
                    radius=Particle_Radius * Visualize_Ratio * Scale_Ratio,
                    color=Boundary_Color)
        cnt += 1
        if cnt > 1:
            break
        if not os.path.exists('debug'):
            os.mkdir('debug')
        filename = f'debug/frame_{cnt:05d}.png'
        gui.show(filename)
