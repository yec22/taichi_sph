import taichi as ti
from .settings import *
from .scene import *
from .utils import *

@ti.data_oriented
class WCSPH:
    def __init__(self):
        self.scene = Scene() # define the scene
        self.dt = ti.field(dtype=ti.float32, shape=()) # WCSPH time step (smaller than PBF)
        self.dt[None] = 2e-4
        self.m = Density0 * Particle_Volume # mass = density * volume
        self.accleration = ti.Vector.field(Dim, dtype=ti.float32)
        ti.root.dense(ti.i, MAX_Particle_Number).place(self.accleration)
        self.bound = np.array(GUI_Resolution) / Scale_Ratio
        self.board = ti.field(dtype=ti.float32, shape=())
        self.board[None] = GUI_Resolution[0] / Scale_Ratio

    # update the velocity and position for each particle
    @ti.kernel
    def step(self, cnt: int):
        self.board[None] = GUI_Resolution[0] / Scale_Ratio
        for i in range(self.scene.N[None]):
            if self.scene.type[i] == SOLID: # moving board
                self.scene.v[i][0] = Board_v0 * ti.sin(60 * np.pi * cnt * self.dt[None])
                self.scene.v[i][1] = 0.0
                self.scene.x[i] += self.scene.v[i] * self.dt[None]
                self.scene.boundary_x[i] = self.scene.x[i]
                self.board[None] = ti.min(self.board[None], self.scene.boundary_x[i][0])
            if self.scene.type[i] == BOUNDARY: # boundary particles always stay still
                continue
            if self.scene.type[i] == FLUID: # fluid particles
                self.scene.v[i] += self.accleration[i] * self.dt[None] # dv = a * dt
                self.scene.x[i] += self.scene.v[i] * self.dt[None] # dx = v * dt
    
    @ti.kernel
    def advection(self):
        for p_i in range(self.scene.N[None]):
            if self.scene.type[p_i] == BOUNDARY:
                continue
            if self.scene.type[p_i] == SOLID:
                continue
            # a = g + Viscosity_Coefficient * Laplacian(v)
            # Laplacian(v) is computed via SPH
            a = ti.Vector.zero(ti.float32, Dim)
            a[Dim-1] = G
            for j in range(self.scene.neighbor_N[p_i]):
                p_j = self.scene.neighbor_idx[p_i, j]
                dist = self.scene.x[p_i] - self.scene.x[p_j]
                v_ij = (self.scene.v[p_i] - self.scene.v[p_j]).dot(dist)
                viscosity_accleration = Viscosity_Coefficient * 2 * (Dim + 2) * (self.m / self.scene.density[p_j]) * v_ij / (dist.dot(dist) + 0.01 * (Support_Radius ** 2)) * smooth_kernel_derivative(dist)
                a += viscosity_accleration
            self.accleration[p_i] = a
    
    @ti.kernel
    def projection(self):
        for p_i in range(self.scene.N[None]):
            if self.scene.type[p_i] == BOUNDARY:
                continue
            if self.scene.type[p_i] == SOLID:
                continue
            # a = - m_i / density_i * Laplacian(pressure)
            # Laplacian(pressure) is computed via SPH
            a = ti.Vector.zero(ti.float32, Dim)
            for j in range(self.scene.neighbor_N[p_i]):
                p_j = self.scene.neighbor_idx[p_i, j]
                dist = self.scene.x[p_i] - self.scene.x[p_j]
                pressure_accleration = -self.m * (self.scene.pressure[p_i] / self.scene.density[p_i] ** 2 + self.scene.pressure[p_j] / self.scene.density[p_j] ** 2) * smooth_kernel_derivative(dist)
                a += pressure_accleration
            self.accleration[p_i] += a

    @ti.kernel
    def compute_pressure(self):
        # density_i = Sigma(m_j * W_ij), which W_ij is given by the kernel function
        for p_i in range(self.scene.N[None]):
            self.scene.density[p_i] = 0.0
            for j in range(self.scene.neighbor_N[p_i]):
                p_j = self.scene.neighbor_idx[p_i, j]
                dist = (self.scene.x[p_i] - self.scene.x[p_j]).norm()
                self.scene.density[p_i] += self.m * smooth_kernel(dist)
            self.scene.density[p_i] = ti.max(self.scene.density[p_i], Density0) # uncompressible fluid
            self.scene.pressure[p_i] = Kappa_Coefficient * (ti.pow(self.scene.density[p_i] / Density0, Exponential_Coefficient) - 1.0)
            self.scene.pressure[p_i] = ti.max(self.scene.pressure[p_i], 0.0) # clamp the negative pressure (constrain the free surface)

    @ti.func
    def simulate_collisions(self, p_i, vec, d):
        # Collision factor, assume roughly (1-c_f)*velocity loss after collision
        c_f = 0.3
        self.scene.x[p_i] += vec * d
        self.scene.v[p_i] -= (1.0 + c_f) * self.scene.v[p_i].dot(vec) * vec

    @ti.kernel
    def handle_boundary(self):
        for p_i in range(self.scene.N[None]):
            if self.scene.type[p_i] == BOUNDARY:
                continue
            if self.scene.type[p_i] == SOLID:
                continue
            if Dim == 2 and self.scene.scene_idx == 1:
                pos = self.scene.x[p_i]
                if pos[0] < Support_Radius:
                    self.simulate_collisions(p_i, ti.Vector([1.0, 0.0]), Support_Radius - pos[0])
                if pos[0] > self.bound[0] - Support_Radius:
                    self.simulate_collisions(p_i, ti.Vector([-1.0, 0.0]), pos[0] - (self.bound[0] - Support_Radius))
                if pos[1] > self.bound[1] - Support_Radius:
                    self.simulate_collisions(p_i, ti.Vector([0.0, -1.0]), pos[1] - (self.bound[1] - Support_Radius))
                if pos[1] < Support_Radius:
                    self.simulate_collisions(p_i, ti.Vector([0.0, 1.0]), Support_Radius - pos[1])

                if (pos[0] > COLUMN1[0][0] and pos[0] < COLUMN1[0][0] + 0.1) and (COLUMN1[1][0] < pos[1] < COLUMN1[1][1]):
                    self.simulate_collisions(p_i, ti.Vector([-1.0, 0.0]), pos[0] - COLUMN1[0][0])
                if (pos[0] < COLUMN1[0][1] and pos[0] > COLUMN1[0][1] - 0.1) and (COLUMN1[1][0] < pos[1] < COLUMN1[1][1]):
                    self.simulate_collisions(p_i, ti.Vector([1.0, 0.0]), COLUMN1[0][1] - pos[0])
                if (COLUMN1[0][0] < pos[0] < COLUMN1[0][1]) and (pos[1] < COLUMN1[1][1] and pos[1] > COLUMN1[1][1] - 0.1):
                    self.simulate_collisions(p_i, ti.Vector([0.0, 1.0]), COLUMN1[1][1] - pos[1])
                
                if (pos[0] > COLUMN2[0][0] and pos[0] < COLUMN2[0][0] + 0.1) and (COLUMN2[1][0] < pos[1] < COLUMN2[1][1]):
                    self.simulate_collisions(p_i, ti.Vector([-1.0, 0.0]), pos[0] - COLUMN2[0][0])
                if (pos[0] < COLUMN2[0][1] and pos[0] > COLUMN2[0][1] - 0.1) and (COLUMN2[1][0] < pos[1] < COLUMN2[1][1]):
                    self.simulate_collisions(p_i, ti.Vector([1.0, 0.0]), COLUMN2[0][1] - pos[0])
                if (COLUMN2[0][0] < pos[0] < COLUMN2[0][1]) and (pos[1] < COLUMN2[1][1] and pos[1] > COLUMN2[1][1] - 0.1):
                    self.simulate_collisions(p_i, ti.Vector([0.0, 1.0]), COLUMN2[1][1] - pos[1])
            
            if Dim == 2 and self.scene.scene_idx == 2:
                pos = self.scene.x[p_i]
                if pos[0] > self.board[None]:
                    self.simulate_collisions(p_i, ti.Vector([-1.0, 0.0]), pos[0] - self.board[None])
                if pos[0] < Support_Radius:
                    self.simulate_collisions(p_i, ti.Vector([1.0, 0.0]), Support_Radius - pos[0])
                if pos[1] > self.bound[1] - Support_Radius:
                    self.simulate_collisions(p_i, ti.Vector([0.0, -1.0]), pos[1] - (self.bound[1] - Support_Radius))
                if pos[1] < Support_Radius:
                    self.simulate_collisions(p_i, ti.Vector([0.0, 1.0]), Support_Radius - pos[1])
                

    # Weakly Compressible SPH Solver
    def solve(self, cnt):
        self.scene.search_neighbors() # search neighbors
        self.compute_pressure() # compute pressure
        self.advection() # compute external and viscosity force
        self.projection() # compute fluid pressure force
        self.step(cnt) # update particle properties
        self.handle_boundary() # handle boundary

