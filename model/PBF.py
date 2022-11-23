import taichi as ti
from .settings import *
from .scene import *
from .utils import *

@ti.data_oriented
class PBF:
    def __init__(self):
        self.scene = Scene() # define the scene
        self.dt = ti.field(dtype=ti.float32, shape=()) # PBF time step (larger than WCSPH)
        self.dt[None] = 5e-3
        self.m = Density0 * Particle_Volume # mass = density * volume
        self.accleration = ti.Vector.field(Dim, dtype=ti.float32)
        self.old_x = ti.Vector.field(Dim, dtype=ti.float32)
        self.lambdas = ti.field(dtype=ti.float32)
        self.position_deltas = ti.Vector.field(Dim, dtype=ti.float32)
        ti.root.dense(ti.i, MAX_Particle_Number).place(self.accleration, self.old_x, self.lambdas, self.position_deltas)
        self.bound = np.array(GUI_Resolution) / Scale_Ratio
        self.board = ti.field(dtype=ti.float32, shape=())
        self.board[None] = GUI_Resolution[0] / Scale_Ratio

    # update the velocity and position for each particle
    @ti.kernel
    def step(self, cnt: int):
        self.board[None] = GUI_Resolution[0] / Scale_Ratio
        for i in range(self.scene.N[None]):
            if self.scene.type[i] == SOLID: # moving board
                self.scene.v[i][0] = 3.0 * Board_v0 * ti.sin(1.0 * np.pi * cnt * self.dt[None])
                self.scene.v[i][1] = 0.0
                self.scene.x[i] += self.scene.v[i] * self.dt[None]
                self.scene.boundary_x[i] = self.scene.x[i]
                self.board[None] = ti.min(self.board[None], self.scene.boundary_x[i][0])
            if self.scene.type[i] == BOUNDARY: # boundary particles always stay still
                continue
            if self.scene.type[i] == FLUID: # fluid particles
                self.scene.v[i] = (self.scene.x[i] - self.old_x[i]) / self.dt[None] # v = dx / dt

    
    @ti.kernel
    def advection(self):
        for p_i in range(self.scene.N[None]):
            if self.scene.type[p_i] == BOUNDARY:
                continue
            if self.scene.type[p_i] == SOLID:
                continue
            self.old_x[p_i] = self.scene.x[p_i] # store the old position
        
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

            self.scene.v[p_i] += self.accleration[p_i] * self.dt[None] # update v
            self.scene.x[p_i] += self.scene.v[p_i] * self.dt[None] # update x
    

    @ti.kernel
    def modify(self):
        # compute lambdas
        for p_i in range(self.scene.N[None]):
            if self.scene.type[p_i] == BOUNDARY:
                continue
            if self.scene.type[p_i] == SOLID:
                continue
            grad_i = ti.Vector.zero(ti.float32, Dim)
            sum_gradient_sqr = 0.0
            density_constraint = 0.0

            for j in range(self.scene.neighbor_N[p_i]):
                p_j = self.scene.neighbor_idx[p_i, j]
                dist = self.scene.x[p_i] - self.scene.x[p_j]
                grad_j = spiky_gradient(dist, H)
                grad_i += grad_j
                sum_gradient_sqr += grad_j.dot(grad_j)
                density_constraint += poly6_value(dist.norm(), H)

            density_constraint = (self.m * density_constraint / Density0) - 1.0

            sum_gradient_sqr += grad_i.dot(grad_i)
            self.lambdas[p_i] = -density_constraint / (sum_gradient_sqr + Lambda_Epsilon)
        
        # compute position deltas
        for p_i in range(self.scene.N[None]):
            if self.scene.type[p_i] == BOUNDARY:
                continue
            if self.scene.type[p_i] == SOLID:
                continue
            pos_delta_i = ti.Vector.zero(ti.float32, Dim)
            for j in range(self.scene.neighbor_N[p_i]):
                p_j = self.scene.neighbor_idx[p_i, j]
                dist = self.scene.x[p_i] - self.scene.x[p_j]
                lambda_i = self.lambdas[p_i]
                lambda_j = self.lambdas[p_j]
                scorr_ij = compute_scorr(dist)
                pos_delta_i += (lambda_i + lambda_j + scorr_ij) * spiky_gradient(dist, H)

            pos_delta_i /= Density0
            self.position_deltas[p_i] = pos_delta_i
        
        # apply position deltas
        for i in range(self.scene.N[None]):
            if self.scene.type[i] == BOUNDARY:
                continue
            if self.scene.type[i] == SOLID:
                continue
            self.scene.x[i] += self.position_deltas[i]


    @ti.func
    def simulate_collisions(self, p_i, vec, d):
        self.scene.x[p_i] += vec * d

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
                

    # Possion Based Fluid Solver
    def solve(self, cnt):
        self.advection() # compute external and viscosity force
        self.handle_boundary() # handle boundary
        self.scene.search_neighbors() # search neighbors
        for _ in range(PBF_Num_Iters):
            self.modify()
            self.handle_boundary() # handle boundary
        self.step(cnt) # update particle properties

