import math

# const variables
Dim = 2
Particle_Radius = 0.05
Particle_Volume = 0.8 * (2 * Particle_Radius) ** Dim
Particle_Color = 0x0000ff
Boundary_Color = 0x7f7f7f
Support_Radius = 4.0 * Particle_Radius
MAX_Particle_Number = 50000
MAX_Particle_Per_Grid = 100
MAX_Neighbor_Number = 100
Density0 = 1000.0
Scale_Ratio = 50
Visualize_Ratio = 1.0
GUI_Resolution = (512, 512)
Background_Color = 0xffffff
G = -9.8
Viscosity_Coefficient = 0.05
Exponential_Coefficient = 7.0
Kappa_Coefficient = 50.0
FLUID = 0
BOUNDARY = 1
COLUMN1 = [[3.3, 3.8], [Support_Radius + Particle_Radius, Support_Radius + 3.0]]
COLUMN2 = [[6.5, 7.0], [Support_Radius + Particle_Radius, Support_Radius + 3.0]]
PBF_Num_Iters = 4
H = 1.0
Spiky_Grad_Factor = -45.0 / math.pi
Poly6_Factor = 315.0 / 64.0 / math.pi
Lambda_Epsilon = 100.0
Corr_DeltaQ_Coeff = 0.3
CorrK = 0.001