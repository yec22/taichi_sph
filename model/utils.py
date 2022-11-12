import taichi as ti
import numpy as np
from .settings import *

# smooth kernel function (cubic spline actually)
# reference: https://github.com/erizmr/SPH_Taichi.git
@ti.func
def smooth_kernel(r_norm):
    res = ti.cast(0.0, ti.float32)
    h = Support_Radius
    k = 1.0
    if Dim == 1:
        k = 4 / 3
    elif Dim == 2:
        k = 40 / 7 / np.pi
    elif Dim == 3:
        k = 8 / np.pi
    k /= h ** Dim
    q = r_norm / h
    if q <= 1.0:
        if q <= 0.5:
            q2 = q * q
            q3 = q2 * q
            res = k * (6.0 * q3 - 6.0 * q2 + 1)
        else:
            res = k * 2 * ti.pow(1 - q, 3.0)
    return res

# derivative of smooth kernel function
@ti.func
def smooth_kernel_derivative(r):
    h = Support_Radius
    k = 1.0
    if Dim == 1:
        k = 4 / 3
    elif Dim == 2:
        k = 40 / 7 / np.pi
    elif Dim == 3:
        k = 8 / np.pi
    k = 6.0 * k / h ** Dim
    r_norm = r.norm()
    q = r_norm / h
    res = ti.Vector.zero(ti.float32, Dim)
    if r_norm > 1e-5 and q <= 1.0:
        grad_q = r / (r_norm * h)
        if q <= 0.5:
            res = k * q * (3.0 * q - 2.0) * grad_q
        else:
            factor = 1.0 - q
            res = k * (-factor * factor) * grad_q
    return res

@ti.func
def spiky_gradient(r, h):
    res = ti.Vector.zero(ti.float32, Dim)
    r_len = r.norm()
    if 0 < r_len and r_len < h:
        x = (h - r_len) / (h * h * h)
        g_factor = Spiky_Grad_Factor * x * x
        res = r * g_factor / r_len
    return res


@ti.func
def poly6_value(s, h):
    res = 0.0
    if 0 < s and s < h:
        x = (h * h - s * s) / (h * h * h)
        res = Poly6_Factor * x * x * x
    return res


@ti.func
def compute_scorr(pos_ji):
    # Eq (13)
    x = poly6_value(pos_ji.norm(), H) / poly6_value(Corr_DeltaQ_Coeff * H, H)
    # pow(x, 4)
    x = x * x
    x = x * x
    return (-CorrK) * x