import torch
import functools
from differential_geometry import (Metric, Christoffel, Geodesic,)

G = 1.0
C = 1.0


def minkowski(x: torch.Tensor) -> torch.Tensor:
    
    r     = x[1]
    theta = x[2]
    sin_theta = torch.sin(theta)
    sin2 = torch.clamp(sin_theta * sin_theta, min=1e-12)

    g = torch.zeros(4, 4, dtype=torch.float64, device=x.device)
    g[0, 0] = -1.0
    g[1, 1] =  1.0
    g[2, 2] =  r ** 2
    g[3, 3] =  r ** 2 * sin2
    return g


def schwarzschild(x: torch.Tensor, M: float = 1.0) -> torch.Tensor:
   
    r     = x[1]
    theta = x[2]
    rs    = 2 * G * M

    r = torch.clamp(r, min=rs * 1.01)
    sin_theta = torch.sin(theta)
    sin2 = torch.clamp(sin_theta * sin_theta, min=1e-12)

    g = torch.zeros(4, 4, dtype=torch.float64, device=x.device)
    g[0, 0] = -(1.0 - rs / r)
    g[1, 1] =  1.0 / (1.0 - rs / r)
    g[2, 2] =  r ** 2
    g[3, 3] =  r ** 2 * sin2
    return g


def kerr(x: torch.Tensor, M: float = 1.0,
         a: float = 0.5) -> torch.Tensor:
   
    r     = x[1]
    theta = x[2]
    rs    = 2 * G * M

    Sigma = r**2 + a**2 * torch.cos(theta)**2
    Delta = r**2 - rs * r + a**2

    
    Delta = torch.where(
        torch.abs(Delta) < 1e-6,
        torch.tensor(1e-6, dtype=x.dtype, device=x.device),
        Delta
    )

    sin_theta = torch.sin(theta)
    sin2 = torch.clamp(sin_theta * sin_theta, min=1e-12)

    g = torch.zeros(4, 4, dtype=torch.float64, device=x.device)
    g[0, 0] = -(1.0 - rs * r / Sigma)
    g[1, 1] =  Sigma / Delta
    g[2, 2] =  Sigma
    g[3, 3] =  (r**2 + a**2 + rs*r*a**2*torch.sin(theta)**2/Sigma) \
               * sin2
    g[0, 3] = -rs * r * a * sin2 / Sigma
    g[3, 0] =  g[0, 3]
    return g


def flrw(x: torch.Tensor, a_scale: float = 1.0,
         k: float = 0.0) -> torch.Tensor:
   
    r     = x[1]
    theta = x[2]

    sin_theta = torch.sin(theta)
    sin2 = torch.clamp(sin_theta * sin_theta, min=1e-12)

    g = torch.zeros(4, 4, dtype=torch.float64, device=x.device)
    g[0, 0] = -1.0
    g[1, 1] =  a_scale**2 / (1.0 - k * r**2)
    g[2, 2] =  a_scale**2 * r**2
    g[3, 3] =  a_scale**2 * r**2 * sin2
    return g




def schwarzschild_radius(M: float) -> float:
    return 2 * G * M

def isco_radius(M: float) -> float:
    return 6 * G * M

def photon_sphere_radius(M: float) -> float:
    return 3 * G * M

def orbital_velocity(r: float, M: float) -> float:
    return (G * M / r) ** 0.5

def gravitational_redshift(r: float, M: float) -> float:
    rs = schwarzschild_radius(M)
    return 1.0 / (1.0 - rs / r) ** 0.5 - 1.0

def proper_time_ratio(r: float, M: float) -> float:
    rs = schwarzschild_radius(M)
    return (1.0 - rs / r) ** 0.5

def circular_orbit_velocity(r: float, M: float) -> torch.Tensor:
    rs    = schwarzschild_radius(M)
    f     = 1.0 - rs / r
    omega = (G * M / r**3) ** 0.5
    ut    = 1.0 / (f * (1 - omega**2 * r**2 / f)) ** 0.5
    uphi  = omega * ut

    device = r.device if isinstance(r, torch.Tensor) else torch.device("cpu")
    return torch.tensor([ut, 0.0, 0.0, uphi],
                        dtype=torch.float64, device=device)


