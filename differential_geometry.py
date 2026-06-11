import torch
from torch.autograd.functional import jacobian


class Metric:
    def __init__(self, func):
        self.func = func

    def g(self, x: torch.Tensor) -> torch.Tensor:
        return self.func(x)

    def g_inv(self, x: torch.Tensor) -> torch.Tensor:
        g = self.g(x)
        try:
            return torch.inverse(g)
        except RuntimeError:
            return torch.linalg.pinv(g)

    def inner(self, x: torch.Tensor,
              u: torch.Tensor,
              v: torch.Tensor) -> torch.Tensor:
        return torch.einsum('mn,m,n->', self.g(x), u, v)

    def norm_squared(self, x: torch.Tensor,
                     u: torch.Tensor) -> torch.Tensor:
        return self.inner(x, u, u)

    def norm(self, x: torch.Tensor,
             u: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.abs(self.norm_squared(x, u)))

    def lower(self, x: torch.Tensor,
              u: torch.Tensor) -> torch.Tensor:
        return torch.einsum('mn,n->m', self.g(x), u)

    def raise_index(self, x: torch.Tensor,
                    u: torch.Tensor) -> torch.Tensor:
        return torch.einsum('mn,n->m', self.g_inv(x), u)

    def is_symmetric(self, x: torch.Tensor,
                     tol: float = 1e-8) -> bool:
        g = self.g(x)
        return torch.max(torch.abs(g - g.T)).item() < tol

    def determinant(self, x: torch.Tensor) -> torch.Tensor:
        return torch.det(self.g(x))

    def signature(self, x: torch.Tensor) -> tuple:
        eigenvalues = torch.linalg.eigvalsh(self.g(x))
        p = int((eigenvalues > 0).sum().item())
        q = int((eigenvalues < 0).sum().item())
        return p, q


class Christoffel:

    def __init__(self, metric: Metric):
        self.metric      = metric
        self._const_cache = None

    def gamma(self, x: torch.Tensor) -> torch.Tensor:
        if self._const_cache is not None:
            return self._const_cache

        g     = self.metric.g(x)
        try:
            g_inv = self.metric.g_inv(x)
        except RuntimeError:
            g_inv = torch.linalg.pinv(g)

        def g_wrapper(x_):
            return self.metric.g(x_)

        dg = jacobian(g_wrapper, x)

        if torch.max(torch.abs(dg)).item() < 1e-10:
            self._const_cache = torch.zeros_like(dg)
            return self._const_cache

        term = (
            dg.permute(0, 2, 1)
            + dg.permute(0, 1, 2)
            - dg.permute(2, 0, 1)
        )

        return 0.5 * torch.einsum('ms,snr->mnr', g_inv, term)

    def check_symmetry(self, x: torch.Tensor,
                       tol: float = 1e-8) -> bool:
        Γ = self.gamma(x)
        return torch.max(torch.abs(Γ - Γ.permute(0, 2, 1))).item() < tol


class Geodesic:

    def __init__(self, christoffel: Christoffel):
        self.ch = christoffel

    def acceleration(self, x: torch.Tensor,
                     v: torch.Tensor) -> torch.Tensor:
        Γ = self.ch.gamma(x)
        return -torch.einsum('mnr,n,r->m', Γ, v, v)

    def integrate(self, x0: torch.Tensor,
              v0: torch.Tensor,
              steps: int = 1000,
              dl: float = 0.01) -> tuple:

        device = x0.device  # ← берём device из входных данных

        x = x0.clone().to(torch.float64).to(device)
        v = v0.clone().to(torch.float64).to(device)

        positions  = [x.clone()]
        velocities = [v.clone()]

        for _ in range(steps):
            try:
                k1x = v
                k1v = self.acceleration(x, v)

                k2x = v + 0.5*dl*k1v
                k2v = self.acceleration(x + 0.5*dl*k1x, v + 0.5*dl*k1v)

                k3x = v + 0.5*dl*k2v
                k3v = self.acceleration(x + 0.5*dl*k2x, v + 0.5*dl*k2v)

                k4x = v + dl*k3v
                k4v = self.acceleration(x + dl*k3x, v + dl*k3v)
            except Exception:
                break

            x = x + (dl/6.0) * (k1x + 2*k2x + 2*k3x + k4x)
            v = v + (dl/6.0) * (k1v + 2*k2v + 2*k3v + k4v)

            if not torch.isfinite(x).all() or not torch.isfinite(v).all():
                break

            positions.append(x.clone())
            velocities.append(v.clone())

        return torch.stack(positions), torch.stack(velocities)


def numerical_grad(f, x: torch.Tensor,
                   dx: float = 1e-6) -> torch.Tensor:
    x    = x.clone().detach().to(torch.float64)
    grad = torch.zeros_like(x)

    for i in range(x.numel()):
        xp, xm = x.clone(), x.clone()
        xp[i] += dx
        xm[i] -= dx
        grad[i] = (f(xp) - f(xm)) / (2.0 * dx)

    return grad