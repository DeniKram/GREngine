# bundle.py
import torch
import functools


def make_bundle(x0, v0, M=1.0, n=6, metric="schwarzschild", a=0.0, device=None):
    if device is None:
        device = x0.device

    from metrics import circular_orbit_velocity

    bundle = []
    if metric == "kerr":
        radii = torch.linspace(5.8, 12.0, steps=n,
                               dtype=torch.float64, device=device)
        inclinations = torch.linspace(-0.10, 0.10, steps=n,
                                      dtype=torch.float64, device=device)
    else:
        radii = torch.linspace(8.0, 16.0, steps=n,
                               dtype=torch.float64, device=device)
        inclinations = torch.linspace(-0.05, 0.05, steps=n,
                                      dtype=torch.float64, device=device)

    for i in range(n):
        xi = x0.clone()

        r = radii[i]
        xi[1] = r
        xi[2] = torch.pi/2 + inclinations[i]
        xi[3] = float(i) / n * 2.0 * torch.pi

        vi = circular_orbit_velocity(r, M).to(device)

        if metric == "kerr":
            sign = 1.0 if i < n // 2 else -1.0
            spin_factor = 1.0 + a * 0.18 * sign
            vi[3] = sign * abs(vi[3]) * spin_factor
            label = "prograde" if sign > 0 else "retrograde"
        else:
            vi[3] *= 1.0 + (torch.rand(1, device=device).item() - 0.5) * 0.03
            vi[1] += (torch.rand(1, device=device).item() - 0.5) * 0.02
            vi[2] += (torch.rand(1, device=device).item() - 0.5) * 0.008
            label = "timelike"

        bundle.append((xi, vi, label))

    return bundle