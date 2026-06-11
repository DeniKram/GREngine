import argparse
import torch
import functools
import numpy as np
from differential_geometry import Metric, Christoffel, Geodesic
from metrics import schwarzschild, kerr, flrw, minkowski
from bundle import make_bundle
from visual import BlackHoleVisualizer


def main():
    parser = argparse.ArgumentParser(
        description="Visualize geodesics for different metrics."
    )
    parser.add_argument(
        "--metric",
        choices=["schwarzschild", "kerr", "flrw", "minkowski"],
        default="schwarzschild",
        help="Metric to use for geodesic integration.",
    )
    parser.add_argument(
        "--M",
        type=float,
        default=1.0,
        help="Mass parameter for Schwarzschild/Kerr metrics.",
    )
    parser.add_argument(
        "--a",
        type=float,
        default=0.5,
        help="Spin parameter for Kerr metric.",
    )
    parser.add_argument(
        "--schwarzschild",
        action="store_true",
        help="Use Schwarzschild metric.",
    )
    parser.add_argument(
        "--kerr",
        action="store_true",
        help="Use Kerr metric.",
    )
    parser.add_argument(
        "--flrw",
        action="store_true",
        help="Use FLRW metric.",
    )
    parser.add_argument(
        "--minkowski",
        action="store_true",
        help="Use Minkowski metric.",
    )
    args = parser.parse_args()

    explicit_metric = None
    if args.kerr:
        explicit_metric = "kerr"
    elif args.flrw:
        explicit_metric = "flrw"
    elif args.minkowski:
        explicit_metric = "minkowski"
    elif args.schwarzschild:
        explicit_metric = "schwarzschild"

    if explicit_metric is not None:
        args.metric = explicit_metric

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"metric: {args.metric}")

    M = args.M
    rs = 2.0 * M

    metric_map = {
        "schwarzschild": functools.partial(schwarzschild, M=M),
        "kerr": functools.partial(kerr, M=M, a=args.a),
        "flrw": functools.partial(flrw, a_scale=1.0, k=0.0),
        "minkowski": functools.partial(minkowski),
    }

    metric = Metric(metric_map[args.metric])
    Г = Christoffel(metric)
    geo = Geodesic(Г)

    x0 = torch.tensor([0.0, 10.0, torch.pi/2, 0.0],
                      dtype=torch.float64, device=device)
    v0 = torch.zeros(4, dtype=torch.float64, device=device)

    bundle = make_bundle(x0, v0, M=M, n=6,
                         metric=args.metric, a=args.a,
                         device=device)

    traj = []
    for i, (xi, vi, label) in enumerate(bundle):
        print(f"geodesic {i+1}/{len(bundle)}...")
        pos, _ = geo.integrate(xi, vi, steps=1200, dl=0.75)
        traj.append((pos, label))


    visualizer = BlackHoleVisualizer(
        M=M,
        title=f"Geodesic Bundle — {args.metric.capitalize()}"
    )
    visualizer.show(traj)


if __name__ == "__main__":
    main()
