# visual.py
import numpy as np
from vispy import app, scene
from vispy.scene import visuals


COLORS = {
    "circular": (0.2, 0.8, 1.0, 1.0),   
    "plunge":   (1.0, 0.3, 0.3, 1.0),   
    "null":     (1.0, 0.9, 0.2, 1.0),  
    "prograde": (0.2, 0.9, 0.5, 0.95),  
    "retrograde": (1.0, 0.45, 0.45, 0.95), 
    "default":  (0.6, 0.6, 0.6, 1.0),   
}


def sph_to_xyz(traj):
    
    t   = traj.cpu().numpy()
    r   = t[:, 1]
    the = t[:, 2]
    phi = t[:, 3]
    x   = r * np.sin(the) * np.cos(phi)
    y   = r * np.sin(the) * np.sin(phi)
    z   = r * np.cos(the)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def circle_pts(radius, n=200, tilt_x=0.0, tilt_y=0.0):
    
    phi = np.linspace(0, 2*np.pi, n)
    x   = radius * np.cos(phi)
    y   = radius * np.sin(phi)
    z   = np.zeros_like(phi)

    if tilt_x:
        y, z = (y*np.cos(tilt_x) - z*np.sin(tilt_x),
                y*np.sin(tilt_x) + z*np.cos(tilt_x))
    if tilt_y:
        x, z = (x*np.cos(tilt_y) + z*np.sin(tilt_y),
               -x*np.sin(tilt_y) + z*np.cos(tilt_y))

    pts = np.stack([x, y, z], axis=1).astype(np.float32)
    return np.vstack([pts, pts[:1]])


class BlackHoleVisualizer:

    def __init__(self, M=1.0,
                 title="GR-engine — Geodesic Bundle"):
        self.M    = M
        self.rs   = 2.0 * M
        self.isco = 6.0 * M

        self.canvas = scene.SceneCanvas(
            title=title,
            size=(1200, 900),
            bgcolor="#05060f",
            show=True
        )
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.ArcballCamera(
            fov=35, distance=65, up='+z'
        )
        self.view.camera.elevation = 25
        self.view.camera.azimuth   = 40

    def _add_line(self, pts, color, width=1.5, antialias=True):
        self.view.add(visuals.Line(
            pos=pts, color=color,
            width=width, method="gl",
            antialias=antialias
        ))

    def draw_starfield(self, n=800, radius=120.0, spread=24.0):
        rng = np.random.default_rng(42)
        u = rng.uniform(-1.0, 1.0, size=n)
        phi = rng.uniform(0.0, 2*np.pi, size=n)
        theta = np.arccos(u)
        r = radius + rng.uniform(-spread, spread, size=n)
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)

        self.view.add(visuals.Markers(
            pos=np.stack([x, y, z], axis=1).astype(np.float32),
            face_color=(1.0, 1.0, 1.0, 0.14),
            size=2.0, edge_color=None
        ))

    def draw_axes(self, length=24.0):
        axes = {
            'x': ((length, 0.0, 0.0), (1.0, 0.4, 0.4, 1.0)),
            'y': ((0.0, length, 0.0), (0.4, 1.0, 0.4, 1.0)),
            'z': ((0.0, 0.0, length), (0.4, 0.6, 1.0, 1.0)),
        }
        for direction, (endpoint, color) in axes.items():
            pts = np.array([[0.0, 0.0, 0.0], endpoint], dtype=np.float32)
            self._add_line(pts, color=color, width=2.0)

    def draw_black_hole(self):
        
        for tilt in np.linspace(0, np.pi, 5):
            self._add_line(
                circle_pts(self.rs, tilt_x=tilt),
                color=(1.0, 0.38, 0.08, 0.42),
                width=1.0
            )
        self._add_line(
            circle_pts(self.rs),
            color=(1.0, 0.5, 0.12, 1.0),
            width=3.0
        )

    def draw_trajectories(self, results, rs_cutoff=None):
        
        if rs_cutoff is None:
            rs_cutoff = self.rs * 1.1

        for pos, label in results:
            xyz    = sph_to_xyz(pos)
            r_vals = np.linalg.norm(xyz, axis=1)
            mask   = r_vals > rs_cutoff
            xyz    = xyz[mask][::3]

            if len(xyz) < 5:
                continue

            base = np.array(COLORS.get(label, COLORS["default"]), dtype=np.float32)
            alphas = np.linspace(0.14, base[3], len(xyz), dtype=np.float32)
            colors = np.repeat(base[None, :], len(xyz), axis=0)
            colors[:, 3] = alphas

            self._add_line(
                xyz,
                color=colors,
                width=2.2 if label == "null" else 1.6
            )

    def show(self, results=None):
        
        self.draw_black_hole()

        if results is not None:
            self.draw_trajectories(results)

        app.run()