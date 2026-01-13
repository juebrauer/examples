"""
2D Optimizer Playground — PySide6 + Matplotlib (2x4 grid)

Features:
- Complex random loss surface f(x, y) with many local minima
- One optimizer per subplot (2 rows x 4 columns)
- Optimizers:
    1) GD
    2) GD + Momentum (Polyak)
    3) GD + Nesterov Momentum
    4) Adagrad
    5) RMSProp
    6) Adam
    7) AdamW
    8) Lion
- SPACE: generate a new random surface and restart
- Click: set a new shared starting point (all optimizers restart from there)
- Speed slider: timer interval in ms/step
- Best optimizer is highlighted with green title

Run:
    python neural_network_optimizers_demo.py
"""

import sys
import math
import numpy as np

from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QSlider
)

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas


# -----------------------------
# Complex loss surface with many local minima
# -----------------------------
class ErrorSurface:
    """
    f(x, y) is constructed as:

        f(x, y) = 1/2 * v^T A v
                  + Σ_k a_k * sin(wx_k*x + phix_k) * sin(wy_k*y + phiy_k)
                  + Σ_j b_j * exp( -||v - m_j||^2 / (2 sigma_j^2) )

    with v=[x,y].
    
    Enhanced with more sine waves and Gaussian bumps to create many local minima.
    """

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

        # Positive definite quadratic matrix A
        M = rng.normal(size=(2, 2))
        A = M.T @ M
        A += np.eye(2) * 0.3
        self.A = A

        # More sin ripples for complex landscape with many local minima
        self.num_sin = 15
        self.sin_a = rng.uniform(-0.9, 0.9, size=self.num_sin)
        self.sin_wx = rng.uniform(0.8, 3.0, size=self.num_sin)
        self.sin_wy = rng.uniform(0.8, 3.0, size=self.num_sin)
        self.sin_phix = rng.uniform(0, 2 * math.pi, size=self.num_sin)
        self.sin_phiy = rng.uniform(0, 2 * math.pi, size=self.num_sin)

        # More Gaussian bumps for additional local minima/maxima
        self.num_gauss = 8
        self.gauss_b = rng.uniform(-1.5, 1.5, size=self.num_gauss)
        self.gauss_mx = rng.uniform(-2.5, 2.5, size=self.num_gauss)
        self.gauss_my = rng.uniform(-2.5, 2.5, size=self.num_gauss)
        self.gauss_sigma = rng.uniform(0.4, 1.0, size=self.num_gauss)

    def f(self, x: float, y: float) -> float:
        v = np.array([x, y], dtype=float)

        # Quadratic term: 1/2 v^T A v
        quad = 0.5 * float(v.T @ self.A @ v)

        # Sin ripples
        s = 0.0
        for k in range(self.num_sin):
            sx = math.sin(self.sin_wx[k] * x + self.sin_phix[k])
            sy = math.sin(self.sin_wy[k] * y + self.sin_phiy[k])
            s += self.sin_a[k] * sx * sy

        # Gaussians
        g = 0.0
        for j in range(self.num_gauss):
            dx = x - self.gauss_mx[j]
            dy = y - self.gauss_my[j]
            sig = self.gauss_sigma[j]
            g += self.gauss_b[j] * math.exp(-(dx * dx + dy * dy) / (2.0 * sig * sig))

        return quad + s + g

    def f_vec(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Vectorized f over meshgrid arrays (for fast contour precompute)."""
        # Quadratic term: 1/2 v^T A v with v=[X,Y]
        A11 = float(self.A[0, 0])
        A12 = float(self.A[0, 1])
        A22 = float(self.A[1, 1])
        quad = 0.5 * (A11 * X * X + 2.0 * A12 * X * Y + A22 * Y * Y)

        s = 0.0
        for k in range(self.num_sin):
            s += self.sin_a[k] * np.sin(self.sin_wx[k] * X + self.sin_phix[k]) * np.sin(
                self.sin_wy[k] * Y + self.sin_phiy[k]
            )

        g = 0.0
        for j in range(self.num_gauss):
            dx = X - self.gauss_mx[j]
            dy = Y - self.gauss_my[j]
            sig = self.gauss_sigma[j]
            g += self.gauss_b[j] * np.exp(-(dx * dx + dy * dy) / (2.0 * sig * sig))

        return quad + s + g

    def grad(self, x: float, y: float) -> np.ndarray:
        """
        Gradient ∇f = [df/dx, df/dy].

        - Quadratic:
            f_q = 1/2 v^T A v  =>  ∇f_q = A v

        - Sin term:
            f_s = a * sin(wx x + phix) * sin(wy y + phiy)
            df/dx = a * wx * cos(wx x + phix) * sin(wy y + phiy)
            df/dy = a * wy * sin(wx x + phix) * cos(wy y + phiy)

        - Gaussian term:
            f_g = b * exp( -r^2 / (2 sigma^2) ), r^2=(x-mx)^2+(y-my)^2
            df/dx = f_g * (-(x-mx)/(sigma^2))
            df/dy = f_g * (-(y-my)/(sigma^2))
        """
        v = np.array([x, y], dtype=float)
        g = self.A @ v

        # Sin ripples
        for k in range(self.num_sin):
            a = self.sin_a[k]
            wx = self.sin_wx[k]
            wy = self.sin_wy[k]
            phix = self.sin_phix[k]
            phiy = self.sin_phiy[k]

            sx = math.sin(wx * x + phix)
            sy = math.sin(wy * y + phiy)
            cx = math.cos(wx * x + phix)
            cy = math.cos(wy * y + phiy)

            g[0] += a * wx * cx * sy
            g[1] += a * wy * sx * cy

        # Gaussians
        for j in range(self.num_gauss):
            b = self.gauss_b[j]
            mx = self.gauss_mx[j]
            my = self.gauss_my[j]
            sig = self.gauss_sigma[j]

            dx = x - mx
            dy = y - my
            expo = math.exp(-(dx * dx + dy * dy) / (2.0 * sig * sig))
            f_g = b * expo

            g[0] += f_g * (-(dx) / (sig * sig))
            g[1] += f_g * (-(dy) / (sig * sig))

        return g


# -----------------------------
# Optimizer states + step rules
# -----------------------------
@dataclass
class OptimizerState:
    name: str
    x: np.ndarray
    path: List[np.ndarray] = field(default_factory=list)

    # Generic buffers used by some optimizers
    v: np.ndarray = field(default_factory=lambda: np.zeros(2))
    m: np.ndarray = field(default_factory=lambda: np.zeros(2))
    s: np.ndarray = field(default_factory=lambda: np.zeros(2))
    t: int = 0

    def record(self):
        self.path.append(self.x.copy())


def step_gd(st: OptimizerState, g: np.ndarray, lr: float):
    st.x = st.x - lr * g


def step_momentum(st: OptimizerState, g: np.ndarray, lr: float, beta: float):
    st.v = beta * st.v + g
    st.x = st.x - lr * st.v


def step_nesterov(st: OptimizerState, surface: ErrorSurface, lr: float, beta: float):
    # lookahead gradient
    look = st.x - lr * beta * st.v
    g = surface.grad(float(look[0]), float(look[1]))
    st.v = beta * st.v + g
    st.x = st.x - lr * st.v


def step_adagrad(st: OptimizerState, g: np.ndarray, lr: float, eps: float = 1e-8):
    st.s = st.s + g * g
    st.x = st.x - lr * g / (np.sqrt(st.s) + eps)


def step_rmsprop(st: OptimizerState, g: np.ndarray, lr: float, beta: float, eps: float = 1e-8):
    st.s = beta * st.s + (1.0 - beta) * (g * g)
    st.x = st.x - lr * g / (np.sqrt(st.s) + eps)


def step_adam(st: OptimizerState, g: np.ndarray, lr: float, beta1: float, beta2: float, eps: float = 1e-8):
    st.t += 1
    st.m = beta1 * st.m + (1 - beta1) * g
    st.s = beta2 * st.s + (1 - beta2) * (g * g)

    mhat = st.m / (1 - beta1 ** st.t)
    shat = st.s / (1 - beta2 ** st.t)

    st.x = st.x - lr * mhat / (np.sqrt(shat) + eps)


def step_adamw(
    st: OptimizerState, g: np.ndarray, lr: float, weight_decay: float,
    beta1: float, beta2: float, eps: float = 1e-8
):
    st.t += 1
    st.m = beta1 * st.m + (1 - beta1) * g
    st.s = beta2 * st.s + (1 - beta2) * (g * g)

    mhat = st.m / (1 - beta1 ** st.t)
    shat = st.s / (1 - beta2 ** st.t)

    st.x = st.x - lr * (mhat / (np.sqrt(shat) + eps) + weight_decay * st.x)


def step_lion(
    st: OptimizerState, g: np.ndarray, lr: float,
    beta1: float, beta2: float, weight_decay: float
):
    st.t += 1
    update = np.sign(beta1 * st.m + (1 - beta1) * g)
    st.m = beta2 * st.m + (1 - beta2) * g
    st.x = st.x - lr * (update + weight_decay * st.x)


# -----------------------------
# Main Window
# -----------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("2D Optimizer Playground (Enhanced)")

        self.rng = np.random.default_rng(42)

        # Hyperparameters
        self.lr = 0.08
        self.momentum_beta = 0.9
        self.rms_beta = 0.9
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.weight_decay = 0.01
        self.lion_beta1 = 0.9
        self.lion_beta2 = 0.99
        self.lion_wd = 0.01

        self.max_steps = 500
        self.xmin, self.xmax = -3.0, 3.0
        self.ymin, self.ymax = -3.0, 3.0

        # Figure with 2x4 grid - maximize plot area
        self.fig = Figure(figsize=(14, 7))
        # Adjust spacing to maximize plot area
        self.fig.subplots_adjust(left=0.04, right=0.98, top=0.95, bottom=0.05, 
                                hspace=0.25, wspace=0.25)
        self.axes = self.fig.subplots(2, 4)
        self.axes = self.axes.flatten()
        self.canvas = FigureCanvas(self.fig)
        self.canvas.mpl_connect("button_press_event", self.on_canvas_click)

        # Info label
        self.info = QLabel()
        self.info.setWordWrap(True)
        self.info.setMaximumHeight(40)

        # Compact speed slider
        self.speed_label = QLabel("25 ms/step")
        self.speed_label.setMinimumWidth(80)
        
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(0)
        self.speed_slider.setMaximum(2000)
        self.speed_slider.setValue(25)
        self.speed_slider.setMaximumHeight(30)
        self.speed_slider.valueChanged.connect(self.on_speed_changed)

        speed_row = QHBoxLayout()
        speed_row.addWidget(QLabel("Speed:"))
        speed_row.addWidget(self.speed_slider)
        speed_row.addWidget(self.speed_label)

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.addWidget(self.canvas, stretch=100)  # Give most space to canvas
        layout.addLayout(speed_row)
        layout.addWidget(self.info)
        layout.setSpacing(5)
        layout.setContentsMargins(5, 5, 5, 5)
        self.setCentralWidget(root)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.on_tick)

        self.reset_demo()
        self.on_speed_changed(self.speed_slider.value())
        self.timer.start()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.reset_demo()
        elif event.key() in (Qt.Key_Plus, Qt.Key_Equal):
            self.speed_slider.setValue(max(self.speed_slider.minimum(), self.speed_slider.value() - 50))
        elif event.key() == Qt.Key_Minus:
            self.speed_slider.setValue(min(self.speed_slider.maximum(), self.speed_slider.value() + 50))

    def on_speed_changed(self, value: int):
        # value is milliseconds per optimization step (timer interval)
        # Use minimum of 1ms to keep timer running
        actual_interval = max(1, int(value))
        self.timer.setInterval(actual_interval)
        self.speed_label.setText(f"{value} ms/step")

    def reset_demo(self):
        self.surface = ErrorSurface(self.rng)

        # Shared starting point for fairness
        x0 = self.rng.uniform(self.xmin * 0.7, self.xmax * 0.7)
        y0 = self.rng.uniform(self.ymin * 0.7, self.ymax * 0.7)
        self.start = np.array([x0, y0], dtype=float)

        # Create states (8 optimizers)
        self.optimizers = [
            ("GD", OptimizerState("GD", self.start)),
            ("Momentum", OptimizerState("Momentum", self.start)),
            ("Nesterov", OptimizerState("Nesterov", self.start)),
            ("Adagrad", OptimizerState("Adagrad", self.start)),
            ("RMSProp", OptimizerState("RMSProp", self.start)),
            ("Adam", OptimizerState("Adam", self.start)),
            ("AdamW", OptimizerState("AdamW", self.start)),
            ("Lion", OptimizerState("Lion", self.start)),
        ]

        self.step_count = 0

        # Precompute contour grid (once per restart) — vectorized (fast)
        gx = np.linspace(self.xmin, self.xmax, 220)
        gy = np.linspace(self.ymin, self.ymax, 220)
        X, Y = np.meshgrid(gx, gy)
        Z = self.surface.f_vec(X, Y)

        self.X, self.Y, self.Z = X, Y, Z

        # Initial record (starting point)
        for _, st in self.optimizers:
            st.path = []
            st.v = np.zeros(2)
            st.m = np.zeros(2)
            st.s = np.zeros(2)
            st.t = 0
            st.record()

        self.redraw(full=True)

    def clamp_to_domain(self, x: np.ndarray) -> np.ndarray:
        x0 = float(np.clip(x[0], self.xmin, self.xmax))
        x1 = float(np.clip(x[1], self.ymin, self.ymax))
        return np.array([x0, x1], dtype=float)

    def on_tick(self):
        if self.step_count >= self.max_steps:
            return

        # One gradient step per timer tick (speed controls spacing between steps)
        for name, st in self.optimizers:
            if name == "Nesterov":
                step_nesterov(st, self.surface, lr=self.lr, beta=self.momentum_beta)
            else:
                g = self.surface.grad(float(st.x[0]), float(st.x[1]))

                if name == "GD":
                    step_gd(st, g, lr=self.lr)
                elif name == "Momentum":
                    step_momentum(st, g, lr=self.lr, beta=self.momentum_beta)
                elif name == "Adagrad":
                    step_adagrad(st, g, lr=self.lr)
                elif name == "RMSProp":
                    step_rmsprop(st, g, lr=self.lr, beta=self.rms_beta)
                elif name == "Adam":
                    step_adam(st, g, lr=self.lr, beta1=self.adam_beta1, beta2=self.adam_beta2)
                elif name == "AdamW":
                    step_adamw(
                        st, g, lr=self.lr, weight_decay=self.weight_decay,
                        beta1=self.adam_beta1, beta2=self.adam_beta2
                    )
                elif name == "Lion":
                    step_lion(
                        st, g, lr=self.lr,
                        beta1=self.lion_beta1, beta2=self.lion_beta2,
                        weight_decay=self.lion_wd
                    )
                else:
                    step_gd(st, g, lr=self.lr)

            st.x = self.clamp_to_domain(st.x)
            st.record()

        self.step_count += 1
        self.redraw(full=False)

    def on_canvas_click(self, event):
        # Left-click sets a new shared starting point (for all optimizers)
        if event.button != 1:
            return
        if event.inaxes is None:
            return
        if event.xdata is None or event.ydata is None:
            return

        x = float(np.clip(event.xdata, self.xmin, self.xmax))
        y = float(np.clip(event.ydata, self.ymin, self.ymax))
        self.set_start_point(np.array([x, y], dtype=float))

    def set_start_point(self, start_xy: np.ndarray):
        self.start = start_xy.copy()

        # Reset optimizer states (keep the same surface/contours)
        self.optimizers = [
            ("GD", OptimizerState("GD", self.start)),
            ("Momentum", OptimizerState("Momentum", self.start)),
            ("Nesterov", OptimizerState("Nesterov", self.start)),
            ("Adagrad", OptimizerState("Adagrad", self.start)),
            ("RMSProp", OptimizerState("RMSProp", self.start)),
            ("Adam", OptimizerState("Adam", self.start)),
            ("AdamW", OptimizerState("AdamW", self.start)),
            ("Lion", OptimizerState("Lion", self.start)),
        ]
        self.step_count = 0

        for _, st in self.optimizers:
            st.record()

        # Rebuild artists and redraw once
        self.redraw(full=True)

    def init_artists(self):
        """(Re)build the expensive background (contours) once per restart/start-change."""
        self._artists = []
        for ax, (name, st) in zip(self.axes, self.optimizers):
            ax.clear()
            ax.contourf(self.X, self.Y, self.Z, levels=30, cmap='viridis')
            ax.set_xlim(self.xmin, self.xmax)
            ax.set_ylim(self.ymin, self.ymax)
            ax.set_title(name, fontsize=12, fontweight='bold')
            ax.set_xlabel("x", fontsize=9)
            ax.set_ylabel("y", fontsize=9)
            ax.tick_params(labelsize=8)

            # Start point (shared)
            start_pt, = ax.plot([self.start[0]], [self.start[1]], 
                               marker="o", markersize=6, color='red', 
                               markeredgecolor='white', markeredgewidth=1)

            # Path and current point (fast-updated artists)
            path_line, = ax.plot([], [], linewidth=2, color='yellow', alpha=0.7)
            cur_pt, = ax.plot([], [], marker="o", markersize=5, color='cyan',
                            markeredgecolor='white', markeredgewidth=1)

            # Textbox (fast-updated)
            txt = ax.text(
                0.02, 0.98, "f=...",
                transform=ax.transAxes,
                va="top", ha="left",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7)
            )

            self._artists.append({
                "ax": ax,
                "name": name,
                "start_pt": start_pt,
                "path_line": path_line,
                "cur_pt": cur_pt,
                "txt": txt,
            })

    def redraw(self, full: bool):
        # Full redraw rebuilds contours + artists (expensive). Non-full just updates lines/text (fast).
        if full or not getattr(self, "_artists", None) or len(self._artists) != len(self.optimizers):
            self.init_artists()

        # Find the best optimizer (lowest loss)
        vals = [(name, self.surface.f(float(st.x[0]), float(st.x[1]))) 
                for name, st in self.optimizers]
        best_name = min(vals, key=lambda t: t[1])[0]

        for art, (name, st) in zip(self._artists, self.optimizers):
            P = np.array(st.path, dtype=float)

            art["start_pt"].set_data([self.start[0]], [self.start[1]])
            art["path_line"].set_data(P[:, 0], P[:, 1])
            art["cur_pt"].set_data([P[-1, 0]], [P[-1, 1]])

            val = self.surface.f(float(st.x[0]), float(st.x[1]))
            art["txt"].set_text(f"f={val:.3f}")

            # Highlight best optimizer with green title
            if name == best_name:
                art["ax"].set_title(name, fontsize=12, fontweight='bold', color='green')
            else:
                art["ax"].set_title(name, fontsize=12, fontweight='bold', color='black')

        # Global info line: show top-3 best (lowest loss) at the moment
        vals.sort(key=lambda t: t[1])
        best = " | ".join([f"{n}: {v:.3f}" for n, v in vals[:3]])

        self.info.setText(
            f"Steps: {self.step_count}/{self.max_steps}    Best: {best}    "
            f"(SPACE: new surface | Click: set start | +/-: adjust speed)"
        )

        self.canvas.draw_idle()


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(1400, 850)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()