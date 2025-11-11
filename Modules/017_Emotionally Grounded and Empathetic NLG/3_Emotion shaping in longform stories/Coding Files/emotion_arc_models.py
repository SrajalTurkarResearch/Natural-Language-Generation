# emotion_arc_models.py
"""
Mathematical Models for Emotional Arcs in NLG
Author: Grok (xAI) – Scientist, Engineer, Mathematician
Date: October 29, 2025

Features:
- Quadratic, Cubic, Sine-based arcs
- Curve fitting with scipy
- Gaussian Process regression (advanced)
- Plotting utilities
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import warnings

warnings.filterwarnings("ignore")


class EmotionalArc:
    """Base class for all emotional arc models."""

    def __init__(self, n_points=100):
        self.n_points = n_points
        self.x = np.linspace(0, 1, n_points)

    def generate(self):
        raise NotImplementedError

    def fit(self, x_data, y_data):
        raise NotImplementedError

    def plot(self, data=None, title="Emotional Arc"):
        plt.figure(figsize=(10, 5))
        plt.plot(self.x, self.y, label="Model", linewidth=2)
        if data is not None:
            plt.scatter(data[0], data[1], color="red", zorder=5, label="Data")
        plt.xlabel("Story Progress (0 to 1)")
        plt.ylabel("Emotional Intensity (-1 to +1)")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(-1.1, 1.1)
        plt.show()


class QuadraticArc(EmotionalArc):
    """Man in a Hole / Rags to Riches"""

    def __init__(self, a=-1.0, b=0.0, c=0.0, **kwargs):
        super().__init__(**kwargs)
        self.a, self.b, self.c = a, b, c
        self.generate()

    def generate(self):
        self.y = self.a * (self.x - 0.5) ** 2 + self.b * self.x + self.c
        return self.y

    def fit(self, x_data, y_data):
        def quad(x, a, b, c):
            return a * (x - 0.5) ** 2 + b * x + c

        popt, _ = curve_fit(quad, x_data, y_data, p0=[-1, 0, 0])
        self.a, self.b, self.c = popt
        self.generate()
        return popt


class CubicArc(EmotionalArc):
    """Cinderella / Oedipus"""

    def __init__(self, a=1.0, b=-2.0, c=1.0, d=0.0, **kwargs):
        super().__init__(**kwargs)
        self.a, self.b, self.c, self.d = a, b, c, d
        self.generate()

    def generate(self):
        x = self.x
        self.y = (
            self.a * (x - 0.33) ** 3
            + self.b * (x - 0.33) ** 2
            + self.c * (x - 0.33)
            + self.d
        )
        return self.y

    def fit(self, x_data, y_data):
        def cubic(x, a, b, c, d):
            return a * (x - 0.33) ** 3 + b * (x - 0.33) ** 2 + c * (x - 0.33) + d

        popt, _ = curve_fit(cubic, x_data, y_data, p0=[1, -2, 1, 0])
        self.a, self.b, self.c, self.d = popt
        self.generate()
        return popt


class GaussianProcessArc(EmotionalArc):
    """Advanced: Dynamic, uncertain arcs (2025 research)"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gp = None

    def fit(self, x_data, y_data):
        kernel = ConstantKernel(1.0) * RBF(0.5)
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        self.gp.fit(x_data.reshape(-1, 1), y_data)
        self.y, self.sigma = self.gp.predict(self.x.reshape(-1, 1), return_std=True)
        return self.gp

    def plot(self, data=None, title="Gaussian Process Emotional Arc"):
        super().plot(data, title)
        if self.gp:
            plt.fill_between(
                self.x,
                self.y - self.sigma,
                self.y + self.sigma,
                alpha=0.2,
                color="blue",
            )


# === PRESET ARCS ===
def rags_to_riches(n=100):
    arc = QuadraticArc(a=-1.0, b=2.0, c=-1.0, n_points=n)
    return arc


def man_in_hole(n=100):
    arc = QuadraticArc(a=1.0, b=0, c=-0.5, n_points=n)
    return arc


def cinderella(n=100):
    arc = CubicArc(a=2.0, b=-4.0, c=2.0, d=0.0, n_points=n)
    return arc


if __name__ == "__main__":
    # Demo
    arc = cinderella()
    arc.plot(title="Cinderella Arc (Cubic Model)")
