"""Starre Kugeln, eine horizontale Bodenebene und ein Impulslöser.

Einheiten: Meter, Kilogramm, Sekunden, Radiant. Die z-Achse zeigt nach oben.
Die Reihenfolge in World.step ist gleichzeitig der Lesefaden für die Lehre.
"""

from dataclasses import dataclass, field
from itertools import combinations
import math

import numpy as np


def vector(value):
    """Eine eigene, endliche 3D-Kopie vermeidet geteilte Zustandsvariablen."""
    result = np.array(value, dtype=float, copy=True)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError("Erwartet wird ein endlicher Vektor mit drei Komponenten.")
    return result


@dataclass(eq=False)
class Sphere:
    radius: float = 0.5
    mass: float = 1.0                  # 0 bedeutet unbeweglich
    position: object = (0, 0, 1)
    velocity: object = (0, 0, 0)
    angular_velocity: object = (0, 0, 0)
    restitution: float = 0.5           # Rückprall: 0 = unelastisch, 1 = elastisch
    friction: float = 0.4
    color: tuple = (0.25, 0.6, 0.95)
    name: str = "Kugel"
    orientation: np.ndarray = field(default_factory=lambda: np.eye(3), init=False)
    force: np.ndarray = field(default_factory=lambda: np.zeros(3), init=False)
    torque: np.ndarray = field(default_factory=lambda: np.zeros(3), init=False)

    def __post_init__(self):
        for key in ("radius", "mass", "restitution", "friction"):
            if not math.isfinite(getattr(self, key)):
                raise ValueError(f"{key} muss endlich sein.")
        if self.radius <= 0 or self.mass < 0 or self.friction < 0:
            raise ValueError("radius > 0, mass >= 0 und friction >= 0 erforderlich.")
        if not 0 <= self.restitution <= 1:
            raise ValueError("restitution muss zwischen 0 und 1 liegen.")
        for key in ("position", "velocity", "angular_velocity"):
            setattr(self, key, vector(getattr(self, key)))
        if self.mass == 0 and (np.any(self.velocity) or np.any(self.angular_velocity)):
            raise ValueError("Statische Kugeln dürfen keine Geschwindigkeit haben.")
        color = vector(self.color)
        if np.any(color < 0) or np.any(color > 1):
            raise ValueError("RGB-Farbwerte müssen zwischen 0 und 1 liegen.")
        self.color = tuple(color)

    @property
    def inverse_mass(self):
        return 1 / self.mass if self.mass else 0.0

    @property
    def inverse_inertia(self):
        # Vollkugel: I = 2/5 m r². In jeder Raumrichtung gleich groß!
        return 2.5 * self.inverse_mass / self.radius**2

    def apply_force(self, force, point=None):
        """Kraft bis zum nächsten step; point ist ein Punkt in Weltkoordinaten.

        Vor jedem step erneut aufrufen, wenn die Kraft dauerhaft wirken soll.
        Ohne point greift die Kraft im Schwerpunkt an.
        """
        force = vector(force)
        arm = np.zeros(3) if point is None else vector(point) - self.position
        self.force += force
        self.torque += np.cross(arm, force)

    def apply_torque(self, torque):
        self.torque += vector(torque)

    def apply_impulse(self, impulse, point=None):
        """Momentane Geschwindigkeitsänderung; Impuls in Newtonsekunden."""
        impulse = vector(impulse)
        arm = np.zeros(3) if point is None else vector(point) - self.position
        self.velocity += impulse * self.inverse_mass
        self.angular_velocity += np.cross(arm, impulse) * self.inverse_inertia


class World:
    def __init__(self, gravity=(0, 0, -9.81), floor_z=0.0,
                 floor_restitution=0.5, floor_friction=0.6,
                 max_substep=1 / 240, solver_iterations=6):
        self.gravity = vector(gravity)
        if floor_z is not None and not math.isfinite(floor_z):
            raise ValueError("floor_z muss endlich oder None sein.")
        if not math.isfinite(floor_restitution) or not 0 <= floor_restitution <= 1:
            raise ValueError("floor_restitution muss zwischen 0 und 1 liegen.")
        if not math.isfinite(floor_friction) or floor_friction < 0:
            raise ValueError("floor_friction muss endlich und >= 0 sein.")
        if not math.isfinite(max_substep) or max_substep <= 0:
            raise ValueError("max_substep muss endlich und > 0 sein.")
        if not isinstance(solver_iterations, int) or solver_iterations < 1:
            raise ValueError("solver_iterations muss eine positive ganze Zahl sein.")
        self.floor_z = floor_z
        self.floor_restitution = floor_restitution
        self.floor_friction = floor_friction
        self.max_substep = max_substep
        self.solver_iterations = solver_iterations
        self.bodies = []
        self.time = 0.0

    def add(self, body):
        if not isinstance(body, Sphere):
            raise TypeError("Diese Version unterstützt Sphere-Objekte.")
        if body in self.bodies:
            raise ValueError("Der Körper ist bereits in der Welt.")
        self.bodies.append(body)
        return body

    def step(self, dt):
        """Expliziter Simulationsschritt; benötigt weder Qt noch eine Echtzeituhr."""
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt muss endlich und > 0 sein.")
        substeps = math.ceil(dt / self.max_substep)
        h = dt / substeps
        for _ in range(substeps):
            # 1. Kräfte -> Geschwindigkeit -> Position (semi-impliziter Euler).
            for body in self.bodies:
                if body.mass == 0:
                    continue
                body.velocity += (self.gravity + body.force * body.inverse_mass) * h
                body.angular_velocity += body.torque * body.inverse_inertia * h
                body.position += body.velocity * h
                self._rotate(body, h)

            # 2. Kontakte finden und mehrfach lösen, damit auch Stapel reagieren.
            for _ in range(self.solver_iterations):
                for a, b, normal, depth in self._contacts():
                    self._resolve(a, b, normal, depth)

        # 3. Externe Kräfte gelten für genau diesen öffentlichen step-Aufruf.
        for body in self.bodies:
            body.force.fill(0)
            body.torque.fill(0)
        self.time += dt

    @staticmethod
    def _rotate(body, h):
        # Rodrigues: eine kleine Drehung um die momentane Winkelgeschwindigkeit.
        speed = np.linalg.norm(body.angular_velocity)
        if speed == 0:
            return
        x, y, z = body.angular_velocity / speed
        skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        angle = speed * h
        rotation = np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * (skew @ skew)
        body.orientation = rotation @ body.orientation

    def _contacts(self):
        # Normalen zeigen immer von b nach a; b=None steht für den Boden.
        if self.floor_z is not None:
            for a in self.bodies:
                depth = self.floor_z + a.radius - a.position[2]
                if a.mass and depth >= 0:
                    yield a, None, np.array([0., 0., 1.]), depth
        # Bewusst O(n²): für kleine Lehrszenen leicht lesbar und ausreichend.
        for a, b in combinations(self.bodies, 2):
            if a.mass == 0 and b.mass == 0:
                continue
            delta = a.position - b.position
            distance = np.linalg.norm(delta)
            depth = a.radius + b.radius - distance
            if depth >= 0:
                normal = delta / distance if distance > 1e-12 else np.array([1., 0., 0.])
                yield a, b, normal, depth

    def _resolve(self, a, b, normal, depth):
        inv_b = b.inverse_mass if b is not None else 0.0
        inv_sum = a.inverse_mass + inv_b

        # Kleine Durchdringungen separat korrigieren, ohne Energie einzuspeisen.
        correction = 0.8 * max(depth - 1e-5, 0) * normal / inv_sum
        a.position += correction * a.inverse_mass
        if b is not None:
            b.position -= correction * inv_b

        arm_a = -normal * a.radius
        arm_b = normal * b.radius if b is not None else np.zeros(3)

        def relative_velocity():
            va = a.velocity + np.cross(a.angular_velocity, arm_a)
            vb = b.velocity + np.cross(b.angular_velocity, arm_b) if b is not None else 0
            return va - vb

        def impulse_pair(impulse):
            a.apply_impulse(impulse, a.position + arm_a)
            if b is not None:
                b.apply_impulse(-impulse, b.position + arm_b)

        normal_speed = np.dot(relative_velocity(), normal)
        if normal_speed >= 0:          # Bereits auseinanderfliegende Körper
            return
        restitution = min(a.restitution, b.restitution if b is not None else self.floor_restitution)
        if normal_speed > -0.5:        # Ruhende Kontakte sollen nicht ewig hüpfen.
            restitution = 0.0
        # Kugelkontakt: Normalimpuls geht durch den Schwerpunkt, daher kein Drehmoment.
        normal_impulse = -(1 + restitution) * normal_speed / inv_sum
        impulse_pair(normal_impulse * normal)

        # Coulomb-Reibung am Kontaktpunkt erzeugt zusätzlich Rotation.
        relative = relative_velocity()
        tangent = relative - np.dot(relative, normal) * normal
        speed = np.linalg.norm(tangent)
        if speed < 1e-12:
            return
        tangent /= speed
        denominator = inv_sum + a.inverse_inertia * np.linalg.norm(np.cross(arm_a, tangent))**2
        if b is not None:
            denominator += b.inverse_inertia * np.linalg.norm(np.cross(arm_b, tangent))**2
        friction = math.sqrt(a.friction * (b.friction if b is not None else self.floor_friction))
        friction_impulse = min(speed / denominator, friction * normal_impulse)
        impulse_pair(-friction_impulse * tangent)
