"""Minimal PySide6 lunar lander (no Gymnasium).

Controls:
- Left Arrow: fire left thruster
- Right Arrow: fire right thruster
- Up Arrow: fire bottom thruster

Goal: land softly on the uneven moon surface.
"""

from __future__ import annotations

import math
import random
import sys
import time
from dataclasses import dataclass
from typing import List, Tuple

from PySide6 import QtCore, QtGui, QtWidgets


@dataclass
class Vec2:
    x: float
    y: float

    def __add__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> "Vec2":
        return Vec2(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar: float) -> "Vec2":
        return Vec2(self.x / scalar, self.y / scalar)

    def __iadd__(self, other: "Vec2") -> "Vec2":
        self.x += other.x
        self.y += other.y
        return self

    def __isub__(self, other: "Vec2") -> "Vec2":
        self.x -= other.x
        self.y -= other.y
        return self

    def __imul__(self, scalar: float) -> "Vec2":
        self.x *= scalar
        self.y *= scalar
        return self

    def iadd(self, other: "Vec2") -> None:
        self.x += other.x
        self.y += other.y

    def isub(self, other: "Vec2") -> None:
        self.x -= other.x
        self.y -= other.y

    def imul(self, scalar: float) -> None:
        self.x *= scalar
        self.y *= scalar

    def to_qpointf(self) -> QtCore.QPointF:
        return QtCore.QPointF(self.x, self.y)

    @staticmethod
    def from_qpointf(p: QtCore.QPointF) -> "Vec2":
        return Vec2(p.x(), p.y())


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


@dataclass
class Terrain:
    points: List[Tuple[float, float]]  # (x, y)
    width: float

    def height_at(self, x: float) -> float:
        if not self.points:
            return 0.0
        x = clamp(x, 0.0, self.width)
        step = self.points[1][0] - self.points[0][0] if len(self.points) > 1 else self.width
        idx = int(x // step)
        idx = max(0, min(idx, len(self.points) - 2))
        x0, y0 = self.points[idx]
        x1, y1 = self.points[idx + 1]
        if x1 == x0:
            return y0
        t = (x - x0) / (x1 - x0)
        return lerp(y0, y1, t)

    @staticmethod
    def generate(width: float, base_y: float, amplitude: float, step: float, seed: int | None = None) -> "Terrain":
        rng = random.Random(seed)
        n = int(width // step) + 1
        raw = [rng.uniform(-1.0, 1.0) for _ in range(n)]

        smooth = raw[:]
        for _ in range(5):
            nxt = smooth[:]
            for i in range(1, n - 1):
                nxt[i] = (smooth[i - 1] + 2.0 * smooth[i] + smooth[i + 1]) / 4.0
            smooth = nxt

        points: List[Tuple[float, float]] = []
        for i in range(n):
            x = i * step
            y = base_y + amplitude * smooth[i]
            points.append((x, y))

        return Terrain(points=points, width=width)


@dataclass(frozen=True)
class LandingPad:
    x0: float
    x1: float
    y: float

    @property
    def cx(self) -> float:
        return 0.5 * (self.x0 + self.x1)

    @property
    def width(self) -> float:
        return self.x1 - self.x0


@dataclass
class Lander:
    pos: Vec2
    vel: Vec2
    angle: float
    ang_vel: float

    radius: float = 14.0
    mass: float = 1.0
    inertia: float = 1.0


class LanderWidget(QtWidgets.QWidget):
    telemetryUpdated = QtCore.Signal(dict)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(900, 520)

        self._t_last = time.perf_counter()
        self._timer = QtCore.QTimer(self)
        self._timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self._tick)
        self._timer.start(int(1000 / 60))

        self.world_w = 900.0
        self.world_h = 520.0

        # Global slow-motion factor for easier manual control.
        # 1.0 = real-time, 0.5 = half-speed.
        self.time_scale = 0.55

        # Tune these for "hand-flyable" dynamics in class.
        self.gravity = Vec2(0.0, -140.0)
        self.main_thrust = 650.0
        self.side_thrust = 180.0
        self.side_torque = 420.0

        # Damps angular velocity as: w *= exp(-angular_damping * dt)
        self.angular_damping = 3.2
        self.max_ang_vel = 1.2

        self.max_main_burn = 1.0
        self.max_side_burn = 1.0

        self.key_left = False
        self.key_right = False
        self.key_up = False

        self.status_text: str | None = None
        self._frozen = False

        # Filled on first ground contact (either landing or crash).
        self.last_contact: dict | None = None

        self.success_max_abs_vy = 60.0
        self.success_max_abs_vx = 60.0
        self.success_max_abs_angle = math.radians(15.0)

        self.terrain = Terrain(points=[], width=self.world_w)
        self.landing_pad = LandingPad(0.0, 0.0, 0.0)
        self.lander = self._spawn_lander()
        self.reset()

    def reset(self) -> None:
        rng = random.Random()
        seed = rng.randrange(0, 10_000_000)

        self.terrain = Terrain.generate(
            width=self.world_w,
            base_y=95.0,
            amplitude=50.0,
            step=18.0,
            seed=seed,
        )

        pad_width = 140.0
        pad_margin = 40.0
        pad_x0 = rng.uniform(pad_margin, self.world_w - pad_margin - pad_width)
        pad_x1 = pad_x0 + pad_width

        pad_y = self.terrain.height_at(0.5 * (pad_x0 + pad_x1))

        flattened: List[Tuple[float, float]] = []
        for x, y in self.terrain.points:
            if pad_x0 <= x <= pad_x1:
                flattened.append((x, pad_y))
            else:
                flattened.append((x, y))
        self.terrain = Terrain(points=flattened, width=self.world_w)
        self.landing_pad = LandingPad(x0=pad_x0, x1=pad_x1, y=pad_y)

        self.lander = self._spawn_lander()
        self.status_text = None
        self._frozen = False
        self.last_contact = None
        self.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)

    def _spawn_lander(self) -> Lander:
        return Lander(
            pos=Vec2(self.world_w * 0.25, self.world_h * 0.78),
            vel=Vec2(0.0, 0.0),
            angle=0.25,
            ang_vel=0.0,
            radius=14.0,
            mass=1.0,
            inertia=0.9,
        )

    def _world_to_screen(self, p: Vec2) -> QtCore.QPointF:
        return QtCore.QPointF(p.x, self.height() - p.y)

    def _screen_to_world_y(self, y_screen: float) -> float:
        return self.height() - y_screen

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(900, 520)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.isAutoRepeat():
            return
        k = event.key()
        if k == QtCore.Qt.Key.Key_Left:
            self.key_left = True
        elif k == QtCore.Qt.Key.Key_Right:
            self.key_right = True
        elif k == QtCore.Qt.Key.Key_Up:
            self.key_up = True
        elif k == QtCore.Qt.Key.Key_Escape:
            QtWidgets.QApplication.quit()
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.isAutoRepeat():
            return
        k = event.key()
        if k == QtCore.Qt.Key.Key_Left:
            self.key_left = False
        elif k == QtCore.Qt.Key.Key_Right:
            self.key_right = False
        elif k == QtCore.Qt.Key.Key_Up:
            self.key_up = False
        else:
            super().keyReleaseEvent(event)

    def _tick(self) -> None:
        now = time.perf_counter()
        dt = now - self._t_last
        self._t_last = now

        dt = clamp(dt, 0.0, 1.0 / 20.0)
        dt *= self.time_scale

        if not self._frozen:
            self._step_physics(dt)

        self._emit_telemetry()

        self.update()

    def _emit_telemetry(self) -> None:
        lander = self.lander
        ground = self.terrain.height_at(lander.pos.x)
        altitude = max(0.0, lander.pos.y - (ground + lander.radius))
        ang = ((lander.angle + math.pi) % (2.0 * math.pi)) - math.pi
        in_pad = self.landing_pad.x0 <= lander.pos.x <= self.landing_pad.x1
        payload = {
            "pos_x": lander.pos.x,
            "pos_y": lander.pos.y,
            "vel_x": lander.vel.x,
            "vel_y": lander.vel.y,
            "angle_rad": ang,
            "angle_deg": math.degrees(ang),
            "ang_vel": lander.ang_vel,
            "altitude": altitude,
            "status": self.status_text or "FLYING",
            "in_landing_pad": in_pad,
        }

        if self.last_contact is not None:
            payload["last_contact"] = self.last_contact
        self.telemetryUpdated.emit(payload)

    def _step_physics(self, dt: float) -> None:
        lander = self.lander

        force = Vec2(self.gravity.x * lander.mass, self.gravity.y * lander.mass)
        torque = 0.0

        burn_main = self.max_main_burn if self.key_up else 0.0
        burn_left = self.max_side_burn if self.key_left else 0.0
        burn_right = self.max_side_burn if self.key_right else 0.0

        c = math.cos(lander.angle)
        s = math.sin(lander.angle)
        up_dir = Vec2(-s, c)
        right_dir = Vec2(c, s)

        if burn_main > 0.0:
            force += up_dir * (self.main_thrust * burn_main)

        if burn_left > 0.0:
            force += right_dir * (self.side_thrust * burn_left)
            torque -= self.side_torque * burn_left

        if burn_right > 0.0:
            force -= right_dir * (self.side_thrust * burn_right)
            torque += self.side_torque * burn_right

        acc = force / lander.mass
        ang_acc = torque / lander.inertia

        lander.vel += acc * dt
        lander.ang_vel += ang_acc * dt

        lander.ang_vel *= math.exp(-self.angular_damping * dt)
        lander.ang_vel = clamp(lander.ang_vel, -self.max_ang_vel, self.max_ang_vel)

        lander.pos += lander.vel * dt
        lander.angle += lander.ang_vel * dt

        if lander.pos.x < lander.radius:
            lander.pos.x = lander.radius
            lander.vel.x = -0.3 * lander.vel.x
        elif lander.pos.x > self.world_w - lander.radius:
            lander.pos.x = self.world_w - lander.radius
            lander.vel.x = -0.3 * lander.vel.x

        if lander.pos.y > self.world_h - lander.radius:
            lander.pos.y = self.world_h - lander.radius
            lander.vel.y = -0.2 * lander.vel.y

        self._handle_ground_contact()

    def _lander_local_point_to_world(self, lx: float, ly_down: float) -> Vec2:
        """Convert a point from lander-local drawing coords to world coords.

        Local coords:
        - +x is to the right
        - +y is down (as used in QPainter drawing)

        World coords:
        - +x is to the right
        - +y is up
        """
        lander = self.lander
        c = math.cos(lander.angle)
        s = math.sin(lander.angle)
        # offset = right_dir*lx + up_dir*(-ly_down)
        ox = c * lx + s * ly_down
        oy = s * lx - c * ly_down
        return Vec2(lander.pos.x + ox, lander.pos.y + oy)

    def _ground_penetration_for_point(self, p: Vec2) -> float:
        ground_y = self.terrain.height_at(p.x)
        return ground_y - p.y

    def _contact_points_world(self) -> list[Vec2]:
        # Use leg "feet" (line endpoints) and a body-bottom point.
        # These match the points used in _draw_lander().
        return [
            self._lander_local_point_to_world(-20.0, 18.0),
            self._lander_local_point_to_world(20.0, 18.0),
            self._lander_local_point_to_world(0.0, 10.0),
        ]

    def _handle_ground_contact(self) -> None:
        lander = self.lander
        points = self._contact_points_world()
        penetrations = [self._ground_penetration_for_point(p) for p in points]
        max_pen = max(penetrations) if penetrations else 0.0

        if max_pen <= 0.0:
            return

        vx = lander.vel.x
        vy = lander.vel.y
        speed_x = abs(vx)
        speed_y = abs(vy)
        ang = ((lander.angle + math.pi) % (2.0 * math.pi)) - math.pi
        angle_ok = abs(ang) < self.success_max_abs_angle

        in_pad = self.landing_pad.x0 <= lander.pos.x <= self.landing_pad.x1
        vy_ok = speed_y <= self.success_max_abs_vy
        vx_ok = speed_x <= self.success_max_abs_vx
        soft = vy_ok and vx_ok and angle_ok and in_pad

        angle_deg = math.degrees(ang)
        max_angle_deg = math.degrees(self.success_max_abs_angle)

        def pf(ok: bool) -> str:
            return "PASS" if ok else "FAIL"

        checks = [
            f"on landing pad: {pf(in_pad)}",
            f"|vy| <= {self.success_max_abs_vy:.1f}: {pf(vy_ok)}   (|vy|={speed_y:.1f})",
            f"|vx| <= {self.success_max_abs_vx:.1f}: {pf(vx_ok)}   (|vx|={speed_x:.1f})",
            f"|angle| <= {max_angle_deg:.1f}°: {pf(angle_ok)}   (|angle|={abs(angle_deg):.1f}°)",
        ]

        self.last_contact = {
            "impact_vx": vx,
            "impact_vy": vy,
            "impact_speed_x": speed_x,
            "impact_speed_y": speed_y,
            "impact_angle_deg": angle_deg,
            "impact_in_pad": in_pad,
            "success": soft,
            "checks": checks,
        }

        # Resolve penetration by shifting the lander up in world space.
        lander.pos.y = lander.pos.y + max_pen

        if soft:
            lander.vel = Vec2(0.0, 0.0)
            lander.ang_vel = 0.0
            lander.angle = 0.0
            self.status_text = "LANDED"
            self._frozen = True

            # After snapping angle, resolve any remaining penetration.
            points2 = self._contact_points_world()
            pens2 = [self._ground_penetration_for_point(p) for p in points2]
            max_pen2 = max(pens2) if pens2 else 0.0
            if max_pen2 > 0.0:
                lander.pos.y = lander.pos.y + max_pen2
        else:
            lander.vel = Vec2(0.0, 0.0)
            lander.ang_vel = 0.0
            self.status_text = "CRASHED"
            self._frozen = True

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        _ = event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)

        painter.fillRect(self.rect(), QtGui.QColor(10, 10, 14))

        self._draw_stars(painter)
        self._draw_terrain(painter)
        self._draw_lander(painter)

    def _draw_stars(self, painter: QtGui.QPainter) -> None:
        painter.save()
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor(220, 220, 230))
        rng = random.Random(2)
        for _ in range(90):
            x = rng.randrange(0, self.width())
            y = rng.randrange(0, int(self.height() * 0.7))
            r = 1 if rng.random() < 0.85 else 2
            painter.drawEllipse(QtCore.QPointF(x, y), r, r)
        painter.restore()

    def _draw_terrain(self, painter: QtGui.QPainter) -> None:
        painter.save()
        path = QtGui.QPainterPath()

        x0, y0 = self.terrain.points[0]
        p0 = self._world_to_screen(Vec2(x0, y0))
        path.moveTo(p0)
        for x, y in self.terrain.points[1:]:
            path.lineTo(self._world_to_screen(Vec2(x, y)))

        path.lineTo(self._world_to_screen(Vec2(self.world_w, 0.0)))
        path.lineTo(self._world_to_screen(Vec2(0.0, 0.0)))
        path.closeSubpath()

        painter.setPen(QtGui.QPen(QtGui.QColor(210, 210, 225), 2))
        painter.setBrush(QtGui.QColor(135, 135, 150))
        painter.drawPath(path)

        pad = self.landing_pad
        pad_screen_y = self.height() - pad.y
        painter.setPen(QtGui.QPen(QtGui.QColor(240, 220, 60), 3))
        painter.drawLine(QtCore.QPointF(pad.x0, pad_screen_y), QtCore.QPointF(pad.x1, pad_screen_y))

        painter.setPen(QtGui.QPen(QtGui.QColor(240, 220, 60), 2))
        painter.drawLine(QtCore.QPointF(pad.x0, pad_screen_y), QtCore.QPointF(pad.x0, pad_screen_y - 22))
        painter.drawLine(QtCore.QPointF(pad.x1, pad_screen_y), QtCore.QPointF(pad.x1, pad_screen_y - 22))
        painter.setBrush(QtGui.QColor(240, 220, 60))
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.drawPolygon(
            QtGui.QPolygonF(
                [
                    QtCore.QPointF(pad.x0, pad_screen_y - 22),
                    QtCore.QPointF(pad.x0 + 14, pad_screen_y - 18),
                    QtCore.QPointF(pad.x0, pad_screen_y - 14),
                ]
            )
        )
        painter.drawPolygon(
            QtGui.QPolygonF(
                [
                    QtCore.QPointF(pad.x1, pad_screen_y - 22),
                    QtCore.QPointF(pad.x1 - 14, pad_screen_y - 18),
                    QtCore.QPointF(pad.x1, pad_screen_y - 14),
                ]
            )
        )
        painter.restore()

    def _draw_lander(self, painter: QtGui.QPainter) -> None:
        lander = self.lander
        center = self._world_to_screen(lander.pos)

        painter.save()
        painter.translate(center)
        painter.rotate(-math.degrees(lander.angle))

        body = QtCore.QRectF(-14, -10, 28, 20)
        painter.setPen(QtGui.QPen(QtGui.QColor(200, 200, 210), 2))
        painter.setBrush(QtGui.QColor(120, 120, 140))
        painter.drawRoundedRect(body, 4, 4)

        painter.setPen(QtGui.QPen(QtGui.QColor(180, 180, 190), 2))
        painter.drawLine(QtCore.QPointF(-12, 10), QtCore.QPointF(-20, 18))
        painter.drawLine(QtCore.QPointF(12, 10), QtCore.QPointF(20, 18))

        painter.setBrush(QtGui.QColor(255, 160, 60))
        painter.setPen(QtCore.Qt.PenStyle.NoPen)

        if self.key_up and not self._frozen:
            flame = QtGui.QPolygonF(
                [
                    QtCore.QPointF(-5, 11),
                    QtCore.QPointF(5, 11),
                    QtCore.QPointF(0, 28),
                ]
            )
            painter.drawPolygon(flame)

        if self.key_left and not self._frozen:
            flame = QtGui.QPolygonF(
                [
                    QtCore.QPointF(-16, -4),
                    QtCore.QPointF(-16, 4),
                    QtCore.QPointF(-30, 0),
                ]
            )
            painter.drawPolygon(flame)

        if self.key_right and not self._frozen:
            flame = QtGui.QPolygonF(
                [
                    QtCore.QPointF(16, -4),
                    QtCore.QPointF(16, 4),
                    QtCore.QPointF(30, 0),
                ]
            )
            painter.drawPolygon(flame)

        painter.restore()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Lunar Lander 2D (PySide6 demo)")

        self._contact_text_last: str | None = None

        root = QtWidgets.QWidget(self)
        layout = QtWidgets.QHBoxLayout(root)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        self.sim = LanderWidget(root)
        layout.addWidget(self.sim, 1)

        panel = QtWidgets.QWidget(root)
        panel.setFixedWidth(260)
        v = QtWidgets.QVBoxLayout(panel)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(8)

        title = QtWidgets.QLabel("Controls")
        title.setStyleSheet("font-weight: 600;")
        v.addWidget(title)

        v.addWidget(QtWidgets.QLabel("Arrow Left: left thruster"))
        v.addWidget(QtWidgets.QLabel("Arrow Right: right thruster"))
        v.addWidget(QtWidgets.QLabel("Arrow Up: bottom thruster"))
        v.addSpacing(6)

        self.restart_btn = QtWidgets.QPushButton("Restart")
        self.restart_btn.clicked.connect(self._on_restart)
        v.addWidget(self.restart_btn)

        v.addSpacing(10)
        stats_title = QtWidgets.QLabel("Telemetry")
        stats_title.setStyleSheet("font-weight: 600;")
        v.addWidget(stats_title)

        self.lbl_status = QtWidgets.QLabel("status: FLYING")
        self.lbl_pos = QtWidgets.QLabel("pos: (0.0, 0.0)")
        self.lbl_vel = QtWidgets.QLabel("vel: (0.0, 0.0)")
        self.lbl_angle = QtWidgets.QLabel("angle: 0.0°")
        self.lbl_alt = QtWidgets.QLabel("altitude: 0.0")
        self.lbl_pad = QtWidgets.QLabel("landing pad: no")
        contact_title = QtWidgets.QLabel("Contact")
        contact_title.setStyleSheet("font-weight: 600;")

        self.txt_contact = QtWidgets.QPlainTextEdit()
        self.txt_contact.setReadOnly(True)
        self.txt_contact.setMinimumHeight(110)
        self.txt_contact.setMaximumHeight(170)

        for lbl in [
            self.lbl_status,
            self.lbl_pos,
            self.lbl_vel,
            self.lbl_angle,
            self.lbl_alt,
            self.lbl_pad,
        ]:
            lbl.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            v.addWidget(lbl)

        v.addSpacing(6)
        v.addWidget(contact_title)
        v.addWidget(self.txt_contact)

        v.addStretch(1)

        layout.addWidget(panel, 0)
        self.setCentralWidget(root)

        self.sim.telemetryUpdated.connect(self._on_telemetry)

    def _on_restart(self) -> None:
        self.sim.reset()

    def _on_telemetry(self, t: dict) -> None:
        self.lbl_status.setText(f"status: {t['status']}")
        self.lbl_pos.setText(f"pos: ({t['pos_x']:.1f}, {t['pos_y']:.1f})")
        self.lbl_vel.setText(f"vel: ({t['vel_x']:.1f}, {t['vel_y']:.1f})")
        self.lbl_angle.setText(f"angle: {t['angle_deg']:.1f}°")
        self.lbl_alt.setText(f"altitude: {t['altitude']:.1f}")
        self.lbl_pad.setText(f"landing pad: {'yes' if t['in_landing_pad'] else 'no'}")

        contact = t.get("last_contact")
        if isinstance(contact, dict):
            lines = [
                "impact:",
                f"  vx={contact['impact_vx']:.1f}",
                f"  vy={contact['impact_vy']:.1f}",
                f"  angle={contact['impact_angle_deg']:.1f}°",
                "",
                "checks:",
            ]
            checks = contact.get("checks")
            if isinstance(checks, list):
                lines.extend([f"  {str(x)}" for x in checks])
            else:
                lines.append("(no checks)")
            text = "\n".join(lines)
            if text != self._contact_text_last:
                self._contact_text_last = text
                self.txt_contact.setPlainText(text)
        else:
            text = "impact: -"
            if text != self._contact_text_last:
                self._contact_text_last = text
                self.txt_contact.setPlainText(text)


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.resize(900, 520)
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
