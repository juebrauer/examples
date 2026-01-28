#!/usr/bin/env python3
"""
Robot Arm Simulation Data Recorder (n-DOF, 2D)

Records simulation "camera" images and expert actions to disk.

Directory layout (inside --output):
- images/                     (all PNGs go here)
  - image_000000.png
  - image_000001.png
  - ...
- samples.csv                 (only: image_filename, action)
- metadata.json

Action encoding:
- One-hot vector of length (3 * DOF)
- Index mapping:
    idx = dof_index * 3 + direction
    direction: 0=no change, 1=-1 degree, 2=+1 degree
- Stored in samples.csv as a JSON list string, e.g. "[0,0,1,0,0,0,...]"

CLI:
  python robot_arm_simulator.py --dof 3 --samples 5000 --output ./data_dof3_5k
  python robot_arm_simulator.py --dof 5 --samples 200000 --output ./data_5dof --headless

Requirements:
  PySide6, numpy, opencv-python
"""

import sys
import math
import time
import json
import csv
import random
import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Any

import numpy as np

from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel
from PySide6.QtCore import QTimer
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QPixmap
from PySide6.QtCore import Qt

try:
    import cv2
except ImportError as e:
    raise ImportError("opencv-python is required (pip install opencv-python).") from e


def action_index_to_onehot(action_index: int, action_dim: int) -> List[int]:
    if not (0 <= action_index < action_dim):
        raise ValueError(f"action_index {action_index} out of range for action_dim {action_dim}")
    vec = [0] * action_dim
    vec[action_index] = 1
    return vec


class KinematicChain:
    """n-DOF planar chain with a simple expert action selector (greedy distance reduction)."""

    def __init__(self, num_dof: int = 2, arm_length: float = 80.0):
        self.num_dof = int(num_dof)
        self.arm_length = float(arm_length)
        self.max_reach = self.num_dof * self.arm_length

        self.angles: List[float] = [0.0 for _ in range(self.num_dof)]
        self.joint_positions: List[Tuple[float, float]] = [(0.0, 0.0)]
        self.end_effector_pos: Tuple[float, float] = (0.0, 0.0)

        self.target_pos: Tuple[float, float] = (0.0, 0.0)
        self.target_tolerance = 15.0

        self.angle_step = math.radians(1.0)

        self.scenario_count = 0
        self.current_step = 0
        self.scenario_complete = False

        self.update_positions()
        self.generate_new_scenario()

    def update_positions(self) -> None:
        self.joint_positions = [(0.0, 0.0)]
        x, y = 0.0, 0.0
        cumulative_angle = 0.0

        for i in range(self.num_dof):
            cumulative_angle += self.angles[i]
            x += self.arm_length * math.cos(cumulative_angle)
            y += self.arm_length * math.sin(cumulative_angle)
            self.joint_positions.append((x, y))

        self.end_effector_pos = (x, y)

    def generate_reachable_target(self) -> Tuple[float, float]:
        if self.num_dof == 1:
            angle = random.uniform(0.0, 2.0 * math.pi)
            return (self.arm_length * math.cos(angle), self.arm_length * math.sin(angle))

        max_distance = 0.7 * self.max_reach
        min_distance = 0.2 * self.max_reach
        distance = random.uniform(min_distance, max_distance)
        angle = random.uniform(0.0, 2.0 * math.pi)
        return (distance * math.cos(angle), distance * math.sin(angle))

    def generate_new_scenario(self) -> None:
        self.angles = [random.uniform(-math.pi, math.pi) for _ in range(self.num_dof)]
        self.update_positions()
        self.target_pos = self.generate_reachable_target()

        self.scenario_count += 1
        self.current_step = 0
        self.scenario_complete = False

    def get_distance_to_target(self) -> float:
        dx = self.end_effector_pos[0] - self.target_pos[0]
        dy = self.end_effector_pos[1] - self.target_pos[1]
        return math.sqrt(dx * dx + dy * dy)

    def is_target_reached(self) -> bool:
        return self.get_distance_to_target() <= self.target_tolerance

    def get_action_space_size(self) -> int:
        return 3 * self.num_dof

    def get_expert_action_index(self) -> int:
        """
        Greedy expert:
          For each joint, try +/-1 degree, pick the action that most reduces distance.
          Action encoding:
            idx = dof_index * 3 + direction
            direction: 0=no change, 1=-1 degree, 2=+1 degree
        """
        if self.is_target_reached():
            return 0

        # Simple 1-DOF heuristic
        if self.num_dof == 1:
            tx, ty = self.target_pos
            target_angle = math.atan2(ty, tx)
            current_angle = self.angles[0]
            angle_diff = target_angle - current_angle

            while angle_diff > math.pi:
                angle_diff -= 2.0 * math.pi
            while angle_diff < -math.pi:
                angle_diff += 2.0 * math.pi

            if abs(angle_diff) < math.radians(0.5):
                return 0
            return 2 if angle_diff > 0 else 1

        best_action = 0
        best_distance = self.get_distance_to_target()

        for dof in range(self.num_dof):
            original_angle = self.angles[dof]

            for direction in (1, 2):
                self.angles[dof] = original_angle + (self.angle_step if direction == 2 else -self.angle_step)
                self.update_positions()
                new_distance = self.get_distance_to_target()

                if new_distance < best_distance:
                    best_distance = new_distance
                    best_action = dof * 3 + direction

            self.angles[dof] = original_angle
            self.update_positions()

        return best_action

    def take_action_index(self, action_index: int) -> bool:
        joint_index = action_index // 3
        direction = action_index % 3

        if 0 <= joint_index < len(self.angles):
            if direction == 1:
                self.angles[joint_index] -= self.angle_step
            elif direction == 2:
                self.angles[joint_index] += self.angle_step

            # keep angles bounded (optional)
            self.angles[joint_index] = ((self.angles[joint_index] + 2 * math.pi) % (4 * math.pi)) - 2 * math.pi

        self.update_positions()
        self.current_step += 1

        if self.is_target_reached():
            self.scenario_complete = True
            return True

        return False


class KinematicWidget(QWidget):
    """
    Renders the arm and provides a clean image capture (no text).
    - End effector: blue circle
    - Target: red X
    """

    def __init__(self, chain: KinematicChain):
        super().__init__()
        self.chain = chain

        self.setFixedSize(400, 400)
        self.center_x = 200
        self.center_y = 200

        self.joint_radius = 8
        self.arm_width = 4
        self.angle_line_length = 12

        self.end_effector_radius = 6  # blue circle
        self.target_size = 10         # red X

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(255, 255, 255))

        self._draw_arms(painter)
        self._draw_joints(painter)
        self._draw_end_effector(painter)
        self._draw_target(painter)
        self._draw_overlay_text(painter)

    def _draw_arms(self, painter: QPainter) -> None:
        pen = QPen(QColor(128, 128, 128), self.arm_width)
        painter.setPen(pen)
        for i in range(len(self.chain.joint_positions) - 1):
            sx = self.center_x + self.chain.joint_positions[i][0]
            sy = self.center_y + self.chain.joint_positions[i][1]
            ex = self.center_x + self.chain.joint_positions[i + 1][0]
            ey = self.center_y + self.chain.joint_positions[i + 1][1]
            painter.drawLine(int(sx), int(sy), int(ex), int(ey))

    def _draw_joints(self, painter: QPainter) -> None:
        painter.setBrush(QBrush(QColor(0, 0, 0)))
        painter.setPen(QPen(QColor(0, 0, 0), 1))

        for i, (x, y) in enumerate(self.chain.joint_positions[:-1]):
            jx = self.center_x + x
            jy = self.center_y + y

            painter.drawEllipse(
                int(jx - self.joint_radius),
                int(jy - self.joint_radius),
                2 * self.joint_radius,
                2 * self.joint_radius,
            )

            if i < len(self.chain.angles):
                self._draw_angle_indicator(painter, jx, jy, i)

    def _draw_angle_indicator(self, painter: QPainter, joint_x: float, joint_y: float, joint_index: int) -> None:
        cumulative_angle = sum(self.chain.angles[: joint_index + 1])
        lx = joint_x + self.angle_line_length * math.cos(cumulative_angle)
        ly = joint_y + self.angle_line_length * math.sin(cumulative_angle)

        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.drawLine(int(joint_x), int(joint_y), int(lx), int(ly))

    def _draw_end_effector(self, painter: QPainter) -> None:
        painter.setBrush(QBrush(QColor(0, 0, 255)))
        painter.setPen(QPen(QColor(0, 0, 255), 1))

        ex = self.center_x + self.chain.end_effector_pos[0]
        ey = self.center_y + self.chain.end_effector_pos[1]

        painter.drawEllipse(
            int(ex - self.end_effector_radius),
            int(ey - self.end_effector_radius),
            2 * self.end_effector_radius,
            2 * self.end_effector_radius,
        )

    def _draw_target(self, painter: QPainter) -> None:
        tx = self.center_x + self.chain.target_pos[0]
        ty = self.center_y + self.chain.target_pos[1]

        painter.setPen(QPen(QColor(255, 0, 0), 3))
        half = self.target_size // 2
        painter.drawLine(int(tx - half), int(ty - half), int(tx + half), int(ty + half))
        painter.drawLine(int(tx - half), int(ty + half), int(tx + half), int(ty - half))

        # Optional tolerance circle (light red dashed)
        pen = QPen(QColor(255, 200, 200), 1)
        pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(QBrush())
        d = int(2 * self.chain.target_tolerance)
        painter.drawEllipse(
            int(tx - self.chain.target_tolerance),
            int(ty - self.chain.target_tolerance),
            d,
            d,
        )

    def _draw_overlay_text(self, painter: QPainter) -> None:
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        lines = [
            f"Scenario: {self.chain.scenario_count} | Step: {self.chain.current_step}",
            f"Distance: {self.chain.get_distance_to_target():.1f}px",
        ]
        for i, line in enumerate(lines):
            painter.drawText(10, 20 + i * 15, line)

    def capture_image(self) -> np.ndarray:
        """Capture RGB image without overlay text."""
        pixmap = QPixmap(400, 400)
        pixmap.fill(QColor(255, 255, 255))

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        self._draw_arms(painter)
        self._draw_joints(painter)
        self._draw_end_effector(painter)
        self._draw_target(painter)

        painter.end()

        image = pixmap.toImage()
        w, h = image.width(), image.height()
        ptr = image.constBits()
        arr = np.array(ptr).reshape((h, w, 4))
        rgb = arr[:, :, :3].copy()
        return rgb


def write_outputs(
    output_dir: Path,
    rows: List[Dict[str, Any]],
    chain: KinematicChain,
    collected: int,
    start_time: float,
    headless: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if collected == 0:
        raise RuntimeError("No samples collected; refusing to write empty outputs.")

    # CSV (only two columns)
    csv_path = output_dir / "samples.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_filename", "action"])
        writer.writeheader()
        writer.writerows(rows)

    # Metadata
    metadata = {
        "dof": chain.num_dof,
        "arm_length_px": chain.arm_length,
        "target_tolerance_px": chain.target_tolerance,
        "action_dim": chain.get_action_space_size(),
        "action_encoding": {
            "type": "one_hot",
            "length": chain.get_action_space_size(),
            "index_mapping": "index = dof_index * 3 + direction; direction: 0=no change, 1=-1 degree, 2=+1 degree",
            "stored_in_csv_as": "JSON list string",
        },
        "total_samples": collected,
        "headless": bool(headless),
        "created_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(time.time() - start_time, 3),
        "files": {
            "images_dir": "images/",
            "csv": "samples.csv",
            "metadata": "metadata.json",
        },
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nDone.")
    print(f"  Output directory: {output_dir.resolve()}")
    print(f"  Images: {(output_dir / 'images').resolve()}")
    print(f"  Samples: {collected}")
    print(f"  CSV: {csv_path}")
    print(f"  Metadata: {output_dir / 'metadata.json'}")


class RecorderWindow(QMainWindow):
    """GUI mode: runs the simulation loop via a Qt timer and shows the widget."""

    def __init__(self, chain: KinematicChain, output_dir: Path, target_samples: int):
        super().__init__()

        self.chain = chain
        self.output_dir = output_dir
        self.images_dir = output_dir / "images"
        self.target_samples = int(target_samples)
        self.collected = 0

        self.images_dir.mkdir(parents=True, exist_ok=True)

        self.setWindowTitle(f"Recorder - {chain.num_dof} DOF")
        self.setFixedSize(420, 460)

        central = QWidget()
        layout = QVBoxLayout()
        self.widget = KinematicWidget(chain)
        layout.addWidget(self.widget)

        self.status = QLabel("Starting...")
        layout.addWidget(self.status)

        central.setLayout(layout)
        self.setCentralWidget(central)

        self.rows: List[Dict[str, Any]] = []
        self.start_time = time.time()

        self.timer = QTimer()
        self.timer.timeout.connect(self.step)
        self.timer.start(50)  # ~20 FPS

    def step(self) -> None:
        if self.collected >= self.target_samples:
            self.finish_and_quit()
            return

        image = self.widget.capture_image()

        action_index = self.chain.get_expert_action_index()
        action_dim = self.chain.get_action_space_size()
        action_onehot = action_index_to_onehot(action_index, action_dim)

        filename = f"image_{self.collected:06d}.png"
        rel_path = f"images/{filename}"
        out_path = self.images_dir / filename
        cv2.imwrite(str(out_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        # CSV row: only two columns
        self.rows.append(
            {
                "image_filename": rel_path,
                "action": json.dumps(action_onehot),
            }
        )
        self.collected += 1

        done = self.chain.take_action_index(action_index)
        if done:
            self.chain.generate_new_scenario()

        self.status.setText(f"Collected: {self.collected}/{self.target_samples} | Scenario: {self.chain.scenario_count}")
        self.widget.update()

    def finish_and_quit(self) -> None:
        self.timer.stop()
        write_outputs(
            output_dir=self.output_dir,
            rows=self.rows,
            chain=self.chain,
            collected=self.collected,
            start_time=self.start_time,
            headless=False,
        )
        QApplication.instance().quit()


class HeadlessRecorder:
    """Headless mode: no window, no Qt timers; runs as fast as possible."""

    def __init__(self, chain: KinematicChain, output_dir: Path, target_samples: int):
        self.chain = chain
        self.output_dir = output_dir
        self.images_dir = output_dir / "images"
        self.target_samples = int(target_samples)

        self.images_dir.mkdir(parents=True, exist_ok=True)

        # Offscreen widget (never shown); used for consistent rendering.
        self.widget = KinematicWidget(chain)

        self.rows: List[Dict[str, Any]] = []
        self.collected = 0
        self.start_time = time.time()

    def run(self) -> None:
        action_dim = self.chain.get_action_space_size()

        while self.collected < self.target_samples:
            image = self.widget.capture_image()

            action_index = self.chain.get_expert_action_index()
            action_onehot = action_index_to_onehot(action_index, action_dim)

            filename = f"image_{self.collected:06d}.png"
            rel_path = f"images/{filename}"
            out_path = self.images_dir / filename
            cv2.imwrite(str(out_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            self.rows.append(
                {
                    "image_filename": rel_path,
                    "action": json.dumps(action_onehot),
                }
            )
            self.collected += 1

            done = self.chain.take_action_index(action_index)
            if done:
                self.chain.generate_new_scenario()

            if self.collected % 2000 == 0:
                print(f"[headless] collected {self.collected}/{self.target_samples}")

        write_outputs(
            output_dir=self.output_dir,
            rows=self.rows,
            chain=self.chain,
            collected=self.collected,
            start_time=self.start_time,
            headless=True,
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record simulation images + expert actions.")
    p.add_argument("--dof", type=int, required=True, help="Number of degrees of freedom (>=1).")
    p.add_argument("--samples", type=int, required=True, help="Number of samples to record (>=1).")
    p.add_argument("--output", type=str, required=True, help="Output directory (no extra subfolder).")
    p.add_argument(
        "--headless",
        action="store_true",
        help="Run without showing a window (faster data collection).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.dof < 1:
        print("Error: --dof must be >= 1")
        return 1
    if args.samples < 1:
        print("Error: --samples must be >= 1")
        return 1

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Simulation Data Recorder ===")
    print(f"DOF: {args.dof}")
    print(f"Samples: {args.samples}")
    print(f"Headless: {args.headless}")
    print(f"Output: {out_dir.resolve()}")

    chain = KinematicChain(num_dof=args.dof)

    # Even in headless mode, we still create a QApplication because we render via QPixmap/QPainter.
    app = QApplication(sys.argv)

    if args.headless:
        recorder = HeadlessRecorder(chain=chain, output_dir=out_dir, target_samples=args.samples)
        recorder.run()
        return 0

    win = RecorderWindow(chain=chain, output_dir=out_dir, target_samples=args.samples)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
