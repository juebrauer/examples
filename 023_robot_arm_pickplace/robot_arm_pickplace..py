#!/usr/bin/env python3
"""
Robot Arm Pick-and-Place Simulator + Data Recorder (n-DOF, 2D)

This is an extended version of the user's previous simulator:
- The robot first moves to an object (blue box),
- closes the gripper to grasp it,
- moves to a goal (drop) position,
- opens the gripper to release it.

It can be used interactively (GUI) or headless to record training data (images + expert actions).

Directory layout (inside --output):
- images/                     (all PNGs go here)
  - image_000000.png
  - image_000001.png
  - ...
- samples.csv                 (image_filename, action)
- metadata.json

Action encoding (one-hot):
- Action dimension: (3 * DOF) + 2
- Joint action mapping (same as before):
    idx = dof_index * 3 + direction
    direction: 0=no change, 1=-1 degree, 2=+1 degree
- Two additional discrete gripper actions at the end:
    idx = 3*DOF + 0  -> gripper_open
    idx = 3*DOF + 1  -> gripper_close

CLI examples:
  python robot_arm_pickplace.py --dof 3 --samples 10000 --output ./data_pickplace_dof3_10k
  python robot_arm_pickplace.py --dof 2 --samples 5000 --output ./data_pickplace_dof2_5k --headless

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
from typing import Tuple, List, Dict, Any, Optional

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


class PickPlaceChain:
    """
    n-DOF planar chain with a simple expert policy for pick-and-place.

    Task phases:
      1) approach_object  -> move end effector to object
      2) grasp            -> close gripper (attach object)
      3) move_to_goal     -> move end effector (with object) to goal
      4) release          -> open gripper (drop object)
    """

    PHASE_APPROACH = "approach_object"
    PHASE_GRASP = "grasp"
    PHASE_TO_GOAL = "move_to_goal"
    PHASE_RELEASE = "release"

    def __init__(self, num_dof: int = 2, arm_length: float = 80.0):
        self.num_dof = int(num_dof)
        self.arm_length = float(arm_length)
        self.max_reach = self.num_dof * self.arm_length

        self.angles: List[float] = [0.0 for _ in range(self.num_dof)]
        self.joint_positions: List[Tuple[float, float]] = [(0.0, 0.0)]
        self.end_effector_pos: Tuple[float, float] = (0.0, 0.0)

        # Object + goal (world coordinates, centered at origin)
        self.object_pos: Tuple[float, float] = (0.0, 0.0)
        self.goal_pos: Tuple[float, float] = (0.0, 0.0)

        # Gripper state
        self.gripper_closed: bool = False
        self.holding_object: bool = False

        # Tolerances
        self.approach_tolerance = 14.0
        self.goal_tolerance = 14.0

        self.angle_step = math.radians(1.0)

        # Episode bookkeeping
        self.episode_count = 0
        self.current_step = 0
        self.episode_complete = False

        self.phase: str = self.PHASE_APPROACH

        self.update_positions()
        self.generate_new_episode()

    # ----------------- geometry -----------------

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

        # If holding: object follows end effector
        if self.holding_object:
            self.object_pos = self.end_effector_pos

    def _reachable_point(self, min_r: float, max_r: float) -> Tuple[float, float]:
        r = random.uniform(min_r, max_r)
        a = random.uniform(0.0, 2.0 * math.pi)
        return (r * math.cos(a), r * math.sin(a))

    def generate_new_episode(self) -> None:
        # Randomize arm pose
        self.angles = [random.uniform(-math.pi, math.pi) for _ in range(self.num_dof)]
        self.gripper_closed = False
        self.holding_object = False

        self.update_positions()

        # Place object and goal within reachable workspace but not too close to each other
        min_r = 0.25 * self.max_reach
        max_r = 0.75 * self.max_reach

        obj = self._reachable_point(min_r, max_r)
        goal = self._reachable_point(min_r, max_r)

        # Ensure some separation to make the task visible
        for _ in range(30):
            if math.hypot(goal[0] - obj[0], goal[1] - obj[1]) > 0.25 * self.max_reach:
                break
            goal = self._reachable_point(min_r, max_r)

        self.object_pos = obj
        self.goal_pos = goal

        self.phase = self.PHASE_APPROACH
        self.episode_count += 1
        self.current_step = 0
        self.episode_complete = False

    # ----------------- distances / phase logic -----------------

    @staticmethod
    def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return math.hypot(dx, dy)

    def _active_target(self) -> Optional[Tuple[float, float]]:
        if self.phase == self.PHASE_APPROACH:
            return self.object_pos
        if self.phase == self.PHASE_TO_GOAL:
            return self.goal_pos
        return None

    def is_close_to_object(self) -> bool:
        return self._dist(self.end_effector_pos, self.object_pos) <= self.approach_tolerance

    def is_close_to_goal(self) -> bool:
        return self._dist(self.end_effector_pos, self.goal_pos) <= self.goal_tolerance

    def get_action_space_size(self) -> int:
        return 3 * self.num_dof + 2

    def _joint_expert_towards(self, target: Tuple[float, float]) -> int:
        """
        Greedy: try +/- angle_step for each joint, choose the one that reduces distance most.
        Returns a JOINT action index in [0, 3*DOF).
        """
        # If already close, no-op
        if self._dist(self.end_effector_pos, target) <= min(self.approach_tolerance, self.goal_tolerance) * 0.6:
            return 0

        best_action = 0
        best_distance = self._dist(self.end_effector_pos, target)

        for dof in range(self.num_dof):
            original_angle = self.angles[dof]

            for direction in (1, 2):  # -step, +step
                self.angles[dof] = original_angle + (self.angle_step if direction == 2 else -self.angle_step)
                self.update_positions()
                new_distance = self._dist(self.end_effector_pos, target)

                if new_distance < best_distance:
                    best_distance = new_distance
                    best_action = dof * 3 + direction

            # reset
            self.angles[dof] = original_angle
            self.update_positions()

        return best_action

    def get_expert_action_index(self) -> int:
        """
        Expert policy over the full action space.
        - In movement phases: choose best joint adjustment
        - In grasp/release phases: emit the gripper action once we are close enough
        """
        joint_action_dim = 3 * self.num_dof
        gripper_open_idx = joint_action_dim + 0
        gripper_close_idx = joint_action_dim + 1

        if self.episode_complete:
            return 0

        if self.phase == self.PHASE_APPROACH:
            # Move towards object; when close, transition to grasp
            if self.is_close_to_object():
                self.phase = self.PHASE_GRASP
                return 0
            return self._joint_expert_towards(self.object_pos)

        if self.phase == self.PHASE_GRASP:
            # Ensure we're close; if drifted away, go back to approach
            if not self.is_close_to_object():
                self.phase = self.PHASE_APPROACH
                return 0
            # Close gripper to attach
            if not self.gripper_closed:
                return gripper_close_idx
            # Once closed, if attached, move to goal
            if self.holding_object:
                self.phase = self.PHASE_TO_GOAL
            else:
                # If somehow not attached, try again
                return gripper_close_idx
            return 0

        if self.phase == self.PHASE_TO_GOAL:
            if self.is_close_to_goal():
                self.phase = self.PHASE_RELEASE
                return 0
            return self._joint_expert_towards(self.goal_pos)

        if self.phase == self.PHASE_RELEASE:
            # Open gripper to drop; then finish episode
            if self.gripper_closed:
                return gripper_open_idx
            # Episode ends after open (object stays where released)
            self.episode_complete = True
            return 0

        # Fallback
        return 0

    def take_action_index(self, action_index: int) -> bool:
        """
        Applies action, updates positions, phase transitions, and episode completion.
        Returns True if the episode completed this step.
        """
        joint_action_dim = 3 * self.num_dof
        gripper_open_idx = joint_action_dim + 0
        gripper_close_idx = joint_action_dim + 1

        # Joint actions
        if 0 <= action_index < joint_action_dim:
            joint_index = action_index // 3
            direction = action_index % 3

            if 0 <= joint_index < len(self.angles):
                if direction == 1:
                    self.angles[joint_index] -= self.angle_step
                elif direction == 2:
                    self.angles[joint_index] += self.angle_step

                # keep angles bounded (optional)
                self.angles[joint_index] = ((self.angles[joint_index] + 2 * math.pi) % (4 * math.pi)) - 2 * math.pi

        # Gripper actions
        elif action_index == gripper_open_idx:
            self.gripper_closed = False
            if self.holding_object:
                # Drop: object remains at current location
                self.holding_object = False

        elif action_index == gripper_close_idx:
            self.gripper_closed = True
            # Attach only if close enough
            if self.is_close_to_object():
                self.holding_object = True
                self.object_pos = self.end_effector_pos

        # Update positions and bookkeeping
        self.update_positions()
        self.current_step += 1

        # If released, check if object is sufficiently close to goal (optional "success" condition)
        if self.episode_complete:
            return True

        # Auto-finish if we've released and object ended near goal
        if self.phase == self.PHASE_RELEASE and (not self.gripper_closed) and (not self.holding_object):
            # This is the first step AFTER opening (expert will do this quickly).
            # Mark complete next time get_expert_action_index runs, but we can also mark here.
            self.episode_complete = True
            return True

        return False


class PickPlaceWidget(QWidget):
    """
    Renders the arm + object + goal and provides a clean image capture (no overlay).
    - End effector: blue circle
    - Object: blue box
    - Goal: green X + light green tolerance circle
    - Active target (phase): red X (object or goal) to make it obvious where the expert is heading
    - Gripper: simple two-finger visualization (open vs closed)
    """

    def __init__(self, chain: PickPlaceChain):
        super().__init__()
        self.chain = chain

        self.setFixedSize(420, 420)
        self.center_x = 210
        self.center_y = 210

        self.joint_radius = 8
        self.arm_width = 4
        self.angle_line_length = 12

        self.end_effector_radius = 6

        self.object_size = 16
        self.goal_size = 10

        # gripper visuals
        self.gripper_finger_len = 10
        self.gripper_open_gap = 8
        self.gripper_closed_gap = 2

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(255, 255, 255))

        self._draw_goal(painter)
        self._draw_object(painter)

        self._draw_arms(painter)
        self._draw_joints(painter)
        self._draw_end_effector(painter)
        self._draw_gripper(painter)
        self._draw_overlay_text(painter)

    def _world_to_screen(self, p: Tuple[float, float]) -> Tuple[float, float]:
        return (self.center_x + p[0], self.center_y + p[1])

    def _draw_arms(self, painter: QPainter) -> None:
        pen = QPen(QColor(128, 128, 128), self.arm_width)
        painter.setPen(pen)
        for i in range(len(self.chain.joint_positions) - 1):
            sx, sy = self._world_to_screen(self.chain.joint_positions[i])
            ex, ey = self._world_to_screen(self.chain.joint_positions[i + 1])
            painter.drawLine(int(sx), int(sy), int(ex), int(ey))

    def _draw_joints(self, painter: QPainter) -> None:
        painter.setBrush(QBrush(QColor(0, 0, 0)))
        painter.setPen(QPen(QColor(0, 0, 0), 1))

        for i, (x, y) in enumerate(self.chain.joint_positions[:-1]):
            jx, jy = self._world_to_screen((x, y))

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
        painter.setBrush(QBrush(QColor(0, 0, 0)))
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        ex, ey = self._world_to_screen(self.chain.end_effector_pos)
        painter.drawEllipse(
            int(ex - self.end_effector_radius),
            int(ey - self.end_effector_radius),
            2 * self.end_effector_radius,
            2 * self.end_effector_radius,
        )

    def _draw_gripper(self, painter: QPainter) -> None:
        # simple 2-finger gripper aligned with last link direction
        ex, ey = self._world_to_screen(self.chain.end_effector_pos)
        last_angle = sum(self.chain.angles) if self.chain.num_dof > 0 else 0.0

        # perpendicular direction for finger separation
        px = -math.sin(last_angle)
        py = math.cos(last_angle)

        gap = self.gripper_closed_gap if self.chain.gripper_closed else self.gripper_open_gap

        # Two fingers: start at end-effector, extend forward (along last_angle)
        fx = math.cos(last_angle)
        fy = math.sin(last_angle)

        p1x = ex + px * gap
        p1y = ey + py * gap
        p2x = ex - px * gap
        p2y = ey - py * gap

        q1x = p1x + fx * self.gripper_finger_len
        q1y = p1y + fy * self.gripper_finger_len
        q2x = p2x + fx * self.gripper_finger_len
        q2y = p2y + fy * self.gripper_finger_len

        painter.setPen(QPen(QColor(0, 0, 0), 2))
        painter.drawLine(int(p1x), int(p1y), int(q1x), int(q1y))
        painter.drawLine(int(p2x), int(p2y), int(q2x), int(q2y))

    def _draw_object(self, painter: QPainter) -> None:
        ox, oy = self._world_to_screen(self.chain.object_pos)
        s = self.object_size
        painter.setBrush(QBrush(QColor(30, 90, 255)))
        painter.setPen(QPen(QColor(30, 90, 255), 1))
        painter.drawRect(int(ox - s / 2), int(oy - s / 2), int(s), int(s))

    def _draw_goal(self, painter: QPainter) -> None:
        gx, gy = self._world_to_screen(self.chain.goal_pos)
        painter.setPen(QPen(QColor(0, 160, 0), 3))
        half = self.goal_size // 2
        painter.drawLine(int(gx - half), int(gy - half), int(gx + half), int(gy + half))
        painter.drawLine(int(gx - half), int(gy + half), int(gx + half), int(gy - half))

        # tolerance circle
        pen = QPen(QColor(210, 240, 210), 1)
        pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(QBrush())
        r = self.chain.goal_tolerance
        painter.drawEllipse(int(gx - r), int(gy - r), int(2 * r), int(2 * r))

    def _draw_active_target(self, painter: QPainter) -> None:
        tgt = self.chain._active_target()
        if tgt is None:
            return
        tx, ty = self._world_to_screen(tgt)
        painter.setPen(QPen(QColor(255, 0, 0), 2))
        half = 8
        painter.drawLine(int(tx - half), int(ty - half), int(tx + half), int(ty + half))
        painter.drawLine(int(tx - half), int(ty + half), int(tx + half), int(ty - half))

    def _draw_overlay_text(self, painter: QPainter) -> None:
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        lines = [
            f"Episode: {self.chain.episode_count} | Step: {self.chain.current_step}",
            f"Phase: {self.chain.phase} | Gripper: {'closed' if self.chain.gripper_closed else 'open'}"
            + (" | holding" if self.chain.holding_object else ""),
        ]
        for i, line in enumerate(lines):
            painter.drawText(10, 20 + i * 16, line)

    def capture_image(self) -> np.ndarray:
        """Capture RGB image without overlay text."""
        pixmap = QPixmap(self.width(), self.height())
        pixmap.fill(QColor(255, 255, 255))

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        self._draw_goal(painter)
        self._draw_object(painter)

        self._draw_arms(painter)
        self._draw_joints(painter)
        self._draw_end_effector(painter)
        self._draw_gripper(painter)

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
    chain: PickPlaceChain,
    collected: int,
    start_time: float,
    headless: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if collected == 0:
        raise RuntimeError("No samples collected; refusing to write empty outputs.")

    # CSV
    csv_path = output_dir / "samples.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_filename", "action"])
        writer.writeheader()
        writer.writerows(rows)

    # Metadata
    metadata = {
        "task": "pick_and_place_2d_planar",
        "dof": chain.num_dof,
        "arm_length_px": chain.arm_length,
        "approach_tolerance_px": chain.approach_tolerance,
        "goal_tolerance_px": chain.goal_tolerance,
        "action_dim": chain.get_action_space_size(),
        "action_encoding": {
            "type": "one_hot",
            "length": chain.get_action_space_size(),
            "joint_part": "index = dof_index * 3 + direction; direction: 0=no change, 1=-1 degree, 2=+1 degree",
            "gripper_part": {
                "open": 3 * chain.num_dof + 0,
                "close": 3 * chain.num_dof + 1,
            },
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

    def __init__(self, chain: PickPlaceChain, output_dir: Path, target_samples: int):
        super().__init__()

        self.chain = chain
        self.output_dir = output_dir
        self.images_dir = output_dir / "images"
        self.target_samples = int(target_samples)
        self.collected = 0

        self.images_dir.mkdir(parents=True, exist_ok=True)

        self.setWindowTitle(f"Pick&Place Recorder - {chain.num_dof} DOF")
        self.setFixedSize(440, 520)

        central = QWidget()
        layout = QVBoxLayout()

        self.widget = PickPlaceWidget(chain)
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
        #cv2.imwrite(str(out_path), image)

        self.rows.append(
            {
                "image_filename": rel_path,
                "action": json.dumps(action_onehot),
            }
        )
        self.collected += 1

        done = self.chain.take_action_index(action_index)
        if done:
            self.chain.generate_new_episode()

        self.status.setText(
            f"Collected: {self.collected}/{self.target_samples} | Episode: {self.chain.episode_count} | Phase: {self.chain.phase}"
        )
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

    def __init__(self, chain: PickPlaceChain, output_dir: Path, target_samples: int):
        self.chain = chain
        self.output_dir = output_dir
        self.images_dir = output_dir / "images"
        self.target_samples = int(target_samples)

        self.images_dir.mkdir(parents=True, exist_ok=True)

        # Offscreen widget (never shown); used for consistent rendering.
        self.widget = PickPlaceWidget(chain)

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
                self.chain.generate_new_episode()

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
    p = argparse.ArgumentParser(description="Pick-and-place simulator (record images + expert actions).")
    p.add_argument("--dof", type=int, required=True, help="Number of degrees of freedom (>=1).")
    p.add_argument("--samples", type=int, required=True, help="Number of samples to record (>=1).")
    p.add_argument("--output", type=str, required=True, help="Output directory (no extra subfolder).")
    p.add_argument("--headless", action="store_true", help="Run without showing a window (faster data collection).")
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

    print("=== Pick-and-Place Simulation Data Recorder ===")
    print(f"DOF: {args.dof}")
    print(f"Samples: {args.samples}")
    print(f"Headless: {args.headless}")
    print(f"Output: {out_dir.resolve()}")

    chain = PickPlaceChain(num_dof=args.dof)

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
