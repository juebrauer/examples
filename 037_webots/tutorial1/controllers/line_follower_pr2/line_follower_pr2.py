"""line_follower_pr2 controller.

Phase 1: manual base teleoperation via keyboard.
Phase 2 (next): line following can reuse the same wheel motor setup.
"""

from controller import Keyboard, Robot

print("PR2 Line Follower Controller - Version 5")


MAX_WHEEL_SPEED = 3.0
FORWARD_SPEED = 1.5
TURN_SPEED = 0.8

PR2_CASTER_NAMES = ["fl", "fr", "bl", "br"]
PR2_FORWARD_CASTER_ANGLES = {name: 0.0 for name in PR2_CASTER_NAMES}
PR2_TURN_CASTER_ANGLES = {
    "fl": 2.3561944902,   # 135 degrees
    "fr": 0.7853981634,   # 45 degrees
    "bl": -2.3561944902,  # -135 degrees
    "br": -0.7853981634,  # -45 degrees
}


def detect_pr2_caster_modules(robot):
    modules = []
    try:
        for name in PR2_CASTER_NAMES:
            modules.append(
                {
                    "name": name,
                    "rotation_motor": robot.getDevice(f"{name}_caster_rotation_joint"),
                    "wheel_motors": [
                        robot.getDevice(f"{name}_caster_l_wheel_joint"),
                        robot.getDevice(f"{name}_caster_r_wheel_joint"),
                    ],
                }
            )
        return modules
    except BaseException:
        return []


def detect_drive_motors(robot):
    """Detect robot-left/robot-right drive motors from device names.

    In PR2 caster names, l/r describes the wheel inside one caster module.
    For turning, the controller needs robot-side groups instead.
    """
    left = []
    right = []

    pr2_left_names = [
        "fl_caster_l_wheel_joint",
        "fl_caster_r_wheel_joint",
        "bl_caster_l_wheel_joint",
        "bl_caster_r_wheel_joint",
    ]
    pr2_right_names = [
        "fr_caster_l_wheel_joint",
        "fr_caster_r_wheel_joint",
        "br_caster_l_wheel_joint",
        "br_caster_r_wheel_joint",
    ]

    # First, try the exact PR2 wheel joint names.
    try:
        left = [robot.getDevice(n) for n in pr2_left_names]
        right = [robot.getDevice(n) for n in pr2_right_names]
        return left, right
    except BaseException:
        left = []
        right = []

    # Generic fallback for other robots.
    for i in range(robot.getNumberOfDevices()):
        device = robot.getDeviceByIndex(i)
        name = device.getName().lower()
        if "wheel" not in name:
            continue

        if "_l_" in name or "left" in name:
            left.append(device)
        elif "_r_" in name or "right" in name:
            right.append(device)

    # Fallback for classic two-wheel names.
    if not left and not right:
        try:
            left.append(robot.getDevice("left wheel motor"))
            right.append(robot.getDevice("right wheel motor"))
        except BaseException:
            pass

    return left, right


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def set_side_speed(motors, speed):
    speed = clamp(speed, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED)
    for m in motors:
        # print(f"Motor {m.getName()}: {speed}")
        m.setVelocity(speed)


def set_pr2_caster_angles(caster_modules, angles):
    for module in caster_modules:
        module["rotation_motor"].setPosition(angles[module["name"]])


def set_pr2_wheel_speed(caster_modules, speed):
    speed = clamp(speed, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED)
    for module in caster_modules:
        for motor in module["wheel_motors"]:
            motor.setVelocity(speed)


robot = Robot()
timestep = int(robot.getBasicTimeStep())

pr2_caster_modules = detect_pr2_caster_modules(robot)
left_motors, right_motors = detect_drive_motors(robot)

if not left_motors or not right_motors:
    print("[ERROR] No left/right wheel motor groups found.")
    print("Available devices:")
    for i in range(robot.getNumberOfDevices()):
        d = robot.getDeviceByIndex(i)
        print(f"  - {d.getName()}")
    raise RuntimeError("Cannot control base without wheel motors.")

if pr2_caster_modules:
    set_pr2_caster_angles(pr2_caster_modules, PR2_FORWARD_CASTER_ANGLES)
    for module in pr2_caster_modules:
        for motor in module["wheel_motors"]:
            motor.setPosition(float("inf"))
    set_pr2_wheel_speed(pr2_caster_modules, 0.0)
else:
    for motor in left_motors + right_motors:
        motor.setPosition(float("inf"))
        motor.setVelocity(0.0)

keyboard = Keyboard()
keyboard.enable(timestep)

print("[INFO] PR2 manual drive ready.")
print("[INFO] Controls: W/S/A/D or Arrow Keys, Space=Stop, Q=Quit")
if pr2_caster_modules:
    print("[INFO] PR2 caster steering motors:")
    for module in pr2_caster_modules:
        print(f"  - {module['rotation_motor'].getName()}")
print("[INFO] Robot-left motors:")
for m in left_motors:
    print(f"  - {m.getName()}")
print("[INFO] Robot-right motors:")
for m in right_motors:
    print(f"  - {m.getName()}")

running = True
target_forward = 0.0
target_turn = 0.0
while running and robot.step(timestep) != -1:
    stop_requested = False

    key = keyboard.getKey()
    while key != -1:
        if key in (Keyboard.UP, ord("W"), ord("w")):
            target_forward = FORWARD_SPEED
            target_turn = 0.0
        elif key in (Keyboard.DOWN, ord("S"), ord("s")):
            target_forward = -FORWARD_SPEED
            target_turn = 0.0
        elif key in (Keyboard.LEFT, ord("A"), ord("a")):
            target_forward = 0.0
            target_turn = TURN_SPEED
        elif key in (Keyboard.RIGHT, ord("D"), ord("d")):
            target_forward = 0.0
            target_turn = -TURN_SPEED
        elif key == ord(" "):
            stop_requested = True
        elif key in (ord("Q"), ord("q"), 27):  # Q/q or ESC
            running = False
            stop_requested = True
        key = keyboard.getKey()

    if stop_requested:
        target_forward = 0.0
        target_turn = 0.0
        left_speed = 0.0
        right_speed = 0.0
    elif pr2_caster_modules and target_turn != 0.0 and target_forward == 0.0:
        set_pr2_caster_angles(pr2_caster_modules, PR2_TURN_CASTER_ANGLES)
        set_pr2_wheel_speed(pr2_caster_modules, target_turn)
        continue
    elif pr2_caster_modules:
        set_pr2_caster_angles(pr2_caster_modules, PR2_FORWARD_CASTER_ANGLES)
        set_pr2_wheel_speed(pr2_caster_modules, target_forward)
        continue
    else:
        # Differential-style command mixing.
        left_speed = target_forward - target_turn
        right_speed = target_forward + target_turn

    set_side_speed(left_motors, left_speed)
    set_side_speed(right_motors, right_speed)

# Ensure robot is stopped when leaving the controller loop.
if pr2_caster_modules:
    set_pr2_wheel_speed(pr2_caster_modules, 0.0)
else:
    set_side_speed(left_motors, 0.0)
    set_side_speed(right_motors, 0.0)
