"""line_follower_pr2 controller.

Phase 1: manual base teleoperation via keyboard.
Phase 2 (next): line following can reuse the same wheel motor setup.
"""

from controller import Keyboard, Robot

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

camera_display_enabled = cv2 is not None and np is not None

print("PR2 Line Follower Controller - Version 7")


MAX_WHEEL_SPEED = 3.0
FORWARD_SPEED = 1.5
TURN_SPEED = 0.8
HEAD_TILT_STEP = 0.1
HEAD_TILT_DEFAULT = 1.1
HEAD_TILT_FALLBACK_MIN = -0.5
HEAD_TILT_FALLBACK_MAX = 1.2
CAMERA_DISPLAY_SCALE = 0.25
LINE_FOLLOW_SPEED = 1.0
LINE_FOLLOW_SIDE_SPEED = 0.8
LINE_FOLLOW_ROTATE_SPEED = 0.7
LINE_FOLLOW_ROI_TOP = 0.45
LINE_FOLLOW_ROI_BOTTOM = 0.95
LINE_FOLLOW_WHITE_THRESHOLD = 215
LINE_FOLLOW_MIN_MASK_AREA = 30
LINE_FOLLOW_CENTER_DEADBAND = 0.15
LINE_FOLLOW_SLOPE_DEADBAND = 0.20
LINE_FOLLOW_ROTATE_STEP_DEGREES = 2.0
ACTION_FORWARD = "a1: forward"
ACTION_MOVE_LEFT = "a2: move left"
ACTION_MOVE_RIGHT = "a3: move right"
ACTION_ROTATE_LEFT = f"a4: rotate left {LINE_FOLLOW_ROTATE_STEP_DEGREES:g} deg"
ACTION_ROTATE_RIGHT = f"a5: rotate right {LINE_FOLLOW_ROTATE_STEP_DEGREES:g} deg"
ACTION_LINE_LOST = "STOP: line lost"
ACTION_MANUAL = "MANUAL"

PR2_ARM_UP_POSE = {
    "l_shoulder_pan_joint": 1.2,
    "l_shoulder_lift_joint": 1.25,
    "l_upper_arm_roll_joint": 1.57,
    "l_elbow_flex_joint": 0.0,
    "l_forearm_roll_joint": 0.0,
    "l_wrist_flex_joint": 0.0,
    "l_wrist_roll_joint": 0.0,
    "r_shoulder_pan_joint": -1.2,
    "r_shoulder_lift_joint": 1.25,
    "r_upper_arm_roll_joint": -1.57,
    "r_elbow_flex_joint": 0.0,
    "r_forearm_roll_joint": 0.0,
    "r_wrist_flex_joint": 0.0,
    "r_wrist_roll_joint": 0.0,
}

PR2_CAMERA_NAMES = [
    "high_def_sensor"
    #"l_forearm_cam_sensor",
    #"r_forearm_cam_sensor",
    #"wide_stereo_l_stereo_camera_sensor",
    #"wide_stereo_r_stereo_camera_sensor",
]

PR2_CASTER_NAMES = ["fl", "fr", "bl", "br"]
PR2_FORWARD_CASTER_ANGLES = {name: 0.0 for name in PR2_CASTER_NAMES}
PR2_SIDE_CASTER_ANGLES = {name: 1.5707963268 for name in PR2_CASTER_NAMES}
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


def detect_pr2_cameras(robot, timestep):
    cameras = []
    for name in PR2_CAMERA_NAMES:
        try:
            camera = robot.getDevice(name)
            camera.enable(timestep)
            cameras.append(camera)
        except BaseException:
            pass
    return cameras


def detect_head_tilt_motor(robot):
    try:
        return robot.getDevice("head_tilt_joint")
    except BaseException:
        return None


def detect_pr2_arm_motors(robot):
    motors = {}
    for name in PR2_ARM_UP_POSE:
        try:
            motors[name] = robot.getDevice(name)
        except BaseException:
            pass
    return motors


def get_motor_position_limits(motor, fallback_min, fallback_max):
    try:
        min_position = motor.getMinPosition()
        max_position = motor.getMaxPosition()
        if min_position < max_position:
            return min_position, max_position
    except BaseException:
        pass

    return fallback_min, fallback_max


def show_camera_images(cameras, line_follow_enabled):
    global camera_display_enabled

    if not camera_display_enabled:
        return None, False, "STOP: no camera display"

    line_error = None
    line_visible = False
    selected_action = ACTION_LINE_LOST

    try:
        for camera in cameras:
            image = camera.getImage()
            if image is None:
                continue

            width = camera.getWidth()
            height = camera.getHeight()
            frame = np.frombuffer(image, dtype=np.uint8).reshape((height, width, 4))
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            frame_bgr_small = cv2.resize(
                frame_bgr,
                (
                    max(1, int(width * CAMERA_DISPLAY_SCALE)),
                    max(1, int(height * CAMERA_DISPLAY_SCALE)),
                ),
                interpolation=cv2.INTER_AREA,
            )
            detection = detect_white_line(frame_bgr_small)
            if line_follow_enabled:
                action = select_line_follow_action(detection)
            else:
                action = ACTION_MANUAL
            draw_line_detection(frame_bgr_small, detection, action)
            if detection["visible"] and line_error is None:
                line_error = detection["error"]
                line_visible = True
                selected_action = action

            cv2.imshow(camera.getName(), frame_bgr_small)

        cv2.waitKey(1)
        return line_error, line_visible, selected_action
    except BaseException as exc:
        camera_display_enabled = False
        print(f"[WARNING] OpenCV camera display disabled: {exc}")
        return None, False, "STOP: display error"


def detect_white_line(frame_bgr):
    height, width = frame_bgr.shape[:2]
    roi_top = int(height * LINE_FOLLOW_ROI_TOP)
    roi_bottom = int(height * LINE_FOLLOW_ROI_BOTTOM)
    roi = frame_bgr[roi_top:roi_bottom, :]

    if roi.size == 0:
        return {"visible": False, "error": 0.0, "roi": (0, 0, width, height), "mask": None}

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    mask = np.where(gray >= LINE_FOLLOW_WHITE_THRESHOLD, 255, 0).astype(np.uint8)

    moments = cv2.moments(mask)
    if moments["m00"] < LINE_FOLLOW_MIN_MASK_AREA:
        return {
            "visible": False,
            "error": 0.0,
            "roi": (0, roi_top, width, roi_bottom),
            "mask": mask,
        }

    center_x = int(moments["m10"] / moments["m00"])
    center_y = roi_top + int(moments["m01"] / moments["m00"])
    error = (center_x - (width / 2.0)) / (width / 2.0)
    ys, xs = np.nonzero(mask)
    slope = 0.0
    regression_line = None
    if len(xs) >= 2 and np.ptp(ys) > 0:
        slope, intercept = np.polyfit(ys, xs, 1)
        x_top = int(intercept)
        x_bottom = int(slope * (roi.shape[0] - 1) + intercept)
        regression_line = (
            (clamp(x_top, 0, width - 1), roi_top),
            (clamp(x_bottom, 0, width - 1), roi_bottom - 1),
        )

    return {
        "visible": True,
        "error": clamp(error, -1.0, 1.0),
        "center": (center_x, center_y),
        "roi": (0, roi_top, width, roi_bottom),
        "mask": mask,
        "slope": slope,
        "regression_line": regression_line,
    }


def select_line_follow_action(detection):
    if not detection["visible"]:
        return ACTION_LINE_LOST

    slope = detection.get("slope", 0.0)
    if slope < -LINE_FOLLOW_SLOPE_DEADBAND:
        return ACTION_ROTATE_LEFT
    if slope > LINE_FOLLOW_SLOPE_DEADBAND:
        return ACTION_ROTATE_RIGHT

    error = detection["error"]
    if error > LINE_FOLLOW_CENTER_DEADBAND:
        return ACTION_MOVE_RIGHT
    if error < -LINE_FOLLOW_CENTER_DEADBAND:
        return ACTION_MOVE_LEFT

    return ACTION_FORWARD


def draw_line_detection(frame_bgr, detection, action):
    left, top, right, bottom = detection["roi"]
    mask = detection.get("mask")
    if mask is not None:
        roi = frame_bgr[top:bottom, left:right]
        green_overlay = np.zeros_like(roi)
        green_overlay[:, :] = (0, 255, 0)
        mask_bool = mask > 0
        roi[mask_bool] = cv2.addWeighted(roi, 0.35, green_overlay, 0.65, 0)[mask_bool]

    color = (0, 255, 0) if detection["visible"] else (0, 0, 255)
    cv2.rectangle(frame_bgr, (left, top), (right - 1, bottom - 1), color, 1)
    cv2.line(
        frame_bgr,
        (frame_bgr.shape[1] // 2, top),
        (frame_bgr.shape[1] // 2, bottom - 1),
        (255, 0, 0),
        1,
    )
    if detection["visible"]:
        cv2.circle(frame_bgr, detection["center"], 4, (0, 0, 255), -1)
        regression_line = detection.get("regression_line")
        if regression_line is not None:
            cv2.line(frame_bgr, regression_line[0], regression_line[1], (0, 255, 255), 2)

    cv2.putText(
        frame_bgr,
        action,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame_bgr,
        action,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


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


def execute_line_follow_action(action, caster_modules, left_motors, right_motors):
    left_speed = 0.0
    right_speed = 0.0

    if caster_modules:
        if action == ACTION_FORWARD:
            set_pr2_caster_angles(caster_modules, PR2_FORWARD_CASTER_ANGLES)
            set_pr2_wheel_speed(caster_modules, LINE_FOLLOW_SPEED)
        elif action == ACTION_MOVE_LEFT:
            set_pr2_caster_angles(caster_modules, PR2_SIDE_CASTER_ANGLES)
            set_pr2_wheel_speed(caster_modules, LINE_FOLLOW_SIDE_SPEED)
        elif action == ACTION_MOVE_RIGHT:
            set_pr2_caster_angles(caster_modules, PR2_SIDE_CASTER_ANGLES)
            set_pr2_wheel_speed(caster_modules, -LINE_FOLLOW_SIDE_SPEED)
        elif action == ACTION_ROTATE_LEFT:
            set_pr2_caster_angles(caster_modules, PR2_TURN_CASTER_ANGLES)
            set_pr2_wheel_speed(caster_modules, LINE_FOLLOW_ROTATE_SPEED)
        elif action == ACTION_ROTATE_RIGHT:
            set_pr2_caster_angles(caster_modules, PR2_TURN_CASTER_ANGLES)
            set_pr2_wheel_speed(caster_modules, -LINE_FOLLOW_ROTATE_SPEED)
        else:
            set_pr2_wheel_speed(caster_modules, 0.0)
        return

    if action == ACTION_FORWARD:
        left_speed = LINE_FOLLOW_SPEED
        right_speed = LINE_FOLLOW_SPEED
    elif action == ACTION_ROTATE_LEFT:
        left_speed = -LINE_FOLLOW_ROTATE_SPEED
        right_speed = LINE_FOLLOW_ROTATE_SPEED
    elif action == ACTION_ROTATE_RIGHT:
        left_speed = LINE_FOLLOW_ROTATE_SPEED
        right_speed = -LINE_FOLLOW_ROTATE_SPEED

    set_side_speed(left_motors, left_speed)
    set_side_speed(right_motors, right_speed)


def set_pr2_arm_pose(arm_motors, pose):
    configured = []
    for name, target_position in pose.items():
        motor = arm_motors.get(name)
        if motor is None:
            continue

        min_position, max_position = get_motor_position_limits(
            motor,
            target_position,
            target_position,
        )
        motor.setPosition(clamp(target_position, min_position, max_position))
        configured.append(name)

    return configured


robot = Robot()
timestep = int(robot.getBasicTimeStep())
print("timestep: ", timestep)

pr2_caster_modules = detect_pr2_caster_modules(robot)
left_motors, right_motors = detect_drive_motors(robot)
pr2_cameras = detect_pr2_cameras(robot, timestep)
head_tilt_motor = detect_head_tilt_motor(robot)
pr2_arm_motors = detect_pr2_arm_motors(robot)
head_tilt_min, head_tilt_max = HEAD_TILT_FALLBACK_MIN, HEAD_TILT_FALLBACK_MAX
target_head_tilt = HEAD_TILT_DEFAULT

if head_tilt_motor is not None:
    head_tilt_min, head_tilt_max = get_motor_position_limits(
        head_tilt_motor,
        HEAD_TILT_FALLBACK_MIN,
        HEAD_TILT_FALLBACK_MAX,
    )
    target_head_tilt = clamp(HEAD_TILT_DEFAULT, head_tilt_min, head_tilt_max)
    head_tilt_motor.setPosition(target_head_tilt)

configured_arm_motors = set_pr2_arm_pose(pr2_arm_motors, PR2_ARM_UP_POSE)

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
print(
    "[INFO] Controls: F=Toggle line following, W/S/A/D or Arrow Keys=Manual drive, "
    "P=Head up, L=Head down, Space=Stop, Q=Quit"
)
if pr2_caster_modules:
    print("[INFO] PR2 caster steering motors:")
    for module in pr2_caster_modules:
        print(f"  - {module['rotation_motor'].getName()}")
if head_tilt_motor is not None:
    print(
        f"[INFO] Head tilt motor: {head_tilt_motor.getName()} "
        f"({head_tilt_min:.2f} to {head_tilt_max:.2f} rad)"
    )
else:
    print("[INFO] Head tilt motor 'head_tilt_joint' not found.")
if configured_arm_motors:
    print("[INFO] PR2 arms moved upward to clear the head camera view:")
    for name in configured_arm_motors:
        print(f"  - {name}: {PR2_ARM_UP_POSE[name]:.2f} rad")
else:
    print("[INFO] No PR2 arm motors found for the upward camera-clearance pose.")
print("[INFO] Robot-left motors:")
for m in left_motors:
    print(f"  - {m.getName()}")
print("[INFO] Robot-right motors:")
for m in right_motors:
    print(f"  - {m.getName()}")
if pr2_cameras:
    print("[INFO] PR2 cameras enabled:")
    for camera in pr2_cameras:
        print(f"  - {camera.getName()} ({camera.getWidth()}x{camera.getHeight()})")
    if cv2 is None or np is None:
        print("[INFO] Install opencv-python and numpy to show camera images in OpenCV windows.")
else:
    print("[INFO] No PR2 cameras found with the configured names.")

running = True
line_follow_enabled = True
target_forward = 0.0
target_turn = 0.0
while running and robot.step(timestep) != -1:
    line_error, line_visible, line_action = show_camera_images(pr2_cameras, line_follow_enabled)

    stop_requested = False

    key = keyboard.getKey()
    while key != -1:
        if key in (Keyboard.UP, ord("W"), ord("w")):
            line_follow_enabled = False
            target_forward = FORWARD_SPEED
            target_turn = 0.0
        elif key in (Keyboard.DOWN, ord("S"), ord("s")):
            line_follow_enabled = False
            target_forward = -FORWARD_SPEED
            target_turn = 0.0
        elif key in (Keyboard.LEFT, ord("A"), ord("a")):
            line_follow_enabled = False
            target_forward = 0.0
            target_turn = TURN_SPEED
        elif key in (Keyboard.RIGHT, ord("D"), ord("d")):
            line_follow_enabled = False
            target_forward = 0.0
            target_turn = -TURN_SPEED
        elif key in (ord("F"), ord("f")):
            line_follow_enabled = not line_follow_enabled
            target_forward = 0.0
            target_turn = 0.0
            print(f"[INFO] Line following enabled: {line_follow_enabled}")
        elif key in (ord("P"), ord("p")) and head_tilt_motor is not None:
            target_head_tilt = clamp(
                target_head_tilt + HEAD_TILT_STEP,
                head_tilt_min,
                head_tilt_max,
            )
            head_tilt_motor.setPosition(target_head_tilt)
            print(f"{target_head_tilt=}")
        elif key in (ord("L"), ord("l")) and head_tilt_motor is not None:
            target_head_tilt = clamp(
                target_head_tilt - HEAD_TILT_STEP,
                head_tilt_min,
                head_tilt_max,
            )
            head_tilt_motor.setPosition(target_head_tilt)
            print(f"{target_head_tilt=}")
        elif key == ord(" "):
            line_follow_enabled = False
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
    elif line_follow_enabled:
        execute_line_follow_action(
            line_action if line_visible else ACTION_LINE_LOST,
            pr2_caster_modules,
            left_motors,
            right_motors,
        )
        continue
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


print("Controller stopped!")

# Ensure robot is stopped when leaving the controller loop.
if pr2_caster_modules:
    set_pr2_wheel_speed(pr2_caster_modules, 0.0)
else:
    set_side_speed(left_motors, 0.0)
    set_side_speed(right_motors, 0.0)

if cv2 is not None:
    cv2.destroyAllWindows()
