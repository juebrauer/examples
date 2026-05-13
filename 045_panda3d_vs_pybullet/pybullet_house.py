import pybullet as p
import pybullet_data
import time
import random
import math


def create_box(size, position, color, mass=0, name=""):
    half_extents = [s / 2 for s in size]

    collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=half_extents,
    )

    visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=half_extents,
        rgbaColor=color,
    )

    body_id = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=position,
    )

    return body_id


def create_house():
    wall_height = 1.5
    wall_thickness = 0.12

    floor_id = create_box(
        size=[12, 10, 0.05],
        position=[0, 0, -0.025],
        color=[0.72, 0.72, 0.72, 1],
        name="floor",
    )

    # Außenwände
    create_box([12, wall_thickness, wall_height], [0, -5, wall_height / 2], [0.92, 0.92, 0.92, 1])
    create_box([12, wall_thickness, wall_height], [0, 5, wall_height / 2], [0.92, 0.92, 0.92, 1])
    create_box([wall_thickness, 10, wall_height], [-6, 0, wall_height / 2], [0.92, 0.92, 0.92, 1])
    create_box([wall_thickness, 10, wall_height], [6, 0, wall_height / 2], [0.92, 0.92, 0.92, 1])

    # Zentraler Gang: x von -1 bis 1 bleibt frei
    # Linke/rechte Zimmerbereiche
    # Vertikale Trennwände zwischen Gang und Zimmern mit Türöffnungen
    for x in [-1.0, 1.0]:
        # unten
        create_box([wall_thickness, 2.0, wall_height], [x, -4.0, wall_height / 2], [0.75, 0.82, 1.0, 1])
        # mitte unten
        create_box([wall_thickness, 1.2, wall_height], [x, -1.4, wall_height / 2], [0.75, 0.82, 1.0, 1])
        # mitte oben
        create_box([wall_thickness, 1.2, wall_height], [x, 1.4, wall_height / 2], [0.75, 0.82, 1.0, 1])
        # oben
        create_box([wall_thickness, 2.0, wall_height], [x, 4.0, wall_height / 2], [0.75, 0.82, 1.0, 1])

    # Horizontale Trennwände: 3 Zimmer links, 3 Zimmer rechts
    for y in [-1.7, 1.7]:
        create_box([5.0, wall_thickness, wall_height], [-3.5, y, wall_height / 2], [0.8, 1.0, 0.8, 1])
        create_box([5.0, wall_thickness, wall_height], [3.5, y, wall_height / 2], [0.8, 1.0, 0.8, 1])

    # Möbel links unten
    create_box([1.3, 0.6, 0.45], [-4.2, -3.8, 0.225], [0.45, 0.25, 0.12, 1])
    create_box([0.7, 0.7, 0.7], [-2.5, -4.0, 0.35], [0.2, 0.45, 0.8, 1])
    create_box([0.9, 0.4, 0.45], [-5.0, -2.4, 0.225], [0.6, 0.3, 0.2, 1])

    # Möbel links mitte
    create_box([1.2, 0.8, 0.35], [-4.5, 0.0, 0.175], [0.25, 0.5, 0.25, 1])
    create_box([0.5, 0.5, 0.9], [-2.4, -0.7, 0.45], [0.7, 0.7, 0.25, 1])
    create_box([1.0, 0.35, 0.5], [-3.0, 0.9, 0.25], [0.5, 0.25, 0.15, 1])

    # Möbel links oben
    create_box([1.5, 0.7, 0.35], [-4.1, 3.9, 0.175], [0.55, 0.35, 0.2, 1])
    create_box([0.6, 1.0, 0.45], [-2.4, 2.6, 0.225], [0.8, 0.35, 0.35, 1])

    # Möbel rechts unten
    create_box([1.2, 0.6, 0.5], [4.3, -3.7, 0.25], [0.5, 0.25, 0.1, 1])
    create_box([0.8, 0.8, 0.8], [2.6, -3.4, 0.4], [0.2, 0.55, 0.8, 1])
    create_box([0.8, 0.35, 0.55], [5.1, -2.3, 0.275], [0.6, 0.25, 0.25, 1])

    # Möbel rechts mitte
    create_box([1.4, 0.8, 0.35], [4.2, 0.1, 0.175], [0.35, 0.5, 0.25, 1])
    create_box([0.5, 0.5, 0.9], [2.5, 0.8, 0.45], [0.75, 0.75, 0.25, 1])
    create_box([0.9, 0.4, 0.45], [5.0, -0.9, 0.225], [0.45, 0.25, 0.15, 1])

    # Möbel rechts oben
    create_box([1.5, 0.8, 0.35], [4.0, 3.7, 0.175], [0.55, 0.3, 0.2, 1])
    create_box([0.7, 1.0, 0.45], [2.5, 2.6, 0.225], [0.8, 0.4, 0.4, 1])
    create_box([0.5, 0.5, 1.0], [5.1, 4.1, 0.5], [0.25, 0.6, 0.6, 1])

    return floor_id


def create_robot(position):
    radius = 0.22
    height = 0.25

    collision = p.createCollisionShape(
        p.GEOM_CYLINDER,
        radius=radius,
        height=height,
    )

    visual = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=radius,
        length=height,
        rgbaColor=[1.0, 0.25, 0.1, 1],
    )

    robot_id = p.createMultiBody(
        baseMass=2.0,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=position,
    )

    p.changeDynamics(
        robot_id,
        -1,
        lateralFriction=1.0,
        linearDamping=0.2,
        angularDamping=0.9,
    )

    return robot_id


def get_yaw(body_id):
    _, orn = p.getBasePositionAndOrientation(body_id)
    _, _, yaw = p.getEulerFromQuaternion(orn)
    return yaw


def set_yaw(body_id, yaw):
    pos, _ = p.getBasePositionAndOrientation(body_id)
    orn = p.getQuaternionFromEuler([0, 0, yaw])
    p.resetBasePositionAndOrientation(body_id, pos, orn)


def drive_forward(robot_id, speed):
    yaw = get_yaw(robot_id)

    vx = math.cos(yaw) * speed
    vy = math.sin(yaw) * speed

    p.resetBaseVelocity(
        robot_id,
        linearVelocity=[vx, vy, 0],
        angularVelocity=[0, 0, 0],
    )


def has_collision(robot_id, floor_id):
    contacts = p.getContactPoints(bodyA=robot_id)

    for contact in contacts:
        body_a = contact[1]
        body_b = contact[2]

        other_body = body_b if body_a == robot_id else body_a

        if other_body != floor_id:
            return True

    return False


def back_up(robot_id, distance=0.25):
    pos, orn = p.getBasePositionAndOrientation(robot_id)
    yaw = get_yaw(robot_id)

    new_pos = [
        pos[0] - math.cos(yaw) * distance,
        pos[1] - math.sin(yaw) * distance,
        pos[2],
    ]

    p.resetBasePositionAndOrientation(robot_id, new_pos, orn)
    p.resetBaseVelocity(robot_id, [0, 0, 0], [0, 0, 0])


def main():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1 / 240)

    floor_id = create_house()

    robot_id = create_robot([0, -4.2, 0.2])
    set_yaw(robot_id, math.pi / 2)

    p.resetDebugVisualizerCamera(
        cameraDistance=11,
        cameraYaw=0,
        cameraPitch=-65,
        cameraTargetPosition=[0, 0, 0],
    )

    speed = 1.0
    turn_steps = 0
    turn_direction = 1

    while True:
        if turn_steps > 0:
            yaw = get_yaw(robot_id)
            set_yaw(robot_id, yaw + turn_direction * 0.08)
            p.resetBaseVelocity(robot_id, [0, 0, 0], [0, 0, 0])
            turn_steps -= 1

        else:
            drive_forward(robot_id, speed)

            if has_collision(robot_id, floor_id):
                back_up(robot_id, distance=0.3)

                turn_direction = random.choice([-1, 1])
                turn_steps = random.randint(25, 70)

        p.stepSimulation()
        time.sleep(1 / 240)


if __name__ == "__main__":
    main()