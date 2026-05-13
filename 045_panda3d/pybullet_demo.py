import pybullet as p
import pybullet_data
import time
import math


p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.resetSimulation()
p.setGravity(0, 0, -9.81)

p.loadURDF("plane.urdf")

robot_id = p.loadURDF(
    "franka_panda/panda.urdf",
    basePosition=[0, 0, 0],
    useFixedBase=True
)

p.resetDebugVisualizerCamera(
    cameraDistance=2.2,
    cameraYaw=45,
    cameraPitch=-30,
    cameraTargetPosition=[0, 0, 0.6],
)

num_joints = p.getNumJoints(robot_id)

while True:
    t = time.time()

    for joint_index in range(min(7, num_joints)):
        target_angle = math.sin(t + joint_index * 0.5) * 0.6

        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=joint_index,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target_angle,
            force=80,
        )

    p.stepSimulation()
    time.sleep(1 / 240)