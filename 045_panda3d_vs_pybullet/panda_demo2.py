from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import AmbientLight, DirectionalLight, Vec3, Vec4, Quat
import pybullet as p
import pybullet_data
import math


class RobotArmDemo(ShowBase):
    def __init__(self):
        super().__init__()
        self.disableMouse()

        self.camera.setPos(3, -7, 4)
        self.camera.lookAt(0, 0, 1)

        self.setup_light()
        self.setup_pybullet()
        self.create_visual_arm()

        self.taskMgr.add(self.update, "update")

    def setup_light(self):
        ambient = AmbientLight("ambient")
        ambient.setColor(Vec4(0.4, 0.4, 0.4, 1))
        self.render.setLight(self.render.attachNewNode(ambient))

        sun = DirectionalLight("sun")
        sun.setDirection(Vec3(-2, -4, -5))
        sun.setColor(Vec4(0.8, 0.8, 0.8, 1))
        self.render.setLight(self.render.attachNewNode(sun))

    def setup_pybullet(self):
        p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        self.robot_id = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=[0, 0, 0],
            useFixedBase=True
        )

        self.num_joints = p.getNumJoints(self.robot_id)

    def make_box(self, name, scale, color):
        box = self.loader.loadModel("models/box")
        box.setName(name)
        box.setScale(*scale)
        box.setColor(*color)
        box.reparentTo(self.render)
        return box

    def create_visual_arm(self):
        self.links = []

        for i in range(self.num_joints):
            visual = self.make_box(
                f"link_{i}",
                scale=(0.05, 0.05, 0.15),
                color=(0.2, 0.6, 1.0, 1)
            )
            self.links.append(visual)

        floor = self.loader.loadModel("models/box")
        floor.setScale(4, 4, 0.03)
        floor.setPos(0, 0, -0.03)
        floor.setColor(0.4, 0.4, 0.4, 1)
        floor.reparentTo(self.render)

    def update(self, task):
        t = task.time

        # Erste 7 Gelenke des Franka-Panda-Arms bewegen
        for joint_index in range(min(7, self.num_joints)):
            target_angle = math.sin(t + joint_index * 0.5) * 0.7

            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_angle,
                force=80
            )

        p.stepSimulation()

        # PyBullet-Positionen nach Panda3D übertragen
        for i, visual in enumerate(self.links):
            state = p.getLinkState(self.robot_id, i)

            if state is None:
                continue

            pos = state[0]
            orn = state[1]  # PyBullet: x, y, z, w

            visual.setPos(pos[0], pos[1], pos[2])

            # Panda3D Quat: w, x, y, z
            quat = Quat(orn[3], orn[0], orn[1], orn[2])
            visual.setQuat(quat)

        return Task.cont


app = RobotArmDemo()
app.run()