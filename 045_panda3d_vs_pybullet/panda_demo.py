from direct.showbase.ShowBase import ShowBase
from direct.task import Task

class MyApp(ShowBase):
    def __init__(self):
        super().__init__()

        # Standardmodell laden
        self.model = self.loader.loadModel("models/panda")

        # Modell an Szene anhängen
        self.model.reparentTo(self.render)

        # Position und Skalierung
        self.model.setScale(0.5)
        self.model.setPos(0, 10, 0)

        # Kamera etwas zurücksetzen
        self.camera.setPos(0, -30, 6)

        # Rotations-Task registrieren
        self.taskMgr.add(self.rotate_model, "rotate_model")

    def rotate_model(self, task):
        angle = task.time * 50
        self.model.setH(angle)
        return Task.cont

app = MyApp()
app.run()