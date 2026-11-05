"""Optionales PySide6/OpenGL-Fenster ohne eigene Simulationsschleife.

Bewusst OpenGL 2.1 (Desktop): feste Beleuchtung statt zusätzlicher Shader-Dateien.
Die Physik importiert dieses Modul nicht und funktioniert auch ohne Qt.
"""

import math
import numpy as np
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QApplication
from OpenGL import GL


def _sphere_mesh(rings=12, sectors=24):
    """Einheitskugel aus Dreiecken; Positionen sind zugleich Oberflächennormalen."""
    def point(i, j):
        latitude = math.pi * i / rings
        longitude = 2 * math.pi * j / sectors
        return (math.sin(latitude) * math.cos(longitude),
                math.sin(latitude) * math.sin(longitude), math.cos(latitude))

    vertices = []
    for i in range(rings):
        for j in range(sectors):
            a, b = point(i, j), point(i + 1, j)
            c, d = point(i + 1, j + 1), point(i, j + 1)
            vertices.extend((a, b, c, a, c, d))
    return np.array(vertices, dtype=np.float32)


class _View(QOpenGLWidget):
    def __init__(self, world, eye, target, floor_extent):
        super().__init__()
        self.world = world
        self.eye = np.array(eye, dtype=float)
        self.target = np.array(target, dtype=float)
        direction = self.target - self.eye
        if (not np.isfinite(self.eye).all() or not np.isfinite(self.target).all()
                or np.linalg.norm(direction) < 1e-9
                or np.linalg.norm(np.cross(direction, (0, 0, 1))) < 1e-9):
            raise ValueError("Die Kamera benötigt eine endliche, nicht vertikale Blickrichtung.")
        self.floor_extent = floor_extent
        self.mesh = _sphere_mesh()
        self.setWindowTitle("yaRoboSim – Skriptgesteuerte 3D-Simulation")
        self.resize(1000, 750)

    def initializeGL(self):
        GL.glClearColor(0.07, 0.09, 0.13, 1)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_NORMALIZE)
        GL.glEnable(GL.GL_COLOR_MATERIAL)
        GL.glColorMaterial(GL.GL_FRONT_AND_BACK, GL.GL_AMBIENT_AND_DIFFUSE)
        GL.glEnable(GL.GL_LIGHT0)
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_AMBIENT, (0.3, 0.3, 0.3, 1))

    def paintGL(self):
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        # Perspektive, z nach oben. Qt verwaltet den Viewport einschließlich HiDPI.
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        near = 0.1
        top = near * math.tan(math.radians(45) / 2)
        right = top * self.width() / max(self.height(), 1)
        GL.glFrustum(-right, right, -top, top, near, 500)
        forward = self.target - self.eye
        forward /= np.linalg.norm(forward)
        side = np.cross(forward, (0, 0, 1))
        side /= np.linalg.norm(side)
        up = np.cross(side, forward)
        view = np.eye(4)
        view[:3, :3] = np.array([side, up, -forward])
        view[:3, 3] = -view[:3, :3] @ self.eye
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadMatrixf(view.T.astype(np.float32))
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION, (4, -6, 12, 1))

        self._floor()
        for body in self.world.bodies:
            GL.glPushMatrix()
            transform = np.eye(4)
            transform[:3, :3] = body.orientation * body.radius
            transform[:3, 3] = body.position
            GL.glMultMatrixf(transform.T.astype(np.float32))
            GL.glEnable(GL.GL_LIGHTING)
            GL.glColor3f(*body.color)
            GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
            GL.glEnableClientState(GL.GL_NORMAL_ARRAY)
            GL.glVertexPointer(3, GL.GL_FLOAT, 0, self.mesh)
            GL.glNormalPointer(GL.GL_FLOAT, 0, self.mesh)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, len(self.mesh))
            GL.glDisableClientState(GL.GL_NORMAL_ARRAY)
            GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
            # Drei farbige Radien machen die sonst unsichtbare Rotation sichtbar.
            GL.glDisable(GL.GL_LIGHTING)
            GL.glBegin(GL.GL_LINES)
            for axis, color in zip(np.eye(3), ((1, .2, .2), (.2, 1, .2), (.3, .5, 1))):
                GL.glColor3f(*color)
                GL.glVertex3f(*(axis * 0.98))
                GL.glVertex3f(*(axis * 1.2))
            GL.glEnd()
            GL.glPopMatrix()

    def _floor(self):
        GL.glDisable(GL.GL_LIGHTING)
        z = self.world.floor_z
        if z is None:
            return
        size = self.floor_extent
        GL.glColor3f(0.22, 0.26, 0.30)
        GL.glBegin(GL.GL_QUADS)
        for x, y in ((-size, -size), (size, -size), (size, size), (-size, size)):
            GL.glVertex3f(x, y, z)
        GL.glEnd()
        GL.glColor3f(0.35, 0.40, 0.45)
        GL.glBegin(GL.GL_LINES)
        for i in range(-size, size + 1):
            GL.glVertex3f(i, -size, z + .002)
            GL.glVertex3f(i, size, z + .002)
            GL.glVertex3f(-size, i, z + .002)
            GL.glVertex3f(size, i, z + .002)
        GL.glEnd()


class Viewer:
    """Nur Darstellung. Im Hauptthread erzeugen und render() selbst aufrufen.

    render() liefert False, sobald das Fenster geschlossen wurde.
    eye und target konfigurieren die feste Kamera direkt aus Python.
    """

    def __init__(self, world, eye=(12, -18, 11), target=(0, 0, 3), floor_extent=12):
        if not isinstance(floor_extent, int) or floor_extent < 1:
            raise ValueError("floor_extent muss eine positive ganze Zahl sein.")
        fmt = QSurfaceFormat()
        fmt.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
        fmt.setVersion(2, 1)
        fmt.setDepthBufferSize(24)
        # Vor QApplication setzen, damit Qt den richtigen Kontext erzeugt.
        if QApplication.instance() is None:
            QSurfaceFormat.setDefaultFormat(fmt)
        self.app = QApplication.instance() or QApplication([])
        self.widget = _View(world, eye, target, floor_extent)
        self.widget.setFormat(fmt)
        self.widget.show()
        self.app.processEvents()
        if not self.widget.isValid():
            self.widget.close()
            raise RuntimeError("Kein Desktop-OpenGL-Kontext verfügbar. --headless verwenden.")

    def render(self):
        self.app.processEvents()
        if not self.widget.isVisible():
            return False
        self.widget.repaint()
        return True

    def close(self):
        self.widget.close()
        self.app.processEvents()
