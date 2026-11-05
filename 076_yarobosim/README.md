# yaRoboSim – ein kleiner 3D-Simulator für Python

Die erste Version ist ein bewusst überschaubarer Ausgangspunkt für Forschung
und Lehre: Szene und Simulationsschleife stehen im eigenen Python-Skript.
Das optionale Fenster zeigt ausschließlich die Welt, ohne Buttons oder Editor.
Die Physik ist selbst implementiert; es wird keine externe Physik-Engine versteckt.

## Starten

Python 3.10 oder neuer. Im Verzeichnis `076_yarobosim`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python falling_spheres.py
```

Zwölf starre Kugeln fallen auf einen Boden, aufeinander und auf ein festes
kugelförmiges Hindernis. Unterschiedliche Massen, Rückprall und Reibung bestimmen
ihre Bewegung. Farbige Markierungen auf den Kugeln zeigen die Rotation.
Nach 15 Simulationssekunden endet das Skript. Früheres Schließen des Fensters
beendet die Schleife ebenfalls. Die Dauer ist frei wählbar:

```bash
python falling_spheres.py --seconds 30
```

Ohne Fenster benötigt man **nur NumPy**, keinen Displayserver und kein Qt:

```bash
python -m pip install -r requirements-headless.txt
python falling_spheres.py --headless --seconds 10
```

Headless läuft ohne künstliche Wartezeit so schnell, wie die Berechnung erlaubt.
Die Simulation verwendet dieselben Zeitschritte und dieselbe Szene wie mit Fenster.
Unter Windows lautet die Aktivierung der Umgebung `.venv\Scripts\activate`.

## Eine eigene Simulation

Das Skript im Projektverzeichnis ablegen; dann ist `yarobosim` direkt importierbar.

```python
from yarobosim import Sphere, World

world = World(gravity=(0, 0, -9.81), floor_z=0)
ball = world.add(Sphere(radius=0.5, mass=2, position=(0, 0, 5)))
other = world.add(Sphere(radius=0.7, mass=1, position=(0.5, 0, 2)))

dt = 1 / 120
for step in range(1200):
    # Kraft in Newton; vor jedem Schritt neu setzen, solange sie wirken soll.
    ball.apply_force((0.2, 0, 0))
    world.step(dt)
    # Zustand direkt als NumPy-Vektoren lesen, z.B. für Messungen oder Regelung.
    print(world.time, ball.position, ball.velocity)
```

Rendering zur eigenen Schleife hinzufügen:

```python
from yarobosim.rendering import Viewer

viewer = Viewer(world, eye=(12, -18, 11), target=(0, 0, 3))
try:
    for step in range(1200):
        world.step(1 / 120)
        if step % 2 == 0 and not viewer.render():
            break
finally:
    viewer.close()
```

`Viewer` verändert weder Physik noch Simulationszeit. Die Schleife bleibt im
Skript; es gibt keinen versteckten Qt-Timer und keinen Aufruf von `app.exec()`.
Qt und Viewer müssen im Hauptthread laufen. Dieses Minimalbeispiel enthält keine
Echtzeit-Taktung; `falling_spheres.py` zeigt die Taktung mit `time.perf_counter()`.
Bei langsamer Berechnung läuft die Anzeige entsprechend langsamer, ohne die
Physikschritte zu vergrößern oder auszulassen.

## Kleine API

| Eigenschaft / Methode | Bedeutung |
| --- | --- |
| `Sphere(radius, mass, position, velocity, ...)` | Starre Vollkugel |
| `mass=0` | Unbeweglicher Körper mit unendlicher effektiver Masse |
| `restitution` | Rückprall zwischen 0 und 1; beim Kontakt gilt der kleinere Wert |
| `friction` | Coulomb-Reibwert; Kontaktwert ist das geometrische Mittel |
| `angular_velocity` | Winkelgeschwindigkeit als Weltvektor in rad/s |
| `orientation` | 3×3-Drehmatrix: lokale Achsen in Weltkoordinaten |
| `apply_force(force, point=None)` | Kraft bis zum nächsten `world.step`; optionaler Weltpunkt erzeugt Drehmoment |
| `apply_torque(torque)` | Zusätzliches Drehmoment in Nm bis zum nächsten Schritt |
| `apply_impulse(impulse, point=None)` | Sofortiger Stoß in Ns, optional mit Drehimpuls |
| `World(floor_z=None)` | Welt ohne Boden; sonst unendliche horizontale Ebene |
| `world.add(body)` | Körper hinzufügen; gibt denselben Körper zurück |
| `world.step(dt)` | Physik um `dt` Sekunden voranbringen |
| `world.time`, `world.bodies` | Simulationszeit und Körperliste |
| `viewer.render()` | Aktuellen Zustand zeichnen, Ereignisse bearbeiten; False nach Schließen |

Alle Längen sind in Metern, Massen in Kilogramm, Zeiten in Sekunden.
**z zeigt nach oben.** Zustände sind absichtlich direkt zugänglich; bei eigenen
Änderungen gültige 3D-NumPy-Vektoren verwenden. Radius und Masse nach dem Aufbau
einer Szene möglichst unverändert lassen. Für statische Körper Position setzen,
Geschwindigkeit aber auf null lassen.

## Wie die Physik funktioniert

Der Lesefaden beginnt in `yarobosim/physics.py` bei `World.step`:

1. Ein öffentlicher Schritt wird in Schritte von höchstens `max_substep=1/240 s`
   unterteilt. Externe Kräfte bleiben während aller Teilschritte konstant.
2. Aus `a = g + F/m` wird zuerst die Geschwindigkeit und daraus die Position
   berechnet: semi-impliziter Euler. Analog gilt für Rotation `alpha = torque/I`
   mit dem Vollkugel-Trägheitsmoment `I = 2/5 m r²`.
3. Kontakte entstehen über Kugelabstände beziehungsweise die Höhe über dem Boden.
4. Ein Normalimpuls verändert die Geschwindigkeiten beider Körper gleich und
   entgegengesetzt. Bei reibungsfreien, elastischen Zweikörperstößen bleiben
   linearer Impuls und kinetische Energie erhalten.
5. Ein begrenzter tangentialer Reibimpuls verändert Translation **und** Rotation.
   Dadurch geht Gleiten in Rollen über.
6. Kleine Überschneidungen werden entlang der Kontaktnormalen korrigiert. Mehrere
   Löserdurchläufe verbessern Kontakte zwischen mehreren Körpern. Sehr langsame
   Stöße werden unelastisch behandelt, damit ruhende Kugeln nicht ständig hüpfen.
7. Am Ende des öffentlichen Schrittes werden Kraft- und Drehmomentspeicher geleert.

Die drei wesentlichen Dateien sind `physics.py` (Dynamik), `rendering.py`
(Darstellung) und `falling_spheres.py` (Szene und eigene Schleife).
Die Physik lässt sich unabhängig von der Darstellung erweitern und prüfen.

## Grenzen und mögliche nächste Schritte

Diese Version unterstützt **Kugeln und eine horizontale Bodenebene**. Würfel,
beliebige Meshes, Gelenke, Motoren und Robotermodelle sind noch nicht enthalten.
Sie bildet eine Basis für deren spätere Implementierung; sie ist noch kein
vollständiger Robotersimulator mit Roboterarmen oder URDF-Import.

Die Kollisionserkennung prüft alle Kugelpaare, also O(n²). Geeignet sind kleine
Lehrszenen, keine tausenden Objekte. Es gibt keine kontinuierliche
Kollisionserkennung: sehr schnelle oder sehr kleine Kugeln können zwischen
Zeitschritten durch andere Kugeln tunneln. Dann `max_substep` verkleinern;
Faustregel: Weg pro Teilschritt deutlich kleiner als der kleinste Radius.
Sehr große öffentliche `dt` erzeugen entsprechend viele Teilschritte.

Stapel sind näherungsweise stabil; bei vielen Körpern, großen Massenunterschieden
oder starken Überlappungen sind Artefakte möglich. Es gibt kein Sleeping,
keine Luftreibung und keinen Rollwiderstand. Eine ideal rollende Kugel kann
deshalb dauerhaft rollen. Die Bodenphysik ist unendlich ausgedehnt; das sichtbare
Raster ist nur ein begrenzter Ausschnitt. Inelastizität, Reibung und die
Kontaktstabilisierung dissipieren Energie bewusst.

Das Rendering verwendet einfaches **Desktop-OpenGL 2.1** mit fester Beleuchtung.
Ein OpenGL-fähiger Grafiktreiber und eine Qt-taugliche Desktop-Sitzung sind nötig;
OpenGL ES und reine Core-Profile werden nicht unterstützt. Unter Linux können
zusätzlich Qt/xcb-Systembibliotheken erforderlich sein. `QT_QPA_PLATFORM=offscreen`
ersetzt keinen OpenGL-fähigen Displayserver für `QOpenGLWidget`.
Bei Darstellungsproblemen zuerst `--headless` verwenden.

Als nächste Lehrschritte bieten sich separate Kontaktgeometrien für Quader,
danach Distanz-/Drehgelenke und einfache Motorregler an.

## Überprüfen

Tests benötigen nur NumPy und die Python-Standardbibliothek:

```bash
python -m unittest discover -s tests -v
```

Geprüft werden freier Fall mit Konvergenz, elastischer Stoß unterschiedlicher
Massen, Rückprall, ruhender Stapel, Reibung und Rollen, Kraft/Drehmoment über
Teilschritte, statische Körper und degenerierte Anfangslagen.

Technische Referenzen für die Darstellung:
[Qt QOpenGLWidget](https://doc.qt.io/qtforpython-6/PySide6/QtOpenGLWidgets/QOpenGLWidget.html),
[Qt QSurfaceFormat](https://doc.qt.io/qtforpython-6/PySide6/QtGui/QSurfaceFormat.html),
[PyOpenGL](https://pyopengl.sourceforge.net/documentation/).
