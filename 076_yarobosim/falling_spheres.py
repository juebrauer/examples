"""Eigene Szene + eigene Schleife. Aufruf: python falling_spheres.py [--headless]."""

import argparse
import math
import time
import numpy as np

from yarobosim import Sphere, World


def create_world():
    world = World()
    rng = np.random.default_rng(7)
    colors = [(0.95, 0.35, 0.20), (0.20, 0.65, 0.95), (0.95, 0.75, 0.20)]
    # Drei versetzte Säulen: alle Kugeln starten über dem Boden, ohne Überlappung.
    for layer in range(4):
        for column in range(3):
            world.add(Sphere(
                name=f"Kugel {layer}-{column}", radius=0.45,
                mass=1 + 0.25 * column,
                position=((column - 1) * 1.2 + 0.2 * (layer % 2),
                          rng.uniform(-0.2, 0.2), 2.5 + layer * 1.15),
                velocity=(rng.uniform(-0.25, 0.25), rng.uniform(-0.1, 0.1), 0),
                restitution=0.65, friction=0.45, color=colors[column],
            ))
    # Unbewegliches Hindernis: die fallenden Kugeln stoßen daran und aneinander.
    world.add(Sphere(name="Hindernis", radius=1.0, mass=0,
                     position=(0, 0, 0.7), color=(0.5, 0.55, 0.6)))
    return world


def run(seconds=15.0, headless=False):
    world = create_world()
    viewer = None
    if not headless:
        # Kein Qt-/OpenGL-Import bei headless=True.
        from yarobosim.rendering import Viewer
        viewer = Viewer(world)

    dt = 1 / 120
    render_every = 2                     # 120 Physikschritte, 60 Bilder pro Sekunde
    started = time.perf_counter()
    try:
        for step in range(math.ceil(seconds / dt)):
            # Eigene Steuerung hier, z.B. vor jedem step:
            # world.bodies[0].apply_force((1, 0, 0))
            world.step(min(dt, seconds - world.time))
            viewer.widget.eye += np.array( [0,0,0.01] )
            if viewer is not None and step % render_every == 0:
                if not viewer.render():
                    break
                # Nur die Darstellung takten; die Physik erhält immer feste dt.
                remaining = started + world.time - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        if viewer is not None:
            viewer.close()

    print(f"Simulierte Zeit: {world.time:.3f} s, Körper: {len(world.bodies)}")
    for body in world.bodies[:3]:
        print(f"{body.name}: Position = {body.position.round(3)} m")
    return world


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="Ohne Fenster simulieren")
    parser.add_argument("--seconds", type=float, default=15.0, help="Dauer in Simulationssekunden")
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds muss endlich und > 0 sein")
    run(args.seconds, args.headless)
