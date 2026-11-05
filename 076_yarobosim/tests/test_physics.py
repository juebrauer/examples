"""Physikalische Eigenschaften prüfen, unabhängig von Fenster und Grafikkarte."""

import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from yarobosim import Sphere, World


class PhysicsTests(unittest.TestCase):
    def test_free_fall_converges_to_analytic_solution(self):
        errors = []
        for h in (1 / 120, 1 / 240):
            world = World(floor_z=None, max_substep=h)
            ball = world.add(Sphere(position=(0, 0, 10)))
            for _ in range(round(1 / h)):
                world.step(h)
            self.assertAlmostEqual(ball.velocity[2], -9.81, places=10)
            errors.append(abs(ball.position[2] - (10 - 9.81 / 2)))
        self.assertLess(errors[1], errors[0] * 0.51)
        self.assertLess(errors[1], 0.021)

    def test_elastic_unequal_mass_collision_conserves_momentum_and_energy(self):
        world = World(gravity=(0, 0, 0), floor_z=None)
        a = world.add(Sphere(position=(-1, 0, 0), velocity=(2, 0, 0),
                             mass=2, restitution=1, friction=0))
        b = world.add(Sphere(position=(1, 0, 0), velocity=(-1, 0, 0),
                             mass=1, restitution=1, friction=0))
        momentum = a.mass * a.velocity + b.mass * b.velocity
        energy = sum(0.5 * x.mass * np.dot(x.velocity, x.velocity) for x in (a, b))
        for _ in range(120):
            world.step(1 / 120)
        np.testing.assert_allclose(a.mass * a.velocity + b.mass * b.velocity, momentum)
        self.assertAlmostEqual(sum(0.5 * x.mass * np.dot(x.velocity, x.velocity)
                                   for x in (a, b)), energy)
        np.testing.assert_allclose(a.velocity, (0, 0, 0), atol=1e-12)
        np.testing.assert_allclose(b.velocity, (3, 0, 0), atol=1e-12)

    def test_floor_rebound(self):
        world = World(gravity=(0, 0, 0), floor_restitution=0.8)
        ball = world.add(Sphere(position=(0, 0, .5), velocity=(0, 0, -2), restitution=.8))
        world.step(1 / 240)
        self.assertAlmostEqual(ball.velocity[2], 1.6)

    def test_resting_stack_is_stable(self):
        world = World()
        for i in range(3):
            world.add(Sphere(position=(0, 0, .5 + i), restitution=0))
        for _ in range(600):
            world.step(1 / 120)
        for i, ball in enumerate(world.bodies):
            self.assertAlmostEqual(ball.position[2], .5 + i, delta=.004)
            self.assertLess(np.linalg.norm(ball.velocity), .03)

    def test_floor_friction_produces_rolling_without_adding_energy(self):
        world = World()
        ball = world.add(Sphere(position=(0, 0, .5), velocity=(2, 0, 0), restitution=0))
        for _ in range(120):
            world.step(1 / 120)
        self.assertGreater(ball.angular_velocity[1], 0)
        self.assertLess(ball.velocity[0], 2)
        contact_velocity = ball.velocity + np.cross(ball.angular_velocity, (0, 0, -.5))
        self.assertAlmostEqual(contact_velocity[0], 0, places=6)
        inertia = .4 * ball.mass * ball.radius**2
        energy = .5 * ball.mass * np.dot(ball.velocity, ball.velocity)
        energy += .5 * inertia * np.dot(ball.angular_velocity, ball.angular_velocity)
        self.assertLessEqual(energy, 2.0001)

    def test_force_torque_and_reset_over_substeps(self):
        world = World(gravity=(0, 0, 0), floor_z=None)
        ball = world.add(Sphere(mass=2, position=(0, 0, 0)))
        ball.apply_force((0, 4, 0), point=(.5, 0, 0))
        world.step(.1)
        np.testing.assert_allclose(ball.velocity, (0, .2, 0), atol=1e-12)
        np.testing.assert_allclose(ball.angular_velocity, (0, 0, 1), atol=1e-12)
        world.step(.1)
        np.testing.assert_allclose(ball.velocity, (0, .2, 0), atol=1e-12)
        np.testing.assert_allclose(ball.angular_velocity, (0, 0, 1), atol=1e-12)
        np.testing.assert_allclose(ball.orientation.T @ ball.orientation, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(ball.orientation), 1)

    def test_static_body_does_not_move(self):
        world = World(gravity=(0, 0, 0), floor_z=None)
        fixed = world.add(Sphere(position=(0, 0, 0), mass=0))
        ball = world.add(Sphere(position=(-1, 0, 0), velocity=(2, 0, 0)))
        fixed.apply_force((20, 0, 0))
        world.step(.02)
        np.testing.assert_array_equal(fixed.position, (0, 0, 0))
        np.testing.assert_array_equal(fixed.velocity, (0, 0, 0))
        self.assertLess(ball.velocity[0], 0)

    def test_coincident_spheres_separate_without_nan(self):
        world = World(gravity=(0, 0, 0), floor_z=None)
        a = world.add(Sphere())
        b = world.add(Sphere())
        world.step(.01)
        self.assertGreater(np.linalg.norm(a.position - b.position), .99)
        self.assertTrue(np.isfinite(a.position).all())

    def test_invalid_inputs(self):
        for kwargs in ({"radius": 0}, {"mass": -1}, {"position": (0, 0)},
                       {"restitution": 1.1}, {"mass": float('nan')},
                       {"mass": 0, "velocity": (1, 0, 0)}):
            with self.assertRaises(ValueError):
                Sphere(**kwargs)
        world = World()
        for dt in (0, -1, float('inf'), float('nan')):
            with self.assertRaises(ValueError):
                world.step(dt)
        body = world.add(Sphere())
        with self.assertRaises(ValueError):
            world.add(body)


if __name__ == "__main__":
    unittest.main()
