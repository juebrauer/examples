"""Ten fixed, synthetic 2D teaching datasets; no download or sklearn required."""
import numpy as np


DATASETS = (
    "Two Gaussian blobs", "Three Gaussian blobs", "Five Gaussian blobs",
    "Anisotropic blobs", "Two moons", "Concentric circles", "XOR",
    "Two spirals", "Parallel stripes", "Checkerboard",
)


def load_dataset(name):
    """Return fresh (points, class indices, class count), always in the same order.

    Each call has its own fixed RNG, independent of model seeds and mouse clicks.
    Coordinates fit the canvas [-1, 1]². Every dataset contains 300 points.
    """
    rng = np.random.default_rng(42)
    count = 300
    if name in DATASETS[:4]:
        classes = {DATASETS[0]: 2, DATASETS[1]: 3, DATASETS[2]: 5,
                   DATASETS[3]: 3}[name]
        labels = np.arange(count) % classes
        angles = np.arange(classes) * 2 * np.pi / classes
        centers = 0.62 * np.column_stack((np.cos(angles), np.sin(angles)))
        noise = rng.normal(0, 0.12, (count, 2))
        if name == "Anisotropic blobs":
            noise = noise @ np.array([[1.5, 0.8], [0.0, 0.4]])
        points = centers[labels] + noise
    elif name == "Two moons":
        classes = 2
        labels = np.arange(count) % classes
        angle = rng.uniform(0, np.pi, count)
        points = np.column_stack((np.cos(angle), np.sin(angle)))
        points[labels == 1] = np.column_stack((1 - np.cos(angle[labels == 1]),
                                             0.5 - np.sin(angle[labels == 1])))
        points = (points - [0.5, 0.25]) * 0.55 + rng.normal(0, 0.035, (count, 2))
    elif name in ("Concentric circles", "Two spirals"):
        classes = 2
        labels = np.arange(count) % classes
        if name == "Concentric circles":
            angle = rng.uniform(0, 2 * np.pi, count)
            radius = np.where(labels == 0, 0.35, 0.78)
        else:
            t = np.tile(np.linspace(0.08, 1, count // 2), (2, 1)).T.flatten()
            angle = 2.5 * np.pi * t + labels * np.pi
            radius = 0.85 * t
        points = radius[:, None] * np.column_stack((np.cos(angle), np.sin(angle)))
        points += rng.normal(0, 0.025, (count, 2))
    elif name in ("XOR", "Parallel stripes", "Checkerboard"):
        points = rng.uniform(-0.9, 0.9, (count, 2))
        classes = 3 if name == "Parallel stripes" else 2
        if name == "XOR":
            labels = (points[:, 0] * points[:, 1] >= 0).astype(int)
        elif name == "Parallel stripes":
            labels = (np.floor((points[:, 0] + 0.9) / 0.3).astype(int) % classes)
        else:
            labels = (np.floor((points + 0.9) / 0.45).sum(axis=1).astype(int) % 2)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    order = rng.permutation(count)
    return np.clip(points[order], -0.98, 0.98), labels[order], classes
