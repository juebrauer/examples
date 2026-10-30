"""Reproducible 2D tasks, without dependencies on the user interface."""
from dataclasses import dataclass

import torch

TASKS = ("Circle-Klassifikation", "Berge-und-Täler-Regression")


@dataclass(frozen=True)
class DataConfig:
    task: str = TASKS[0]
    train_samples: int = 512
    test_samples: int = 256
    seed: int = 42

    def __post_init__(self):
        if self.task not in TASKS:
            raise ValueError("Unbekannte Aufgabe.")
        if min(self.train_samples, self.test_samples) < 1:
            raise ValueError("Training und Test benötigen jeweils mindestens einen Punkt.")
        if not 0 <= self.seed < 2**31:
            raise ValueError("Seed muss zwischen 0 und 2³¹−1 liegen.")


def target_function(x: torch.Tensor, task: str) -> torch.Tensor:
    """Return [N, 1] labels/values on the square [-1, 1]²."""
    a, b = x[:, 0], x[:, 1]
    if task == TASKS[0]:
        return (a.square() + b.square() >= 0.55**2).to(x.dtype)[:, None]
    if task != TASKS[1]:
        raise ValueError("Unbekannte Aufgabe.")
    z = (0.45 * torch.sin(3 * torch.pi * a) * torch.cos(2 * torch.pi * b)
         + 0.25 * torch.sin(5 * torch.pi * (a + b))
         + 0.8 * torch.exp(-18 * ((a - 0.35)**2 + (b + 0.25)**2))
         - 0.7 * torch.exp(-22 * ((a + 0.4)**2 + (b - 0.35)**2)))
    return z[:, None]


def make_data(config: DataConfig):
    generator = torch.Generator().manual_seed(config.seed)

    def sample(n):
        if config.task == TASKS[0]:
            # Approximately balanced disk / annulus, sampled uniformly in area.
            labels = (torch.arange(n) % 2)[torch.randperm(n, generator=generator)]
            u = torch.rand(n, generator=generator)
            radius = torch.sqrt(torch.where(labels == 0, u * 0.55**2,
                                             0.55**2 + u * (1 - 0.55**2)))
            angle = 2 * torch.pi * torch.rand(n, generator=generator)
            x = torch.stack((radius * angle.cos(), radius * angle.sin()), dim=1)
        else:
            x = 2 * torch.rand(n, 2, generator=generator) - 1
        return x, target_function(x, config.task)

    return sample(config.train_samples), sample(config.test_samples)
