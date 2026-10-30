"""Interchangeable gradient estimators, including backprop-free perturbations.

For independent Rademacher entries u_i in {-1, +1}, E[u uᵀ] = I.
Thus E[(∇L·u)u] = ∇L (up to finite-difference truncation error).
One direction alone does NOT determine the individual partial derivatives.
"""
from collections.abc import Callable
from dataclasses import dataclass
import math

import torch
from torch.nn.utils import parameters_to_vector

from network import loss_output_derivative, loss_value

ALGORITHMS = ("AutoDiff", "EarthQuakeWeights", "EarthQuakeLoss (SPSA)")


class TrainingCancelled(Exception):
    """Cooperative cancellation, including within a perturbation estimate."""


@dataclass(frozen=True)
class GradientConfig:
    algorithm: str = ALGORITHMS[0]
    epsilon: float = 0.01
    directions: int = 32

    def __post_init__(self):
        if self.algorithm not in ALGORITHMS:
            raise ValueError("Unbekannter Gradientenalgorithmus.")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("Die Störungsstärke muss endlich und positiv sein.")
        if self.directions < 1:
            raise ValueError("Mindestens eine Zufallsrichtung ist erforderlich.")


@torch.no_grad()
def copy_parameters(model, flat):
    # copy_ preserves parameter storage and optimizer references.
    offset = 0
    for parameter in model.parameters():
        size = parameter.numel()
        parameter.copy_(flat[offset:offset + size].view_as(parameter))
        offset += size


def estimate_gradient(model, x, y, task, config: GradientConfig,
                      generator: torch.Generator,
                      should_stop: Callable[[], bool] = lambda: False):
    """Return (flat gradient, number of network forward passes).

    EarthQuakeWeights measures the output directional derivative J_z u,
    contracts it with the analytical loss derivative, then multiplies by u.
    EarthQuakeLoss measures the scalar loss directional derivative directly.
    All weights AND biases are perturbed; all pairs use the same mini-batch.
    The original parameters are restored even after cancellation/errors.
    """
    if should_stop():
        raise TrainingCancelled()
    if config.algorithm == "AutoDiff":
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            loss = loss_value(model(x), y, task)
            if not torch.isfinite(loss):
                raise ValueError("Loss ist nicht endlich. Lernrate reduzieren und Netz zurücksetzen.")
            loss.backward()
        return torch.cat([p.grad.detach().flatten() for p in model.parameters()]), 1

    with torch.no_grad():
        base = parameters_to_vector(model.parameters()).clone()
        gradient = torch.zeros_like(base)
        forwards = 0
        try:
            if config.algorithm == "EarthQuakeWeights":
                output = model(x)
                derivative = loss_output_derivative(output, y, task)
                forwards += 1
            for _ in range(config.directions):
                if should_stop():
                    raise TrainingCancelled()
                direction = torch.randint(0, 2, base.shape, generator=generator,
                                          device=base.device).to(base.dtype).mul_(2).sub_(1)
                copy_parameters(model, base + config.epsilon * direction)
                plus = model(x)
                copy_parameters(model, base - config.epsilon * direction)
                minus = model(x)
                forwards += 2
                if config.algorithm == "EarthQuakeWeights":
                    slope = (derivative * ((plus - minus) / (2 * config.epsilon))).sum()
                else:
                    slope = (loss_value(plus, y, task) - loss_value(minus, y, task)) / (2 * config.epsilon)
                if not torch.isfinite(slope):
                    raise ValueError("Störungsmessung ist nicht endlich. ε/Lernrate reduzieren.")
                gradient.add_(direction * slope)
        finally:
            copy_parameters(model, base)
        return gradient / config.directions, forwards


@torch.no_grad()
def assign_gradient(model, flat):
    offset = 0
    for parameter in model.parameters():
        size = parameter.numel()
        parameter.grad = flat[offset:offset + size].view_as(parameter).clone()
        offset += size


def compare_gradients(model, x, y, task, epsilon, directions, seed=123,
                      should_stop=lambda: False):
    """Diagnostics at identical weights/batch, before clipping or updating."""
    saved_grads = [None if p.grad is None else p.grad.detach().clone() for p in model.parameters()]
    try:
        reference, _ = estimate_gradient(model, x, y, task, GradientConfig(),
                                         torch.Generator().manual_seed(seed), should_stop)
        rows = []
        for algorithm in ALGORITHMS[1:]:
            estimate, forwards = estimate_gradient(
                model, x, y, task, GradientConfig(algorithm, epsilon, directions),
                torch.Generator().manual_seed(seed), should_stop)
            norm = reference.norm().item()
            denominator = norm * estimate.norm().item()
            rows.append({"algorithm": algorithm,
                         "cosine": float(reference.dot(estimate)) / denominator if denominator else float("nan"),
                         "relative_error": float((reference - estimate).norm()) / norm if norm else float("nan"),
                         "reference_norm": norm, "estimate_norm": estimate.norm().item(),
                         "forwards": forwards})
        return rows
    finally:
        for parameter, grad in zip(model.parameters(), saved_grads):
            parameter.grad = grad
