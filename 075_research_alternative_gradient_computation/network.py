"""MLP parser and task-specific losses. No Qt or visualization imports."""
import torch
from torch import nn

from data import TASKS

DEFAULT_ARCHITECTURE = "linear 20\ntanh\nlinear 10\ntanh"


def build_mlp(description: str, seed: int = 42) -> nn.Sequential:
    """Two coordinates in, one logit (classification) / value (regression) out."""
    if len(description.splitlines()) > 40:
        raise ValueError("Maximal 40 Zeilen sind erlaubt.")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        layers, width = [], 2
        activations = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid,
                       "gelu": nn.GELU, "silu": nn.SiLU}
        for number, line in enumerate(description.splitlines(), 1):
            words = line.split("#", 1)[0].lower().split()
            if not words:
                continue
            try:
                if len(words) == 2 and words[0] == "linear":
                    new_width = int(words[1])
                    if not 1 <= new_width <= 512:
                        raise ValueError("Breite muss zwischen 1 und 512 liegen.")
                    layers.append(nn.Linear(width, new_width))
                    width = new_width
                elif len(words) == 1 and words[0] in activations:
                    layers.append(activations[words[0]]())
                else:
                    raise ValueError("Erlaubt: linear N, relu, tanh, sigmoid, gelu, silu.")
            except ValueError as error:
                raise ValueError(f"Zeile {number}: {error}") from error
        layers.append(nn.Linear(width, 1))
        return nn.Sequential(*layers)


def loss_value(output, target, task):
    if task == TASKS[0]:
        return nn.functional.binary_cross_entropy_with_logits(output, target)
    return nn.functional.mse_loss(output, target)


def loss_output_derivative(output, target, task):
    """Analytical dL/dz only; does not differentiate through the network."""
    if task == TASKS[0]:
        return (output.sigmoid() - target) / target.numel()
    return 2 * (output - target) / target.numel()
