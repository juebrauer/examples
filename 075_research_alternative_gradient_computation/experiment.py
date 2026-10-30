"""Headless training, evaluation, snapshots and independent random streams."""
from dataclasses import dataclass, asdict
import math
from time import perf_counter

import torch

from data import DataConfig, TASKS, make_data, target_function
from gradients import GradientConfig, TrainingCancelled, assign_gradient, estimate_gradient
from network import DEFAULT_ARCHITECTURE, build_mlp, loss_value


@dataclass(frozen=True)
class TrainConfig:
    gradient: GradientConfig = GradientConfig()
    optimizer: str = "Adam"
    learning_rate: float = 0.01
    batch_size: int = 64
    clip_norm: float = 5.0

    def __post_init__(self):
        if self.optimizer not in ("Adam", "SGD"):
            raise ValueError("Unbekannter Optimizer.")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Lernrate muss endlich und positiv sein.")
        if self.batch_size < 1 or not math.isfinite(self.clip_norm) or self.clip_norm < 0:
            raise ValueError("Ungültige Batch-Größe oder Gradient-Clipping-Grenze.")


class Experiment:
    def __init__(self, data_config=DataConfig(), architecture=DEFAULT_ARCHITECTURE):
        self.data_config = data_config
        self.architecture = architecture
        self.train_data, self.test_data = make_data(data_config)
        self.grid_size = 120
        axis = torch.linspace(-1, 1, self.grid_size)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        self.grid = torch.stack((xx.flatten(), yy.flatten()), dim=1)
        self.truth = target_function(self.grid, data_config.task).reshape(self.grid_size, self.grid_size).numpy()
        self.reset_model()

    def reset_model(self):
        self.model = build_mlp(self.architecture, self.data_config.seed)
        self.batch_rng = torch.Generator().manual_seed(self.data_config.seed + 1)
        self.perturbation_rng = torch.Generator().manual_seed(self.data_config.seed + 2)
        self.optimizer = None
        self.optimizer_key = None
        self.epochs = self.steps = self.forwards = 0
        self.training_seconds = 0.0
        self.history = []
        self.last_config = None
        self.record_metrics()

    def prepare_optimizer(self, config):
        # Retain moments across repeated Learn clicks; reset when switching method.
        key = (config.optimizer, config.gradient.algorithm)
        if self.optimizer_key != key:
            cls = torch.optim.Adam if config.optimizer == "Adam" else torch.optim.SGD
            self.optimizer = cls(self.model.parameters(), lr=config.learning_rate)
            self.optimizer_key = key
        for group in self.optimizer.param_groups:
            group["lr"] = config.learning_rate
        self.last_config = config

    def train_epoch(self, config: TrainConfig, should_stop=lambda: False):
        self.prepare_optimizer(config)
        self.model.train()
        x, y = self.train_data
        order = torch.randperm(len(x), generator=self.batch_rng)
        started = perf_counter()
        try:
            for indices in order.split(config.batch_size):
                if should_stop():
                    raise TrainingCancelled()
                self.optimizer.zero_grad(set_to_none=True)
                gradient, forwards = estimate_gradient(
                    self.model, x[indices], y[indices], self.data_config.task,
                    config.gradient, self.perturbation_rng, should_stop)
                if not torch.isfinite(gradient).all():
                    raise ValueError("Gradient ist nicht endlich. Netz zurücksetzen und Lernrate reduzieren.")
                assign_gradient(self.model, gradient)
                if config.clip_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.clip_norm,
                                                  error_if_nonfinite=True)
                self.optimizer.step()
                self.steps += 1
                self.forwards += forwards
            self.epochs += 1
        finally:
            self.training_seconds += perf_counter() - started
        return self.record_metrics()

    @torch.no_grad()
    def evaluate(self, data):
        x, y = data
        self.model.eval()
        output = self.model(x)
        loss = loss_value(output, y, self.data_config.task).item()
        if not math.isfinite(loss):
            raise ValueError("Loss ist nicht endlich. Netz zurücksetzen und Lernrate reduzieren.")
        metric = ((output >= 0) == (y >= 0.5)).float().mean().item() if self.data_config.task == TASKS[0] else math.sqrt(loss)
        return loss, metric

    def record_metrics(self):
        train_loss, train_metric = self.evaluate(self.train_data)
        test_loss, test_metric = self.evaluate(self.test_data)
        row = {"epoch": self.epochs, "steps": self.steps,
               "train_loss": train_loss, "test_loss": test_loss,
               "train_metric": train_metric, "test_metric": test_metric,
               "seconds": self.training_seconds, "forwards": self.forwards,
               "algorithm": self.last_config.gradient.algorithm if self.last_config else "Initialisierung",
               "config": asdict(self.last_config) if self.last_config else None}
        # Also represent a stopped, partially completed epoch without duplicating it.
        if self.history and self.history[-1]["steps"] == self.steps:
            self.history[-1] = row
        else:
            self.history.append(row)
        return row

    @torch.no_grad()
    def snapshot(self):
        self.model.eval()
        output = self.model(self.grid)
        if self.data_config.task == TASKS[0]:
            output = output.sigmoid()
        return {"prediction": output.reshape(self.grid_size, self.grid_size).numpy().copy(),
                "history": [row.copy() for row in self.history],
                "parameters": sum(p.numel() for p in self.model.parameters())}

    def export_data(self):
        return {"data": asdict(self.data_config), "architecture": self.architecture,
                "training": asdict(self.last_config) if self.last_config else None,
                "history": [row.copy() for row in self.history]}
