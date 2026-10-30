"""Numerical correctness, repeatability and real offscreen Qt lifecycle checks."""
import itertools
import os
from time import monotonic, sleep

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/earthquake-test-mpl")

import pytest
import torch
from torch.nn.utils import parameters_to_vector

from data import DataConfig, TASKS, make_data
from experiment import Experiment, TrainConfig
from gradients import (ALGORITHMS, GradientConfig, TrainingCancelled,
                       compare_gradients, copy_parameters, estimate_gradient)
from network import build_mlp, loss_value

torch.set_num_threads(1)


@pytest.mark.parametrize("task", TASKS)
def test_data_reproducible_disjoint_and_correct(task):
    config = DataConfig(task, 33, 15, 7)
    train, test = make_data(config)
    repeated = make_data(config)
    for a, b in zip((train, test), repeated):
        assert all(torch.equal(u, v) for u, v in zip(a, b))
    assert train[0].shape == (33, 2)
    assert test[1].shape == (15, 1)
    assert not (train[0][:, None] == test[0][None]).all(dim=-1).any()
    if task == TASKS[0]:
        assert torch.equal(train[1][:, 0], (train[0].square().sum(1) >= 0.55**2).float())
        assert abs(train[1].sum().item() - 33 / 2) <= 0.5


@pytest.mark.parametrize("text", ["linear nope", "linear 0", "linear 513", "relu 2", "dropout .5", "exec hi"])
def test_parser_rejects_invalid_descriptions(text):
    with pytest.raises(ValueError, match="Zeile 1"):
        build_mlp(text)


def test_parser_automatic_input_output_and_local_rng():
    rng = torch.random.get_rng_state()
    model = build_mlp("linear 20 # twenty neurons\nrelu\nlinear 10\ntanh")
    assert model[0].in_features == 2 and model[0].out_features == 20
    assert model[-1].out_features == 1
    assert model(torch.zeros(4, 2)).shape == (4, 1)
    assert torch.equal(rng, torch.random.get_rng_state())


@pytest.mark.parametrize("task", TASKS)
def test_autodiff_matches_independent_coordinate_differences(task):
    model = build_mlp("linear 3\ntanh", seed=6).double()
    x, y = make_data(DataConfig(task, 7, 3, 5))[0]
    x, y = x.double(), y.double()
    base = parameters_to_vector(model.parameters()).detach().clone()
    exact, _ = estimate_gradient(model, x, y, task, GradientConfig(), torch.Generator())
    numeric = torch.zeros_like(base)
    epsilon = 1e-5
    with torch.no_grad():
        for i in range(len(base)):
            offset = torch.zeros_like(base)
            offset[i] = epsilon
            copy_parameters(model, base + offset)
            plus = loss_value(model(x), y, task)
            copy_parameters(model, base - offset)
            minus = loss_value(model(x), y, task)
            numeric[i] = (plus - minus) / (2 * epsilon)
        copy_parameters(model, base)
    torch.testing.assert_close(exact, numeric, atol=1e-8, rtol=1e-6)


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("algorithm", ALGORITHMS[1:])
def test_complete_rademacher_average_matches_autodiff_without_backward(monkeypatch, task, algorithm):
    # A linear model has three parameters. Averaging all 2³ sign vectors
    # eliminates cross terms exactly, making this a deterministic math test.
    model = build_mlp("", seed=4).double()
    x, y = make_data(DataConfig(task, 7, 3, 5))[0]
    x, y = x.double(), y.double()
    base = parameters_to_vector(model.parameters()).detach().clone()
    exact, _ = estimate_gradient(model, x, y, task, GradientConfig(), torch.Generator())
    signs = iter(itertools.product((0, 1), repeat=3))
    monkeypatch.setattr(torch, "randint", lambda *args, **kwargs: torch.tensor(next(signs)))

    def forbid_backward(*args, **kwargs):
        raise AssertionError("EarthQuake must not call backward")

    monkeypatch.setattr(torch.Tensor, "backward", forbid_backward)
    grad_enabled = []
    hook = model.register_forward_hook(lambda *args: grad_enabled.append(torch.is_grad_enabled()))
    estimate, forwards = estimate_gradient(model, x, y, task, GradientConfig(algorithm, 1e-4, 8), torch.Generator())
    hook.remove()
    assert not any(grad_enabled)
    assert forwards == (17 if algorithm == "EarthQuakeWeights" else 16)
    torch.testing.assert_close(exact, estimate, rtol=1e-7, atol=1e-8)
    assert torch.equal(base, parameters_to_vector(model.parameters()))


@pytest.mark.parametrize("algorithm", ALGORITHMS[1:])
def test_perturbation_restores_weights_after_error_and_cancel(algorithm):
    model = build_mlp("linear 3\ntanh")
    x, y = make_data(DataConfig(train_samples=8, test_samples=4))[0]
    base = parameters_to_vector(model.parameters()).detach().clone()
    calls = 0

    def fail_during_perturbation(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected forward failure")

    hook = model.register_forward_hook(fail_during_perturbation)
    with pytest.raises(RuntimeError, match="injected"):
        estimate_gradient(model, x, y, TASKS[0], GradientConfig(algorithm), torch.Generator())
    hook.remove()
    assert torch.equal(base, parameters_to_vector(model.parameters()))
    calls = 0

    def stop():
        nonlocal calls
        calls += 1
        return calls >= 3

    with pytest.raises(TrainingCancelled):
        estimate_gradient(model, x, y, TASKS[0], GradientConfig(algorithm), torch.Generator(), stop)
    assert torch.equal(base, parameters_to_vector(model.parameters()))


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_training_learns_and_repeated_calls_continue(algorithm):
    experiment = Experiment(DataConfig(TASKS[0], 128, 128))
    initial = experiment.history[-1]["train_loss"]
    config = TrainConfig(GradientConfig(algorithm, directions=32), batch_size=32)
    for _ in range(100):
        experiment.train_epoch(config)
    assert experiment.history[-1]["train_loss"] < initial * 0.9
    optimizer = experiment.optimizer
    experiment.train_epoch(config)
    assert experiment.optimizer is optimizer
    assert experiment.steps == 101 * 4
    assert experiment.epochs == 101
    expected_forwards = {ALGORITHMS[0]: 1, ALGORITHMS[1]: 65, ALGORITHMS[2]: 64}[algorithm]
    assert experiment.forwards == experiment.steps * expected_forwards


def test_reproducibility_separate_rngs_diagnostic_and_export():
    a, b = Experiment(), Experiment()
    before = parameters_to_vector(a.model.parameters()).detach().clone()
    rng = a.perturbation_rng.get_state().clone()
    rows = compare_gradients(a.model, *a.train_data, TASKS[0], 0.01, 4)
    assert len(rows) == 2
    assert torch.equal(before, parameters_to_vector(a.model.parameters()))
    assert torch.equal(rng, a.perturbation_rng.get_state())
    exported = a.export_data()
    for experiment in (a, b):
        experiment.train_epoch(TrainConfig(GradientConfig(ALGORITHMS[1], directions=2)))
    assert len(exported["history"]) == 1
    assert torch.equal(parameters_to_vector(a.model.parameters()), parameters_to_vector(b.model.parameters()))
    c = Experiment()
    c.train_epoch(TrainConfig())
    assert torch.equal(a.batch_rng.get_state(), c.batch_rng.get_state())


def test_partial_minibatch_and_test_data_never_used_for_training():
    a = Experiment(DataConfig(TASKS[1], 17, 5))
    b = Experiment(DataConfig(TASKS[1], 17, 50))
    config = TrainConfig(batch_size=8)
    for experiment in (a, b):
        experiment.train_epoch(config)
    assert a.steps == 3
    assert torch.equal(parameters_to_vector(a.model.parameters()), parameters_to_vector(b.model.parameters()))


def test_gui_all_methods_tasks_diagnostics_cancel_and_close(monkeypatch, tmp_path):
    from PySide6.QtWidgets import QApplication, QMessageBox
    from ui import ExperimentWindow

    app = QApplication.instance() or QApplication([])
    errors = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: errors.append(args[-1]))
    window = ExperimentWindow()
    window.show()

    def wait_for_worker():
        deadline = monotonic() + 20
        while window.worker is not None and monotonic() < deadline:
            app.processEvents()
            sleep(0.005)
        assert window.worker is None, "Worker failed to finish"
        assert not errors, errors

    try:
        window.train_samples.setValue(32)
        window.test_samples.setValue(16)
        window.directions.setValue(2)
        window.epochs.setValue(2)
        for task in TASKS:
            window.task.setCurrentText(task)
            for algorithm in ALGORITHMS:
                window.algorithm.setCurrentText(algorithm)
                window.reset_model()
                window.start_worker(False)
                assert not window.train_button.isEnabled()
                wait_for_worker()
                assert window.experiment.epochs == 2
                assert window.train_button.isEnabled()
                for mode in range(3):
                    window.view_mode.setCurrentIndex(mode)
                    window.canvas.draw()
        window.start_worker(True)
        wait_for_worker()
        assert "Kosinus=" in window.diagnostic_label.text()
        window.remember_curve()
        assert len(window.remembered) == 1
        window.grab().save(str(tmp_path / "gui.png"))
        window.epochs.setValue(100000)
        window.start_worker(False)
        window.stop_training()
        wait_for_worker()
        assert window.experiment.epochs < 100002
        window.start_worker(False)
        window.close()
        wait_for_worker()
        assert not window.isVisible()
    finally:
        if window.worker is not None:
            window.stop_training()
            wait_for_worker()
        window.close()
