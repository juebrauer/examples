"""Behavior checks for the teaching demo; can run with Qt's offscreen backend."""
import unittest

import numpy as np
import torch
from PySide6.QtCore import QEvent, QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication

from datasets import DATASETS, load_dataset
from mlp_core import (build_mlp, classification_loss, evaluate, make_optimizer,
                      predict_classes, train_one_epoch)
from mlp_demo import DemoWindow


class CoreTests(unittest.TestCase):
    def test_fixed_datasets_are_independent_of_other_randomness(self):
        self.assertEqual(len(DATASETS), 10)
        for name in DATASETS:
            x, y, classes = load_dataset(name)
            np.random.seed(100)
            np.random.random(100)
            torch.rand(100)
            again_x, again_y, again_classes = load_dataset(name)
            np.testing.assert_array_equal(x, again_x)
            np.testing.assert_array_equal(y, again_y)
            self.assertEqual(classes, again_classes)
            self.assertEqual(x.shape, (300, 2))
            self.assertTrue(np.isfinite(x).all())
            self.assertTrue((np.abs(x) <= 1).all())
            self.assertEqual(set(y), set(range(classes)))

    def test_mse_uses_probabilities_and_one_hot_targets(self):
        logits = torch.tensor([[2., -1., 0.], [-2., 1., 3.]], requires_grad=True)
        labels = torch.tensor([0, 2])
        target = torch.tensor([[1., 0., 0.], [0., 0., 1.]])
        expected = ((logits.softmax(1) - target) ** 2).mean()
        loss = classification_loss(logits, labels, "MSELoss")
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad.abs().sum().item(), 0)
        with self.assertRaises(ValueError):
            classification_loss(logits, labels, "unknown")

    def test_parser_and_fixed_output(self):
        for n in range(2, 6):
            model = build_mlp("linear 10\nrelu\ndropout 0.1\nlinear 5\nlogistic", n)
            self.assertEqual(tuple(model(torch.zeros(7, 2)).shape), (7, n))
        for invalid in ("linear 0", "linear x", "dropout 1", "dropout nan", "relu 5", "unknown"):
            with self.assertRaises(ValueError):
                build_mlp(invalid, 3)

    def test_learns_and_predicts_without_dropout_noise(self):
        for loss in ("CrossEntropyLoss", "MSELoss"):
            with self.subTest(loss=loss):
                self.check_learning(loss)

    def check_learning(self, loss):
        torch.manual_seed(7)
        model = build_mlp("linear 12\nrelu\ndropout 0.1", 2)
        x = torch.tensor([[-1., -1.], [-0.8, -0.9], [1., 1.], [0.8, 0.9]])
        y = torch.tensor([0, 0, 1, 1])
        optimizer = make_optimizer(model, "Adam", 0.03)
        before, _ = evaluate(model, x, y, loss)
        for _ in range(80):
            train_one_epoch(model, optimizer, x, y, loss)
        after, accuracy = evaluate(model, x, y, loss)
        self.assertLess(after, before)
        self.assertEqual(accuracy, 1.0)
        self.assertTrue(torch.equal(predict_classes(model, x), predict_classes(model, x)))


class UiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_draw_train_continue_and_reset(self):
        window = DemoWindow()
        window.clear_points()
        window.show()
        self.app.processEvents()
        rect = window.canvas.plot_rect()
        QTest.mouseClick(window.canvas, Qt.MouseButton.LeftButton,
                         pos=QPoint(int(rect.left() + rect.width() * 0.2), int(rect.top() + rect.height() * 0.2)))
        self.assertEqual(len(window.canvas.points), 1)
        self.assertAlmostEqual(window.canvas.points[0][0], -0.6, places=2)
        self.assertAlmostEqual(window.canvas.points[0][1], 0.6, places=2)
        window.selected_class.setCurrentIndex(1)
        QTest.mouseClick(window.canvas, Qt.MouseButton.RightButton,
                         pos=QPoint(int(rect.left() + rect.width() * 0.8), int(rect.top() + rect.height() * 0.8)))
        self.assertEqual(len(window.canvas.points), 11)
        self.assertEqual(window.canvas.labels.count(1), 10)
        self.assertEqual(window.epochs.value(), 1)
        QTest.keyPress(window.epochs, Qt.Key.Key_Space)
        self.assertTrue(window.timer.isActive())
        window.training_step()
        self.assertFalse(window.timer.isActive())
        QTest.keyRelease(window.epochs, Qt.Key.Key_Space)
        self.assertFalse(window.timer.isActive())
        self.assertEqual(window.epochs_trained, 1)
        optimizer = window.optimizer
        window.epochs.setValue(3)
        window.start_training()
        for _ in range(3):
            window.training_step()
        self.assertEqual(window.epochs_trained, 4)
        self.assertIs(window.optimizer, optimizer)
        self.assertEqual([row[0] for row in window.history], [0, 1, 2, 3, 4])
        loss, accuracy = evaluate(window.model, *window.tensors())
        self.assertAlmostEqual(window.history[-1][1], loss)
        self.assertAlmostEqual(window.history[-1][2], 100 * (1 - accuracy))
        window.regions.setFocus()
        checked = window.regions.isChecked()
        QTest.keyPress(window.regions, Qt.Key.Key_Space)
        self.assertEqual(window.regions.isChecked(), checked)
        self.assertTrue(window.timer.isActive())
        QTest.keyRelease(window.regions, Qt.Key.Key_Space)
        self.assertTrue(window.timer.isActive())  # Releasing must not interrupt N epochs.
        for _ in range(3):
            window.training_step()
        self.assertEqual(window.epochs_trained, 7)
        self.assertFalse(window.timer.isActive())
        # Holding the key must not start a second batch after the first completes.
        repeat = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Space,
                           Qt.KeyboardModifier.NoModifier, " ", True, 1)
        self.app.sendEvent(window.regions, repeat)
        self.assertFalse(window.timer.isActive())
        self.assertEqual(window.regions.isChecked(), checked)
        self.assertFalse(window.canvas.background.isNull())
        window.num_classes.setValue(5)
        self.assertIsNone(window.model)
        self.assertEqual(window.history, [])
        window.reset_model()
        self.assertEqual(window.model[-1].out_features, 5)
        window.selected_class.setCurrentIndex(4)
        QTest.mouseClick(window.canvas, Qt.MouseButton.LeftButton, pos=rect.center().toPoint())
        window.num_classes.setValue(2)
        self.assertEqual(window.num_classes.value(), 5)
        self.assertEqual(len(window.canvas.points), 12)
        window.reset_model()
        window.epochs.setValue(100)
        window.start_training()
        window.training_step()
        window.stop_training()
        self.assertFalse(window.timer.isActive())
        self.assertTrue(window.train_button.isEnabled())
        window.grab().save('/tmp/mlp_demo_smoke.png')
        window.clear_points()
        self.assertEqual(len(window.canvas.points), 0)
        self.assertEqual(window.history, [])
        before = window.architecture.toPlainText().count("\n")
        QTest.keyClick(window.architecture, Qt.Key.Key_Return)
        self.assertEqual(window.architecture.toPlainText().count("\n"), before + 1)
        self.assertFalse(window.timer.isActive())
        before_text = window.architecture.toPlainText()
        QTest.keyClick(window.architecture, Qt.Key.Key_Space)
        self.assertEqual(len(window.architecture.toPlainText()), len(before_text) + 1)
        self.assertFalse(window.timer.isActive())
        window.close()

    def test_presets_loss_switch_and_reproducible_training(self):
        window = DemoWindow()
        try:
            self.assertEqual(len(window.canvas.points), 300)
            self.assertEqual(window.dataset_choice.count(), 11)
            for name in DATASETS:
                window.dataset_choice.setCurrentText(name)
                self.assertEqual(window.num_classes.value(), load_dataset(name)[2])
                self.assertEqual(window.model[-1].out_features, window.num_classes.value())
                self.assertFalse(window.num_classes.isEnabled())
            initial = {k: v.clone() for k, v in window.model.state_dict().items()}
            original_points = list(window.canvas.points)

            def train_three():
                window.epochs.setValue(3)
                window.start_training()
                self.assertFalse(window.loss_choice.isEnabled())
                self.assertFalse(window.model_seed.isEnabled())
                for _ in range(3):
                    window.training_step()
                self.assertFalse(window.num_classes.isEnabled())
                return list(window.history), {k: v.clone() for k, v in window.model.state_dict().items()}

            ce_history, _ = train_three()
            for loss in ("MSELoss", "CrossEntropyLoss", "MSELoss"):
                window.loss_choice.setCurrentText(loss)
                self.assertEqual(window.epochs_trained, 0)
                self.assertEqual(window.history, [])
                self.assertEqual(len(window.optimizer.state), 0)
                for k, v in window.model.state_dict().items():
                    self.assertTrue(torch.equal(initial[k], v))
                history, weights = train_three()
                measured, _ = evaluate(window.model, *window.tensors(), loss)
                self.assertEqual(history[-1][1], measured)
                if loss == "CrossEntropyLoss":
                    self.assertEqual(history, ce_history)
                window.reset_button.click()
                repeat_history, repeat_weights = train_three()
                self.assertEqual(history, repeat_history)
                for k in weights:
                    self.assertTrue(torch.equal(weights[k], repeat_weights[k]))

            window.model_seed.setValue(43)
            self.assertTrue(any(not torch.equal(v, initial[k])
                                for k, v in window.model.state_dict().items()))
            window.canvas.points[0] = [0., 0.]
            window.reload_button.click()
            self.assertEqual(window.canvas.points, original_points)
            self.assertEqual(window.epochs_trained, 0)
            window.grab().save('/tmp/mlp_demo_presets.png')
        finally:
            window.close()


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
