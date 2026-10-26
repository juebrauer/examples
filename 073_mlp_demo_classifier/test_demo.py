"""Behavior checks for the teaching demo; can run with Qt's offscreen backend."""
import unittest

import torch
from PySide6.QtCore import QEvent, QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication

from mlp_core import build_mlp, evaluate, make_optimizer, predict_classes, train_one_epoch
from mlp_demo import DemoWindow


class CoreTests(unittest.TestCase):
    def test_parser_and_fixed_output(self):
        for n in range(2, 6):
            model = build_mlp("linear 10\nrelu\ndropout 0.1\nlinear 5\nlogistic", n)
            self.assertEqual(tuple(model(torch.zeros(7, 2)).shape), (7, n))
        for invalid in ("linear 0", "linear x", "dropout 1", "dropout nan", "relu 5", "unknown"):
            with self.assertRaises(ValueError):
                build_mlp(invalid, 3)

    def test_learns_and_predicts_without_dropout_noise(self):
        torch.manual_seed(7)
        model = build_mlp("linear 12\nrelu\ndropout 0.1", 2)
        x = torch.tensor([[-1., -1.], [-0.8, -0.9], [1., 1.], [0.8, 0.9]])
        y = torch.tensor([0, 0, 1, 1])
        optimizer = make_optimizer(model, "Adam", 0.03)
        before, _ = evaluate(model, x, y)
        for _ in range(80):
            train_one_epoch(model, optimizer, x, y)
        after, accuracy = evaluate(model, x, y)
        self.assertLess(after, before)
        self.assertEqual(accuracy, 1.0)
        self.assertTrue(torch.equal(predict_classes(model, x), predict_classes(model, x)))


class UiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_draw_train_continue_and_reset(self):
        window = DemoWindow()
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


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
