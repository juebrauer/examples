"""Ohne API und ohne Audiogeräte: SDK-Toolschleife, Events, Audio und Qt-Ansicht."""

import asyncio
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from agents.realtime import RealtimeRunner
from agents.realtime.model_events import RealtimeModelToolCallEvent
from agents.realtime.testing import ScriptedRealtimeModel
from PySide6.QtWidgets import QApplication

from agent_backend import RealtimeService, create_agent
from audio_io import AudioIO
from bestelldaten import BestellDaten, create_database
from chat_ui import ChatWindow, STYLE, TypewriterLabel


class TranscriptTests(unittest.TestCase):
    def test_streaming_und_korrigiertes_finales_transkript(self):
        events = []
        service = RealtimeService(events.append)
        for raw in (
            {"type": "input_audio_buffer.speech_started", "item_id": "u1"},
            {"type": "conversation.item.input_audio_transcription.delta", "item_id": "u1", "delta": "Zeige Bayer"},
            {"type": "conversation.item.input_audio_transcription.completed", "item_id": "u1", "transcript": "Zeige Bayern."},
            {"type": "response.output_audio_transcript.delta", "item_id": "a1", "delta": "Hier "},
            {"type": "response.output_audio_transcript.delta", "item_id": "a1", "delta": "sind die Daten."},
            {"type": "response.output_audio_transcript.done", "item_id": "a1", "transcript": "Hier sind die Daten."},
        ):
            service._raw(raw)
        self.assertEqual(service.texts["u1"], "Zeige Bayern.")
        self.assertEqual(service.texts["a1"], "Hier sind die Daten.")
        self.assertEqual(service.finals, {"u1", "a1"})
        self.assertEqual(events[-1]["final"], True)
        self.assertEqual({e["id"] for e in events}, {"u1", "a1"})


class SessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_service_text_stummschalten_und_sauberes_beenden(self):
        class FakeAudio:
            def __init__(self, *args):
                self.microphone = asyncio.Queue()
                self.closed = False
                self.muted = False
            @property
            def accepting_input(self):
                return not self.muted and not self.closed
            def start(self):
                pass
            def clear_output(self):
                pass
            def set_muted(self, value):
                self.muted = value
            def close(self):
                self.closed = True

        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "openai_api_key.txt").write_text("nur-test-kein-echter-schluessel")
            model = ScriptedRealtimeModel(strict=False)
            runner = RealtimeRunner(create_agent(BestellDaten(root / "db", root / "charts")),
                                    model=model, config={"tracing_disabled": True})
            events = []
            service = RealtimeService(events.append)
            with patch("agent_backend.ROOT", root), patch("agent_backend.AudioIO", FakeAudio), \
                    patch("agent_backend.create_runner", return_value=runner), \
                    patch("agent_backend.create_database"):
                task = asyncio.create_task(service.run())
                for _ in range(100):
                    if any(e["type"] == "connected" for e in events):
                        break
                    await asyncio.sleep(0.02)
                self.assertTrue(any(e["type"] == "connected" for e in events))
                await service.commands.put(("text", "Zeige Bayern"))
                await service.commands.put(("mute", True))
                for _ in range(100):
                    if service.audio.muted:
                        break
                    await asyncio.sleep(0.02)
                self.assertTrue(service.audio.muted)
                self.assertTrue(any(e.get("text") == "Zeige Bayern" for e in events))
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertTrue(service.audio.closed)
                self.assertTrue(model.closed)
                self.assertIsNone(service.session)
                self.assertFalse(Path(service.chart_cache.name).exists())

    async def test_sdk_fuehrt_angefordertes_tool_aus_und_liefert_diagramm(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            create_database(root / "orders.sqlite3")
            data = BestellDaten(root / "orders.sqlite3", root / "charts")
            args = json.dumps(dict(gruppierung="jahr", kennzahl="bestellungen",
                                   bundesland=None, artikel=None, jahr=None))
            # Anstelle eines kostenpflichtigen Modells liefert der Testtransport
            # eine echte SDK-Werkzeuganforderung. Das SDK muss sie selbst ausführen.
            model = ScriptedRealtimeModel(strict=False, connect_events=[
                RealtimeModelToolCallEvent(name="balkendiagramm_erstellen", call_id="c1", arguments=args),
            ])
            runner = RealtimeRunner(create_agent(data), model=model,
                                    config={"tracing_disabled": True})
            seen = []
            service = RealtimeService(seen.append)
            service.chart_cache = SimpleNamespace(name=str(root / "charts"))
            async with await runner.run() as session:
                async def collect():
                    async for event in session:
                        if event.type == "tool_start":
                            seen.append({"type": "tool", "name": event.tool.name})
                        elif event.type == "tool_end":
                            service._tool_result(event)
                            return
                # Periodische Schleifenaktivität funktioniert auch in Test-Sandboxes.
                task = asyncio.create_task(collect())
                for _ in range(200):
                    if task.done():
                        break
                    await asyncio.sleep(0.05)
                await asyncio.wait_for(task, 1)
            self.assertEqual([e["type"] for e in seen], ["tool", "chart"])
            self.assertTrue(seen[1]["image"].startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertTrue(model.closed)

    async def test_audio_unterbrechung_loescht_buffer_und_fortschritt(self):
        played = []
        audio = AudioIO(asyncio.get_running_loop(), lambda *args: played.append(args))
        audio.play(b"\x01\x00" * 480, "a1", 0)
        output = bytearray(960)
        timing = SimpleNamespace(outputBufferDacTime=0, currentTime=0)
        audio._speaker(output, 480, timing, None)
        self.assertEqual(output, b"\x01\x00" * 480)
        audio.clear_output()
        await asyncio.sleep(0.05)
        self.assertEqual(played, [])
        audio._speaker(output, 480, timing, None)
        self.assertEqual(output, bytes(960))
        audio._enqueue_input(b"test")
        audio.set_muted(True)
        self.assertTrue(audio.microphone.empty())
        audio.close()


class UiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.app.setStyleSheet(STYLE)

    def test_typewriter_korrektur_und_blockcursor(self):
        label = TypewriterLabel()
        label.update_text("Bayer")
        self.assertIn("█", label.text())
        label.update_text("Bayern!", True)
        for _ in range(10):
            label.tick()
        self.assertEqual(label.text(), "Bayern!")
        self.assertFalse(label.timer.isActive())
        label.close()

    def test_verlauf_rollen_diagramm_speichern_und_fenstergroesse(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            create_database(root / "orders.sqlite3")
            result = BestellDaten(root / "orders.sqlite3", root / "preview").balkendiagramm_erstellen("jahr")
            png = Path(result["datei"]).read_bytes()
            with patch("chat_ui.audio_devices", return_value=([], [])):
                window = ChatWindow()
            window.show()
            window.handle_event({"type": "message", "id": "u1", "role": "user", "text": "Zeige ein Diagramm.", "final": True})
            window.handle_event({"type": "tool", "name": "balkendiagramm_erstellen", "arguments": "{}"})
            window.handle_event({"type": "chart", "image": png, "filename": "vergleich.png", "title": "Bestellungen pro Jahr"})
            card = window.rows[-1][1]
            with patch("chat_ui.ROOT", root):
                card.save()
            self.assertEqual((root / "diagramme" / "vergleich.png").read_bytes(), png)
            self.assertEqual(window.messages["u1"].role, "user")
            self.assertEqual(window.rows[-2][1].objectName(), "tool")
            for width in (1180, 760):
                window.resize(width, 860)
                self.app.processEvents()
                self.assertLessEqual(card.width(), window.canvas.width())
            window.close()


if __name__ == "__main__":
    unittest.main()
