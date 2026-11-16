"""Echo-Sperre ohne Audiogeräte und ohne API prüfen."""

import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from audio_io import AudioIO


class EchoGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_sperre_bis_geraet_und_nachhall_fertig_sind(self):
        audio = AudioIO(asyncio.get_running_loop(), lambda *args: None)
        with patch("audio_io.monotonic", return_value=10.0) as clock:
            audio._enqueue_input(b"vorher")
            self.assertFalse(audio.microphone.empty())
            audio.play(b"\x01\x00" * 480, "a1", 0)
            self.assertTrue(audio.microphone.empty())
            self.assertFalse(audio.accepting_input)
            audio._enqueue_input(b"echo")
            output = bytearray(960)
            # 100 ms Gerätelatenz + 20 ms Audio + 500 ms Nachhall.
            timing = SimpleNamespace(outputBufferDacTime=10.1, currentTime=10.0)
            audio._speaker(output, 480, timing, None)
            self.assertEqual(output, b"\x01\x00" * 480)
            clock.return_value = 10.3
            audio._microphone(b"echo", 480, None, None)
            audio._enqueue_input(b"verspaeteter callback")
            self.assertTrue(audio.microphone.empty())
            audio.clear_output()  # Unterbrechen darf den Nachhall nicht freigeben.
            self.assertFalse(audio.accepting_input)
            clock.return_value = 10.7
            self.assertTrue(audio.accepting_input)
            audio._enqueue_input(b"naechste frage")
            self.assertEqual(audio.microphone.get_nowait(), b"naechste frage")
            audio.set_muted(True)
            self.assertFalse(audio.accepting_input)
            audio.set_muted(False)
            self.assertTrue(audio.accepting_input)
            audio.close()
            self.assertFalse(audio.accepting_input)


if __name__ == "__main__":
    unittest.main()
