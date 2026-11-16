"""Mikrofon und Lautsprecher: PCM16, Mono, 24 kHz. Kein Qt, kein OpenAI-Code."""

import asyncio
from collections import deque
from threading import Lock
from time import monotonic

import sounddevice as sd

SAMPLE_RATE = 24000
BLOCK_SIZE = 480  # 20 ms
ECHO_TAIL_SECONDS = 0.5  # Nachhall nach der tatsächlich hörbaren Ausgabe abwarten.


def audio_devices():
    """Geräte für die beiden Auswahlfelder der UI."""
    devices = sd.query_devices()
    inputs = [(i, d["name"]) for i, d in enumerate(devices) if d["max_input_channels"]]
    outputs = [(i, d["name"]) for i, d in enumerate(devices) if d["max_output_channels"]]
    return inputs, outputs


class AudioIO:
    def __init__(self, loop, on_play, input_device=None, output_device=None):
        self.loop = loop
        self.on_play = on_play
        self.input_device = input_device
        self.output_device = output_device
        self.microphone = asyncio.Queue(maxsize=50)
        self.muted = False
        self.closed = False
        self._output = deque()
        self._lock = Lock()
        self._generation = 0
        self._echo_until = 0.0
        self.input_stream = None
        self.output_stream = None

    def start(self):
        try:
            self.output_stream = sd.RawOutputStream(
                device=self.output_device, samplerate=SAMPLE_RATE, channels=1,
                dtype="int16", blocksize=BLOCK_SIZE, callback=self._speaker,
            )
            self.input_stream = sd.RawInputStream(
                device=self.input_device, samplerate=SAMPLE_RATE, channels=1,
                dtype="int16", blocksize=BLOCK_SIZE, callback=self._microphone,
            )
            self.output_stream.start()
            self.input_stream.start()
        except Exception:
            self.close()
            raise

    def _microphone(self, data, frames, timing, status):
        if self.accepting_input:
            self.loop.call_soon_threadsafe(self._enqueue_input, bytes(data))

    @property
    def accepting_input(self):
        # Halbduplex: Lautsprecher dürfen keine neue Benutzerrunde auslösen.
        with self._lock:
            return (not self.closed and not self.muted and not self._output
                    and monotonic() >= self._echo_until)

    def _enqueue_input(self, data):
        if not self.accepting_input:
            return
        if self.microphone.full():
            self.microphone.get_nowait()  # Bei Netzverzögerung keine alten Sekunden nachsenden.
        self.microphone.put_nowait(data)

    def set_muted(self, muted):
        self.muted = muted
        while not self.microphone.empty():
            self.microphone.get_nowait()

    def play(self, data, item_id, content_index):
        with self._lock:
            self._output.append((data, item_id, content_index))
        # Diese Methode läuft im Event-Loop: alte Mikrofonblöcke verwerfen,
        # bevor der Sender sie während der Antwort nachreichen kann.
        while not self.microphone.empty():
            self.microphone.get_nowait()

    def clear_output(self):
        with self._lock:
            self._output.clear()
            self._generation += 1

    def _speaker(self, outdata, frames, timing, status):
        result = bytearray(len(outdata))  # Stille, wenn gerade keine Antwort läuft.
        played = []
        offset = 0
        with self._lock:
            generation = self._generation
            while self._output and offset < len(result):
                data, item_id, index = self._output.popleft()
                count = min(len(data), len(result) - offset)
                result[offset:offset + count] = data[:count]
                if count < len(data):
                    self._output.appendleft((data[count:], item_id, index))
                # Zeitpunkt, zu dem diese Samples voraussichtlich hörbar waren.
                delay = max(0, timing.outputBufferDacTime - timing.currentTime)
                played.append((item_id, index, count, delay + (offset + count) / 2 / SAMPLE_RATE))
                offset += count
            if offset:
                # Der Puffer kann schon leer sein, während das Gerät noch spielt.
                self._echo_until = max(
                    self._echo_until,
                    monotonic() + delay + offset / 2 / SAMPLE_RATE + ECHO_TAIL_SECONDS,
                )
        outdata[:] = bytes(result)
        for item_id, index, count, delay in played:
            self.loop.call_soon_threadsafe(
                self._schedule_progress, generation, delay, item_id, index, count,
            )

    def _schedule_progress(self, generation, delay, item_id, index, count):
        self.loop.call_later(delay, self._report_progress, generation, item_id, index, count)

    def _report_progress(self, generation, item_id, index, count):
        if not self.closed and generation == self._generation:
            self.on_play(item_id, index, count / 2 / SAMPLE_RATE * 1000)

    def close(self):
        self.closed = True
        self.clear_output()
        for stream in (self.input_stream, self.output_stream):
            if stream is not None:
                try:
                    stream.abort()
                except sd.PortAudioError:
                    pass
                try:
                    stream.close()
                except sd.PortAudioError:
                    pass
