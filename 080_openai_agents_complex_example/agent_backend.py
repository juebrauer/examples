"""Der Lehrteil: Tools → RealtimeAgent → RealtimeRunner → Session-Ereignisse.

Kein Qt-Import. Die UI erhält nur einfache Ereignis-Dictionaries über emit().
Der Agent entscheidet selbst, welche Funktion er mit welchen Parametern aufruft.
"""

import asyncio
from contextlib import suppress
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from agents import function_tool, set_tracing_disabled
from agents.realtime import RealtimeAgent, RealtimeRunner
from agents.realtime.model import RealtimePlaybackTracker
from agents.realtime.model_inputs import RealtimeModelSendRawMessage

from audio_io import AudioIO
from bestelldaten import ARTIKEL, BestellDaten, ROOT, create_database

MODEL = "gpt-realtime-2.1"


def create_agent(data: BestellDaten) -> RealtimeAgent:
    # 1. Diese Liste beschreibt die verfügbaren Fähigkeiten.
    #    Es gibt KEIN if 'Diagramm' in frage: ... in unserem Programm!
    tools = [
        function_tool(data.tabellen_anzeigen),
        function_tool(data.bestellungen_auswerten),
        function_tool(data.balkendiagramm_erstellen),
    ]
    return RealtimeAgent(
        name="Bestellanalyst",
        instructions=(
            "Sprich natürlich, kurz und auf Deutsch. Du analysierst fiktive Bestellungen. "
            "Nutze für Datenfragen deine Werkzeuge; erfinde keine Zahlen. "
            "Erkunde bei der ersten Datenfrage Tabellen und Zeitraum mit tabellen_anzeigen. "
            "Eine Tabellenzeile ist eine Bestellung, menge ist die Stückzahl. Umsatz ist "
            "Menge mal Einzelpreis und wird in EUR ausgegeben. Unterscheide diese Kennzahlen. "
            f"Artikel: {', '.join(ARTIKEL)}. Filter sind optional; ohne Filter gilt alles. "
            "Monat bedeutet Jahr-Monat. Nutze den Gesprächskontext für Folgefragen. "
            "Für Diagramme nutze balkendiagramm_erstellen. Das Bild erscheint automatisch "
            "in der Oberfläche. Erkläre das Ergebnis; lies keine Dateipfade vor. "
            "Bei leeren Ergebnissen und Werkzeugfehlern erkläre die Ursache."
        ),
        tools=tools,
    )


def create_runner(data: BestellDaten) -> RealtimeRunner:
    # 2. Ein Modell verarbeitet Sprache, wählt Tools und antwortet mit Sprache.
    return RealtimeRunner(create_agent(data), config={
        "tracing_disabled": True,
        "model_settings": {
            "model_name": MODEL,
            "output_modalities": ["audio"],
            "tool_choice": "auto",
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "transcription": {"model": "gpt-4o-mini-transcribe", "language": "de"},
                    "noise_reduction": {"type": "far_field"},
                    "turn_detection": {
                        "type": "semantic_vad", "eagerness": "medium",
                        "create_response": True, "interrupt_response": True,
                    },
                },
                "output": {"format": {"type": "audio/pcm", "rate": 24000}, "voice": "marin"},
            },
        },
    })


class RealtimeService:
    """Verbindet SDK und Audio; läuft vollständig außerhalb des UI-Threads."""

    def __init__(self, emit, input_device=None, output_device=None):
        self.emit = emit
        self.input_device = input_device
        self.output_device = output_device
        self.commands = asyncio.Queue()
        self.audio = None
        self.session = None
        self.texts = {}
        self.roles = {}
        self.finals = set()
        self.seen_charts = set()
        self.pending_text_ids = []
        self.chart_cache = None

    async def run(self):
        key_path = ROOT / "openai_api_key.txt"
        try:
            api_key = key_path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError) as error:
            raise ValueError(f"API-Schlüsseldatei nicht lesbar: {key_path}") from error
        if not api_key:
            raise ValueError(f"API-Schlüsseldatei ist leer: {key_path}")
        set_tracing_disabled(True)
        self.emit({"type": "status", "text": "Datenbank vorbereiten …"})
        await asyncio.to_thread(create_database, ROOT / "bestellungen.sqlite3")
        # Diagramme bleiben als Vorschau erhalten, bis die UI den Verlauf schließt.
        self.chart_cache = TemporaryDirectory(prefix="bestellagent-")
        data = BestellDaten(ROOT / "bestellungen.sqlite3", Path(self.chart_cache.name))
        tracker = RealtimePlaybackTracker()
        self.audio = AudioIO(asyncio.get_running_loop(), tracker.on_play_ms,
                             self.input_device, self.output_device)
        self.emit({"type": "status", "text": f"Verbinde mit {MODEL} …"})
        runner = create_runner(data)
        try:
            # 3. Das SDK betreibt die Agentenschleife und führt angeforderte Tools aus.
            async with await runner.run(model_config={
                "api_key": api_key, "playback_tracker": tracker,
            }) as session:
                self.session = session
                self.audio.start()  # Dieser Code läuft bereits im Hintergrundthread.
                self.emit({"type": "connected"})
                tasks = [asyncio.create_task(self._events()),
                         asyncio.create_task(self._send_microphone()),
                         asyncio.create_task(self._commands())]
                try:
                    done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        task.result()
                finally:
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            self.session = None
            self.audio.close()
            self.chart_cache.cleanup()

    async def _send_microphone(self):
        while True:
            pcm = await self.audio.microphone.get()
            if self.audio.accepting_input:
                await self.session.send_audio(pcm)

    async def _commands(self):
        while True:
            command, value = await self.commands.get()
            if command == "mute":
                self.audio.set_muted(value)
                # Keine halbfertige Aufnahme bei einer Sprechpause nachsenden.
                if value:
                    await self.session.model.send_event(RealtimeModelSendRawMessage(
                        message={"type": "input_audio_buffer.clear"},
                    ))
            elif command == "interrupt":
                self.audio.clear_output()
                await self.session.interrupt()
                self.emit({"type": "status", "text": "Antwort gestoppt · Du kannst gleich weitersprechen."})
            elif command == "text":
                self.audio.clear_output()
                await self.session.interrupt()
                local_id = "text-" + uuid4().hex
                self.pending_text_ids.append(local_id)
                self._text(local_id, "user", value, final=True)
                await self.session.send_message(value)

    def _text(self, item_id, role, text, final=False):
        if not text and item_id not in self.texts:
            return
        old = self.texts.get(item_id)
        was_final = item_id in self.finals
        if final:
            self.finals.add(item_id)
        if old == text and was_final == (item_id in self.finals):
            return
        self.texts[item_id] = text
        self.roles[item_id] = role
        self.emit({"type": "message", "id": item_id, "role": role,
                   "text": text, "final": item_id in self.finals})

    def _raw(self, raw):
        """Streaming-Transkripte beider Sprecher; IDs verhindern doppelte Blasen."""
        kind = raw.get("type", "")
        item_id = raw.get("item_id")
        if kind == "input_audio_buffer.speech_started":
            self._text(item_id, "user", "", final=False)
            self.emit({"type": "listening", "id": item_id})
        elif kind == "conversation.item.input_audio_transcription.delta":
            self._text(item_id, "user", self.texts.get(item_id, "") + raw.get("delta", ""))
        elif kind == "conversation.item.input_audio_transcription.completed":
            self._text(item_id, "user", raw.get("transcript", ""), final=True)
        elif kind == "conversation.item.input_audio_transcription.failed":
            self._text(item_id, "user", "[Sprache konnte nicht transkribiert werden]", final=True)
        elif kind in ("response.output_audio_transcript.delta", "response.output_text.delta"):
            self._text(item_id, "assistant", self.texts.get(item_id, "") + raw.get("delta", ""))
        elif kind in ("response.output_audio_transcript.done", "response.output_text.done"):
            self._text(item_id, "assistant", raw.get("transcript", raw.get("text", "")), final=True)
        elif kind in ("conversation.item.created", "conversation.item.added"):
            item = raw.get("item", {})
            if item.get("role") == "user" and self.pending_text_ids:
                if any(c.get("type") == "input_text" for c in item.get("content", [])):
                    local = self.pending_text_ids.pop(0)
                    # Text wurde bereits sofort in der UI angezeigt.
                    self.texts[item["id"]] = self.texts[local]
                    self.finals.add(item["id"])
        elif kind == "response.done":
            response = raw.get("response", {})
            if response.get("status") == "failed":
                self.emit({"type": "error", "text": str(response.get("status_details"))})

    def _tool_result(self, event):
        output = event.output
        # Das SDK gibt dict-Rückgaben je nach Version als JSON-String weiter.
        if isinstance(output, str):
            with suppress(ValueError):
                output = json.loads(output)
        if not isinstance(output, dict) or "datei" not in output:
            return
        path = Path(output["datei"]).resolve()
        if (path.parent == Path(self.chart_cache.name).resolve()
                and path.is_file() and path not in self.seen_charts):
            self.seen_charts.add(path)
            self.emit({"type": "chart", "image": path.read_bytes(),
                       "filename": path.name, "title": "Bestellauswertung"})

    async def _events(self):
        # 4. Ereignisse anzeigen/abspielen; die Werkzeugwahl bleibt beim Modell.
        async for event in self.session:
            if event.type == "raw_model_event" and event.data.type == "raw_server_event":
                self._raw(event.data.data)
            elif event.type == "tool_start":
                self.emit({"type": "tool", "name": event.tool.name, "arguments": event.arguments})
            elif event.type == "tool_end":
                self._tool_result(event)
            elif event.type == "audio":
                self.audio.play(event.audio.data, event.item_id, event.content_index)
            elif event.type == "audio_interrupted":
                self.audio.clear_output()
                self.emit({"type": "interrupted", "id": event.item_id})
            elif event.type == "error":
                self.emit({"type": "error", "text": str(event.error)})
            elif event.type == "raw_model_event" and event.data.type == "exception":
                raise event.data.exception
