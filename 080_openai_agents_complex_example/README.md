# Bestellagent mit Sprache und PySide6

Eine einfache Desktop-Oberfläche für natürliche Gespräche über fiktive
Bestellungen. Das Modell **gpt-realtime-2.1** hört das Mikrofon, entscheidet selbst
über Werkzeugaufrufe und antwortet über den Lautsprecher. Kein separater
Textagent und keine Kommandozeilenschalter.

## Installation und Start

Python 3.10+:

```bash
cd 080_openai_agents_complex_example
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 agents_demo.py
```

Lege `openai_api_key.txt` neben dem Skript an: nur der API-Schlüssel, ohne
Anführungszeichen. Er wird ausschließlich aus dieser Datei gelesen und nicht aus
einer Umgebungsvariable. Die Schlüsseldatei wird von Git ignoriert.

Unter Linux benötigt `sounddevice` PortAudio (z.B. Ubuntu/Debian:
`sudo apt install libportaudio2`). Für Qt ist eine grafische Desktop-Sitzung
mit den üblichen Qt-Systembibliotheken erforderlich.

## Bedienung

- Mikrofon und Lautsprecher auswählen, dann **Gespräch starten** anklicken.
  Erst dann werden die Audiogeräte geöffnet und die Verbindung hergestellt.
- Natürlich sprechen; die automatische Sprecherkennung erkennt Sprechpausen.
  Während der KI-Ausgabe und für eine halbe Sekunde Nachhall danach wird das
  Mikrofon automatisch gesperrt. Danach kannst du die nächste Frage stellen.
- **Mikrofon an/aus** schaltet die Aufnahmeübermittlung um. Alternativ kann man
  unten eine Frage tippen und mit Enter senden. Eine neue Textfrage unterbricht
  eine laufende Antwort.
- **Hör auf zu quatschen!** stoppt die laufende Antwort und leert den lokalen
  Audiopuffer. Die Verbindung bleibt bestehen; nach der Nachhallpause kannst du
  weitersprechen.
- KI-Nachrichten stehen links, User-Nachrichten rechts. Beide erscheinen im
  Typewriter-Stil mit Blockcursor. Werkzeugaufrufe samt Parametern erscheinen gelb.
- Ein erzeugtes Diagramm erscheint direkt im Verlauf. **Vergrößern** öffnet eine
  Ansicht mit Scrollen und Zoom. **In diagramme speichern** sichert das PNG in
  `diagramme/` neben dem Skript. Vorher liegt es nur im temporären Vorschauordner.
- **Gespräch beenden** trennt die Verbindung und schließt die Audiogeräte.
  Der sichtbare Verlauf bleibt erhalten. Ein neuer Start eröffnet eine neue
  Modellsitzung. Auch das Schließen des Fensters beendet die aktive Verbindung.

Die automatische Mikrofonsperre verhindert, dass die Lautsprecherausgabe erneut
als Benutzereingabe gesendet wird. Es ist Halbduplex ohne akustische
Echounterdrückung: Unterbrechen per Sprache ist während der Ausgabe gesperrt;
eine Textfrage kann die Antwort weiterhin unterbrechen.

Die User-Texte sind Transkripte und können korrigiert werden oder etwas verzögert
erscheinen. Der Transkriptionsdienst liefert die Anzeige; das Realtime-Modell
versteht die Audioeingabe direkt. Die Typewriter-Animation ist nicht lippensynchron.

Beispiele:

- „Welche Tabellen haben wir und aus welchen Jahren stammen die Daten?“
- „Wie viele Bestellungen gibt es pro Bundesland?“
- „Erstelle dafür ein Balkendiagramm.“
- „Davon bitte nur Notebooks im Jahr 2024.“
- „Zeige den Umsatz pro Artikel in Bayern im Jahr 2025.“

## Für Studierende: Wo passiert was?

| Datei | Aufgabe |
| --- | --- |
| `agents_demo.py` | Startet die Anwendung. |
| `agent_backend.py` | Der eigentliche SDK-Ablauf: Werkzeugliste, Instructions, RealtimeAgent, RealtimeRunner und Session-Ereignisse. Kein Qt-Code. |
| `bestelldaten.py` | SQLite-Erzeugung, lesende Auswertungen und Diagrammerzeugung. Kein OpenAI- oder UI-Code. |
| `audio_io.py` | Mikrofon/Lautsprecher mit PCM16, Mono, 24 kHz. Kein Qt- oder OpenAI-Code. |
| `chat_ui.py` | Darstellung und Bedienung. Ein Hintergrundthread verbindet die UI mit dem asynchronen Backend. |

Der Kern steht in `create_agent()` und `create_runner()` in `agent_backend.py`:

```python
tools = [
    function_tool(data.tabellen_anzeigen),
    function_tool(data.bestellungen_auswerten),
    function_tool(data.balkendiagramm_erstellen),
]
agent = RealtimeAgent(name="Bestellanalyst", instructions=..., tools=tools)
runner = RealtimeRunner(agent, config=...)

async with await runner.run(model_config={"api_key": api_key}) as session:
    # Audio senden und gleichzeitig Ereignisse empfangen.
    # Das SDK führt die vom Modell angeforderten Tools aus.
    ...
```

Es gibt keine Zuordnung wie `if "Diagramm" in frage`. Das Modell erhält die
Toolbeschreibungen und Parameterschemas. Es fordert passende Tools an; das SDK
führt die lokalen Python-Funktionen aus und schickt die Rückgaben zurück. Die UI
beobachtet `tool_start`/`tool_end`, zeigt Aufrufe an und lädt Diagrammvorschauen.
Sie entscheidet nicht, welches Werkzeug verwendet wird.

Die Realtime-Session hält den Gesprächskontext während der Verbindung. Der frühere
CLI-Chat und seine SQLite-Chatpersistenz sind ersetzt; eine alte
`chatverlauf.sqlite3` wird nicht mehr gelesen oder verändert.

## Daten, Schlüssel und Kosten

`bestellungen.sqlite3` wird beim ersten Gespräch automatisch angelegt, wenn sie
fehlt. Bestehende Daten bleiben erhalten. Sie enthält 16 Tabellen (eine je
Bundesland) mit Bestellungen für **2021–2025** und den Artikeln Notebook, Monitor,
Tastatur, Maus, PC. Verteilungen, Ländergewichte und Preise sind frei erfunden.
Eine Zeile ist eine Bestellung eines Artikels mit einer Menge.

Bestellanzahl ist `COUNT(*)`, Stückzahl ist `SUM(menge)`, Umsatz wird in ganzen
Cent berechnet und in Euro ausgegeben. Werkzeuge öffnen SQLite ausschließlich
lesend, verwenden feste Tabellenzuordnungen und binden Filter als SQL-Parameter.

Mikrofon-Audio, eingegebene Texte, Instructions, Werkzeugbeschreibungen und
Werkzeugergebnisse werden an die OpenAI API übertragen. Die künstlich erzeugte
KI-Stimme kommt von dort zurück. Für die Anzeige des gesprochenen User-Texts ist
`gpt-4o-mini-transcribe` als Eingabetranskription aktiviert. Das Sprachmodell bleibt
`gpt-realtime-2.1`. API-Nutzung einschließlich Audio/Transkription verursacht Kosten.
SDK-Tracing ist deaktiviert. Diagramme werden lokal erzeugt; das Modell erhält die
Auswertungszahlen und den temporären Dateipfad, aber nicht automatisch das Bild.

## Prüfungen

```bash
QT_QPA_PLATFORM=offscreen python3 -m unittest -v
```

Die Tests laufen ohne API-Schlüssel und ohne Audiogeräte. Sie prüfen Daten,
Filter und SQL-Summen, die tatsächliche SDK-Toolschleife mit Testtransport,
Transkriptkorrekturen, Audiopuffer bei Unterbrechungen sowie Typewriter,
Diagrammvorschau und Speichern in der Qt-Oberfläche.

Zusätzlich wurde ein echter `gpt-realtime-2.1`-Aufruf mit Werkzeugwahl,
Diagrammerzeugung, Antworttranskript und Audioempfang geprüft. Das ersetzt keinen
Hörtest mit dem Mikrofon und den Lautsprechern am jeweiligen Arbeitsplatz.

## Offizielle Dokumentation

- [GPT-Realtime-2.1](https://developers.openai.com/api/docs/models/gpt-realtime-2.1)
- [Realtime-Architektur](https://developers.openai.com/api/docs/guides/realtime)
- [Voice Agents](https://developers.openai.com/api/docs/guides/voice-agents)
