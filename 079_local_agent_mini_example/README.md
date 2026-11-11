# Lokaler Agent mit Ollama und OpenAI Agents SDK

Das Mini-Beispiel enthält alle Daten und den gesamten Python-Code in
`agents_mini_demo.py`. Es zeigt denselben Ablauf wie Beispiel 078, verwendet aber
ein lokal installiertes Modell. Kein OpenAI-API-Schlüssel erforderlich.

## Start

1. [Ollama](https://ollama.com) installieren und starten.
2. Das Modell einmal herunterladen: `ollama pull qwen3:8b` (ca. 5 GB).
3. Im Beispielordner ausführen:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 agents_mini_demo.py
```

Falls Ollama noch nicht als Dienst läuft, in einem zweiten Terminal
`ollama serve` starten. Das Modell benötigt ausreichend freien RAM bzw. VRAM;
die Geschwindigkeit hängt von der Hardware ab.

## Ausprobieren

- Wie viele Bestellungen gibt es insgesamt in Bayern?
- Und welcher Artikel wird dort am häufigsten bestellt?
- Vergleiche die Verkäufe von Bayern und Berlin als Balkendiagramm.

`ende` beendet den Chat. Das Diagramm wird als `mini_vergleich.png` neben dem
Skript gespeichert; jeder neue Vergleich ersetzt das vorherige Bild.

## Warum ist das lokal?

`AsyncOpenAI` ist hier ein Client für das OpenAI-kompatible Nachrichtenformat.
Sein `base_url` zeigt ausdrücklich auf `http://localhost:11434/v1`, also Ollama
auf dem eigenen Rechner. `OpenAIChatCompletionsModel` verbindet diesen Client
mit dem Agents SDK. Der Wert `api_key="ollama"` ist nur ein Platzhalter.
SDK-Tracing ist ausgeschaltet.

`Runner` übermittelt die Frage an Ollama, führt die vom Modell angeforderten
Python-Werkzeuge aus und schickt deren Rückgabewerte wieder an Ollama. Das lokale
Modell formuliert die Antwort. In dieser Konfiguration werden keine Chat- oder
Werkzeugdaten an OpenAI gesendet. Die Installation und der Modelldownload benötigen
Internet; danach können diese Werkzeuge mit dem lokalen Modell offline arbeiten.

Die Konstante `model="qwen3:8b"` kann gegen ein anderes **lokal installiertes
Modell mit Tool-Calling-Unterstützung** ausgetauscht werden. `/no_think` in den
Instructions verkürzt bei Qwen3 die Antworten durch Abschalten des Denkmodus.
Ein Cloud-Modell würde die lokale Verarbeitung aufheben.

Quellen: [OpenAI Agents SDK: Modellanbieter](https://developers.openai.com/api/docs/guides/agents/models),
[Ollama: OpenAI-Kompatibilität](https://docs.ollama.com/api/openai-compatibility),
[Ollama: Tool Calling](https://docs.ollama.com/capabilities/tool-calling).
