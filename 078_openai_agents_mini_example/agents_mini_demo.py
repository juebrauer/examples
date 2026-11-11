"""Mini-Demo: Frage → Agent → Python-Werkzeug → Daten → Antwort.

Installation: pip install openai-agents matplotlib
Start:        python3 agents_mini_demo.py
API-Schlüssel: openai_api_key.txt neben dieser Datei (API-Aufrufe kosten Geld).
"""

from pathlib import Path
from agents import Agent, Runner, function_tool, set_default_openai_key, set_tracing_disabled


# 1. Unsere Daten liegen hier in Python. Der Agent sieht sie NICHT automatisch.
#    Das könnten ebenso Daten aus einer Datei, Datenbank oder Web-API sein.
BESTELLUNGEN = {
    "Bayern": {"Notebook": 120, "Monitor": 80, "Maus": 200},
    "Berlin": {"Notebook": 90, "Monitor": 60, "Maus": 150},
    "Hessen": {"Notebook": 70, "Monitor": 50, "Maus": 110},
}


# 2. Der Dekorator macht die Funktion zu einem Werkzeug für den Agenten.
#    Name, Docstring und Typen erklären dem Modell, was es damit tun kann.
@function_tool
def bestellungen_lesen(bundesland: str) -> dict:
    """Liefert fiktive Bestellzahlen pro Artikel für Bayern, Berlin oder Hessen."""
    daten = BESTELLUNGEN.get(bundesland)
    ergebnis = daten if daten is not None else {"fehler": "Unbekanntes Bundesland"}
    print(f"\n  [Python-Werkzeug] bestellungen_lesen({bundesland!r})")
    print(f"  [Daten an den Agenten] {ergebnis}\n")
    return ergebnis


@function_tool
def verkaeufe_vergleichen(bundesland1: str, bundesland2: str) -> dict:
    """Vergleicht zwei Bundesländer (Bayern, Berlin, Hessen) pro Artikel als PNG-Balkendiagramm."""
    print(f"\n  [Python-Werkzeug] verkaeufe_vergleichen({bundesland1!r}, {bundesland2!r})")
    if bundesland1 not in BESTELLUNGEN or bundesland2 not in BESTELLUNGEN:
        return {"fehler": "Bitte zwei Bundesländer aus Bayern, Berlin und Hessen wählen."}
    if bundesland1 == bundesland2:
        return {"fehler": "Bitte zwei unterschiedliche Bundesländer wählen."}

    import matplotlib
    matplotlib.use("Agg")  # Speichert Bilder auch ohne Desktop-Fenster.
    import matplotlib.pyplot as plt

    # Die Funktion holt die Daten selbst; das Modell muss keine Zahlen übergeben.
    artikel = list(BESTELLUNGEN[bundesland1])
    positionen = list(range(len(artikel)))
    fig, ax = plt.subplots(figsize=(8, 5))
    for land, versatz in ((bundesland1, -0.2), (bundesland2, 0.2)):
        werte = [BESTELLUNGEN[land][a] for a in artikel]
        balken = ax.bar([x + versatz for x in positionen], werte, width=0.4, label=land)
        ax.bar_label(balken)
    ax.set_xticks(positionen, artikel)
    ax.set_ylabel("Anzahl Bestellungen")
    ax.set_title(f"{bundesland1} und {bundesland2} – fiktive Verkäufe")
    ax.legend()
    fig.tight_layout()
    datei = Path(__file__).with_name("mini_vergleich.png").resolve()
    fig.savefig(datei)  # Ein neuer Vergleich ersetzt das vorherige Bild.
    plt.close(fig)
    print(f"  [Diagramm gespeichert] {datei}\n")
    return {"datei": str(datei), "daten": {
        bundesland1: BESTELLUNGEN[bundesland1], bundesland2: BESTELLUNGEN[bundesland2],
    }}


def main():
    key_datei = Path(__file__).with_name("openai_api_key.txt")
    if not key_datei.is_file() or not key_datei.read_text(encoding="utf-8").strip():
        raise SystemExit("Bitte den API-Schlüssel in openai_api_key.txt neben dem Skript eintragen.")
    set_default_openai_key(key_datei.read_text(encoding="utf-8").strip())
    set_tracing_disabled(True)

    # 3. Der Agent kennt die Beschreibung des Werkzeugs und kann es anfordern.
    agent = Agent(
        name="Mini-Bestellagent",
        model="gpt-4.1-mini",
        instructions=(
            "Antworte kurz auf Deutsch. Du analysierst fiktive Bestellungen. "
            "Lies Bestellzahlen immer mit bestellungen_lesen; erfinde keine Zahlen. "
            "Verfügbare Bundesländer: Bayern, Berlin, Hessen. "
            "Nutze für Diagrammvergleiche verkaeufe_vergleichen. "
            "Nenne danach den zurückgegebenen Bildpfad. "
            "Erkläre anhand der erhaltenen Daten deine Antwort."
        ),
        tools=[bestellungen_lesen, verkaeufe_vergleichen],
    )

    print("Beispiel: Wie viele Bestellungen gibt es insgesamt in Bayern?")
    print("Folgefrage: Und welcher Artikel wird dort am häufigsten bestellt?")
    print("Diagramm: Vergleiche die Verkäufe von Bayern und Berlin als Balkendiagramm.")
    print("Mit 'ende' beenden. Eingaben und Werkzeugergebnisse gehen an die OpenAI API.\n")
    verlauf = []
    while True:
        frage = input("Du> ").strip()
        if frage.lower() == "ende":
            break
        if not frage:
            continue
        verlauf.append({"role": "user", "content": frage})

        # 4. Runner fragt das Modell, führt angeforderte Werkzeuge lokal aus und
        #    sendet deren Ergebnis zurück. Das Modell formuliert daraus die Antwort.
        ergebnis = Runner.run_sync(agent, verlauf)
        print(f"Agent> {ergebnis.final_output}\n")
        verlauf = ergebnis.to_input_list()  # Kontext für Folgefragen behalten.


if __name__ == "__main__":
    main()
