#!/usr/bin/env python3
"""Fiktive SQLite-Daten und lokale Werkzeuge – ohne UI und ohne OpenAI-Aufrufe."""

from contextlib import closing
from datetime import date, timedelta
from pathlib import Path
import random
import sqlite3
from typing import Literal
from uuid import uuid4


ROOT = Path(__file__).resolve().parent
ARTIKEL = ("Notebook", "Monitor", "Tastatur", "Maus", "PC")
# Die Gewichte sind frei erfunden, keine Bevölkerungs- oder Marktdaten.
BUNDESLAENDER = {
    "Baden-Württemberg": ("baden_wuerttemberg", 7),
    "Bayern": ("bayern", 9),
    "Berlin": ("berlin", 4),
    "Brandenburg": ("brandenburg", 3),
    "Bremen": ("bremen", 1),
    "Hamburg": ("hamburg", 3),
    "Hessen": ("hessen", 5),
    "Mecklenburg-Vorpommern": ("mecklenburg_vorpommern", 2),
    "Niedersachsen": ("niedersachsen", 6),
    "Nordrhein-Westfalen": ("nordrhein_westfalen", 12),
    "Rheinland-Pfalz": ("rheinland_pfalz", 4),
    "Saarland": ("saarland", 1),
    "Sachsen": ("sachsen", 4),
    "Sachsen-Anhalt": ("sachsen_anhalt", 2),
    "Schleswig-Holstein": ("schleswig_holstein", 3),
    "Thüringen": ("thueringen", 2),
}
Gruppierung = Literal["bundesland", "artikel", "jahr", "monat"]
Kennzahl = Literal["bestellungen", "stueckzahl", "umsatz"]


def create_database(path: Path, start_year: int = 2021, seed: int = 42) -> bool:
    """Erzeugt fünf vollständige Jahre; eine vorhandene Datei bleibt erhalten."""
    if path.exists():
        return False
    if not 1 <= start_year <= 9994:
        raise ValueError("Das Startjahr muss zwischen 1 und 9994 liegen.")
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    prices = dict(zip(ARTIKEL, (109900, 24900, 5900, 2900, 89900)))
    # Exklusives Anlegen verhindert versehentliches Überschreiben.
    path.touch(exist_ok=False)
    try:
        with closing(sqlite3.connect(path)) as connection, connection:
            for _, (table, weight) in BUNDESLAENDER.items():
                connection.execute(f'''CREATE TABLE "{table}" (
                    id INTEGER PRIMARY KEY,
                    bestelldatum TEXT NOT NULL,
                    artikel TEXT NOT NULL CHECK (
                        artikel IN ('Notebook', 'Monitor', 'Tastatur', 'Maus', 'PC')),
                    menge INTEGER NOT NULL CHECK (menge > 0),
                    einzelpreis_cent INTEGER NOT NULL CHECK (einzelpreis_cent > 0)
                )''')
                rows = []
                for year in range(start_year, start_year + 5):
                    for month in range(1, 13):
                        first = date(year, month, 1)
                        following = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
                        # Leichtes Wachstum und mehr Bestellungen im November/Dezember.
                        count = round(rng.randint(12, 22) * weight
                                      * (1 + 0.08 * (year - start_year))
                                      * (1.4 if month >= 11 else 1.0))
                        for _ in range(count):
                            item = rng.choices(ARTIKEL, weights=(3, 3, 2, 4, 2))[0]
                            day = first + timedelta(days=rng.randrange((following - first).days))
                            price = round(prices[item] * rng.uniform(0.85, 1.15))
                            rows.append((day.isoformat(), item, rng.randint(1, 5), price))
                connection.executemany(
                    f'INSERT INTO "{table}" (bestelldatum, artikel, menge, einzelpreis_cent) '
                    'VALUES (?, ?, ?, ?)', rows,
                )
                connection.execute(f'CREATE INDEX "idx_{table}_datum" ON "{table}" (bestelldatum)')
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return True


class BestellDaten:
    """Lokale Werkzeuge: ausschließlich lesende, parametrisierte SQL-Abfragen."""

    def __init__(self, db_path: Path, chart_dir: Path):
        self.db_path = db_path.resolve()
        self.chart_dir = chart_dir.resolve()

    def connect(self):
        connection = sqlite3.connect(self.db_path.as_uri() + "?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        return connection

    def tabellen_anzeigen(self) -> dict:
        """Listet die tatsächlichen Tabellen, Spalten und Datenzeiträume auf."""
        with closing(self.connect()) as connection:
            actual = {r[0] for r in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )}
            tables = []
            for state, (table, _) in BUNDESLAENDER.items():
                if table not in actual:
                    raise ValueError(f"Die erwartete Tabelle {table} fehlt in {self.db_path}.")
                row = connection.execute(
                    f'SELECT COUNT(*) AS bestellungen, MIN(bestelldatum) AS von, '
                    f'MAX(bestelldatum) AS bis FROM "{table}"'
                ).fetchone()
                columns = [dict(r) for r in connection.execute(f'PRAGMA table_info("{table}")')]
                tables.append({"bundesland": state, "tabelle": table, **dict(row), "spalten": columns})
        return {"datenbank": str(self.db_path), "tabellen": tables, "fiktive_daten": True}

    def bestellungen_auswerten(
        self, gruppierung: Gruppierung = "bundesland", kennzahl: Kennzahl = "bestellungen",
        bundesland: str | None = None, artikel: str | None = None,
        jahr: int | None = None,
    ) -> dict:
        """Aggregiert Bestellanzahl, Stückzahl oder Umsatz mit optionalen Filtern.

        Eine Tabellenzeile ist eine Bestellung eines Artikels mit einer Menge.
        Bundesland und Artikel müssen exakt wie in tabellen_anzeigen bzw. ARTIKEL heißen.
        """
        if gruppierung not in ("bundesland", "artikel", "jahr", "monat"):
            raise ValueError("Unbekannte Gruppierung.")
        if kennzahl not in ("bestellungen", "stueckzahl", "umsatz"):
            raise ValueError("Unbekannte Kennzahl.")
        if bundesland is not None and bundesland not in BUNDESLAENDER:
            raise ValueError(f"Bundesland muss eines von {list(BUNDESLAENDER)} sein.")
        if artikel is not None and artikel not in ARTIKEL:
            raise ValueError(f"Artikel muss einer von {ARTIKEL} sein.")
        if jahr is not None and not 1 <= jahr <= 9999:
            raise ValueError("Ungültiges Jahr.")
        # Tabellennamen stammen ausschließlich aus der festen Zuordnung oben.
        unions, params = [], []
        for state, (table, _) in BUNDESLAENDER.items():
            if bundesland is None or state == bundesland:
                unions.append(f'SELECT ? AS bundesland, bestelldatum, artikel, menge, '
                              f'einzelpreis_cent FROM "{table}"')
                params.append(state)
        conditions = []
        if artikel is not None:
            conditions.append("artikel = ?")
            params.append(artikel)
        if jahr is not None:
            conditions.append("substr(bestelldatum, 1, 4) = ?")
            params.append(f"{jahr:04d}")
        group = {"bundesland": "bundesland", "artikel": "artikel",
                 "jahr": "substr(bestelldatum, 1, 4)",
                 "monat": "substr(bestelldatum, 1, 7)"}[gruppierung]
        aggregate = {"bestellungen": "COUNT(*)", "stueckzahl": "SUM(menge)",
                     "umsatz": "SUM(menge * einzelpreis_cent)"}[kennzahl]
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        order = "gruppe ASC" if gruppierung in ("jahr", "monat") else "wert DESC, gruppe ASC"
        sql = (f"WITH alle AS ({' UNION ALL '.join(unions)}) "
               f"SELECT {group} AS gruppe, {aggregate} AS wert FROM alle{where} "
               f"GROUP BY {group} ORDER BY {order}")
        with closing(self.connect()) as connection:
            rows = [dict(r) for r in connection.execute(sql, params)]
        # Geld wird in SQLite in ganzen Cent berechnet; erst zur Ausgabe umrechnen.
        total = sum(r["wert"] for r in rows)
        if kennzahl == "umsatz":
            for row in rows:
                row["wert"] = round(row["wert"] / 100, 2)
            total = round(total / 100, 2)
        return {"gruppierung": gruppierung, "kennzahl": kennzahl,
                "einheit": "EUR" if kennzahl == "umsatz" else "Anzahl",
                "filter": {"bundesland": bundesland, "artikel": artikel, "jahr": jahr},
                "daten": rows, "gesamt": total, "fiktive_daten": True}

    def balkendiagramm_erstellen(
        self, gruppierung: Gruppierung = "bundesland", kennzahl: Kennzahl = "bestellungen",
        bundesland: str | None = None, artikel: str | None = None,
        jahr: int | None = None,
    ) -> dict:
        """Speichert ein PNG-Balkendiagramm aus echten Abfrageergebnissen."""
        from matplotlib.figure import Figure

        result = self.bestellungen_auswerten(gruppierung, kennzahl, bundesland, artikel, jahr)
        if not result["daten"]:
            raise ValueError("Keine Bestellungen für diese Filter gefunden; kein Diagramm erzeugt.")
        labels = [row["gruppe"] for row in result["daten"]]
        values = [row["wert"] for row in result["daten"]]
        title = {"bestellungen": "Bestellungen", "stueckzahl": "Bestellte Stückzahl",
                 "umsatz": "Umsatz"}[kennzahl]
        filters = ", ".join(str(value) for value in (bundesland, artikel, jahr) if value is not None)
        fig = Figure(figsize=(12, max(4, len(labels) * 0.32)))
        ax = fig.subplots()
        try:
            bars = ax.barh(labels, values, color="#2463a0")
            ax.invert_yaxis()
            ax.bar_label(bars, fmt="%.2f" if kennzahl == "umsatz" else "%.0f", padding=4, fontsize=8)
            ax.set_xlabel(f"{title} ({result['einheit']})")
            ax.set_title(f"{title} nach {gruppierung.capitalize()}" + (f" – {filters}" if filters else ""))
            ax.margins(x=0.2)
            ax.set_axisbelow(True)
            ax.grid(axis="x", alpha=0.2)
            fig.text(0.01, 0.01, "Fiktive Demodaten · eine Zeile = eine Bestellung", fontsize=8)
            fig.tight_layout(rect=(0, 0.025, 1, 1))
            self.chart_dir.mkdir(parents=True, exist_ok=True)
            path = self.chart_dir / f"{kennzahl}_{gruppierung}_{uuid4().hex[:10]}.png"
            fig.savefig(path, dpi=140)
        finally:
            fig.clear()
        return {"datei": str(path), **result}

