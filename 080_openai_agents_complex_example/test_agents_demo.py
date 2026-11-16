"""Lokale Tests ohne API-Schlüssel: python3 -m unittest -v"""

from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest

from bestelldaten import ARTIKEL, BUNDESLAENDER, BestellDaten, create_database


class DatenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.temp.name) / "test.sqlite3"
        create_database(cls.path)
        cls.data = BestellDaten(cls.path, Path(cls.temp.name) / "charts")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_16_tabellen_mit_fuenf_jahren_und_artikeln(self):
        with closing(sqlite3.connect(self.path)) as connection:
            tables = connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            self.assertEqual(len(tables), 16)
            for table, _ in BUNDESLAENDER.values():
                years = connection.execute(f'SELECT DISTINCT substr(bestelldatum, 1, 4) FROM "{table}"').fetchall()
                self.assertEqual({r[0] for r in years}, {str(y) for y in range(2021, 2026)})
                items = connection.execute(f'SELECT DISTINCT artikel FROM "{table}"').fetchall()
                self.assertEqual({r[0] for r in items}, set(ARTIKEL))

    def test_gruppierungen_haben_identische_gesamtsummen(self):
        for metric in ("bestellungen", "stueckzahl", "umsatz"):
            totals = [self.data.bestellungen_auswerten(group, metric)["gesamt"]
                      for group in ("bundesland", "artikel", "jahr", "monat")]
            self.assertTrue(all(value == totals[0] for value in totals))

    def test_filter_und_umsatz_gegen_direkte_abfrage(self):
        with closing(sqlite3.connect(self.path)) as connection:
            count, quantity, cents = connection.execute(
                "SELECT COUNT(*), SUM(menge), SUM(menge * einzelpreis_cent) FROM bayern "
                "WHERE artikel = 'Notebook' AND bestelldatum BETWEEN '2024-01-01' AND '2024-12-31'"
            ).fetchone()
        for metric, expected in (("bestellungen", count), ("stueckzahl", quantity), ("umsatz", cents / 100)):
            result = self.data.bestellungen_auswerten("bundesland", metric, "Bayern", "Notebook", 2024)
            self.assertEqual(result["daten"], [{"gruppe": "Bayern", "wert": expected}])

    def test_monate_und_leere_ergebnisse(self):
        result = self.data.bestellungen_auswerten("monat", jahr=2023)
        self.assertEqual([r["gruppe"] for r in result["daten"]], [f"2023-{m:02d}" for m in range(1, 13)])
        empty = self.data.bestellungen_auswerten(jahr=1900)
        self.assertEqual(empty["gesamt"], 0)
        self.assertEqual(empty["daten"], [])

    def test_ungueltige_filter_und_schreibschutz(self):
        for kwargs in ({"bundesland": "Bayern'; DROP TABLE bayern; --"},
                       {"artikel": "Smartphone"}, {"gruppierung": "ungueltig"}):
            with self.assertRaises(ValueError):
                self.data.bestellungen_auswerten(**kwargs)
        with closing(self.data.connect()) as connection:
            with self.assertRaises(sqlite3.OperationalError):
                connection.execute("DELETE FROM bayern")

    def test_vorhandene_datenbank_bleibt_erhalten(self):
        self.assertFalse(create_database(self.path, start_year=2010, seed=123))
        self.assertEqual(len(self.data.bestellungen_auswerten("jahr")["daten"]), 5)
        self.assertEqual(self.data.bestellungen_auswerten("jahr")["daten"][0]["gruppe"], "2021")

    def test_diagramm_ist_png(self):
        result = self.data.balkendiagramm_erstellen("artikel", bundesland="Berlin", jahr=2023)
        self.assertEqual(Path(result["datei"]).read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        with self.assertRaises(ValueError):
            self.data.balkendiagramm_erstellen(jahr=1900)


if __name__ == "__main__":
    unittest.main()
