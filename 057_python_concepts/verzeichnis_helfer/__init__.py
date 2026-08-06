"""Kleine Hilfsfunktionen für die Arbeit mit Verzeichnissen.

Die Funktionen des Moduls :mod:`verzeichnis_helfer.verzeichnisse` werden hier
erneut exportiert. Dadurch können sie bequem direkt aus dem Paket importiert
werden::

    from verzeichnis_helfer import dateien_filtern, dateien_zaehlen
"""

__docformat__ = "google"

from .verzeichnisse import (
    dateien_filtern,
    dateien_zaehlen,
    groesste_dateien,
    menschenlesbare_groesse,
    verzeichnisgroesse,
)

__all__ = [
    "dateien_filtern",
    "dateien_zaehlen",
    "groesste_dateien",
    "menschenlesbare_groesse",
    "verzeichnisgroesse",
]
