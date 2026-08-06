"""Einfache Funktionen für die Arbeit mit Dateien und Verzeichnissen.

Das Modul verwendet ausschließlich die Python-Standardbibliothek. Pfade dürfen
als Zeichenkette oder als :class:`pathlib.Path` übergeben werden. Filter werden
als Glob-Muster angegeben, zum Beispiel ``"*.csv"`` oder ``"bild_?.png"``.

Ein kurzes Beispiel::

    from verzeichnis_helfer import dateien_filtern, verzeichnisgroesse

    csv_dateien = dateien_filtern("messwerte", "*.csv")
    groesse = verzeichnisgroesse("messwerte")
    print(f"{len(csv_dateien)} CSV-Dateien, insgesamt {groesse} Bytes")
"""

__docformat__ = "google"

from pathlib import Path
from typing import TypeAlias


Pfad: TypeAlias = str | Path
"""Ein Dateisystempfad als Zeichenkette oder :class:`pathlib.Path`."""


def _pruefe_verzeichnis(pfad: Pfad) -> Path:
    """Wandle *pfad* in einen geprüften Verzeichnispfad um."""
    verzeichnis = Path(pfad)
    if not verzeichnis.is_dir():
        raise NotADirectoryError(f"Kein Verzeichnis: {verzeichnis}")
    return verzeichnis


def dateien_filtern(
    pfad: Pfad,
    muster: str = "*",
    *,
    rekursiv: bool = False,
) -> list[Path]:
    """Gib passende Dateien in einem Verzeichnis zurück.

    Args:
        pfad: Das zu durchsuchende Verzeichnis.
        muster: Ein Glob-Muster wie ``"*.txt"`` oder ``"foto_?.jpg"``.
        rekursiv: Falls ``True``, werden auch Unterverzeichnisse durchsucht.

    Returns:
        Eine nach dem Pfad sortierte Liste passender Dateien. Verzeichnisse
        selbst sind nicht enthalten.

    Raises:
        NotADirectoryError: Wenn *pfad* nicht auf ein Verzeichnis zeigt.

    Examples:
        Nur CSV-Dateien im Verzeichnis ``messwerte`` finden::

            csv_dateien = dateien_filtern("messwerte", "*.csv")
            for datei in csv_dateien:
                print(datei.name)
    """
    verzeichnis = _pruefe_verzeichnis(pfad)
    treffer = verzeichnis.rglob(muster) if rekursiv else verzeichnis.glob(muster)
    return sorted(datei for datei in treffer if datei.is_file())


def dateien_zaehlen(
    pfad: Pfad,
    muster: str = "*",
    *,
    rekursiv: bool = True,
) -> int:
    """Zähle Dateien, die zu einem Muster passen.

    Args:
        pfad: Das zu untersuchende Verzeichnis.
        muster: Optionaler Dateifilter als Glob-Muster. Der Standard ``"*"``
            berücksichtigt alle Dateien.
        rekursiv: Gibt an, ob Unterverzeichnisse einbezogen werden.

    Returns:
        Die Anzahl der passenden Dateien.

    Examples:
        ``dateien_zaehlen("messwerte", "*.csv")`` zählt alle CSV-Dateien im
        Verzeichnis und seinen Unterverzeichnissen.
    """
    return len(dateien_filtern(pfad, muster, rekursiv=rekursiv))


def verzeichnisgroesse(pfad: Pfad, *, rekursiv: bool = True) -> int:
    """Berechne die Gesamtgröße aller Dateien in Bytes.

    Args:
        pfad: Das zu untersuchende Verzeichnis.
        rekursiv: Gibt an, ob Dateien in Unterverzeichnissen mitzählen.

    Returns:
        Die Summe der Dateigrößen in Bytes.

    Raises:
        NotADirectoryError: Wenn *pfad* nicht auf ein Verzeichnis zeigt.
        OSError: Wenn eine Datei nicht gelesen werden kann.

    Examples:
        Die Größe lesbar ausgeben::

            groesse = verzeichnisgroesse("messwerte")
            print(menschenlesbare_groesse(groesse))
    """
    dateien = dateien_filtern(pfad, rekursiv=rekursiv)
    return sum(datei.stat().st_size for datei in dateien)


def groesste_dateien(
    pfad: Pfad,
    anzahl: int = 5,
    *,
    rekursiv: bool = True,
) -> list[tuple[Path, int]]:
    """Ermittle die größten Dateien eines Verzeichnisses.

    Args:
        pfad: Das zu untersuchende Verzeichnis.
        anzahl: Maximale Anzahl der Ergebnisse. Muss mindestens null sein.
        rekursiv: Gibt an, ob Unterverzeichnisse einbezogen werden.

    Returns:
        Eine absteigend sortierte Liste aus ``(Pfad, Größe_in_Bytes)``-Tupeln.

    Raises:
        ValueError: Wenn *anzahl* negativ ist.

    Examples:
        Die drei größten Dateien anzeigen::

            for datei, groesse in groesste_dateien(".", 3):
                print(datei, menschenlesbare_groesse(groesse))
    """
    if anzahl < 0:
        raise ValueError("anzahl darf nicht negativ sein")

    dateien_mit_groesse = [
        (datei, datei.stat().st_size)
        for datei in dateien_filtern(pfad, rekursiv=rekursiv)
    ]
    dateien_mit_groesse.sort(key=lambda eintrag: eintrag[1], reverse=True)
    return dateien_mit_groesse[:anzahl]


def menschenlesbare_groesse(anzahl_bytes: int) -> str:
    """Formatiere eine Byte-Anzahl mit einer passenden binären Einheit.

    Args:
        anzahl_bytes: Eine nichtnegative Größe in Bytes.

    Returns:
        Eine Zeichenkette wie ``"850 B"``, ``"1.5 KiB"`` oder ``"2.0 MiB"``.

    Raises:
        ValueError: Wenn *anzahl_bytes* negativ ist.

    Examples:
        >>> menschenlesbare_groesse(1536)
        '1.5 KiB'
    """
    if anzahl_bytes < 0:
        raise ValueError("anzahl_bytes darf nicht negativ sein")

    groesse = float(anzahl_bytes)
    einheiten = ("B", "KiB", "MiB", "GiB", "TiB")
    for einheit in einheiten:
        if groesse < 1024 or einheit == einheiten[-1]:
            if einheit == "B":
                return f"{int(groesse)} {einheit}"
            return f"{groesse:.1f} {einheit}"
        groesse /= 1024

    raise AssertionError("Keine passende Einheit gefunden")
