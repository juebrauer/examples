def read_file(file_path):
    """
    Liest den Inhalt einer Datei und gibt ihn als String zurück.
    """
    with open(file_path, 'r') as file:
        return file.read()


def find_term_occurrences(text, term, context_chars=30):
    """
    Findet alle Vorkommen eines Suchbegriffs im Text mit Kontext.

    :param text: Der gesamte Text.
    :param term: Der Suchbegriff (case-insensitive).
    :param context_chars: Anzahl der Zeichen vor und nach dem Begriff im Kontext.
    :return: Liste von Kontextinformationen (vorher, Begriff, nachher).
    """
    occurrences = []
    search_start = 0
    term_lower = term.lower()
    text_lower = text.lower()

    while True:
        position = text_lower.find(term_lower, search_start)
        if position == -1:
            break

        # Kontext extrahieren
        start = max(0, position - context_chars)
        end = min(len(text), position + len(term) + context_chars)
        before = text[start:position].strip()
        match = text[position:position + len(term)]
        after = text[position + len(term):end].strip()

        occurrences.append({"before": before, "match": match, "after": after})

        search_start = position + len(term)

    return occurrences


def print_occurrences(occurrences):
    """
    Gibt alle gefundenen Vorkommen mit Kontext aus.
    """
    for occurrence in occurrences:
        print(occurrence["before"], occurrence["match"], occurrence["after"])


def search_term_in_file(file_path, term, context_chars=30):
    """
    Hauptfunktion zum Suchen und Anzeigen von Begriffsvorkommen in einer Datei.

    Diese Funktion nutzt andere Funktionen wie z.B.
    die Funktion `read_file` oder die Funktion `find_term_occurrences`
    """
    text = read_file(file_path)
    occurrences = find_term_occurrences(text, term, context_chars)
    print_occurrences(occurrences)

# Google-Style
def add_numbers_google_style(a: int, b: int) -> int:
    """
    Add two integers together.

    Args:
        a (int): The first integer.
        b (int): The second integer.

    Returns:
        int: The sum of the two integers.
    """
    return a + b

# NumPy-Style
def add_numbers_numpy_style(a: int, b: int) -> int:
    """
    Add two integers together.

    Parameters
    ----------
    a : int
        The first integer.
    b : int
        The second integer.

    Returns
    -------
    int
        The sum of the two integers.
    """
    return a + b

# ReStructuredText-Style
def add_numbers_restructuredtext_style(a: int, b: int) -> int:
    """
    Add two integers together.

    :param a: Der 1. Summand
    :type a: int    
    :param b: Der 2. Summand    
    :type b: int
    :return: Die Summe!
    :rtype: int
    """
    return a + b

# PlainText-Style
def add_numbers_plaintext_style(a: int, b: int) -> int:
    """
    Add two integers together.

    param a: Der 1. Summand
    param b: Der 2. Summand    
    return: Die Summe!    
    """
    return a + b

# Beispielaufruf
search_term_in_file("example.txt", "Python")
