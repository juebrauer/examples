#include <stdio.h>

// Ein enum gibt zusammengehörigen Zahlen verständliche Namen.
// Der erste Raum hat den Wert 1. Danach zählt C automatisch weiter.
enum Raum
{
    RAUM_BERLIN = 1,
    RAUM_HAMBURG,    // Wert 2
    RAUM_MUENCHEN   // Wert 3
};

// Ein struct fasst verschiedene Daten zu einer Buchung zusammen.
struct Buchung
{
    enum Raum raum;
    char name[50];
    int tag;
    int monat;
    int jahr;
    int startstunde;
    int dauer_in_stunden;
    int anzahl_personen;
};

// Der Rückgabetyp ist struct Buchung: Die Funktion liefert eine ganze Buchung.
// void in der Parameterliste bedeutet, dass die Funktion keine Argumente erwartet.
struct Buchung erstelle_buchung(void)
{
    // Alle Daten dieser Beispielbuchung stehen in einer einzigen Variablen.
    // Die Feldnamen vor dem Gleichheitszeichen zeigen, wohin jeder Wert gehört.
    struct Buchung buchung = {
        .raum = RAUM_HAMBURG,
        .name = "Anna Weber",
        .tag = 20,
        .monat = 11,
        .jahr = 2026,
        .startstunde = 10,
        .dauer_in_stunden = 2,
        .anzahl_personen = 6
    };

    // Auch die enthaltene Zeichenkette wird als Teil des structs zurückgegeben.
    return buchung;
}

// Ein struct kann wie eine Zahl als Argument übergeben werden.
// Die Funktion erhält eine Kopie. Änderungen daran würden das Original
// in main nicht verändern. void als Rückgabetyp bedeutet: keine Rückgabe.
void zeige_buchung(struct Buchung buchung)
{
    // Mit dem Punkt greifen wir auf einzelne Felder des structs zu.
    printf("Buchung von: %s\n", buchung.name);
    printf("Raumnummer: %d\n", buchung.raum);

    // Das enum speichert eine Zahl. Mit switch geben wir den Raumnamen aus.
    switch (buchung.raum)
    {
        case RAUM_BERLIN:
            printf("Raum: Berlin\n");
            break;
        case RAUM_HAMBURG:
            printf("Raum: Hamburg\n");
            break;
        case RAUM_MUENCHEN:
            printf("Raum: München\n");
            break;
        default:
            printf("Unbekannter Raum\n");
            break;
    }

    printf("Datum: %02d.%02d.%04d\n", buchung.tag, buchung.monat, buchung.jahr);
    printf("Beginn: %d Uhr\n", buchung.startstunde);
    printf("Dauer: %d Stunden\n", buchung.dauer_in_stunden);
    printf("Personen: %d\n", buchung.anzahl_personen);
}

// Der Stern im Parametertyp bedeutet: buchung ist ein Zeiger auf ein struct.
// Die Funktion erhält eine Kopie der Adresse und verändert darüber das Original.
void aendere_buchung(struct Buchung *buchung)
{
    // Mit -> greifen wir über einen Zeiger auf die Felder des structs zu.
    // buchung->raum bedeutet dasselbe wie (*buchung).raum.
    buchung->raum = RAUM_BERLIN;
    buchung->dauer_in_stunden = 3;
    buchung->anzahl_personen = 8;
}

int main(void)
{
    // Die Funktion gibt ein struct zurück, das wir in einer Variablen speichern.
    struct Buchung meine_buchung = erstelle_buchung();

    // Hier übergeben wir das gesamte struct an die Ausgabefunktion.
    printf("Vor der Änderung:\n");
    zeige_buchung(meine_buchung);

    // & liefert die Adresse unserer Buchung. So kann die Funktion
    // die Felder von meine_buchung direkt verändern.
    aendere_buchung(&meine_buchung);

    // Die Änderungen bleiben auch nach dem Funktionsaufruf erhalten.
    printf("\nNach der Änderung:\n");
    zeige_buchung(meine_buchung);

    return 0;
}
