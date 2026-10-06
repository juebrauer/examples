#include <stdio.h>

#define MAX_ANZAHL 20

int main(void)
{
    int zahlen[MAX_ANZAHL];  // Hier speichern wir die Zahlen
    int anzahl = 0;          // Wie viele Zahlen wir schon gelesen haben

    // Datei zum Lesen öffnen ("r" = read)
    FILE *datei = fopen("umsaetze.txt", "r");
    if (datei == NULL)
    {
        printf("Fehler: Datei konnte nicht geoeffnet werden!\n");
        return 1;
    }

    // Zahlen einlesen, bis die Datei zu Ende ist
    // fscanf gibt 1 zurück, wenn eine Zahl erfolgreich gelesen wurde
    while (anzahl < MAX_ANZAHL && fscanf(datei, "%d", &zahlen[anzahl]) == 1)
    {
        anzahl = anzahl + 1;
    }

    // Datei wieder schließen
    fclose(datei);

    // Bitte Bubble-Sort hier implementieren!
   

    // Alle Zahlen ausgeben
    printf("Es wurden %d Zahlen eingelesen:\n", anzahl);
    for (int i = 0; i < anzahl; i++)
    {
        printf("%d\n", zahlen[i]);
    }

    // Bubble Sort: Wir vergleichen immer zwei Nachbarn.
    // Ist der linke groesser als der rechte, tauschen wir die beiden.
    // Nach jedem Durchgang steht die groesste Zahl ganz hinten.
    for (int durchgang = 0; durchgang < anzahl - 1; durchgang++)
    {
        for (int i = 0; i < anzahl - 1 - durchgang; i++)
        {
            if (zahlen[i] > zahlen[i + 1])
            {
                int hilfe = zahlen[i];
                zahlen[i] = zahlen[i + 1];
                zahlen[i + 1] = hilfe;
            }
        }
    }

    // Sortierte Zahlen ausgeben
    printf("\nSortiert:\n");
    for (int i = 0; i < anzahl; i++)
    {
        printf("%d\n", zahlen[i]);
    }

    // Sortierte Zahlen in eine Datei schreiben ("w" = write)
    FILE *ausgabe = fopen("umsaetze_sortiert.txt", "w");
    if (ausgabe == NULL)
    {
        printf("Fehler: Datei konnte nicht geschrieben werden!\n");
        return 1;
    }

    for (int i = 0; i < anzahl; i++)
    {
        fprintf(ausgabe, "%d\n", zahlen[i]);
    }

    fclose(ausgabe);
    printf("\nErgebnis wurde in umsaetze_sortiert.txt geschrieben.\n");

    return 0;
}
