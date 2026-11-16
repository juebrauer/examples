#include <float.h>
#include <stdio.h>

int main(void)
{
    // unsigned int kann keine negativen Zahlen speichern.
    // Beim Subtrahieren von 1 von 0 springt der Wert zur größten darstellbaren Zahl.
    unsigned int zahl = 0;
    printf("unsigned int vorher: %u\n", zahl);
    zahl = zahl - 1;
    printf("unsigned int nach - 1: %u\n\n", zahl);

    // FLT_MIN ist die kleinste positive normale float-Zahl.
    // Noch kleinere Beträge können mit geringerer Genauigkeit als subnormale Zahlen gespeichert werden.
    // Wiederholtes Halbieren führt bei üblicher Rundung schließlich zu 0 (Underflow).
    float wert = 1.0f;
    int i = 0;    
    while (wert > 0.0f)
    {
        wert = wert / 2.0f;
        printf("%.50f (i=%d)\n", wert, i);
        i++;
    }

    printf("FLOAT_MIN:\n");
    printf("%.50f\n", FLT_MIN);

    return 0;
}
