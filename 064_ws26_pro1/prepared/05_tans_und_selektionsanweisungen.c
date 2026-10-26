
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX 100

int main(void) {
    int tans[MAX], tan, doppelt;
    int auswahl, N;

    printf("TAN-Generator\n");
    printf("1 - Kleine TAN-Liste  (5 TANs)\n");
    printf("2 - Mittlere TAN-Liste (10 TANs)\n");
    printf("3 - Grosse TAN-Liste  (20 TANs)\n");
    printf("4 - XL-TAN-Liste      (100 TANs)\n");
    printf("Auswahl: ");
    scanf("%d", &auswahl);

    // Menüauswahl mit switch-case
    switch (auswahl) {
        case 1: N = 5;   break;
        case 2: N = 10;  break;
        case 3: N = 20;  break;
        case 4: N = 100; break;
        default:
            printf("Ungueltige Auswahl!\n");
            return 1;
    }

    srand(time(NULL));

    for (int i = 0; i < N; i++) {
        do {
            tan = rand() % 900000 + 100000;
            doppelt = 0;

            for (int j = 0; j < i; j++)
                if (tans[j] == tan) doppelt = 1;

        } while (doppelt);

        tans[i] = tan;
    }

    printf("\n       TAN-LISTE\n\n");

    for (int i = 0; i < N; i++)
        printf("%06d%s", tans[i],
               (i + 1) % 5 ? "  " : "\n");

    return 0;
}
