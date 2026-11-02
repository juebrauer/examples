#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <termios.h>
#include <unistd.h>

int main(void)
{
    struct termios vorher;
    struct termios nachher;
    int zeichen;

    // Terminal-Einstellungen sichern (dieser Teil ist Linux-spezifisch).
    if (tcgetattr(STDIN_FILENO, &vorher) == -1)
    {
        perror("Terminal konnte nicht gelesen werden");
        return 1;
    }

    nachher = vorher;

    // ICANON aus: Eingaben sofort lesen, ohne auf Enter zu warten.
    // ECHO aus: Wir geben die Zeichen selbst aus.
    // ISIG aus: Auch Strg+C wird als Zeichen gelesen (Wert 3).
    nachher.c_lflag &= ~(ICANON | ECHO | ISIG);
    nachher.c_cc[VMIN] = 1;
    nachher.c_cc[VTIME] = 0;

    printf("0: Piepen | 1: Bildschirm leeren | 3: Zufallsfarbe\n");
    printf("Esc oder Strg+C: Beenden\n\n");
    fflush(stdout);

    if (tcsetattr(STDIN_FILENO, TCSANOW, &nachher) == -1)
    {
        perror("Terminal konnte nicht umgestellt werden");
        return 1;
    }

    srand(time(NULL));

    while ((zeichen = getchar()) != EOF)
    {
        if (zeichen == 27 || zeichen == 3) // Esc oder Strg+C
        {
            break;
        }
        else if (zeichen == '0')
        {
            printf("\a"); // Signalton (abhaengig von den Terminal-Einstellungen)
        }
        else if (zeichen == '1')
        {
            printf("\033[2J\033[H"); // Bildschirm loeschen, Cursor nach oben
        }
        else if (zeichen == '3')
        {
            int farbe = 31 + rand() % 7; // Farbnummer von 31 bis 37
            printf("\033[%dm", farbe);  // Neue Textfarbe einstellen
        }
        else
        {
            putchar(zeichen);
        }

        fflush(stdout); // Ausgabe sofort anzeigen
    }

    printf("\033[0m\n"); // Normale Farbe wiederherstellen
    if (tcsetattr(STDIN_FILENO, TCSANOW, &vorher) == -1)
    {
        perror("Terminal konnte nicht zurueckgestellt werden");
        return 1;
    }

    return 0;
}
