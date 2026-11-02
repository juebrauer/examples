#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <termios.h>
#include <unistd.h>

void clear()
{
    printf("\033[2J\033[H");
}

void set_random_color()
{
    int farbe = 31 + rand() % 7; // Farbnummer von 31 bis 37
    printf("\033[%dm", farbe);
}

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

    clear();

    for (int i=1; i<=10; i++)
    {
        set_random_color();
        printf("Hallo!\n");
    }

}