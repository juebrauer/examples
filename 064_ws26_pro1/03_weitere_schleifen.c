#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void bsp1()
{
    srand(time(NULL));

   
    int zufallszahl;
    char weiter;

    //printf("Wahrheitswert=%d \n", (weiter='j'));


    do
    {
        zufallszahl = rand() % 6 + 1;
        printf("%d ", zufallszahl);

        printf("Noch eine Zufallszahl erzeugen? (j/n): ");
        scanf(" %c", &weiter);
    } while ((weiter=='j') || (weiter=='J'));

    printf("Programm vom Benutzer beendet!\n");
}


void bsp2()
{
    srand(time(NULL));

   
    int zufallszahl;
    char weiter;

    printf("Willst du eine 1. Zufallszahl erzeugen? (j/n): ");
    scanf(" %c", &weiter);

    while ((weiter=='j') || (weiter=='J'))
    {
        zufallszahl = rand() % 6 + 1;
        printf("%d ", zufallszahl);

        printf("Noch eine Zufallszahl erzeugen? (j/n): ");
        scanf(" %c", &weiter);
    } 
    printf("Programm vom Benutzer beendet!\n");

}

void bsp3()
{
    int i = 1;
    start:
    printf("i=%d\n", i);
    i++;
    if (i<=10)
        goto start;
    printf("Schleife beendet!\n");

}

void bsp4()
{    
    int i = 1;

    while (i < 10) {
        i = i * 2;
        printf("%d ", i);
        i = i + 5;
    }
}

void bsp5()
{
    int i = 1;
    int summe = 0;

    while (i <= 5) {
        if (i % 2 == 0) {
            summe = summe + i;
        }
        i++;
    } 
}


int main(void)
{
    //bsp1();

    //for (char i=0; i<1000; i++)
    //    printf("%d ", i);

    //bsp2();

    //bsp3();

    //bsp4();

    bsp5();
    
}