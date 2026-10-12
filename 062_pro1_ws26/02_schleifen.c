#include <stdio.h>
#include <stdlib.h>
#include <time.h>


void bsp1()
{
    srand(5);
    int zufallszahl;
    int i;
    i = 1;
    for (;;)
    {
        zufallszahl = rand();
        printf("Zufallszahl #%d: %d\n", i, zufallszahl);
        i++;
        
        if (i==4)
            break;
    }
    printf("Jetzt bin ich wieder aus der Schleife raus!\n");
}

void bsp2()
{
    printf("CLOCKS_PER_SEC=%ld\n", CLOCKS_PER_SEC);
    srand( time(NULL) );
    int zufallszahl;
    for (int i=1; i<=3; i++)
    {
        zufallszahl = rand();
        printf("Zufallszahl #%d: %d\n", i, zufallszahl);
        for (int j=1; j<=20; j++)
        {
            printf("-");
        }
        printf("\n");
    }
    printf("Jetzt bin ich wieder aus der Schleife raus!\n");
}


int bsp3()
{
    //printf("10 / 7 = %d\n", 10/7);
    //printf("10 %% 7 = %d\n", 10%7);

    srand( time(NULL) );
    int zufallszahl;
    for (int i=1; i<=10; i++)
    {
        zufallszahl = rand() % 6 + 1;
        printf("%d ", zufallszahl);
    }
    printf("\n");
    return 42;
}



int main()
{
   int a = bsp3();
   printf("Rückgabewert der Funktion bsp3 ist: %d\n", a);


}