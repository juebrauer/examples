#include <stdio.h>
#include <stdlib.h>
#include <time.h>



void tan_block_generator(int n_rows, int n_cols)
{
    int genzahlen[1000];

    if (n_rows*n_cols > 1000)
    {
        printf("Tut mir leid, so grosse TANs kann ich nicht verarbeiten!\n");
        return;
    }

    /*  
        n_rows = 3
        n_cols = 2

        22 45
        99 17
        22

        genzahlen:
        0     1     2     3     4    5    ...
        22    45    99    17    
    */

    int counter = 0;
    for (int y=0; y<n_rows; y++)
    {
        for (int x=0; x<n_cols; x++)
        {

            int gueltig;
            int zufallszahl;

            do {
            
                zufallszahl = rand() % 10 + 1;

                // Gibt es die zufallszahl schon im Array?
                // bzw. haben wir sie vorher schon erzeugt?
                gueltig = 1;
            
                
                for (int i=0; i<counter; i++)
                {
                    if (zufallszahl == genzahlen[i])
                    {
                        gueltig = 0;
                        break;
                    }
                }                

            } while (gueltig == 0);

             genzahlen[counter] = zufallszahl;
             counter++;

             printf("%6d ", zufallszahl);
        }
        printf("\n");
        
    }
}

void tan_block_generator2(int N, int n_cols)
{
    for (int i=1; i<=N; i++)
    {
        int zufallszahl = rand() % 1000000 + 1;
        printf("%010d%s",
               zufallszahl,
               i % n_cols == 0 ? "\n" : " ");
    }
}

int main()
{
    srand(time(NULL));
    int auswahl;
    printf("TAN Generator\n");
    printf("1 - Kleine TAN-Liste (5 TANs)\n");
    printf("2 - Mittlere TAN-Liste (10 TANs)\n");
    printf("3 - Grosse TAN-Liste (20 TANs)\n");
    printf("4 - XL TAN-Liste (100 TANs)\n");
    printf("Deine Auswahl: ");
    scanf("%d", &auswahl);

    int N;
    /*
    if (auswahl==1)
        N = 5;
    else if (auswahl==2)
        N = 10;
    else if (auswahl==3)
        N = 20;
    else if (auswahl==4)
        N = 100;
    else {
        printf("Ungültige Auswahl!");
        return -1;
    } */

    switch (auswahl) {
        case 1 : N = 5; break;
        case 2 : N = 10; break;
        case 3 : N = 20; break;
        case 4 : N = 100; break;
        default:
            printf("Ungültige Auswahl!");
            return -1;
    }

    tan_block_generator2(N, 3);

    //tan_block_generator(rows, cols);
 
    int a = 12;
    int b = 6432;
    int c;
    /*
    if (a<b)
        c = a;
    else
        c = b;
        */
    c = a<b ? a : b; // "Syntactic Sugar"

    printf("Kleinere Zahl von %d und %d ist %d\n", a,b,c);
}