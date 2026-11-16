#include <stdio.h>  // damit wir printf() haben
#include <stdlib.h> // damit wir rand() haben
#include <time.h>   // damit wir time() haben

#define ANZAHL_SILBEN 4
#define MAX_LAENGE_SILBE 6

int main()
{
    srand(time(NULL));

    //                                                 0       1     2      3
    char silben[ANZAHL_SILBEN][MAX_LAENGE_SILBE] =
         { {'H','o','\0'}, "Haaaa", "Hi", "Hu" };

    for (int wort_nr=1; wort_nr<=3; wort_nr++)
    {
        printf("Jetzt kommt das nächste Wort #%d\n", wort_nr);
        for (int silbe_nr=1; silbe_nr<=4; silbe_nr++)
        {
            int zz = rand() % ANZAHL_SILBEN;
            printf("%s", silben[zz]);
        }
        printf("\n");
    }


}