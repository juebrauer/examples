#include <stdio.h>
#include <stdlib.h>
#include <time.h>


void tan_block_generator1(int n_rows, int n_cols)
{
    for (int y=1; y<=n_rows; y++)
    {
        for (int x=1; x<=n_cols; x++)
        {
             int zufallszahl = rand() % 1000000 + 1;
             printf("%6d ", zufallszahl);
        }
        printf("\n");
        break;
    }
}


void tan_block_generator2(int n_rows, int n_cols)
{
    for (int i=1; i<=n_rows*n_cols; i++)
    {
        int zufallszahl = rand() % 1000000 + 1;
        printf("%010d ", zufallszahl);
        if (i % n_cols == 0)
        {
            printf("\n");
        }
    }
}


int main()
{
    srand(time(NULL));

    int rows;
    int cols;

    printf("TAN-Block-Generator\n");
    printf("Anzahl Zeilen:");
    scanf("%d", &rows);
    printf("Anzahl Spalten: ");
    scanf("%d", &cols);


    tan_block_generator1(rows, cols);
    tan_block_generator2(rows, cols);

   
}