#include <stdio.h>

#define ROWS 3
#define COLS 5

int main()
{
    char spielfeld[ROWS][COLS];

    for (int y=0; y<ROWS; y++)
    {
        for (int x=0; x<COLS; x++)
        {
            spielfeld[y][x] = '*';
        }
    }
    
    
    for (int y=0; y<ROWS; y++)
    {
        for (int x=0; x<COLS; x++)
        {
            printf("y=%d x=%d --> %p\n", y,x, &(spielfeld[y][x]));
        }
        printf("\n");
    }

    char* p = &(spielfeld[0][0]);
    for (int y=0; y<ROWS; y++)
    {
        for (int x=0; x<COLS; x++)
        {
            printf("%c", *p);
            p = p + 1;
        }
        printf("\n");
    }

}