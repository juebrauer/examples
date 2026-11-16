#include <stdio.h>


void vertausche_werte(int* x, int* y)
{
    int hilfe = *x;
    *x = *y;
    *y = hilfe;

}


int main()
{
    int a = 42;
    int b = 999;

    vertausche_werte(&a,&b);

    printf("a=%d b=%d\n", a,b);
}