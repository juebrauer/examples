#include <stdio.h>

int main()
{
    int a,b;
    a = 10;
    b = a--;
    printf("b=%d\n", b);
    printf("a=%d\n", a);

    a = 2;
    a = a << 2;
    printf("a=%d\n", a);

    a = 7;
    b = 10;
    printf("a & b = %d\n", a & b);

    unsigned char c = 5;
    printf("c = %b\n", c);

    c = ~c;
    printf("~c=%d (Dezimal)\n", c); 
    printf("~c = %b (Binär)\n", c);


    a = 4;
    b = 3;

    c = ++a * (((((b++))))) + 2;

    printf("a=%d, b=%d, c=%d\n", a, b, c);

    return 0;
}
