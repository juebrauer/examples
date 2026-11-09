#include <stdio.h>

int main()
{
    int a;
    int* p; // Deklaration eines Pointers

    a = 42;
    p = &a; // Hier wird & Adressoperator genannt
    printf("Wert von a=%d\n", a);
    printf("Variable a ist gespeichert hier: %p\n", p);

    // * wird Derefenzierungsoperator / Load-Operator genannt
    printf("Wert von a mit Hilfe des Pointers: %d\n", *p);

    printf("Anzahl der Bytes, um a zu speichern: %ld\n", sizeof(a));
    printf("Anzahl der Bytes, um p zu speichern: %ld\n", sizeof(p));

    double A[10];
    for (int i=0; i<10; i++)
        A[i] = 100 + i;

    /*
    for (int i=0; i<10; i++)
    {
        printf("A[%d] = %f, Adresse: %p\n", i, A[i], &(A[i]) );
    }
    */

    double* q = &(A[0]);
    
    for (int i=0; i<10; i++)
    {
        printf("A[%d] = %f, Adresse: %p\n", i, *(q+i), q+i );
    }



}