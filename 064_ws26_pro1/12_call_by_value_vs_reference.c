#include <stdio.h>
#include <string.h>

void analyze_name(char* A,  // oder: char A[]
                  int* vokale,
                  int* konsonanten)
{
    int i = 0;
    *vokale = 0;
    *konsonanten = 0;
    while (A[i] != '\0')
    {
        char c = A[i];
        printf("Zeichen #%d: %c\n", i, c);

        // Damit wir im Folgenden nicht Gross-/Kleinschreibung
        // berücksichtigen müssen
        if (c >='A' && c<='Z')
            c = c-'A' + 'a';

        if ((c=='a') || (c=='e') || (c=='i') || (c=='o') || (c=='u'))
        {
            (*vokale)++;
        }
        else if (c>='a'&& c<='z')
        {
            (*konsonanten)++;
        }

        i++;
    }

    printf("Ergebnis innerhalb der Funktion:\n");
    printf("Vokale     : %d\n", *vokale);
    printf("Konsonanten: %d\n", *konsonanten);
  
}

int main()
{
    char name[100];

    char name2[] = "ABCD";
    char* name3 = "ABCD";

    printf("Bitte gib deinen Namen ein:\n");
    fgets(name, sizeof(name), stdin);
    int N = strcspn(name, "\r\n");
    name[ N ] = '\0';

    printf("Alle 100 Zeichen des Arrays name[100]:\n");
    for (int i=0; i<=99; i++)
    {
        printf("%c", name[i]);
        
    }
    printf("\n");


    printf("Dein Name: %s\n", name);

    int vokale, konsonanten;
    vokale = 42;
    konsonanten = 43;

    // Call-by-reference
    analyze_name(name, &vokale, &konsonanten);

    printf("Ergebnis innerhalb der main Funktion:\n");
    printf("Vokale     : %d\n", vokale);
    printf("Konsonanten: %d\n", konsonanten);
}