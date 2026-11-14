#include <stdio.h>
#include <string.h>

/*
 * C passes arguments by value. Passing addresses lets a function change
 * the caller's variables (often called "call by reference" using pointers).
 * This simple example counts ASCII letters (A-Z, a-z), without umlauts.
 */
void analyze_name(char *name,
                  int *length,
                  int *consonants,
                  int *vowels)
{
    /* Dereference the pointers to initialize the caller's variables. */
    *length = 0;
    *consonants = 0;
    *vowels = 0;

    for (int i = 0; name[i] != '\0'; i++)
    {
        char c = name[i];
        printf("Processing character: %c\n", c);
        (*length)++;

        if (c >= 'A' && c <= 'Z')
            c = c - 'A' + 'a';

        if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u')
            (*vowels)++;
        else if (c >= 'a' && c <= 'z')
            (*consonants)++;
        /* Spaces and hyphens count towards length, but are not letters. */
    }

    /* No return value: all three results were written through pointers. */
}

int main(void)
{
    char name[100];
    int length;
    int consonants;
    int vowels;

    printf("Enter your name (ASCII letters, max. 99 characters): ");
    if (fgets(name, sizeof(name), stdin) == NULL)
        return 1;

    /* Remove the newline added when the user presses Enter. */
    name[strcspn(name, "\r\n")] = '\0';

    /* & gives each variable's address to the function. */
    analyze_name(name, &length, &consonants, &vowels);

    printf("Name: %s\n", name);
    printf("Length: %d\n", length);
    printf("Consonants: %d\n", consonants);
    printf("Vowels: %d\n", vowels);

    return 0;
}
