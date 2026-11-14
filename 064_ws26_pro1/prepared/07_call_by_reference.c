#include <locale.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

/*
 * Analyze decoded characters instead of individual UTF-8 bytes.
 * Umlauts are counted separately from the vowels a, e, i, o, u.
 * All four results are written into the caller's variables using pointers.
 */
void analyze_name(const wchar_t *name, int *length, int *consonants,
                  int *vowels, int *umlauts)
{
    *length = 0;
    *consonants = 0;
    *vowels = 0;
    *umlauts = 0;

    for (int i = 0; name[i] != L'\0'; i++)
    {
        wchar_t c = name[i];
        printf("Processing character: %lc\n", (wint_t)c);
        (*length)++;

        if (c >= L'A' && c <= L'Z')
            c = c - L'A' + L'a';

        /* L marks a wide character literal, including non-ASCII letters. */
        if (c == L'ä' || c == L'ö' || c == L'ü' ||
            c == L'Ä' || c == L'Ö' || c == L'Ü')
            (*umlauts)++;
        else if (c == L'a' || c == L'e' || c == L'i' ||
                 c == L'o' || c == L'u')
            (*vowels)++;
        else if ((c >= L'a' && c <= L'z') || c == L'ß' || c == L'ẞ')
            (*consonants)++;
        /* Other characters contribute only to length. */
    }
}

int main(void)
{
    char input[100];
    wchar_t name[100];
    int length;
    int consonants;
    int vowels;
    int umlauts;

    /* Use the terminal environment's encoding (a UTF-8 locale is required). */
    if (setlocale(LC_CTYPE, "") == NULL)
    {
        fprintf(stderr, "Cannot activate the environment's locale.\n");
        return 1;
    }

    printf("Enter your name (max. 99 bytes; umlauts use 2 bytes in UTF-8): ");
    if (fgets(input, sizeof(input), stdin) == NULL)
        return 1;

    if (strchr(input, '\n') == NULL)
    {
        int next = getchar();
        if (next != '\n' && next != EOF)
        {
            fprintf(stderr, "Name is too long (maximum: 99 bytes).\n");
            return 1;
        }
    }
    input[strcspn(input, "\r\n")] = '\0';

    /* Convert multibyte input to wide characters: ü becomes one element. */
    if (mbstowcs(name, input, sizeof(name) / sizeof(name[0])) == (size_t)-1)
    {
        fprintf(stderr, "Cannot decode the name. Please use a UTF-8 locale.\n");
        return 1;
    }

    /* & passes addresses; * in analyze_name accesses the result variables. */
    analyze_name(name, &length, &consonants, &vowels, &umlauts);

    printf("Name: %s\n", input);
    printf("Length: %d\n", length);
    printf("Consonants: %d\n", consonants);
    printf("Vowels (a, e, i, o, u): %d\n", vowels);
    printf("Umlauts: %d\n", umlauts);

    return 0;
}
