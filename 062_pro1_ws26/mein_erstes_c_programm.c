#include <stdio.h>
#include <string.h>

int main()
{
        char name[100];
	
	printf("Wie heisst du? ");
	scanf("%99s", name);
	
	printf("Hallo %s\n", name);
	printf("Dein Name hat %ld Zeichen.\n", strlen(name));
	
	return 0;
}
