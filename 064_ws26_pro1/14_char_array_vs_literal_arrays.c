#include <stdio.h>

void print_my_string(char* str) {
   printf("String passed is: %s\n", str);
}

int main() {
   char str1[] = "HELLO";
   char* str2 = "HELLO";

   print_my_string(str1);
   print_my_string(str2);

   str1[0] = 'h';
   print_my_string(str1);

   str2[0] = 'h';
   print_my_string(str2);

   printf("Programmende!\n");

} // end main