C and C++ programming exercises for beginners


# Schedule

- For week #1  please do exercises: G1,G2,L1,L2
- For week #2  please do exercises: L3,L4,L5,R1
- For week #3  please do exercises: L6,L7,L8
- For week #4  please do exercises: S1,A1
- For week #5  please do exercises: F1,F2
- For week #6  please do exercises: Y1,N1
- For week #7  please do exercises: N2,T1
- For week #8  please do exercises: D1,D2
- For week #9  please do exercises: I1,I2
- For week #10 please do exercises: O1
- For week #11 please do exercises: O2 (Xmas exercise!)
- For week #12 please do exercises: O3,O4,O5


# Comment

Solutions for all the exercises are provided here!

This means that you have to show self-discipline yourself.

First try to solve the exercise by your own!

If you do not try it, the learning effect will be much smaller!

Only have a look into the sample code if you cannot manage to find a starting point, solve the exercise, or if you want to see another solution approach after having solved the exercise by your own.


# Table of Contents

## G: Getting started
- [G1: A first program](#g1-a-first-program)
- [G2: What is good source code formating?](#g2-what-is-good-source-code-formating)

## L: Loops
- [L1: A simple do-while Loop](#l1-a-simple-do-while-loop)
- [L2: Multiplication Table](#l2-multiplication-table)
- [L3: for-loops](#l3-for-loops)
- [L4: Two nested for-loops](#l4-two-nested-for-loops)
- [L5: Transforming one loop into another](#l5-transforming-one-loop-into-another)
- [L6: Drawing a rectangle](#l6-drawing-a-rectangle)
- [L7: Drawing a rectangle with one loop](#l7-drawing-a-rectangle-with-one-loop)
- [L8: Drawing a triangle](#l8-drawing-a-triangle)

## R: Random numbers
- [R1: Random numbers in a defined range](#r1-random-numbers-in-a-defined-range)

## S: Selections
- [S1: A simple calculator](#s1-simple-calculator)

## A: Animations
- [A1: Growing rectangle](#a1-growing-rectangle)

## F: Functions
- [F1: Call-by-reference](#f1-call-by-reference)
- [F2: Recursive functions](#f2-recursive-functions)

## Y: Arrays
- [Y1: Writing blocks into a 2D array](#y1-writing-blocks-into-a-2d-array)

## N: Strings
- [N1: Word generator given three syllables from user input](#n1-word-generator-given-three-syllables-from-user-input)
- [N2: Word generator given syllables defined in code](#n2-word-generator-given-syllables-defined-in-code)

## T: Structs, Enums
- [T1: A pizza order system](#t1-a-pizza-order-system)

## D: Dynamic memory allocation
- [D1: Allocating a 1D array dynamically](#d1-allocating-a-1d-array-dynamically)
- [D2: Notebook](#d2-notebook)

## I: File operations
- [I1: Count number of lines in a text file](#i1-count-number-of-lines-in-a-text-file)
- [I2: Read in numbers from file and write numbers sorted into another](#i2-read-in-numbers-from-file-and-write-numbers-sorted-into-another)

## O: Object Oriented Programming in C++
- [O1: NumberBox - A first class for storing numbers](#o1-numberbox---a-first-class-for-storing-numbers)
- [O2: Xmas party simulator](#o2-xmas-party-simulator)
- [O3: printf() vs. cout](#o3-printf-vs-cout)
- [O4: Pointers vs. References](#o4-pointers-vs-references)
- [O5: Shallow vs. Deep copy](#o5-shallow-vs-deep-copy)
- [O6: PacMan with a Ghost class](#o6-pacman-with-a-ghost-class)



# G1: A first program

Write a C program that:
- reads three integer (int) numbers from the user using scanf_s()
- computes the sum, product and average
- and outputs all three values (sum, product, average)!

Play with this program with different numbers for input! Does it always compute correctly?


# G2: What is good source code formating?

Copy the following source code into a C project and compile it. Hint: for some compilers you need to delete the two spaces in `< stdio.h >`. Run it and try to understand what this program is supposed to compute. Then reformat the code such that it is easier to understand the code.

**Linux:**
```c
#include < stdio.h > 
  int main() { float c, r; printf("C:"); scanf("%f", &c);
   printf("r:"); scanf("%f", &r); int y = 0; do { c = c + c*(r / 100.0f);
   y = y + 1; printf("\ny=%d: c=%.1f", y, c); } while (y<10); printf("\nGood bye!\n");
  }
```

**Windows:**
```c
#include < stdio.h > 
#include < conio.h >
int main() { float c, r; printf("C:"); scanf_s("%f", &c);
printf("r:"); scanf_s("%f", &r); int y = 0; do { c = c + c*(r / 100.0f);
y = y + 1; printf("\ny=%d: c=%.1f", y, c); } while (y<10); printf("\nGood bye!\n");
printf("Press any key to exit this program."); _getch(); }
```

Do you want to get an impression how a real Code Style Guide looks like? Then have a look into [Google's C++ Style Guide](https://google.github.io/styleguide/cppguide.html).


# L1: A simple do-while Loop

Write a C program that uses the do-while statement to output the following:

```
Counting from 1 to 10:
1 2 3 4 5 6 7 8 9 10
Counting from -5 to -15:
-5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15
Good bye!
Press any key to exit this program.
```


# L2: Multiplication Table

Write a C program using two nested do-while loops that outputs the following multiplication table:

```
1 x 1 = 1       1 x 2 = 2       1 x 3 = 3       1 x 4 = 4       1 x 5 = 5       1 x 6 = 6       1 x 7 = 7       1 x 8 = 8       1 x 9 = 9       1 x 10 = 10

2 x 1 = 2       2 x 2 = 4       2 x 3 = 6       2 x 4 = 8       2 x 5 = 10      2 x 6 = 12      2 x 7 = 14      2 x 8 = 16      2 x 9 = 18      2 x 10 = 20

3 x 1 = 3       3 x 2 = 6       3 x 3 = 9       3 x 4 = 12      3 x 5 = 15      3 x 6 = 18      3 x 7 = 21      3 x 8 = 24      3 x 9 = 27      3 x 10 = 30

4 x 1 = 4       4 x 2 = 8       4 x 3 = 12      4 x 4 = 16      4 x 5 = 20      4 x 6 = 24      4 x 7 = 28      4 x 8 = 32      4 x 9 = 36      4 x 10 = 40

5 x 1 = 5       5 x 2 = 10      5 x 3 = 15      5 x 4 = 20      5 x 5 = 25      5 x 6 = 30      5 x 7 = 35      5 x 8 = 40      5 x 9 = 45      5 x 10 = 50

6 x 1 = 6       6 x 2 = 12      6 x 3 = 18      6 x 4 = 24      6 x 5 = 30      6 x 6 = 36      6 x 7 = 42      6 x 8 = 48      6 x 9 = 54      6 x 10 = 60

7 x 1 = 7       7 x 2 = 14      7 x 3 = 21      7 x 4 = 28      7 x 5 = 35      7 x 6 = 42      7 x 7 = 49      7 x 8 = 56      7 x 9 = 63      7 x 10 = 70

8 x 1 = 8       8 x 2 = 16      8 x 3 = 24      8 x 4 = 32      8 x 5 = 40      8 x 6 = 48      8 x 7 = 56      8 x 8 = 64      8 x 9 = 72      8 x 10 = 80

9 x 1 = 9       9 x 2 = 18      9 x 3 = 27      9 x 4 = 36      9 x 5 = 45      9 x 6 = 54      9 x 7 = 63      9 x 8 = 72      9 x 9 = 81      9 x 10 = 90

10 x 1 = 10     10 x 2 = 20     10 x 3 = 30     10 x 4 = 40     10 x 5 = 50     10 x 6 = 60     10 x 7 = 70     10 x 8 = 80     10 x 9 = 90     10 x 10 = 100


Good bye!
Press any key to exit this program.
```

# L3: for-loops

Write a program that uses two for-loops (not nested) in order to output the following:

```
Output for first for-loop:
75 70 65 60 55 50 45 40 35 30 25 20 15

Output for second for-loop:
1 2 4 5 8 10 11 13 16 17 19 20 22 23 25 26 29 31 32 34 37 38 40 41 43 44 46 47        
```

So the second for-loop shall output all numbers between 1 and 50 except numbers that are divisible by 3 or 7.


# L4: Two nested for-loops

Write a C program that first outputs lines 1-10 using two nested for-loops and then outputs line 11-20 using two nested for-loops again.

```
0
0 1
0 1 2
0 1 2 3
0 1 2 3 4
0 1 2 3 4 5
0 1 2 3 4 5 6
0 1 2 3 4 5 6 7
0 1 2 3 4 5 6 7 8
0 1 2 3 4 5 6 7 8 9
*
* *
* * *
* * * *
* * * * *
* * * * * *
* * * * * * *
* * * * * * * *
* * * * * * * * *
* * * * * * * * * *
```


# L5: Transforming one loop into another

Write a C program that outputs the numbers 17,20,23,26,29 using 1.) a for-loop, 2.) using a while-loop and 3.) using a do-while loop:

```
1. With a for-loop:
17 20 23 26 29

2. With a while-loop:
17 20 23 26 29

3. With a do-while-loop:
17 20 23 26 29
```


# L6: Drawing a rectangle

Write a C program that asks the user to enter the height and width of an rectangle and then draws a corresponding rectangle:

```
Height of rectangle? 10
Width of rectangle? 20
####################
#                  #
#                  #
#                  #
#                  #
#                  #
#                  #
#                  #
#                  #
####################

Height of rectangle? 5
Width of rectangle? 30
##############################
#                            #
#                            #
#                            #
##############################

Height of rectangle? 10
Width of rectangle? 40
########################################
#                                      #
#                                      #
#                                      #
#                                      #
#                                      #
#                                      #
#                                      #
#                                      #
########################################
```


# L7: Drawing a rectangle with one loop

In exercise L6 the task was to draw a rectangle of a specified size. Probably your solution makes use of several loops. Can you do the same using only one loop? (yes it is possible!)


# L8: Drawing a triangle

Write a C program where the user can enter the number of desired rows and your program draws a corresponding triangle:

```
Number of rows for triangle? 3
                                   *
                                  ***
                                 *****

Number of rows for triangle? 8
                                    *
                                   ***
                                  *****
                                 *******
                                *********
                               ***********
                              *************
                             ***************

Number of rows for triangle? 5
                                    *
                                   ***
                                  *****
                                 *******
                                *********

Number of rows for triangle? 15
                                    *
                                   ***
                                  *****
                                 *******
                                *********
                               ***********
                              *************
                             ***************
                            *****************
                           *******************
                          *********************
                         ***********************
                        *************************
                       ***************************
                      *****************************
```


# R1: Random numbers in a defined range

Write a C program that asks the user to enter a lower and upper border of an interval and then generates and outputs 10 random numbers in that interval.

The program shall stop only if the lower and upper border entered are equal.

```
Enter lower border: 0
Enter upper border: 5
I will generate 10 random numbers between 0 and 5 ...
5       5       4       4       5       4       0       0       4       2

Enter lower border: 10
Enter upper border: 12
I will generate 10 random numbers between 10 and 12 ...
12      12      11      10      11      12      11      12      10      10

Enter lower border: 0
Enter upper border: 1
I will generate 10 random numbers between 0 and 1 ...
1       0       0       1       0       0       1       0       0       1

Enter lower border: -10
Enter upper border: -5
I will generate 10 random numbers between -10 and -5 ...
-5      -10     -5      -10     -7      -6      -5      -9      -9      -10

Enter lower border: 0
Enter upper border: 0


Good bye!
Press any key to exit this program.
```


# S1: Simple calculator

Write a simple calculator in C that reads in a single character using the _getch() method with

```c
char c = _getch();
```

and only accepts as input the operator characters + - * / and =. Only the numbers 0,1,2,...,9 will be accepted.

The program shall compute the expression and output the current result if the operator character = is entered by the user.

When started, the calculator will expect a digit 0,...,9 and then an operator character + - * / =, then a digit, then an operator character, etc.

If a wrong input is entered, e.g., two times a digit or two times an operator character, the new input shall be ignored and using

```c
printf("\a");
```

a beep sound can be played to inform the user that the input was wrong.

**Note: Use switch-case statements to distinguish different cases!**

Example program run:

```
Simple calculator:
Valid inputs are + - * /  = and digits 0,...,9

Your input:
1 + 2 + 3
= 6.00 + 2 + 1
= 9.00 / 2
= 4.50 * 3 * 8
= 108.00 / 9 / 2 + 1
= 7.00 - 5 - 1 - 1 - 1 - 1
= -2.00 - 8
= -10.00 + 9
= -1.00 + 1
= 0.00 + 5
= 5.00
```



# A1: Growing rectangle

Write an ASCII animation in C where we can see a growing rectangle.

**Hints:**

- With `system("cls")` you can clear the console.
- With `Sleep(500)` you can wait for 500ms.
- And here is a code snippet that shows you how to position the cursor at some position (x,y) in a Windows console:

```c
#include <windows.h>
                
// Set cursor position in console
// see http://www.dreamincode.net/forums/topic/153240-console-cursor-coordinates/ 
void set_cursor_position(int x, int y)
{
   //Initialize the coordinates
   COORD coord = { x, y };
   //Set the position
   SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), coord);

} // set_cursor_position            
```


# F1: Call-by-reference

Write a program in C where you use 3 nested for loops to run through all combinations (n1,n2,n3) where n1,n2,n3 are between 1 and 5.

Write a function statistics() that allows you to pass n1,n2,n3 by value and a variable sum and avg by reference.

This function shall compute the sum and average of n1,n2,n3 and store it in sum and average respectively.

Then for each combination (n1,n2,n3) call the function statistics() and after the function has returned, output the computed sum and average.

```
n1=0 n2=0 n3=0 --> sum=0, avg=0.00
n1=0 n2=0 n3=1 --> sum=1, avg=0.33
n1=0 n2=0 n3=2 --> sum=2, avg=0.67
n1=0 n2=0 n3=3 --> sum=3, avg=1.00
n1=0 n2=0 n3=4 --> sum=4, avg=1.33
n1=0 n2=0 n3=5 --> sum=5, avg=1.67
n1=0 n2=1 n3=0 --> sum=1, avg=0.33
n1=0 n2=1 n3=1 --> sum=2, avg=0.67
n1=0 n2=1 n3=2 --> sum=3, avg=1.00
n1=0 n2=1 n3=3 --> sum=4, avg=1.33
n1=0 n2=1 n3=4 --> sum=5, avg=1.67
n1=0 n2=1 n3=5 --> sum=6, avg=2.00
n1=0 n2=2 n3=0 --> sum=2, avg=0.67
n1=0 n2=2 n3=1 --> sum=3, avg=1.00
n1=0 n2=2 n3=2 --> sum=4, avg=1.33
n1=0 n2=2 n3=3 --> sum=5, avg=1.67
n1=0 n2=2 n3=4 --> sum=6, avg=2.00
n1=0 n2=2 n3=5 --> sum=7, avg=2.33
n1=0 n2=3 n3=0 --> sum=3, avg=1.00
n1=0 n2=3 n3=1 --> sum=4, avg=1.33
n1=0 n2=3 n3=2 --> sum=5, avg=1.67
n1=0 n2=3 n3=3 --> sum=6, avg=2.00
...
```


# F2: Recursive functions

Write a function fibonacci_recursive(int n) that computes the [Fibonacci number](https://en.wikipedia.org/wiki/Fibonacci_number) Fn for a given n using recursive function calls, i.e., fibonacci_resurvive() will call itself inside the function.

Then implement a second function fibonacci_iterative(int n) that can also compute the Fibonacci number Fn for a given n, but does it without recursive function calls.

Compare both implementations regarding their execution time using the [clock()](http://www.cplusplus.com/reference/ctime/clock/) function.

An example output of your program could look like this:

```
Function test:
F(6)  is according to fibonacci_recursive() 8
F(6)  is according to fibonacci_iterative() 8
F(12) is according to fibonacci_recursive() 144
F(12) is according to fibonacci_iterative() 144

Speed test:
sum according to fibonacci_recursive() is 165580140
time needed using fibonacci_recursive() is 11.7810001373 seconds
sum according to fibonacci_iterative() is 165580140
time needed using fibonacci_iterative() is 0.0000000000 seconds
```


# Y1: Writing blocks into a 2D array

Imagine you want to prepare a randomly generated game board for a small game.

The game board shall consist of a rectangle border and some randomly generated blocks of size 3x3,4x4,5x5, or 6x6 inside the rectangle.

Use a 2D array to store the information for each game board position (x,y) whether there is a free space ' ' or a wall '#'.

Structure your code using functions such that you can guarantee to have a good overview:

Use a function init_board() to "draw" the outer walls into your 2D array.

Use a function add_single_block() with appropriate parameters to insert a single block into your 2D array. Then call this function 20 times inside your program with random (x,y) positions for the blocks and random block sizes of 3x3, 4x4,5x5, or 6x6.

Use another function show_game_board() to visualize the current contents of your 2D array in the console.

Two examples of randomly generated game boards:

```
############################################################
#               ###                                        #
#        ###  #####                                        #
#     ######  ####                                 ###     #
#     ######  ####                                 ###     #
#   #####     ####  ####                           ###     #
#   ###             ####                                   #
#   ###             ####                                   #
#                   ####                                   #
#         #####      #####                                 #
#         #####      #####                                 #
#         #####      #####              ####               #
#      ########      #####             #####               #
#      ########      #####         #########               #
#      ####         ###            #########               #
#      ####         ###            ########                #
#      ####         ###                                    #
#      ####              #####                 ######      #
#      ####              #####                 ######      #
############################################################
```

```
############################################################
#                                                         ##
#               ###                        ###            ##
#               ###      #####             ###            ##
#               ###      #####             ###            ##
#                        ##### ####     ####              ##
#                        ##### ####     ####               #
#                      ####### ####     ########   #####   #
#                      #####   ####     ########   #####   #
#                     ####                  ####   #####   #
#           ####      ####                  ####   #####   #
#           ####      ####                ### ##########   #
#           ####      ####                ### #####        #
#  ###      #####    #####                #########        #
#  ###       ####    #####                 ########        #
#  ###       ####    #####                 ########        #
#            ####    #####                 #####           #
#                    #########             #####   #####   #
#                        #####                     #####   #
############################################################
```


# N1: Word generator given three syllables from user input

Write a C program that reads three syllables from the user and then generates all words that consist of two of the syllables entered.

Example program run #1:

```
Enter syllable #1: do
Enter syllable #2: te
Enter syllable #3: ma
dodo
dote
doma
tedo
tete
tema
mado
mate
mama
```

Example program run #2:

```
Enter syllable #1: lab
Enter syllable #2: or
Enter syllable #3: ja
lablab
labor
labja
orlab
oror
orja
jalab
jaor
jaja
```


# N2: Word generator given syllables defined in code

Write a C program in which you define 6 syllables within your code using a 1D array of string literals.

All syllables that are stored and used in the following to generate words randomly shall be printed to the console at program start.

Then generate 10 words consisting of a random number of syllables. Each word consists at least of 2 syllables, but not more than 5 syllables.

For each generated word also output the number of letters is has.

Example program run:

```
Syllables stored: tan han klam ro ba lo

Generated word #1 has 6 letters : lobaba
Generated word #2 has 8 letters : batantan
Generated word #3 has 11 letters : klamlolohan
Generated word #4 has 14 letters : hanlohanklamro
Generated word #5 has 5 letters : rotan
Generated word #6 has 8 letters : robabaro
Generated word #7 has 6 letters : klamlo
Generated word #8 has 12 letters : tanlotanroba
Generated word #9 has 13 letters : hanhantanloro
Generated word #10 has 10 letters : roroklamro
```


# T1: A pizza order system

Write a C program which implements a pizza order system. The user shall be able to order up to 10 pizzas. For each pizza the user shall be asked which pizza type he wants and which size.

Use enums to represent the different pizza types and a struct to store the information about a single pizza being ordered. An array shall be used to store all (of the up to 10) pizzas that can be ordered.

Example program run:

```
Hello! Welcome to UASKPS! - University of Applied Sciences Kempten Pizza service!
You can order up to 10 pizzas.

Press any key to start your order.
            
Please choose your pizza #1

Available pizza types:
         0 : Pizza Margherita
         1 : Salami Classico
         2 : Funghi
         3 : Hawaiian Dream

Your choice: 3

Available pizza sizes:
         size 0 : Small
         size 1 : Medium
         size 2 : Large
         size 3 : Extra Large
         size 4 : XXL
         size 5 : Party size

Your desired size: 2
Do you want to order another pizza? (y/n) y
  
Please choose your pizza #2

Available pizza types:
         0 : Pizza Margherita
         1 : Salami Classico
         2 : Funghi
         3 : Hawaiian Dream

Your choice: 5
There is no pizza type with number 5
Please enter a number between 0 and 3!

Your choice: 1

Available pizza sizes:
         size 0 : Small
         size 1 : Medium
         size 2 : Large
         size 3 : Extra Large
         size 4 : XXL
         size 5 : Party size

Your desired size: 0
Do you want to order another pizza? (y/n) n
  
Here is your complete order:

Pizza #0: Hawaiian Dream (Large)
Pizza #1: Salami Classico (Small)


Thank you for your order! Press any key to exit.
```


# D1: Allocating a 1D array dynamically

Write a C program that asks the user for the size of a 1D array and then allocates a float array of corresponding size.

The values in the lower half of the array shall be initialized with the value 123.0 and the values in the upper half shall be initialized with the value 456.0.

If the array has an odd size, the middle element shall be initialized with the value 999.0.

After the initialization of the array all values shall be printed in the console and then the array shall be deallocated.

```
Enter the size of the array: 4
Generating your array...
Initializing your array ...
Here are all the values of your array ...
A[0]=123.0
A[1]=123.0
A[2]=456.0
A[3]=456.0
Deallocating the array ...


Press a key to allocate another array.

Enter the size of the array: 5
Generating your array...
Initializing your array ...
Here are all the values of your array ...
A[0]=123.0
A[1]=123.0
A[2]=999.0
A[3]=456.0
A[4]=456.0
Deallocating the array ...
```


# D2: Notebook

Write a program in C which allows to store simple notes. The user can enter as many notes as he likes as long as there is enough memory.

The program shall be able to output all notes stored so far.

Example output after entering four notes:

```
Simple notebook for storing notes.
Menu:
1) Enter a new note
2) Show all notes taken so far
3) Exit program

Notes stored so far:
Note #0 : Buy Xmas present for grandma
Note #1 : Wash car
Note #2 : Buy tickets for cinema
Note #3 : Do more sports!


Enter a key to return to menu.
```


# I1: Count number of lines in a text file

Write a C program that reads in an absolute filename of a text file from the user. Then the program shall analyze how many text lines are there in the file and output the number of lines onto the console.

```
Enter filename with absolute path --> abc.txt
Error! I could not open the specified file abc.txt
Enter filename with absolute path --> testfile.txt
Error! I could not open the specified file testfile.txt
Enter filename with absolute path --> V:\tmp\testfile.txt
The file V:\tmp\testfile.txt contains 4 lines.
Press any key to exit this program.
```


# I2: Read in numbers from file and write numbers sorted into another

Write a C program that reads in an absolute filename of a text file from the user. The text file shall only contain numbers which are sorted or unsorted. The program shall read in the numbers, sort them in ascending order (e.g., using the [Bubble sort](https://en.wikipedia.org/wiki/Bubble_sort) algorithm), and then write the sorted numbers back to another file.

```
Enter filename with absolute path --> V:\tmp\numbers.txt
Number #0: 19.20000
Number #1: 12.00000
Number #2: 13.70000
Number #3: 20.21310
Number #4: 7.87392
Press any key to exit this program.
```

and the new file V:\tmp\numbers.txt.sorted.txt shall then have the following contents (numbers now sorted in ascending order):

```
7.87392
12.00000
13.70000
19.20000
20.21310
```


# O1: NumberBox - A first class for storing numbers

Write a first C++ class NumberBox which is able to store numbers. The constructor of the class shall expect as argument the desired maximum storage capacity.

Then write member functions:

- add_a_number() which shall allow to add a number to the box (if there is still space in the box)
- show_all_numbers_in_box() shall output all numbers currently stored
- get_min_max() which shall allow to retrieve the information about the minimum and maximum value of the numbers stored so far
- get_arithmetic_mean() which shall compute and return the mean of all numbers stored so far

Which member variables / functions should be declared as public?  
Which member variables / functions should be declared as private?

If you have implemented your NumberBox class test it with the following main() function:

```cpp
int main()
{
   NumberBox myBox(10);

   myBox.add_a_number(1.0);
   myBox.add_a_number(4.0);
   myBox.add_a_number(7.0);
   myBox.show_all_numbers_in_box();

   myBox.add_a_number(11.0);
   myBox.add_a_number(44.0);
   myBox.add_a_number(77.0);
   myBox.show_all_numbers_in_box();
      

   double minval, maxval;
   myBox.get_min_max(&minval,&maxval);
   printf("Minimum value of all numbers in box is: %.1f\n", minval);
   printf("Maximum value of all numbers in box is: %.1f\n", maxval);

   printf("Arithmetic mean of all numbers in box is: %.1f\n", myBox.get_arithmetic_mean());


   myBox.add_a_number(100.0);
   myBox.add_a_number(200.0);
   myBox.add_a_number(300.0);
   myBox.add_a_number(400.0);
   myBox.add_a_number(500.0);
   myBox.add_a_number(600.0);

   _getch();          
}
```

The output of your console should look like this:

```
Constructor of class NumberBox was called.
I created a number box which can store up to 10 numbers.

added number 1.0 to your number box
added number 4.0 to your number box
added number 7.0 to your number box
Numbers currently in the box: 1.0 4.0 7.0

added number 11.0 to your number box
added number 44.0 to your number box
added number 77.0 to your number box
Numbers currently in the box: 1.0 4.0 7.0 11.0 44.0 77.0

Minimum value of all numbers in box is: 1.0
Maximum value of all numbers in box is: 77.0
Arithmetic mean of all numbers in box is: 24.0
added number 100.0 to your number box
added number 200.0 to your number box
added number 300.0 to your number box
added number 400.0 to your number box
Sorry! Cannot add another number to your box! The box is full!
Sorry! Cannot add another number to your box! The box is full!
```


# O2: Xmas party simulator

Write a Xmas party simulator in C++ where you model the different object types that interact at Xmas with the help of classes.

There are four important Xmas object types at Xmas:

1. Xmas tree: which can store up to N presents
2. Santa Claus: which produces presents and puts them under a Xmas tree
3. Present: which can be socks, cat, dogs, and smartphones
4. Child: which takes (consumes) Xmas presents and puts them in its private list

Write a base class xmas_object with a method do_something() and derive each of the four classes xmas_tree, santaclaus, present, and child from this base class.

Then write a simulation loop where in each loop one of the Xmas objects is randomly selected and its do_something() method is called.

In the santaclaus class override the do_something() method from the base class such that it generates a new present object and puts it into a list of presents hold by the Xmas tree object each time do_something() is called.

The xmas_tree class shall only output a list of presents currently stored when the do_something() method is called.

The present class is used to store information about which type of present it is: socks, cat, dog, or smartphone. Use a enum for the present type.

An instance of the child class tries to get a present object from the xmas_tree class. If there is no present any more, the xmas_tree class shall return a NULL pointer. If there is still a present, the child object puts the present in its own private list of presents.

Read the number of children at the beginning of the program from the user and generate a corresponding number of child objects (children).

Start the Xmas party simulation with 5 presents already lying under the Xmas tree.

The output of a Xmas party simulation could look like this:

```
Welcome to the Xmas simulator!
How many children do you want to simulate? 2

Xmas Tree:
         There are 5 presents under the tree!
         socks dog cat smartphone socks

Santa Claus:
         Ho ho ho! I have come to bring a new present (socks)!


Xmas Tree:
         There are 6 presents under the tree!
         socks dog cat smartphone socks socks

Santa Claus:
         Ho ho ho! I have come to bring a new present (dog)!


Xmas Tree:
         There are 7 presents under the tree!
         socks dog cat smartphone socks socks dog

Child 0:
         I will try to get another present!
         Here are all the presents I have so far:

         Ooooh now! I hate getting socks!

Xmas Tree:
         There are 6 presents under the tree!
         socks dog cat smartphone socks dog

Child 0:
         I will try to get another present!
         Here are all the presents I have so far:
                  socks
         HURRAY! The new present is a dog!

Xmas Tree:
         There are 5 presents under the tree!
         socks dog cat smartphone socks

Child 0:
         I will try to get another present!
         Here are all the presents I have so far:
                  socks dog
         Ooooh now! I hate getting socks!

Xmas Tree:
         There are 4 presents under the tree!
         socks dog cat smartphone

Child 0:
         I will try to get another present!
         Here are all the presents I have so far:
                  socks dog socks
         HURRAY! The new present is a cat!

Xmas Tree:
         There are 3 presents under the tree!
         socks dog smartphone

Child 0:
         I will try to get another present!
         Here are all the presents I have so far:
                  socks dog socks cat
         HURRAY! The new present is a smartphone!

Xmas Tree:
         There are 2 presents under the tree!
         socks dog

Child 0:
         I will try to get another present!
         Here are all the presents I have so far:
                  socks dog socks cat smartphone
         Ooooh now! I hate getting socks!

Xmas Tree:
         There are 1 presents under the tree!
         dog

Child 1:
         I will try to get another present!
         Here are all the presents I have so far:

         HURRAY! The new present is a dog!

Xmas Tree:
         There are 0 presents under the tree!


Santa Claus:
         Ho ho ho! I have come to bring a new present (socks)!


Xmas Tree:
         There are 1 presents under the tree!
         socks

Child 0:
         I will try to get another present!
         Here are all the presents I have so far:
                  socks dog socks cat smartphone socks
         Ooooh now! I hate getting socks!

Xmas Tree:
         There are 0 presents under the tree!


Santa Claus:
         Ho ho ho! I have come to bring a new present (cat)!


Xmas Tree:
         There are 1 presents under the tree!
         cat
```



# O3: printf() vs. cout

In C++ you can output console information with the function printf() but also with the help of the cout object, which is an instance of the class [ostream](http://www.cplusplus.com/reference/ostream/ostream/).

Conduct some experiments such as outputting:

- variables of different types
- floats with a certain number of digital places
- floats with a certain number of digital places and a fixed field length
- object information

and try to find out the advantages and disadvantages of printf() vs. cout.


# O4: Pointers vs. References

In C there were only pointers. C++ also introduced the concept of "references".

Find out what a reference is by searching [discussion forums](http://stackoverflow.com/questions/57483/what-are-the-differences-between-a-pointer-variable-and-a-reference-variable-in) and experimenting with references.

Then formulate the difference between pointers and references in some few sentences with your own words!


# O5: Shallow vs. Deep Copy

Take the implementation of the NumberBox class from exercise O1.

Then add a new member function copy() that allows to generate a copy of a NumberBox object.

Test your implementation with the following code snippet:

```cpp
int main()
{
   // 1. create a number box #1
   NumberBox* myBox1 = new NumberBox(10);
   myBox1->add_a_number(1.0);
   myBox1->add_a_number(4.0);
   myBox1->add_a_number(7.0);
      
   // 2. get a copy of the number box
   NumberBox* myBox2 = myBox1->copy();

   // 3. add some numbers to number box #2
   myBox2->add_a_number(14.0);
   myBox2->add_a_number(18.0);

   // 4. show content of number box #1 and #2
   printf("\n");
   printf("myBox1: ");
   myBox1->show_all_numbers_in_box();
   printf("myBox2: ");
   myBox2->show_all_numbers_in_box();

   _getch();
}
```

The output of your program should look like this:

```
Constructor of class NumberBox was called.
I created a number box which can store up to 10 numbers.

added number 1.0 to your number box
added number 4.0 to your number box
added number 7.0 to your number box

Constructor of class NumberBox was called.
I created a number box which can store up to 10 numbers.

added number 14.0 to your number box
added number 18.0 to your number box

myBox1: Numbers currently in the box: 1.0 4.0 7.0

myBox2: Numbers currently in the box: 1.0 4.0 7.0 14.0 18.0
```


# Appendix 1: Clear Screen / Position the cursor under Linux

Some people asked me how to clear the screen and position the cursor under Linux.

Here is a solution / [link](https://stackoverflow.com/questions/26423537/how-to-position-the-input-text-cursor-in-c/26423946):

```c
#include "stdio.h"

#define clear() printf("\033[H\033[J")
#define gotoxy(x,y) printf("\033[%d;%dH", (y), (x))

int main(void)
{
      int number;

      clear();
      printf(
         "Enter your number in the box below\n"
         "+-----------------+\n"
         "|                 |\n"
         "+-----------------+\n"
      );
      gotoxy(2, 3);
      scanf("%d", &number);
      return 0;
}
```


# Appendix 2: Position the cursor under Windows

Here is a code snippet that shows how to position the cursor in a Windows console:

```cpp
#include <windows.h>
        
// Set cursor position in console
// see http://www.dreamincode.net/forums/topic/153240-console-cursor-coordinates/ 
void set_cursor_position(int x, int y)
{
   //Initialize the coordinates
   COORD coord = { x, y };
   //Set the position
   SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), coord);

} // set_cursor_position
```