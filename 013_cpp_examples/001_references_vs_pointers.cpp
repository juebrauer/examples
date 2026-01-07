// Example for talking about
// what references are
//  and
// what the differences are compared to pointers
//
// Note: can be executed as a quiz


#include <iostream>
#include <limits>
using namespace std;

void waitForKey(const string& message) {
    cout << "\n--- " << message << " ---\n";
    cout << "Press ENTER to continue...";
    cin.ignore(numeric_limits<streamsize>::max(), '\n');
}

// Normal functions (no lambdas, no auto)
void incrementByReference(int& v) {
    v++;
}

void incrementByPointer(int* v) {
    if (v != nullptr) {
        (*v)++;
    }
}

int main() {

    cout << "===== SECTION 1: Basic concept =====\n";
    int x = 10;

    int& ref = x;     // reference (alias)
    int* ptr = &x;    // pointer (stores address)

    waitForKey("What will x, ref, and *ptr output?");

    cout << "x   = " << x << endl;
    cout << "ref = " << ref << endl;
    cout << "*ptr = " << *ptr << endl;


    cout << "\n===== SECTION 2: Modifying the value =====\n";
    waitForKey("What happens after ref = 20?");

    ref = 20;
    cout << "x = " << x << endl;

    waitForKey("What happens after *ptr = 30?");

    *ptr = 30;
    cout << "x = " << x << endl;


    cout << "\n===== SECTION 3: Reassignment =====\n";
    int y = 100;

    waitForKey("What does ref = y do?");

    ref = y;  // assigns value of y to x
    cout << "x = " << x << ", y = " << y << endl;

    waitForKey("What happens when the pointer is reseated?");

    ptr = &y;
    *ptr = 200;
    cout << "y = " << y << endl;


    cout << "\n===== SECTION 4: Addresses =====\n";
    ptr = &x;
    waitForKey("Are these addresses the same or different?");

    cout << "Address of x:    " << &x << endl;
    cout << "Address via ref: " << &ref << endl;
    cout << "Value stored in ptr (address of x): " << ptr  << endl;
    cout << "Address of ptr itself:              " << &ptr << endl;
    cout << "Value pointed to by ptr:            " << *ptr << endl;


    cout << "\n===== SECTION 5: Nullability =====\n";
    int* nullablePtr = nullptr;

    waitForKey("Can pointers be null? What about references?");

    if (nullablePtr == nullptr) {
        cout << "Pointer is null\n";
    }

    // This would NOT compile:
    // int& badRef = nullptr;


    cout << "\n===== SECTION 6: Function-style usage =====\n";
    int z = 5;

    waitForKey("What will z be after incrementByReference(z)?");
    incrementByReference(z);
    cout << "z = " << z << endl;

    waitForKey("What will z be after incrementByPointer(&z)?");
    incrementByPointer(&z);
    cout << "z = " << z << endl;


    cout << "\n===== SUMMARY =====\n";
    waitForKey("Final recap");

    cout << "- References are aliases, cannot be null, cannot be reseated\n";
    cout << "- Pointers store addresses, can be null, can change targets\n";

    return 0;
}
