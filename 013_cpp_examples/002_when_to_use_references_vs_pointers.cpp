// Example for talking about
// when to use references and when to use pointers
//
// For this example, assume that we need to store the
// information who is currently logged in (only one user can
// be logged into a system at once)
//
// Rule:
// 1. Use a reference when there must always be a valid object.
// 2. Use a pointer when a valid object is not guaranteed.
//
// A reference asserts:
// - “There is an object”
// - “It will not change which object it is”
// - “You don’t need to check”
// Reference = guarantee
//
// A pointer admits uncertainty:
// - “There might be an object”
// - “There might be none”
// - “You must check”
// Pointer = uncertainty
// E.g., pointers can be used for a linked list
// Either they point to the next element (e.g., a struct)
// or they are nullptr (= end of the list)



#include <iostream>
#include <string>
using namespace std;

struct User {
    string name;
};

// This function ALWAYS needs a valid user
void printUserProfile(const User& user) {
    cout << "User profile: " << user.name << endl;
}

// This function may or may not have a logged-in user
void printCurrentUser(const User* user) {
    if (user != nullptr) {
        cout << "Current user: " << user->name << endl;
    } else {
        cout << "No user is currently logged in\n";
    }
}

int main() {
    User u{"Alice"};

    printUserProfile(u);       // valid user must exist
    printCurrentUser(&u);      // someone is logged in
    printCurrentUser(nullptr);     // nobody logged in

    return 0;
}
