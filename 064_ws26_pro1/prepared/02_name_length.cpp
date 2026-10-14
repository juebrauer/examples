#include <iostream>
#include <string>

using namespace std;

int main()
{
    string name;

    cout << "Wie heisst du? ";
    cin >> name;

    cout << "Hallo " << name << "!\n";
    cout << "Dein Name hat " << name.length()
              << " Zeichen.\n";

    for (int i=0; i<=5; i++)
        cout << "Iteration " << i << "\n";  

    return 0;
}

