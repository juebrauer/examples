// Example that shows
// a matrix class with pointers in the add-function
// --> we need to check for nullptr

#include <iostream>

class Matrix
{
public:
    Matrix(int rows, int cols, int startValue=0)
        : rows(rows), cols(cols)
    {
        data = new int[rows * cols];

        for (int y = 0; y < rows; y++)
        {
            for (int x = 0; x < cols; x++)
            {
                int idx = y * cols + x;
                data[idx] = startValue;
            }
        }
    }

    ~Matrix()
    {
        std::cout << "Cleaning up data ..." << std::endl;
        delete[] data;
    }

    // Pointer version: other might be nullptr
    Matrix* add(const Matrix* other) const
    {
        if (other == nullptr)
            return nullptr;

        Matrix* result = new Matrix(rows, cols);

        for (int i = 0; i < rows * cols; i++)
        {
            result->data[i] = data[i] + other->data[i];
        }

        return result;
    }

    void show() const
    {
        for (int y = 0; y < rows; y++)
        {
            for (int x = 0; x < cols; x++)
            {
                std::cout << data[y * cols + x] << " ";
            }
            std::cout << "\n";
        }
    }

private:
    int rows;
    int cols;
    int* data;
};

int main()
{
    Matrix* m1 = new Matrix(3, 5,  1);
    Matrix* m2 = new Matrix(3, 5,  19);
    m1->show();
    m2->show();

    Matrix* m3 = m1->add(m2);
    if (m3)
        m3->show();

    delete m1;
    delete m2;
    delete m3;
}
