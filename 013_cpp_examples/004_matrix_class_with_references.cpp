// Example that shows
// a matrix class with a reference in the add-function
// --> we do not need to check whether other (matrix) is valid

#include <iostream>

class Matrix
{
public:
    Matrix(int rows, int cols)
        : rows(rows), cols(cols)
    {
        data = new int[rows * cols];

        for (int y = 0; y < rows; y++)
        {
            for (int x = 0; x < cols; x++)
            {
                int idx = y * cols + x;
                if (y == 0)
                    data[idx] = 1;
                else
                    data[idx] = 0;
            }
        }
    }

    // Destructor: frees owned memory
    ~Matrix()
    {
        delete[] data;
    }

    // Matrix addition: other must exist → reference
    Matrix* add(const Matrix& other) const
    {
        Matrix* result = new Matrix(rows, cols);

        for (int y = 0; y < rows; y++)
        {
            for (int x = 0; x < cols; x++)
            {
                int idx = y * cols + x;
                result->data[idx] = data[idx] + other.data[idx];
            }
        }

        return result;
    }

    void show() const
    {
        for (int y = 0; y < rows; y++)
        {
            for (int x = 0; x < cols; x++)
            {
                int idx = y * cols + x;
                std::cout << data[idx] << " ";
            }
            std::cout << std::endl;
        }
    }

private:
    int rows;
    int cols;
    int* data;
};


int main()
{
    Matrix m1(3, 5);
    Matrix m2(3, 5);

    m1.show();
    m2.show();

    Matrix* m3 = m1.add(m2);
    m3->show();

    delete m3;
}