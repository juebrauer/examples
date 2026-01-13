// Example that shows
// a matrix class with a reference in the add-function
// --> we do not need to check whether other (matrix) is valid

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
    
    // Referenz-Version: other kann nicht nullptr sein
    Matrix add(const Matrix& other) const
    {
        Matrix result(rows, cols, 0);
        for (int i = 0; i < rows * cols; i++)
        {
            result.data[i] = data[i] + other.data[i];
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
    Matrix m1(3, 5,  1);
    Matrix m2(3, 5,  19);    
    m1.show();
    m2.show();
    
    Matrix m3 = m1.add(m2);
    m3.show();
    
    // No delete necessary - automatic call of destructor!
}