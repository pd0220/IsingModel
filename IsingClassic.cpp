// including used libraries
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>

// ------------------------------------------------------------------------------------------------------------------------------------------------------

// declaring constant(s)
// beta * J ~ coupling
const double betaJ = 2.;
// spatial size of simulation table
const int spatialSize = 5;

// ------------------------------------------------------------------------------------------------------------------------------------------------------

// struct for simulation table
template <typename T>
struct Table
{
    // variables
    int dim;
    std::vector<T> data;

    // constructor(s)
    // default
    Table() : dim{0} {}
    // parametrized #1
    Table(int L) : dim{L}, data{std::vector<T>((L * L), T(0))} {}
    // parametrized #2
    Table(int L, std::vector<T> const &vec) : dim{L}, data{vec}
    {
        if (static_cast<double>(dim * dim) - static_cast<double>(vec.size()))
        {
            std::cout << "Square matrix cannot be created." << std::endl;
            std::exit(-1);
        }
    }
    // copy
    Table(Table const &) = default;
    // move
    Table(Table &&) = default;
    // copy assignment
    Table<T> &operator=(Table const &) = default;
    // move assignment
    Table<T> &operator=(Table &&) = default;
    // initialize with function
    template <typename func>
    Table(func f, int L) : dim{L}
    {
        data.resize(static_cast<size_t>(L * L));
        std::generate(data.begin(), data.end(), f);
    }

    // indexing simplifier
    T &operator[](int index)
    {
        return data[index];
    }
    T const &operator[](int index) const
    {
        return data[index];
    }
    T &operator()(int row, int col)
    {
        return data[row * dim + col];
    }
    T const &operator()(int row, int col) const
    {
        return data[row * dim + col];
    }

    // get table size
    int get_dim() const
    {
        return (*this).dim;
    }

    // ostream
    friend std::ostream &operator<<(std::ostream &o, Table<T> const &table)
    {
        int L = table.get_dim();
        for (int iRow = 0; iRow < L; iRow++)
        {
            for (int iCol = 0; iCol < L; iCol++)
            {
                // structuring
                std::cout << (table(iRow, iCol) > 0 ? " 1" : "-1") << "\t";
            }
            std::cout << "\n";
        }
        return o;
    }
};

// ------------------------------------------------------------------------------------------------------------------------------------------------------

// main function
int main(int, char **)
{
    // random number generation
    std::random_device rd{};
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> distr(0., 1.);
    // random generator lambda
    auto rand = [&distr, &gen]() {
        return (double)distr(gen) > 0.5 ? 1 : -1;
    };

    // initialize spins
    Table table = Table<int>(rand, spatialSize);
    // write table to screen
    std::cout << table << std::endl;
}
