// table sctruct and helper functions for 2D Ising model simulation(s)

// including used libraries
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

// ------------------------------------------------------------------------------------------------------------------------------------------------------

// square function
template <typename T>
auto sq = [](T const &x) { return (T)(x * x); };

// ------------------------------------------------------------------------------------------------------------------------------------------------------

// struct for simulation table
template <typename T>
struct Table
{
    // variables
    std::vector<T> data;
    int dim;

    // constructor(s)
    // default
    Table() : dim{0} {}
    // parametrized #1
    Table(int L) : data{std::vector<T>(sq<int>(L), T(0))}, dim{L} {}
    // parametrized #2
    Table(int L, std::vector<T> const &vec) : data{vec}, dim{L}
    {
        if (static_cast<double>(sq<int>(dim)) - static_cast<double>(vec.size()))
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
        data.resize(static_cast<size_t>(sq<int>(dim)));
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

// write to file
auto WriteToFile = [](auto &file, auto const &data) {
    for (auto &e : data)
        file << e << " ";
    file << "\n";
};
