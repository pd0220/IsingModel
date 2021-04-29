// including used libraries
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <cmath>

// ------------------------------------------------------------------------------------------------------------------------------------------------------

// spatial size of simulation table (use > 1)
const int spatialSize = 50;
// integration time
const int time = 15000;
// thermalisation time
const int thermTime = 1000;
// scale for coupling index
const double scalar = 50.;

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

// write to file
auto WriteToFile = [](auto &file, auto const &data) {
    for (auto &e : data)
        file << e << " ";
    file << "\n";
};

// ------------------------------------------------------------------------------------------------------------------------------------------------------

// main function
int main(int, char **)
{
    // random number generation
    std::random_device rd{};
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> distrReal(0., 1.);
    std::uniform_int_distribution distrInt(0, spatialSize - 1);
    // random generator lambda for spin initialisation
    auto RandSpin = [&distrReal, &gen]() {
        return (double)distrReal(gen) > 0.5 ? 1 : -1;
    };
    // random generator lambda for spin flipping
    auto RandFlip = [&distrReal, &gen]() {
        return (double)distrReal(gen);
    };

    // initialize spins
    Table<int> table = Table<int>(RandSpin, spatialSize);

    // calculate the sign of the energy difference due to a single flip
    auto DeltaE = [&table](int row, int col, int dim) {
        // spin in question
        int s = table(row, col);

        // periodic boundary conditions
        int rowRight = (row + 1) % dim;
        int rowLeft = (row + dim - 1) % dim;
        int colDown = (col + 1) % dim;
        int colUp = (col + dim - 1) % dim;

        // neighbours
        int right = table(rowRight, col);
        int left = table(rowLeft, col);
        int down = table(row, colDown);
        int up = table(row, colUp);

        // quantity proportional to energy difference
        int energy = s * (up + down + left + right);

        // return sign of difference or zero
        return (0 < energy) - (energy < 0);
    };

    // calculate rate
    auto Rate = [&table, &DeltaE](int row, int col, int dim, double coupling) {
        // sign of energy difference due to flip
        int deltaE = DeltaE(row, col, dim);

        // calculate rate
        if (deltaE < 0)
            return 1.;
        else if (deltaE == 0)
            return 0.5;
        else
            return std::exp(-2 * coupling * deltaE);
    };

    // file
    std::ofstream file;
    file.open((std::string) "C:\\Users\\david\\Desktop\\MSc\\Ising model\\Python\\test.txt");
    // simulation
    for (int iCoupling = 0; iCoupling < 100; iCoupling++)
    {
        // reload table
        table = Table<int>(RandSpin, spatialSize);

        // run for given table
        int i = 0;
        while (i < (spatialSize * spatialSize * time))
        {
            // choose random spin
            int row = distrInt(gen);
            int col = distrInt(gen);
            // random number for flippin
            double randVal = distrReal(gen);
            // rate
            double rate = Rate(row, col, spatialSize, (double)(iCoupling / 50.));
            if (rate > randVal)
                table(row, col) *= -1;

            /*
            // write to file
            // time units
            int iTime = i % (spatialSize * spatialSize);
            if (iTime == 0 && i > (spatialSize * spatialSize * thermTime))
            {
                // configurations
                WriteToFile(file, table.data);
            }
            */

            // new step
            i++;
        }
        // averaging magnetisation
        file << iCoupling / 50. << " " << std::accumulate(table.data.begin(), table.data.end(), 0.) / (spatialSize * spatialSize) << std::endl;
    }
    file.close();
}
