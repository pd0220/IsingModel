// 2D Ising model simulation via Metropolis-Hastings algorithm
// serial setup

// include header(s)
#include <random>
#include <cmath>
#include <numeric>
#include <string>
#include "Table.hh"
// time measurement
#include <chrono>

// ------------------------------------------------------------------------------------------------------------------------------------------------------

// constants
// spatial size of simulation table (use > 1)
const int spatialSize = 64;
// integration time
const int intTime = 30000;
// scale for coupling index
const double scalar = 50.;

// ------------------------------------------------------------------------------------------------------------------------------------------------------

// main function
int main()
{
    // random number generation
    std::random_device rd{};
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> distrReal(0., 1.);
    std::uniform_int_distribution<int> distrInt(0, spatialSize - 1);
    // random generator lambda for spin initialisation
    auto RandSpin = [&distrReal, &gen]()
    {
        return (double)distrReal(gen) > 0.5 ? 1 : -1;
    };
    // random generator lambda for spin flipping
    auto RandFlip = [&distrReal, &gen]()
    {
        return (double)distrReal(gen);
    };

    // initialize spins
    Table<int> table = Table<int>(RandSpin, spatialSize);

    // calculate the sign of the energy difference due to a single flip
    auto DeltaE = [&table](int row, int col, int dim)
    {
        // spin in question
        int s = table(row, col);

        // periodic boundary conditions
        int rowRight = (row + 1) % dim, rowLeft = (row + dim - 1) % dim, colDown = (col + 1) % dim, colUp = (col + dim - 1) % dim;

        // neighbours
        int right = table(rowRight, col), left = table(rowLeft, col), down = table(row, colDown), up = table(row, colUp);

        // quantity proportional to energy difference
        int energy = s * (up + down + left + right);

        // return sign of difference or zero (initialize to zero)
        int sign = 0;
        if (energy > 0)
            sign = 1;
        else if (energy < 0)
            sign = -1;
        return sign;
    };

    // calculate rate
    auto Rate = [&table, &DeltaE](int row, int col, int dim, double coupling)
    {
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

    // vector of time measurements
    std::vector<double> timeMeasurement;

    // simulation
    for (int iCoupling = 0; iCoupling < 100; iCoupling += 10)
    {
        // reload table
        table = Table<int>(RandSpin, spatialSize);
        // real coupling
        double coupling = (double)(iCoupling / scalar);

        // TIME #0
        auto start = std::chrono::high_resolution_clock::now();

        // run for given table
        for (int i = 0; i < sq(spatialSize) * intTime; i++)
        {
            // choose random spin
            int row = distrInt(gen);
            int col = distrInt(gen);
            // random number for flipping
            double randVal = distrReal(gen);
            // rate
            double rate = Rate(row, col, spatialSize, coupling);
            if (rate > randVal)
                table(row, col) *= -1;
        }

        // TIME #1
        auto stop = std::chrono::high_resolution_clock::now();

        timeMeasurement.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count());

        // averaging magnetisation
        file << coupling << " " << std::accumulate(table.data.begin(), table.data.end(), 0.) / sq(spatialSize) << std::endl;
    }
    file.close();

    // print computation time
    std::cout << "Mean serial computation time for a single table : "
              << std::accumulate(timeMeasurement.begin(), timeMeasurement.end(), 0.) / static_cast<double>(timeMeasurement.size()) << " ms." << std::endl;
}
