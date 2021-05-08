// 2D Ising model simulation via Metropolis-Hastings algorithm
// parallel setup ~ single checkboard: preventing race conditions

// include header(s)
#include <random>
#include <cmath>
#include <numeric>
#include <string>
#include "Table.hh"
// for parallelisation
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
// time measurement
#include <chrono>

// ------------------------------------------------------------------------------------------------------------------------------------------------------

// constants
// spatial size of simulation table (use > 1 and even)
const int spatialSize = 64;
// integration time
const int intTime = 20000;
// scale for coupling index
const double scalar = 50.;
// number of threads (16 in my setup)
const std::size_t nThread = std::thread::hardware_concurrency();

// ------------------------------------------------------------------------------------------------------------------------------------------------------

// main function
int main()
{
    // random number generation
    std::random_device rd{};
    std::mt19937 gen(rd());
    // [0, 1] ~ real
    std::uniform_real_distribution<double> distrReal(0., 1.);
    // [0, size - 1] ~ integer
    std::uniform_int_distribution<int> distrInt(0, (int)(spatialSize - 1));
    // [0, size / 2 - 1] ~ integer
    std::uniform_int_distribution<int> distrIntSubTableHalf(0, (int)(spatialSize / 2 / nThread - 1));
    // random generator lambda for spin initialisation
    auto RandSpin = [&distrReal, &gen]() { return (double)distrReal(gen) > 0.5 ? 1 : -1; };
    // random generator lambda for spin flipping
    auto RandFlip = [&distrReal, &gen]() { return (double)distrReal(gen); };

    // initialize spins
    Table<int> table = Table<int>(spatialSize);

    // calculate the sign of the energy difference due to a single flip
    auto DeltaE = [&](int row, int col, int dim) {
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
    auto Rate = [&](int row, int col, int dim, double coupling) {
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

    // spin flip ~ site visit
    auto SpinFlip = [&](bool parity, int minRow, double coupling) {
        // choosing row and column according to parity
        // col (or row) can be anything
        int col = distrInt(gen);
        // initialize row (or col)
        int row = distrIntSubTableHalf(gen);
        // even checkboard
        if (parity == 0)
            row = ((col % 2) == 0) ? 2 * row : 2 * row + 1;
        // odd checkboard
        else
            row = ((col % 2) == 0) ? 2 * row + 1 : 2 * row;
        // moving row to appropriate subtable
        row += minRow * spatialSize / nThread;
        // random number for flipping
        double randVal = distrReal(gen);
        // rate
        double rate = Rate(row, col, spatialSize, coupling);
        // flip or not to flip
        if (rate > randVal)
            table(row, col) *= -1;
    };

    // mutex (mutual exclusion)
    std::mutex m;
    // condition variable ~ odd / even board only
    std::condition_variable cv;
    // atomic variables to manage thread waiting
    std::atomic<int> counterEven = 0;
    std::atomic<int> counterOdd = 0;

    // check for wake up
    auto WakeEven = [&]() { return counterEven % nThread == 0; };
    auto WakeOdd = [&]() { return counterOdd % nThread == 0; };

    // Metroplois sweep
    auto MetropolisSweep = [&](int minRow, double coupling) {
        for (int iSweep = 0; iSweep < intTime; iSweep++)
        {
            // visit sites
            // parity: even
            for (int iVisit = 0; iVisit < sq(spatialSize) / nThread / 2; iVisit++)
                SpinFlip(0, minRow, coupling);

            // update counter after sweep and put thread to sleep
            {
                std::unique_lock<std::mutex> lk(m);
                counterEven++;
                //std::cout << "Even waiting " << counterEven % nThread << " " << counterOdd % nThread << std::endl;
                cv.wait(lk, WakeEven);
            }
            cv.notify_all();

            // parity odd
            for (int iVisit = 0; iVisit < sq(spatialSize) / nThread / 2; iVisit++)
                SpinFlip(1, minRow, coupling);

            // update counter after sweep and put thread to sleep
            {
                std::unique_lock<std::mutex> lk(m);
                counterOdd++;
                //std::cout << "Odd waiting " << counterEven % nThread << " " << counterOdd % nThread << std::endl;
                cv.wait(lk, WakeOdd);
            }
            cv.notify_all();
        }
    };

    // file
    std::ofstream file;
    file.open((std::string) "C:\\Users\\david\\Desktop\\MSc\\Ising model\\Python\\testCPU.txt");

    // vector of threads
    std::vector<std::thread> threads(nThread);
    // vector of time measurements
    std::vector<double> timeMeasurement;

    // simulation
    for (int iCoupling = 0; iCoupling < 100; iCoupling += 5)
    {
        // reload table
        table = Table<int>(RandSpin, spatialSize);
        // real coupling
        double coupling = (double)(iCoupling / scalar);

        // TIME #0
        auto start = std::chrono::high_resolution_clock::now();

        // run Metropolis sweeps
        // start threads
        for (int it = 0; it < nThread; it++)
            threads[it] = std::thread(MetropolisSweep, it, coupling);
        // join threads
        for (auto &t : threads)
            t.join();

        // TIME #1
        auto stop = std::chrono::high_resolution_clock::now();

        timeMeasurement.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count());

        // averaging magnetisation
        file << coupling << " " << std::accumulate(table.data.begin(), table.data.end(), 0.) / sq(spatialSize) << std::endl;
    }
    file.close();

    // print computation time
    std::cout << "Mean parrallel computation time for a single table with " << nThread << " threads: "
              << std::accumulate(timeMeasurement.begin(), timeMeasurement.end(), 0.) / static_cast<double>(timeMeasurement.size()) << " ms." << std::endl;
}
