// 2D Ising model simulation via Metropolis-Hastings algorithm
// parallel setup ~ single checkboard: preventing race conditions

// include header(s)
#include <random>
#include <cmath>
#include <numeric>
#include <string>
#include "Table.hh"
// time measurement
#include <chrono>
// CUDA
#include <curand_kernel.h>
#include <curand.h>

// ------------------------------------------------------------------------------------------------------------------------------------------------------

// constants
// spatial size of simulation table (use > 1 and even)
const int spatialSize = 64;
// integration time
const int intTime = (int)1e6;
// scale for coupling index
const float scalar = 50.;
// number of threads
const int nThread = 32;

// ------------------------------------------------------------------------------------------------------------------------------------------------------

// calculate the sign of the energy difference due to a single flip
__device__ int DeltaE(int *table, int row, int col, int dim)
{
    // spin in question
    int s = table[row * spatialSize + col];

    // periodic boundary conditions
    int rowRight = (row + 1) % dim, rowLeft = (row + dim - 1) % dim, colDown = (col + 1) % dim, colUp = (col + dim - 1) % dim;

    // neighbours
    int right = table[rowRight * spatialSize + col], left = table[rowLeft * spatialSize + col], down = table[row * spatialSize + colDown], up = table[row * spatialSize + colUp];

    // quantity proportional to energy difference
    int energy = s * (up + down + left + right);

    // return sign of difference or zero (initialize to zero)
    int sign = 0;
    if (energy > 0)
        sign = 1;
    else if (energy < 0)
        sign = -1;
    return sign;
}

// ------------------------------------------------------------------------------------------------------------------------------------------------------

// calculate rate
__device__ float Rate(int *table, int row, int col, int dim, float coupling)
{
    // sign of energy difference due to flip
    int deltaE = DeltaE(table, row, col, dim);
    // calculate rate
    if (deltaE < 0)
        return 1.;
    else if (deltaE == 0)
        return 0.5;
    else
        return std::exp(-2 * coupling * deltaE);
}

// ------------------------------------------------------------------------------------------------------------------------------------------------------

// square function for integers
__host__ __device__ int Square(int x) { return x * x; }

// ------------------------------------------------------------------------------------------------------------------------------------------------------

// spin flip ~ site visit
__device__ void SpinFlip(int *table, bool parity, int minRow, double coupling, curandState &state)
{
    // choosing row and column according to parity
    // col (or row) can be anything
    int col = static_cast<int>(curand_uniform(&state) * (spatialSize - 1));
    // initialize row (or col)
    int row = static_cast<int>(curand_uniform(&state) * (spatialSize / nThread / 2 - 1));
    // even checkboard
    if (parity == 0)
        row = ((col % 2) == 0) ? 2 * row : 2 * row + 1;
    // odd checkboard
    else
        row = ((col % 2) == 0) ? 2 * row + 1 : 2 * row;
    // moving row to appropriate subtable
    row += minRow * spatialSize / nThread;
    // random number for flipping
    float randVal = curand_uniform(&state);
    // rate
    double rate = Rate(table, row, col, spatialSize, coupling);
    // flip or not to flip
    if (rate > randVal)
        table[row * spatialSize + col] *= -1;
}

// ------------------------------------------------------------------------------------------------------------------------------------------------------

// kernel for Metropolis sweep
__global__ void KernelMetropolisSweep(int *table, curandState *states, float coupling)
{
    // thread index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // initialize cuRAND
    curand_init(42, tid, 0, &states[tid]);

    // run sweeps
    for (int iSweep = 0; iSweep < intTime; iSweep++)
    {
        // synchronise threads
        __syncthreads();

        // visit sites
        // parity: even
        for (int iVisit = 0; iVisit < Square(spatialSize) / nThread / 2; iVisit++)
            SpinFlip(table, 0, tid, coupling, states[tid]);

        // synchronise threads
        __syncthreads();

        // parity: odd
        for (int iVisit = 0; iVisit < Square(spatialSize) / nThread / 2; iVisit++)
            SpinFlip(table, 1, tid, coupling, states[tid]);

        // synchronise threads
        __syncthreads();
    }
}

// ------------------------------------------------------------------------------------------------------------------------------------------------------

// main function
int main(int, char **)
{
    // random number generation
    std::random_device rd{};
    std::mt19937 gen(rd());
    // [0, 1] ~ real
    std::uniform_real_distribution<double> distrReal(0., 1.);
    // random generator lambda for spin initialisation
    auto RandSpin = [&distrReal, &gen]()
    { return (float)distrReal(gen) > 0.5 ? 1 : -1; };

    for (int iCoupling = 0; iCoupling < 100; iCoupling += 5)
    {
        // real coupling
        float coupling = (float)(iCoupling / scalar);
        
        // initialize spins
        // host
        std::vector<int> table(Square(spatialSize));
        std::generate(table.begin(), table.end(), RandSpin);
        // device
        int *tableDev = nullptr;
        // cuRAND states
        curandState *statesDev = nullptr;

        // memory allocation for the device
        cudaError_t err = cudaSuccess;
        err = cudaMalloc((void **)&tableDev, Square(spatialSize) * sizeof(int));
        if (err != cudaSuccess)
        {
            std::cout << "Error allocating CUDA memory (TABLE): " << cudaGetErrorString(err) << std::endl;
            return -1;
        }
        err = cudaMalloc((void **)&statesDev, nThread * sizeof(curandState));
        if (err != cudaSuccess)
        {
            std::cout << "Error allocating CUDA memory (cuRAND): " << cudaGetErrorString(err) << std::endl;
            return -1;
        }

        // copy data onto device
        err = cudaMemcpy(tableDev, table.data(), Square(spatialSize) * sizeof(int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            std::cout << "Error copying memory to device (TABLE): " << cudaGetErrorString(err) << std::endl;
            return -1;
        }

        // start kernel
        KernelMetropolisSweep<<<1, nThread>>>(tableDev, statesDev, coupling);

        // get errors from run
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cout << "CUDA error in kernel call: " << cudaGetErrorString(err) << std::endl;
            return -1;
        }

        // copy data from device
        err = cudaMemcpy(table.data(), tableDev, Square(spatialSize) * sizeof(int), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            std::cout << "Error copying memory to host: " << cudaGetErrorString(err) << std::endl;
            return -1;
        }

        // free memory
        err = cudaFree(tableDev);
        if (err != cudaSuccess)
        {
            std::cout << "Error freeing allocation (TABLE): " << cudaGetErrorString(err) << std::endl;
            return -1;
        }
        err = cudaFree(statesDev);
        if (err != cudaSuccess)
        {
            std::cout << "Error freeing allocation (cuRAND): " << cudaGetErrorString(err) << std::endl;
            return -1;
        }

        // print coupling
        std::cout << "J = " << coupling;
        // print magnetisation
        std::cout << " |M| = " << std::abs(std::accumulate(table.begin(), table.end(), 0.) / Square(spatialSize)) << std::endl;

        // file
        std::ofstream file;
        file.open((std::string) "C:\\Users\\david\\Desktop\\MSc\\Ising model\\Python\\testGPUTable.txt");
        for (int i = 0; i < spatialSize; i++)
        {
            for (int j = 0; j < spatialSize; j++)
            {
                file << table[i * spatialSize + j] << " ";
            }
            file << std::endl;
        }
        file.close();
    }
}