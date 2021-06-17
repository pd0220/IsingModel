// 2D Ising model simulation via Metropolis-Hastings algorithm
// parallel setup ~ single checkboard: preventing race conditions

// include header(s)
#include <random>
#include <cmath>
#include <numeric>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
// time measurement
#include <chrono>
// cuRAND
#include <curand_kernel.h>
#include <curand.h>

// ------------------------------------------------------------------------------------------------------------------------------------------------------

// constants
// spatial size of simulation table (use > 1 and even)
const int spatialSize = 128;
// integration time
const int intTime = (int)2e6;
// scale for coupling index
const float scalar = 50.;
// number of threads per block
const int nThread = 16;
// block size
const int sizeInBlocks = 4;
// number of blocks
const int nBlock = sizeInBlocks * sizeInBlocks;

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

// spin flip ~ site visit for given (row, col)
__device__ void SpinFlip(int *table, bool parity, float coupling, curandState &state, int row, int col)
{
    // random number for flipping
    float randVal = curand_uniform(&state);
    // rate
    float rate = Rate(table, row, col, spatialSize, coupling);
    // flip or not to flip
    if (rate > randVal)
        table[row * spatialSize + col] *= -1;
}

// ------------------------------------------------------------------------------------------------------------------------------------------------------

// kernel for Metropolis sweep ~ even sites
__global__ void KernelMetropolisEven(int *table, curandState *states, float coupling)
{
    // thread index inside the block
    int id = threadIdx.x;
    // block index
    int bid = blockIdx.x;
    // thread index
    int tid = bid * blockDim.x + id;
    // initialize cuRAND
    curand_init(42, tid, 0, &states[tid]);

    // locate block and thread
    int minRow = (int)(bid / sizeInBlocks) * spatialSize / sizeInBlocks;
    int minCol = bid * spatialSize / sizeInBlocks - sizeInBlocks * minRow;
    // move to thread
    minRow += id * spatialSize / sizeInBlocks / nThread;

    __syncthreads();

    for (int irow = minRow; irow < minRow + spatialSize / sizeInBlocks / nThread; irow++)
    {
        for (int icol = (((irow % 2) == 0) ? minCol : minCol + 1); icol < minCol + spatialSize / sizeInBlocks; icol += 2)
        {
            SpinFlip(table, true, coupling, states[tid], irow, icol);
        }
    }
}

// kernel for Metropolis sweep ~ odd sites
__global__ void KernelMetropolisOdd(int *table, curandState *states, float coupling)
{
    // thread index inside the block
    int id = threadIdx.x;
    // block index
    int bid = blockIdx.x;
    // thread index
    int tid = bid * blockDim.x + id;
    // initialize cuRAND
    curand_init(42, tid, 0, &states[tid]);

    // locate block and thread
    int minRow = (int)(bid / sizeInBlocks) * spatialSize / sizeInBlocks;
    int minCol = bid * spatialSize / sizeInBlocks - sizeInBlocks * minRow;
    // move to thread
    minRow += id * spatialSize / sizeInBlocks / nThread;

    __syncthreads();

    for (int irow = minRow; irow < minRow + spatialSize / sizeInBlocks / nThread; irow++)
    {
        for (int icol = (((irow % 2) == 0) ? minCol + 1 : minCol); icol < minCol + spatialSize / sizeInBlocks; icol += 2)
        {
            SpinFlip(table, true, coupling, states[tid], irow, icol);
        }
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

    // initialize spins
    // host
    std::vector<int> table(Square(spatialSize));
    std::generate(table.begin(), table.end(), RandSpin);
    // device
    int *tableDev = nullptr;
    // cuRAND states
    curandState *statesDev = nullptr;

    for (int iCoupling = 70; iCoupling < 100; iCoupling += 10)
    {
        // real coupling
        //float coupling = (float)(iCoupling / scalar);
        float coupling = 10.;
        
        // (re)initialize spins
        // host
        std::generate(table.begin(), table.end(), RandSpin);
        // device
        tableDev = nullptr;
        // cuRAND states
        statesDev = nullptr;

        // memory allocation for the device
        cudaError_t err = cudaSuccess;
        err = cudaMalloc((void **)&tableDev, Square(spatialSize) * sizeof(int));
        if (err != cudaSuccess)
        {
            std::cout << "Error allocating CUDA memory (TABLE): " << cudaGetErrorString(err) << std::endl;
            return -1;
        }
        err = cudaMalloc((void **)&statesDev, nBlock * nThread * sizeof(curandState));
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

        // simulation
        // Metropolis sweeps
        for (int iSweep = 0; iSweep < intTime; iSweep++)
        {
            // even kernel
            KernelMetropolisEven<<<nBlock, nThread>>>(tableDev, statesDev, coupling);

            // odd kernel
            KernelMetropolisOdd<<<nBlock, nThread>>>(tableDev, statesDev, coupling);
        }

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
        std::cout << " |M| = " << std::accumulate(table.begin(), table.end(), 0.) / Square(spatialSize) << std::endl;

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