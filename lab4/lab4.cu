#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/extrema.h>

#define CSC(call) \
do { \
    cudaError_t status = call; \
    if (status != cudaSuccess) { \
        fprintf(stderr, "ERROR is %s:%d. Message: %s\n", __FILE__, __LINE__, cudaGetErrorString(status)); \
        exit(0); \
    } \
} while(0)

struct comparator {												
    __host__ __device__ bool operator()(double a, double b) {
        return fabs(a) < fabs(b);
    }
};

// A[i][j] = A[i + j * n]

__global__ void kernel_swap(double* A, int n, int m, int row1, int row2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int offset = blockDim.x * gridDim.x;

    while (idx < m) {
        double tmp = A[row1 + idx * n];
        A[row1 + idx * n] = A[row2 + idx * n];
        A[row2 + idx * n] = tmp;
        idx += offset;
    }
}

__global__ void kernel_gauss_step(double* A, int n, int m, int x_start, int y_start) {
    // make Gauss algorithm step at A[x_start:][y_start:]
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    for (int i = idx + x_start + 1; i < n; i += offsetx) {
        for (int j = idy + y_start + 1; j < m; j += offsety) {
            A[i + j * n] -= A[i + y_start * n] / A[x_start + y_start * n] * A[x_start + j * n];
        }
    }
}


int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    // read input data
    long long n, m;
    std::cin >> n >> m;
    double* A = new double[n * m];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            std::cin >> A[i + j * n];  // we'll keep matrix by columns
        }
    }

    // put matrix to gpu
    double* dev_A;
    CSC(cudaMalloc(&dev_A, sizeof(double) * n * m)); 
    CSC(cudaMemcpy(dev_A, A, sizeof(double) * n * m, cudaMemcpyHostToDevice));

    // init thrust ptr
    thrust::device_ptr<double> p_A = thrust::device_pointer_cast(dev_A);
    comparator comp;

    // calculating matrix rank
    int rank = 0;
    for (int i = 0; i < m; ++i) {
        // find max value in column (ignore processed rows)
        thrust::device_ptr<double> cur_max = thrust::max_element(
            p_A + n * i + rank,
            p_A + n * (i + 1), 
            comp
        );

        if (fabs(*cur_max) < 1e-7) {
            // this column has only zeroes left -> skip it
            continue;
        }

        if (rank == n - 1) {
            // m > n
            // last row, but not last column => increment rank and break
            ++rank;
            break;
        }

        int cur_max_id = cur_max - p_A - i * n;
        if (cur_max_id != rank) {
            // swap rows
            // row[rank] is current non-null row
            kernel_swap<<<128, 128>>>(dev_A, n, m, rank, cur_max_id);
            CSC(cudaGetLastError());
        }
        
        // make a step of Gauss algorithm 
        // x_start = cur rank (because we skipped null rows)
        // y_start = i = cur column
        kernel_gauss_step<<<dim3(32, 32), dim3(32, 32)>>>(dev_A, n, m, rank, i);
        CSC(cudaGetLastError());
        ++rank;
    }
    std::cout << rank << std::endl;

    // free memory
    CSC(cudaFree(dev_A));
    delete[] A;
    return 0;
}
