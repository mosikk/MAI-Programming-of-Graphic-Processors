#include <iostream>
#include <limits>

#define CSC(call) \
do { \
    cudaError_t status = call; \
    if (status != cudaSuccess) { \
        fprintf(stderr, "ERROR is %s:%d. Message: %s\n", __FILE__, __LINE__, cudaGetErrorString(status)); \
        exit(0); \
    } \
} while(0)

const int MAX_SHARED_MEMORY_SIZE = 1024;

__device__ void B_step(int* arr, int len, int id_real, int id_global, int n_b, int n_m) {
    // makes a step of B(n_b) operation for M(n_m) merge
    // step = comparing arr[id] and arr[id + n_b/2]
    // id_real - real pos of elem in given arr
    // id_global - pos of elem in global array (if arr in shared memory, this ids will be different)
    if (id_global % n_b >= n_b / 2) {
        // arr[id] was / would be already compared with previous element
        return;
    }
    bool is_ascending = ((id_global / n_m) % 2 == 0);
    
    if ((is_ascending && arr[id_real] > arr[id_real + n_b / 2]) || (!is_ascending && arr[id_real] < arr[id_real + n_b / 2])) {
        int tmp = arr[id_real];
        arr[id_real] = arr[id_real + n_b / 2];
        arr[id_real + n_b / 2] = tmp;
    }
}


__global__ void kernel_shared(int* dev_nums, int len, int n_b_begin, int n_m) {
    // make some iterations B(i) of bitonic merge M(n_m)
    // i = n_b_begin, n_b_begin / 2, n_b_begin / 4, ..., 2
    __shared__ int sh_mem[2048];

    // block dim = MAX_SHARED_MEMORY_SIZE
    // each block is a set of MAX_SHARED_MEMORY_SIZE elements
    int idx = blockIdx.x * MAX_SHARED_MEMORY_SIZE;
    int offset = MAX_SHARED_MEMORY_SIZE * gridDim.x;

    while (idx < len) {
        sh_mem[threadIdx.x] = dev_nums[idx + threadIdx.x];
        sh_mem[threadIdx.x + MAX_SHARED_MEMORY_SIZE / 2] = dev_nums[idx + threadIdx.x + MAX_SHARED_MEMORY_SIZE / 2];
        __syncthreads();

        for (int n_b = n_b_begin; n_b >= 2; n_b /= 2) {
            for (int step = n_b / 2; step > 0; step /= 2) {

                int pos = 2 * threadIdx.x - (threadIdx.x & (step - 1)); // magic formula to find element for comparing
                B_step(sh_mem, len, pos, idx + pos, n_b, n_m);
                __syncthreads();
            }
        }

        dev_nums[idx + threadIdx.x] = sh_mem[threadIdx.x];
        dev_nums[idx + threadIdx.x + MAX_SHARED_MEMORY_SIZE / 2] = sh_mem[threadIdx.x + MAX_SHARED_MEMORY_SIZE / 2];
        idx += offset;
    }
}

__global__ void kernel_global(int* dev_nums, int len, int n_b, int n_m) {
    // make one iteration B(n_b) of bitonic merge B(n_m) using global memory
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;

    while (idx < len) {
        B_step(dev_nums, len, idx, idx, n_b, n_m);
        idx += offset;
    }
}


int main() {
    // read binary data
    int n;
    fread(&n, sizeof(int), 1, stdin);
    
    int n_rounded = 1;
    while (n_rounded < n) {
        // we must get an array with len = 2^(n_rounded)
        n_rounded *= 2;
    }

    int* nums = new int[n_rounded];
    fread(nums, sizeof(int), n, stdin);

    // pad array with max values
    // they will be in the end of sorted array, so we won't print them in answer
    for (int i = n; i < n_rounded; ++i) {
        nums[i] = std::numeric_limits<int>::max();
    }

    // put data to gpu
    int* dev_nums;
    CSC(cudaMalloc(&dev_nums, sizeof(int) * n_rounded)); 
    CSC(cudaMemcpy(dev_nums, nums, sizeof(int) * n_rounded, cudaMemcpyHostToDevice));

    // sort numbers
    for (int n_m = 2; n_m <= n_rounded; n_m *= 2) {
        for (int n_b = n_m; n_b >= 2; n_b /= 2) {
            if (n_b == MAX_SHARED_MEMORY_SIZE) {
                // TODO: make shared memory kernek for n_b < MAX_SHARED_MEMORY_SIZE
                kernel_shared<<<512, 512>>>(dev_nums, n_rounded, n_b, n_m);
                CSC(cudaGetLastError());
                break;
            } else {
                kernel_global<<<512, 512>>>(dev_nums, n_rounded, n_b, n_m);
                CSC(cudaGetLastError());
            }
        }
    }

    // copy numbers back to cpu
    CSC(cudaMemcpy(nums, dev_nums, sizeof(int) * n_rounded, cudaMemcpyDeviceToHost));
    CSC(cudaFree(dev_nums));

    // binary output
    fwrite(nums, sizeof(int), n, stdout);
    delete[] nums;
    return 0;
}
