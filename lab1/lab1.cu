#include <iostream>
 
#define CSC(call) \
do { \
	cudaError_t status = call; \
	if (status != cudaSuccess) { \
		fprintf(stderr, "ERROR is %s:%d. Message: %s\n", __FILE__, __LINE__, cudaGetErrorString(status)); \
		exit(0); \
	} \
} while(0)

__global__ void kernel(double *arr1, double *arr2, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int offset = blockDim.x * gridDim.x;
 
	while (idx < n) {
		arr1[idx] -= arr2[idx];
		idx += offset;
	}
}

 
int main() {
    int n;
    std::cin >> n;
 
    double *arr1 = new double[n];
    double *arr2 = new double[n];
	for(int i = 0; i < n; i++) {
        std::cin >> arr1[i];
    }
    for(int i = 0; i < n; i++) {
        std::cin >> arr2[i];
    }
    
	double *dev_arr1, *dev_arr2;
    CSC(cudaMalloc(&dev_arr1, sizeof(double) * n));
	CSC(cudaMemcpy(dev_arr1, arr1, sizeof(double) * n, cudaMemcpyHostToDevice));
 
    CSC(cudaMalloc(&dev_arr2, sizeof(double) * n));
	CSC(cudaMemcpy(dev_arr2, arr2, sizeof(double) * n, cudaMemcpyHostToDevice));

	kernel<<<32, 32>>>(dev_arr1, dev_arr2, n);

    CSC(cudaDeviceSynchronize());
	CSC(cudaGetLastError());
 
    CSC(cudaMemcpy(arr1, dev_arr1, sizeof(double) * n, cudaMemcpyDeviceToHost));
    
	for(int i = 0; i < n; ++i) {
		std::cout << arr1[i] << " ";
    }
	std::cout << "\n";
 
    CSC(cudaFree(dev_arr1));
    CSC(cudaFree(dev_arr2));
 
    delete[] arr1;
    delete[] arr2;
 
	return 0;
}
