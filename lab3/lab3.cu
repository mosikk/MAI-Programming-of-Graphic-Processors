#include <iostream>
#include <fstream>
#include <vector>

#define CSC(call) \
do { \
	cudaError_t status = call; \
	if (status != cudaSuccess) { \
		fprintf(stderr, "ERROR is %s:%d. Message: %s\n", __FILE__, __LINE__, cudaGetErrorString(status)); \
		exit(0); \
	} \
} while(0)

struct point {
    int x, y;
};

const int MAX_CLASSES = 32;
__constant__ double3 constAvgs[MAX_CLASSES];
__constant__ double constCovInv[MAX_CLASSES][3][3];


__global__ void kernel(uchar4 *dev_picture, int nc, int w, int h) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int offset = blockDim.x * gridDim.x;

    // count Mahalanobis distance = argmax[-(ps_minus_avg)^T * cov^(-1) * ps_minus_avg]
    while (idx < w * h) {
		double cur_max_dist = -1e9;
        double cur_max_id = -1;
        uchar4 cur_pixel_value = dev_picture[idx];

        for (int class_id = 0; class_id < nc; ++class_id) {
            double ps_minus_avg[3] = {
                cur_pixel_value.x - constAvgs[class_id].x,
                cur_pixel_value.y - constAvgs[class_id].y,
                cur_pixel_value.z - constAvgs[class_id].z
            };

            // (ps_minus_avg)^T * cov^(-1)
            double ps_minus_avg_cov[3] = {0, 0, 0};
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    ps_minus_avg_cov[i] += ps_minus_avg[j] * constCovInv[class_id][i][j];
                }
            }

            // -[(ps_minus_avg)^T * cov^(-1)] * ps_minus_avg
            double cur_dist = 0;
            for (int i = 0; i < 3; ++i) {
                cur_dist += ps_minus_avg_cov[i] * ps_minus_avg[i];
            }
            cur_dist *= -1;

            // update cur maximum
            if (cur_max_dist < cur_dist) {
                cur_max_dist = cur_dist;
                cur_max_id = class_id;
            }
            
        }
        dev_picture[idx].w = cur_max_id;
        idx += offset;
	}
}

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);

    // read input data
    std::string input_path, output_path;
    int w, h;
    std::cin >> input_path >> output_path;
 
    std::ifstream input_file(input_path, std::ios::in | std::ios::binary);
    if (!input_file.is_open()) {
        std::cerr << "Can't open file " << input_path << std::endl;
        return 0;
    }
 
    input_file.read((char *)&w, sizeof(w));
    input_file.read((char *)&h, sizeof(h));
    uchar4* picture = new uchar4[w * h];
    input_file.read((char *)picture, w * h * sizeof(picture[0]));
    input_file.close();

    int nc;
    std::cin >> nc;

    std::vector<std::vector<point>> classes_points(nc);
    for (int cur_class = 0; cur_class < nc; ++cur_class) {
        int np;
        std::cin >> np;
        for (int j = 0; j < np; ++j) {
            point p;
            std::cin >> p.x >> p.y;
            classes_points[cur_class].push_back(p);
        }
    }
    
    // count avg class statistics
    double3 avgs[MAX_CLASSES];
    for (int cur_class = 0; cur_class < nc; ++cur_class) {
        int cur_sum_x = 0, cur_sum_y = 0, cur_sum_z = 0;
        for (auto& cur_point : classes_points[cur_class]) {
            uchar4 cur_pixel_value = picture[cur_point.y * w + cur_point.x];
            cur_sum_x += cur_pixel_value.x;
            cur_sum_y += cur_pixel_value.y;
            cur_sum_z += cur_pixel_value.z;
        }
        int points_cnt = classes_points[cur_class].size();
        avgs[cur_class] = make_double3(cur_sum_x / points_cnt, cur_sum_y / points_cnt, cur_sum_z / points_cnt);
    }

    // count cov and cov^(-1)
    double cov[MAX_CLASSES][3][3];
    double cov_inv[MAX_CLASSES][3][3];
    // cov3 = [nc x 3 x 3]
    // 3 x 3 is for rgb pixels matrix
    for (int cur_class = 0; cur_class < nc; ++cur_class) {
        for (auto& cur_point : classes_points[cur_class]) {
            uchar4 cur_pixel_value = picture[cur_point.x + cur_point.y * w];
            auto cur_avg = avgs[cur_class];
            double ps_minus_avg[3] = {
                cur_pixel_value.x - cur_avg.x,
                cur_pixel_value.y - cur_avg.y,
                cur_pixel_value.z - cur_avg.z
            };

            // count (ps - avg) * (ps - avg)^T
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    cov[cur_class][i][j] += ps_minus_avg[i] * ps_minus_avg[j];
                }
            }
        }

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                cov[cur_class][i][j] /= (classes_points[cur_class].size() - 1);
            }
        }

        // count cov^(-1) = 1/det * cov^*
        const auto cur_matrix = cov[cur_class];
        double det = (
            cur_matrix[0][0] * cur_matrix[1][1] * cur_matrix[2][2]
            + cur_matrix[0][1] * cur_matrix[1][2] * cur_matrix[2][0]
            + cur_matrix[0][2] * cur_matrix[1][0] * cur_matrix[2][1]
            - cur_matrix[0][2] * cur_matrix[1][1] * cur_matrix[2][0]
            - cur_matrix[0][1] * cur_matrix[1][0] * cur_matrix[2][2]
            - cur_matrix[0][0] * cur_matrix[1][2] * cur_matrix[2][1]
        ); // Cramer's rule
        
        cov_inv[cur_class][0][0] = (cur_matrix[1][1] * cur_matrix[2][2] - cur_matrix[2][1] * cur_matrix[1][2]) / det;
        cov_inv[cur_class][0][1] = (cur_matrix[1][0] * cur_matrix[2][2] - cur_matrix[2][0] * cur_matrix[1][2]) / -det;
        cov_inv[cur_class][0][2] = (cur_matrix[1][0] * cur_matrix[2][1] - cur_matrix[2][0] * cur_matrix[1][1]) / det;
        cov_inv[cur_class][1][0] = (cur_matrix[0][1] * cur_matrix[2][2] - cur_matrix[2][1] * cur_matrix[0][2]) / -det;
        cov_inv[cur_class][1][1] = (cur_matrix[0][0] * cur_matrix[2][2] - cur_matrix[2][0] * cur_matrix[0][2]) / det;
        cov_inv[cur_class][1][2] = (cur_matrix[0][0] * cur_matrix[2][1] - cur_matrix[2][0] * cur_matrix[0][1]) / -det;
        cov_inv[cur_class][2][0] = (cur_matrix[0][1] * cur_matrix[1][2] - cur_matrix[1][1] * cur_matrix[0][2]) / det;
        cov_inv[cur_class][2][1] = (cur_matrix[0][0] * cur_matrix[1][2] - cur_matrix[1][0] * cur_matrix[0][2]) / -det;
        cov_inv[cur_class][2][2] = (cur_matrix[0][0] * cur_matrix[1][1] - cur_matrix[1][0] * cur_matrix[0][1]) / det;
    }

    // copy avgs and cov to constant memory
    CSC(cudaMemcpyToSymbol(constAvgs, avgs, sizeof(double3) * MAX_CLASSES));
    CSC(cudaMemcpyToSymbol(constCovInv, cov_inv, sizeof(double) * MAX_CLASSES * 3 * 3));

    // copy picture to gpu
    uchar4* dev_picture;
    CSC(cudaMalloc(&dev_picture, sizeof(uchar4) * w * h)); 
    CSC(cudaMemcpy(dev_picture, picture, sizeof(uchar4) * w * h, cudaMemcpyHostToDevice));

    // make classification at gpu
    kernel<<<256, 256>>>(dev_picture, nc, w, h);
    CSC(cudaDeviceSynchronize());
	CSC(cudaGetLastError());

    // copy picture back to cpu and write to file
    CSC(cudaMemcpy(picture, dev_picture, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));

    std::ofstream output_file(output_path, std::ios::out | std::ios::binary);
    if (!output_file.is_open()) {
        std::cerr << "Can't open file " << output_path << std::endl;
        return 0;
    }
 
    output_file.write((char *)&w, sizeof(w));
    output_file.write((char *)&h, sizeof(h));
    output_file.write((char *)picture, w * h * sizeof(picture[0]));
    output_file.close();

    // free memory
    CSC(cudaFree(dev_picture));
    delete[] picture;
    return 0;
}
