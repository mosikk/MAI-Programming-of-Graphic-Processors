#include <iostream>
#include <fstream>
 
#define CSC(call) \
do { \
	cudaError_t status = call; \
	if (status != cudaSuccess) { \
		fprintf(stderr, "ERROR is %s:%d. Message: %s\n", __FILE__, __LINE__, cudaGetErrorString(status)); \
		exit(0); \
	} \
} while(0)
 
texture<uchar4, 2, cudaReadModeElementType> tex;
 
__global__ void kernel(uchar4* output_img, int w_cur, int w_new, int h_cur, int h_new) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    double k_w = (double)w_cur / w_new;
    double k_h = (double)h_cur / h_new;
 
    for (int y_new = idy; y_new < h_new; y_new += offsety) {
        for (int x_new = idx; x_new < w_new; x_new += offsetx) {
            // coords at old picture
            int x_cur = floor((x_new + 0.5) * k_w - 0.5);
            int y_cur = floor((y_new + 0.5) * k_h - 0.5);
 
            // relative coords
            double xx = (x_new + 0.5) * k_w - 0.5 - x_cur;
            double yy = (y_new + 0.5) * k_h - 0.5 - y_cur;
 
            // corner cases
            if (xx < 0) {
                ++xx;
                --x_cur;
            }
            if (yy < 0) {
                ++yy;
                --y_cur;
            }
 
            // get necessary points from texture
            uchar4 p_xy = tex2D(tex, x_cur, y_cur);
            uchar4 p_x1y = tex2D(tex, x_cur + 1, y_cur);
            uchar4 p_xy1 = tex2D(tex, x_cur, y_cur + 1);
            uchar4 p_x1y1 = tex2D(tex, x_cur + 1, y_cur + 1);
 
            // calc new pixel value
            double new_x = p_xy.x * (1-xx) * (1-yy) +
                p_x1y.x * xx * (1-yy) + 
                p_xy1.x * (1-xx) * yy + 
                p_x1y1.x * xx * yy;
 
            double new_y = p_xy.y * (1-xx) * (1-yy) +
                p_x1y.y * xx * (1-yy) + 
                p_xy1.y * (1-xx) * yy + 
                p_x1y1.y * xx * yy;
 
            double new_z = p_xy.z * (1-xx) * (1-yy) +
                p_x1y.z * xx * (1-yy) + 
                p_xy1.z * (1-xx) * yy + 
                p_x1y1.z * xx * yy;
 
 
            output_img[y_new * w_new + x_new] = make_uchar4(new_x, new_y, new_z, 255);
        }
    }
}
 
int main() {
    // read input data
    std::string input_path, output_path;
    int w_cur, h_cur, w_new, h_new;
    std::cin >> input_path >> output_path >> w_new >> h_new;
 
    std::ifstream input_file(input_path, std::ios::in | std::ios::binary);
    if (!input_file.is_open()) {
        std::cerr << "Can't open file " << input_path << std::endl;
        return 0;
    }
 
    input_file.read((char *)&w_cur, sizeof(w_cur));
    input_file.read((char *)&h_cur, sizeof(h_cur));
    uchar4* picture = new uchar4[w_cur * h_cur];
    input_file.read((char *)picture, w_cur * h_cur * sizeof(picture[0]));
    input_file.close();
 
    // create array with cur picture
    cudaArray *dev_picture;
    cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
    CSC(cudaMallocArray(&dev_picture, &ch, w_cur, h_cur));
    CSC(cudaMemcpy2DToArray(dev_picture, 0, 0, picture, w_cur * sizeof(uchar4), 
                            w_cur * sizeof(uchar4), h_cur, cudaMemcpyHostToDevice));
 
    // create texture
    tex.normalized = false;
    tex.filterMode = cudaFilterModePoint;
    tex.channelDesc = ch;
    tex.addressMode[0] = cudaAddressModeClamp;
    tex.addressMode[1] = cudaAddressModeClamp;
    CSC(cudaBindTextureToArray(tex, dev_picture, ch));
 
    // create array for new picture
    uchar4* dev_picture_new;
    uchar4* picture_new = new uchar4[w_new * h_new];
    CSC(cudaMalloc(&dev_picture_new, sizeof(uchar4) * w_new * h_new));
 
    // create new picture
    kernel<<<dim3(32, 32), dim3(32, 32)>>>(dev_picture_new, w_cur, w_new, h_cur, h_new);
    CSC(cudaDeviceSynchronize());
    CSC(cudaGetLastError());
 
    // copy new picture, remove everything from gpu
    CSC(cudaMemcpy(picture_new, dev_picture_new, sizeof(uchar4) * w_new * h_new, cudaMemcpyDeviceToHost));
    CSC(cudaUnbindTexture(tex));
    CSC(cudaFreeArray(dev_picture));
    CSC(cudaFree(dev_picture_new));
 
    // write output data
    std::ofstream output_file(output_path, std::ios::out | std::ios::binary);
    if (!output_file.is_open()) {
        std::cerr << "Can't open file " << output_path << std::endl;
        return 0;
    }
 
    output_file.write((char *)&w_new, sizeof(w_new));
    output_file.write((char *)&h_new, sizeof(h_new));
    output_file.write((char *)picture_new, w_new * h_new * sizeof(picture_new[0]));
    output_file.close();

    delete[] picture;
    delete[] picture_new;
    return 0;
}
