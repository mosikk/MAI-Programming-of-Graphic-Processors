#include <iostream>
#include <cmath>

#define CSC(call) \
do { \
    cudaError_t status = call; \
    if (status != cudaSuccess) { \
        fprintf(stderr, "ERROR is %s:%d. Message: %s\n", __FILE__, __LINE__, cudaGetErrorString(status)); \
        exit(0); \
    } \
} while(0)

// вспомогательные структуры
struct vec3 {
    double x;
    double y;
    double z;

    __host__ __device__ vec3() {}
    __host__ __device__ vec3(double x, double y, double z) : x(x), y(y), z(z) {}
};

__host__ __device__ vec3 operator+(vec3 a, vec3 b) {
    return vec3(
        a.x + b.x,
        a.y + b.y,
        a.z + b.z
    );
}

__host__ __device__ vec3 operator-(vec3 a, vec3 b) {
    return vec3(
        a.x - b.x,
        a.y - b.y,
        a.z - b.z
    );
}

__host__ __device__ vec3 operator*(vec3 a, double b) {
    return vec3(
        a.x * b,
        a.y * b,
        a.z * b
    );
}


struct polygon {
    vec3 a;
    vec3 b;
    vec3 c;
    uchar4 color;

    __host__ __device__ polygon() {}
    __host__ __device__ polygon(vec3 a, vec3 b, vec3 c, uchar4 color) : a(a), b(b), c(c), color(color) {}
};

// вспомогательные математические операции
__host__ __device__ double dot(vec3 a, vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ vec3 prod(vec3 a, vec3 b) {
    return vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__host__ __device__ vec3 norm(vec3 v) {
    double l = sqrt(dot(v, v));
    return vec3(
        v.x / l,
        v.y / l,
        v.z / l
    );
}

__host__ __device__ vec3 mult(vec3 a, vec3 b, vec3 c, vec3 v) {
    return vec3(
        a.x * v.x + b.x * v.y + c.x * v.z,
        a.y * v.x + b.y * v.y + c.y * v.z,
        a.z * v.x + b.z * v.y + c.z * v.z
    );
}

// трассировка лучей
__host__ __device__ uchar4 ray(vec3 pos, vec3 dir, vec3 light_pos, uchar4 light_color, polygon* polygons, int polygons_cnt) {
    int i_min = -1;
    double ts_min;
    for (int i = 0; i < polygons_cnt; ++i) {
        vec3 e1 = polygons[i].b - polygons[i].a;
        vec3 e2 = polygons[i].c - polygons[i].a;
        vec3 p = prod(dir, e2);
        double div = dot(p, e1);
        if (fabs(div) < 1e-10)
            continue;
        vec3 t = pos - polygons[i].a;
        double u = dot(p, t) / div;
        if (u < 0.0 || u > 1.0)
            continue;
        vec3 q = prod(t, e1);
        double v = dot(q, dir) / div;
        if (v < 0.0 || v + u > 1.0)
            continue;
        double ts = dot(q, e2) / div; 
        if (ts < 0.0)
            continue;
        if (i_min == -1 || ts < ts_min) {
            i_min = i;
            ts_min = ts;
        }
    }

    if (i_min == -1) {
        return make_uchar4(0, 0, 0, 255);
    }

    // теперь нужно учесть освещение
    // делаем почти все то же самое, но для источника света
    vec3 new_pos = dir * ts_min + pos;
    vec3 new_dir = light_pos - new_pos;
    double length = sqrt(dot(new_dir, new_dir));
    new_dir = norm(new_dir);

    for (int i = 0; i < polygons_cnt; ++i) {
        vec3 e1 = polygons[i].b - polygons[i].a;
        vec3 e2 = polygons[i].c - polygons[i].a;
        vec3 p = prod(new_dir, e2);
        double div = dot(p, e1);
        if (fabs(div) < 1e-10)
            continue;
        vec3 t = new_pos - polygons[i].a;
        double u = dot(p, t) / div;
        if (u < 0.0 || u > 1.0)
            continue;
        vec3 q = prod(t, e1);
        double v = dot(q, new_dir) / div;
        if (v < 0.0 || v + u > 1.0)
            continue;
        double ts = dot(q, e2) / div; 
        if (ts > 0.0 && ts < length && i != i_min) {
            return make_uchar4(0, 0, 0, 255);
        }
    }

    return make_uchar4(
        polygons[i_min].color.x * light_color.x,
        polygons[i_min].color.y * light_color.y,
        polygons[i_min].color.z * light_color.z,
        255
    );
}

// рендеринг
__host__ __device__ void render(vec3 camera_pos, vec3 camera_view,
                                int w, int h, double angle, uchar4* data,
                                vec3 light_pos, uchar4 light_color,
                                polygon* polygons, int polygons_cnt) {
    double dw = 2.0 / (w - 1.0);
    double dh = 2.0 / (h - 1.0);
    double z = 1.0 / tan(angle * M_PI / 360.0);

    // переходим в новый базис, связанный с камерой
    vec3 bz = norm(camera_view - camera_pos);
    vec3 bx = norm(prod(bz, {0.0, 0.0, 1.0}));
    vec3 by = norm(prod(bx, bz));

    for (int i = 0; i < w; ++i) {
        for (int j = 0; j < h; ++j) {
            vec3 v = vec3(-1.0 + dw * i, (-1.0 + dh * j) * h / w, z);
            vec3 dir = mult(bx, by, bz, v);
            data[(h - 1 - j) * w + i] = ray(camera_pos, norm(dir), light_pos, light_color, polygons, polygons_cnt);
        }
    }
}

// рендеринг на гпу
// функция почти такая же как и для цпу, но параллелим вычисления по пикселям
__global__ void kernel_render(vec3 camera_pos, vec3 camera_view,
                                int w, int h, double angle, uchar4* data,
                                vec3 light_pos, uchar4 light_color,
                                polygon* polygons, int polygons_cnt) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    double dw = 2.0 / (w - 1.0);
    double dh = 2.0 / (h - 1.0);
    double z = 1.0 / tan(angle * M_PI / 360.0);

    // переходим в новый базис, связанный с камерой
    vec3 bz = norm(camera_view - camera_pos);
    vec3 bx = norm(prod(bz, {0.0, 0.0, 1.0}));
    vec3 by = norm(prod(bx, bz));

    for (int i = idx; i < w; i += offsetx) {
        for (int j = idy; j < h; j += offsety) {
            vec3 v = vec3(-1.0 + dw * i, (-1.0 + dh * j) * h / w, z);
            vec3 dir = mult(bx, by, bz, v);
            data[(h - 1 - j) * w + i] = ray(camera_pos, norm(dir), light_pos, light_color, polygons, polygons_cnt);
        }
    }
}

// сглаживание
__host__ __device__ void ssaa(uchar4* data, uchar4* ssaa_data, int w, int h, int sqrt_rays_per_pixel) {
    for (int x = 0; x < w; ++x) {
        for (int y = 0; y < h; ++y) {
            uint4 tmp = make_uint4(0, 0, 0, 0);
            for (int i = 0; i < sqrt_rays_per_pixel; ++i) {
                for (int j = 0; j < sqrt_rays_per_pixel; ++j) {
                    uchar4 cur_pixel = data[w * sqrt_rays_per_pixel * (y * sqrt_rays_per_pixel + j) + (x * sqrt_rays_per_pixel + i)];
                    tmp.x += cur_pixel.x;
                    tmp.y += cur_pixel.y;
                    tmp.z += cur_pixel.z;
                }
            }
            int rays_per_pixel = sqrt_rays_per_pixel * sqrt_rays_per_pixel;
            ssaa_data[y * w + x] = make_uchar4(tmp.x / rays_per_pixel, tmp.y / rays_per_pixel, tmp.z / rays_per_pixel, 255);
        }
    }
}

// сглаживание на гпу
// функция почти такая же, но параллелим вычисления по пикселям
__global__ void kernel_ssaa(uchar4* data, uchar4* ssaa_data, int w, int h, int sqrt_rays_per_pixel) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    for (int x = idx; x < w; x += offsetx) {
        for (int y = idy; y < h; y += offsety) {
            uint4 tmp = make_uint4(0, 0, 0, 0);
            for (int i = 0; i < sqrt_rays_per_pixel; ++i) {
                for (int j = 0; j < sqrt_rays_per_pixel; ++j) {
                    uchar4 cur_pixel = data[w * sqrt_rays_per_pixel * (y * sqrt_rays_per_pixel + j) + (x * sqrt_rays_per_pixel + i)];
                    tmp.x += cur_pixel.x;
                    tmp.y += cur_pixel.y;
                    tmp.z += cur_pixel.z;
                }
            }
            int rays_per_pixel = sqrt_rays_per_pixel * sqrt_rays_per_pixel;
            ssaa_data[y * w + x] = make_uchar4(tmp.x / rays_per_pixel, tmp.y / rays_per_pixel, tmp.z / rays_per_pixel, 255);
        }
    }
}


// объекты на сцене
// 0. Сама сцена (пол)
void add_scene_polygons(vec3 a, vec3 b, vec3 c, vec3 d, uchar4 color, polygon* polygons, int begin_id) {
    polygons[begin_id] = polygon(a, b, c, color);
    polygons[begin_id + 1] = polygon(a, c, d, color);
}

// 1. Тетраэдр
void add_tetrahedron_polygons(vec3 center, uchar4 color, double r, polygon* polygons, int begin_id) {
    double a = r * sqrt(3);
    vec3 vertices[] = {
        vec3(center.x - a / 2, 0, center.z - a / sqrt(12)),
        vec3(center.x, center.y + r, center.z - a / sqrt(12)),
        vec3(center.x + a / 2, 0, center.z - a / sqrt(12)),
        vec3(center.x, center.y, center.z + r)
    };

    polygons[begin_id] = polygon(vertices[0], vertices[1], vertices[2], color);
    polygons[begin_id + 1] = polygon(vertices[0], vertices[1], vertices[3], color);
    polygons[begin_id + 2] = polygon(vertices[0], vertices[2], vertices[3], color);
    polygons[begin_id + 3] = polygon(vertices[1], vertices[2], vertices[3], color);
}

// 2. Гексаэдр (по-русски - куб)
void add_hexahedron_polygons(vec3 center, uchar4 color, double r, polygon* polygons, int begin_id) {
    double a = 2 * r / sqrt(3);
    vec3 first_v = vec3(center.x - a / 2, center.y - a / 2, center.z - a / 2);
    vec3 vertices[] = {
        vec3(first_v.x, first_v.y, first_v.z),
        vec3(first_v.x, first_v.y + a, first_v.z),
        vec3(first_v.x + a, first_v.y + a, first_v.z),
        vec3(first_v.x + a, first_v.y, first_v.z),
        vec3(first_v.x, first_v.y, first_v.z + a),
        vec3(first_v.x, first_v.y + a, first_v.z + a),
        vec3(first_v.x + a, first_v.y + a, first_v.z + a),
        vec3(first_v.x + a, first_v.y, first_v.z + a)
    };

    polygons[begin_id] = polygon(vertices[0], vertices[1], vertices[2], color);
    polygons[begin_id + 1] = polygon(vertices[2], vertices[3], vertices[0], color);
    polygons[begin_id + 2] = polygon(vertices[6], vertices[7], vertices[3], color);
    polygons[begin_id + 3] = polygon(vertices[3], vertices[2], vertices[6], color);
    polygons[begin_id + 4] = polygon(vertices[2], vertices[1], vertices[5], color);
    polygons[begin_id + 5] = polygon(vertices[5], vertices[6], vertices[2], color);
    polygons[begin_id + 6] = polygon(vertices[4], vertices[5], vertices[1], color);
    polygons[begin_id + 7] = polygon(vertices[1], vertices[0], vertices[4], color);
    polygons[begin_id + 8] = polygon(vertices[3], vertices[7], vertices[4], color);
    polygons[begin_id + 9] = polygon(vertices[4], vertices[0], vertices[3], color);
    polygons[begin_id + 10] = polygon(vertices[6], vertices[5], vertices[4], color);
    polygons[begin_id + 11] = polygon(vertices[4], vertices[7], vertices[6], color);
}

// 3. Додекаэдр - страшная штука из правильных пятиугольников
void add_dodecahedron_polygons(vec3 center, uchar4 color, double r, polygon* polygons, int begin_id) {
    double a = (1 + sqrt(5)) / 2;
    double b = 1 / a;
    vec3 vertices[] = {
        vec3(-b, 0, a),
        vec3(b, 0, a), 
        vec3(-1, 1, 1), 
        vec3(1, 1, 1), 
        vec3(1, -1, 1), 
        vec3(-1, -1, 1), 
        vec3(0, -a, b), 
        vec3(0, a, b), 
        vec3(-a, -b, 0), 
        vec3(-a, b, 0), 
        vec3(a, b, 0), 
        vec3(a, -b, 0), 
        vec3(0, -a, -b), 
        vec3(0, a, -b), 
        vec3(1, 1, -1), 
        vec3(1, -1, -1), 
        vec3(-1, -1, -1), 
        vec3(-1, 1, -1), 
        vec3(b, 0, -a), 
        vec3(-b, 0, -a)
    };

    for (auto& v: vertices) {
        v.x = v.x * r / sqrt(3) + center.x;
        v.y = v.y * r / sqrt(3) + center.y;
        v.z = v.z * r / sqrt(3) + center.z;
    }

    polygons[begin_id] = polygon(vertices[4], vertices[0], vertices[6], color);
    polygons[begin_id + 1] = polygon(vertices[0], vertices[5], vertices[6], color);
    polygons[begin_id + 2] = polygon(vertices[0], vertices[4], vertices[1], color);
    polygons[begin_id + 3] = polygon(vertices[0], vertices[3], vertices[7], color);
    polygons[begin_id + 4] = polygon(vertices[2], vertices[0], vertices[7], color);
    polygons[begin_id + 5] = polygon(vertices[0], vertices[1], vertices[3], color);
    polygons[begin_id + 6] = polygon(vertices[10], vertices[1], vertices[11], color);
    polygons[begin_id + 7] = polygon(vertices[3], vertices[1], vertices[10], color);
    polygons[begin_id + 8] = polygon(vertices[1], vertices[4], vertices[11], color);
    polygons[begin_id + 9] = polygon(vertices[5], vertices[0], vertices[8], color);
    polygons[begin_id + 10] = polygon(vertices[0], vertices[2], vertices[9], color);
    polygons[begin_id + 11] = polygon(vertices[8], vertices[0], vertices[9], color);
    polygons[begin_id + 12] = polygon(vertices[5], vertices[8], vertices[16], color);
    polygons[begin_id + 13] = polygon(vertices[6], vertices[5], vertices[12], color);
    polygons[begin_id + 14] = polygon(vertices[12], vertices[5], vertices[16], color);
    polygons[begin_id + 15] = polygon(vertices[4], vertices[12], vertices[15], color);
    polygons[begin_id + 16] = polygon(vertices[4], vertices[6], vertices[12], color);
    polygons[begin_id + 17] = polygon(vertices[11], vertices[4], vertices[15], color);
    polygons[begin_id + 18] = polygon(vertices[2], vertices[13], vertices[17], color);
    polygons[begin_id + 19] = polygon(vertices[2], vertices[7], vertices[13], color);
    polygons[begin_id + 20] = polygon(vertices[9], vertices[2], vertices[17], color);
    polygons[begin_id + 21] = polygon(vertices[13], vertices[3], vertices[14], color);
    polygons[begin_id + 22] = polygon(vertices[7], vertices[3], vertices[13], color);
    polygons[begin_id + 23] = polygon(vertices[3], vertices[10], vertices[14], color);
    polygons[begin_id + 24] = polygon(vertices[8], vertices[17], vertices[19], color);
    polygons[begin_id + 25] = polygon(vertices[16], vertices[8], vertices[19], color);
    polygons[begin_id + 26] = polygon(vertices[8], vertices[9], vertices[17], color);
    polygons[begin_id + 27] = polygon(vertices[14], vertices[11], vertices[18], color);
    polygons[begin_id + 28] = polygon(vertices[11], vertices[15], vertices[18], color);
    polygons[begin_id + 29] = polygon(vertices[10], vertices[11], vertices[14], color);
    polygons[begin_id + 30] = polygon(vertices[12], vertices[19], vertices[18], color);
    polygons[begin_id + 31] = polygon(vertices[15], vertices[12], vertices[18], color);
    polygons[begin_id + 32] = polygon(vertices[12], vertices[16], vertices[19], color);
    polygons[begin_id + 33] = polygon(vertices[19], vertices[13], vertices[18], color);
    polygons[begin_id + 34] = polygon(vertices[17], vertices[13], vertices[19], color);
    polygons[begin_id + 35] = polygon(vertices[13], vertices[14], vertices[18], color);
}

void print_default_inputs() {
    std::cout << "100" << std::endl;;
    std::cout << "res/%d.data" << std::endl;
    std::cout << "600 600 120" << std::endl << std::endl;

    std::cout << "7.0 3.0 0.0     2.0 1.0     2.0 6.0 1.0     0.0 0.0" << std::endl;
    std::cout << "2.0 0.0 0.0     0.5 0.1     1.0 4.0 1.0     0.0 0.0" << std::endl << std::endl;

    std::cout << "3.0 3.0 0.5    0.3 0.55 0.0     1.0" << std::endl;
    std::cout << "0.0 0.0 0.0     0.5 0.25 0.55     1.75" << std::endl;
    std::cout << "-3.0 -3.0 0.0     0.0 0.7 0.7     1.5" << std::endl << std::endl;

    std::cout << "-5.0 -5.0 -1.0     -5.0 5.0 -1.0    5.0 5.0 -1.0    5.0 -5.0 -1.0   1.0 0.9 0.25" << std::endl << std::endl;

    std::cout << "-10.0 0.0 15.0     0.3 0.2 0.1" << std::endl << std::endl;

    std::cout << "4" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc >= 2 && std::string(argv[1]) == "--default") {
        print_default_inputs();
        return 0;
    }

    bool use_gpu = true;
    if (argc >= 2 && std::string(argv[1]) == "--cpu") {
        use_gpu = false;
    }

    int frames_number; // кол-во кадров
    char output_path[256]; // куда класть изображения
    int w, h; // размеры экрана
    double angle; // угол обзора
    double r0c, z0c, phi0c, Arc, Azc, wrc, wzc, wphic, prc, pzc; // параметры движения камеры
    double r0n, z0n, phi0n, Arn, Azn, wrn, wzn, wphin, prn, pzn;
    // параметры первого тела (тетраэдра)
    double center1_x, center1_y, center1_z;
    double color1_x, color1_y, color1_z;
    double r1;
    // параметры второго тела (куба)
    double center2_x, center2_y, center2_z;
    double color2_x, color2_y, color2_z;
    double r2;
    // параметры третьего тела (додекаэдра)
    double center3_x, center3_y, center3_z;
    double color3_x, color3_y, color3_z;
    double r3;
    // параметры пола
    double floor1_x, floor1_y, floor1_z, floor2_x, floor2_y, floor2_z;
    double floor3_x, floor3_y, floor3_z, floor4_x, floor4_y, floor4_z;
    double floor_color_x, floor_color_y, floor_color_z;
    // парамтеры источника света (он только один, извините)
    double light_pos_x, light_pos_y, light_pos_z;
    double light_color_x, light_color_y, light_color_z;
    double sqrt_rays_per_pixel; // параметры SSAA (рекурсии нет :с)

    std::cin >> frames_number;
    std::cin >> output_path;
    std::cin >> w >> h >> angle;
    std::cin >> r0c >> z0c >> phi0c >> Arc >> Azc >> wrc >> wzc >> wphic >> prc >> pzc;
    std::cin >> r0n >> z0n >> phi0n >> Arn >> Azn >> wrn >> wzn >> wphin >> prn >> pzn;
    std::cin >> center1_x >> center1_y >> center1_z;
    std::cin >> color1_x >> color1_y >> color1_z;
    std::cin >> r1;
    std::cin >> center2_x >> center2_y >> center2_z;
    std::cin >> color2_x >> color2_y >> color2_z;
    std::cin >> r2;
    std::cin >> center3_x >> center3_y >> center3_z;
    std::cin >> color3_x >> color3_y >> color3_z;
    std::cin >> r3;
    std::cin >> floor1_x >> floor1_y >> floor1_z >> floor2_x >> floor2_y >> floor2_z;
    std::cin >> floor3_x >> floor3_y >> floor3_z >> floor4_x >> floor4_y >> floor4_z;
    std::cin >> floor_color_x >> floor_color_y >> floor_color_z;
    std::cin >> light_pos_x >> light_pos_y >> light_pos_z;
    std::cin >> light_color_x >> light_color_y >> light_color_z;
    std::cin >> sqrt_rays_per_pixel;


    // создаем полигоны для объектов
    polygon polygons[54];
    add_scene_polygons(
        vec3(floor1_x, floor1_y, floor1_z),
        vec3(floor2_x, floor2_y, floor2_z),
        vec3(floor3_x, floor3_y, floor3_z),
        vec3(floor4_x, floor4_y, floor4_z),
        make_uchar4(floor_color_x * 255, floor_color_y * 255, floor_color_z * 255, 255),
        polygons, 0
    );
    add_tetrahedron_polygons(
        vec3(center1_x, center1_y, center1_z),
        make_uchar4(color1_x * 255, color1_y * 255, color1_z * 255, 255),
        r1, polygons, 2
    );
    add_hexahedron_polygons(
        vec3(center2_x, center2_y, center2_z),
        make_uchar4(color2_x * 255, color2_y * 255, color2_z * 255, 255),
        r2, polygons, 6
    );
    add_dodecahedron_polygons(
        vec3(center3_x, center3_y, center3_z),
        make_uchar4(color3_x * 255, color3_y * 255, color3_z * 255, 255),
        r3, polygons, 18
    );

    vec3 light_pos = vec3(light_pos_x, light_pos_y, light_pos_z);
    uchar4 light_color = make_uchar4(light_color_x * 255, light_color_y * 255, light_color_z * 255, 255);

    // выделяем память под результаты
    uchar4* data = (uchar4*)malloc(sizeof(uchar4) * w * h * sqrt_rays_per_pixel * sqrt_rays_per_pixel);
    uchar4* ssaa_data = (uchar4*)malloc(sizeof(uchar4) * w * h);
    uchar4* dev_data;
    uchar4* dev_ssaa_data;
    polygon* dev_polygons;
    char buff[256];
    if (use_gpu) {
        CSC(cudaMalloc(&dev_data, sizeof(uchar4) * w * h * sqrt_rays_per_pixel * sqrt_rays_per_pixel));
        CSC(cudaMalloc(&dev_ssaa_data, sizeof(uchar4) * w * h));
        CSC(cudaMalloc(&dev_polygons, sizeof(polygon) * 54));
        CSC(cudaMemcpy(dev_polygons, polygons, sizeof(polygon) * 54, cudaMemcpyHostToDevice));
    }

    for (int frame = 0; frame < frames_number; ++frame) {
        double t = 2 * M_PI * frame / frames_number;
        vec3 camera_pos, camera_view;

        double rc = r0c + Arc * sin(wrc * t + prc);
        double zc = z0c + Azc * sin(wzc * t + pzc);
        double phic = phi0c + wphic * t;

        double rn = r0n + Arn * sin(wrn * t + prn);
        double zn = z0n + Azn * sin(wzn * t + pzn);
        double phin = phi0n + wphin * t;

        camera_pos.x = rc * cos(phic);
        camera_pos.y = rc * sin(phic);
        camera_pos.z = zc;

        camera_view.x = rn * cos(phin);
        camera_view.y = rn * sin(phin);
        camera_view.z = zn;

        cudaEvent_t start, stop;
        CSC(cudaEventCreate(&start));
        CSC(cudaEventCreate(&stop));
        CSC(cudaEventRecord(start));

        if (use_gpu) {
            kernel_render<<<dim3(16, 16), dim3(16, 16)>>>(
                camera_pos, camera_view, w * sqrt_rays_per_pixel, h * sqrt_rays_per_pixel, angle,
                dev_data, light_pos, light_color, dev_polygons, 54
            );
            CSC(cudaGetLastError());
            kernel_ssaa<<<dim3(16, 16), dim3(16, 16)>>>(dev_data, dev_ssaa_data, w, h, sqrt_rays_per_pixel);
            CSC(cudaGetLastError());
            CSC(cudaMemcpy(ssaa_data, dev_ssaa_data, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));
        } else {
            render(camera_pos, camera_view, w * sqrt_rays_per_pixel, h * sqrt_rays_per_pixel, angle,
                data, light_pos, light_color, polygons, 54
            );
            ssaa(data, ssaa_data, w, h, sqrt_rays_per_pixel);
        }
        CSC(cudaEventRecord(stop));
        CSC(cudaEventSynchronize(stop));
        float time;
        CSC(cudaEventElapsedTime(&time, start, stop));
        CSC(cudaEventDestroy(start));
        CSC(cudaEventDestroy(stop));

        sprintf(buff, output_path, frame);
        FILE* output_file = fopen(buff, "w");
        fwrite(&w, sizeof(int), 1, output_file);
        fwrite(&h, sizeof(int), 1, output_file);
        fwrite(ssaa_data, sizeof(uchar4), w * h, output_file);
        fclose(output_file);

        std::cout << frame+1 << "\t" << time << "\t" << w * h * sqrt_rays_per_pixel * sqrt_rays_per_pixel << std::endl;
    }

    free(data);
    free(ssaa_data);
    if (use_gpu) {
        CSC(cudaFree(dev_data));
        CSC(cudaFree(dev_ssaa_data));
    }
    return 0;
}
