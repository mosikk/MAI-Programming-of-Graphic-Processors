#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <cstdlib>

#include "mpi.h"

// макросы для определения индексов элементов внутри блока
#define _i(i, j, k) (((k) + 1) * (y_block + 2) * (x_block + 2) + ((j) + 1) * (x_block + 2) + (i) + 1)
#define _ix(id) (((id) % (x_block + 2)) - 1)
#define _iy(id) ((((id) % ((y_block + 2) * (x_block + 2)) ) / (x_block + 2)) - 1)
#define _iz(id) (((id) / ((y_block + 2) * (x_block + 2))) - 1)

// макросы для определения индексов блоков
#define _ib(i, j, k) ((k) * (x_grid * y_grid) + (j) * x_grid + (i))
#define _ibx(id) (((id) % (x_grid * y_grid)) % x_grid)
#define _iby(id) (((id) % (x_grid * y_grid)) / x_grid)
#define _ibz(id) ((id) / (x_grid * y_grid))


int main(int argc, char *argv[]) {
    int x_grid, y_grid, z_grid; // размеры сетки процессов
    int x_block, y_block, z_block; // размеры блоков
    std::string output_filename; // файл куда писать ответ
    double eps; // точность
    double lx, ly, lz; // размеры области
    double u_down, u_up, u_left, u_right, u_front, u_back; // граничные условия
    double u_0; // начальное значение
    int numproc, id;

    // инициализация MPI
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Barrier(MPI_COMM_WORLD);

    if (id == 0) {
        // считываем данные в нулевом процессе
        std::cin >> x_grid >> y_grid >> z_grid;
        std::cin >> x_block >> y_block >> z_block;
        std::cin >> output_filename;
        std::cin >> eps;
        std::cin >> lx >> ly >> lz;
        std::cin >> u_down >> u_up >> u_left >> u_right >> u_front >> u_back;
        std::cin >> u_0;
    }

    // передаем входные данные всем процессам
    MPI_Bcast(&x_grid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&y_grid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&z_grid, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Bcast(&x_block, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&y_block, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&z_block, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Bcast(&eps, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(&lx, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ly, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&lz, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(&u_down, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&u_up, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&u_left, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&u_right, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&u_front, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&u_back, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&u_0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // считаем шаги сетки
    double hx = lx / (x_grid * x_block);	
    double hy = ly / (y_grid * y_block);
    double hz = lz / (z_grid * z_block);

    // выделяем память для значений в сетке
    double* data = (double*)malloc(sizeof(double) * (x_block+2) * (y_block+2) * (z_block+2));	
    double* next = (double*)malloc(sizeof(double) * (x_block+2) * (y_block+2) * (z_block+2));
    double* buff = (double*)malloc(sizeof(double) * x_block * y_block * z_block);

    // буффер для отправки сообщений
    int buffer_size;
    MPI_Pack_size((x_block+2) * (y_block+2) * (z_block+2), MPI_DOUBLE, MPI_COMM_WORLD, &buffer_size);
    buffer_size = 12 * (buffer_size + MPI_BSEND_OVERHEAD);
    double* buffer = (double*)malloc(buffer_size);
    MPI_Buffer_attach(buffer, buffer_size);

    // инициализируем блок
    for (int i = 0; i < x_block; ++i) {
        for (int j = 0; j < y_block; ++j) {
            for (int k = 0; k < z_block; ++k) {
                data[_i(i, j, k)] = u_0;
            }
        }
    }

    // используем трехмерную индексацию процессов
    int ib = _ibx(id);
    int jb = _iby(id);
    int kb = _ibz(id);

    bool converge = false;
    double* all_max_diffs = (double*)malloc(sizeof(double) * numproc); // сюда будем класть диффы в каждом блоке
    while (!converge) {
        MPI_Barrier(MPI_COMM_WORLD);

        // обмениваемся граничными блоками
        // отправляем данные след блоку по иксу
        if (ib + 1 < x_grid) {
            for (int j = 0; j < y_block; ++j) {
                for (int k = 0; k < z_block; ++k) {
                    buff[j * z_block + k] = data[_i(x_block-1, j, k)];
                }
            }
            MPI_Bsend(buff, y_block * z_block, MPI_DOUBLE, _ib(ib + 1, jb, kb), id, MPI_COMM_WORLD);
        }

        // отправляем данные пред блоку по иксу
        if (ib - 1 >= 0) {
            for (int j = 0; j < y_block; ++j) {
                for (int k = 0; k < z_block; ++k) {
                    buff[j * z_block + k] = data[_i(0, j, k)];
                }
            }
            MPI_Bsend(buff, y_block * z_block, MPI_DOUBLE, _ib(ib - 1, jb, kb), id, MPI_COMM_WORLD);
        }

        // отправляем данные след блоку по игреку
        if (jb + 1 < y_grid) {
            for (int i = 0; i < x_block; ++i) {
                for (int k = 0; k < z_block; ++k) {
                    buff[i * z_block + k] = data[_i(i, y_block-1, k)];
                }
            }
            MPI_Bsend(buff, x_block * z_block, MPI_DOUBLE, _ib(ib, jb + 1, kb), id, MPI_COMM_WORLD);
        }

        // отправляем данные пред блоку по игреку
        if (jb - 1 >= 0) {
            for (int i = 0; i < x_block; ++i) {
                for (int k = 0; k < z_block; ++k) {
                    buff[i * z_block + k] = data[_i(i, 0, k)];
                }
            }
            MPI_Bsend(buff, x_block * z_block, MPI_DOUBLE, _ib(ib, jb - 1, kb), id, MPI_COMM_WORLD);
        }

        // отправляем данные след блоку по зету
        if (kb + 1 < z_grid) {
            for (int i = 0; i < x_block; ++i) {
                for (int j = 0; j < y_block; ++j) {
                    buff[i * y_block + j] = data[_i(i, j, z_block-1)];
                }
            }
            MPI_Bsend(buff, x_block * y_block, MPI_DOUBLE, _ib(ib, jb, kb + 1), id, MPI_COMM_WORLD);
        }

        // отправляем данные пред блоку по зету
        if (kb - 1 >= 0) {
            for (int i = 0; i < x_block; ++i) {
                for (int j = 0; j < y_block; ++j) {
                    buff[i * y_block + j] = data[_i(i, j, 0)];
                }
            }
            MPI_Bsend(buff, x_block * y_block, MPI_DOUBLE, _ib(ib, jb, kb - 1), id, MPI_COMM_WORLD);
        }

        // принимаем данные со след блока по иксу
        if (ib + 1 < x_grid) {
            MPI_Recv(buff, y_block * z_block, MPI_DOUBLE, _ib(ib + 1, jb, kb), _ib(ib + 1, jb, kb), MPI_COMM_WORLD, &status);
            for (int j = 0; j < y_block; ++j) {
                for (int k = 0; k < z_block; ++k) {
                    data[_i(x_block, j, k)] = buff[j * z_block + k];
                }
            }
        } else {
            // дальше ничего нет - берем граничные условия
            for (int j = 0; j < y_block; ++j) {
                for (int k = 0; k < z_block; ++k) {
                    data[_i(x_block, j, k)] = u_right;
                }
            }
        }

        // принимаем данные с пред блока по иксу
        if (ib - 1 >= 0) {
            MPI_Recv(buff, y_block * z_block, MPI_DOUBLE, _ib(ib - 1, jb, kb), _ib(ib - 1, jb, kb), MPI_COMM_WORLD, &status);
            for (int j = 0; j < y_block; ++j) {
                for (int k = 0; k < z_block; ++k) {
                    data[_i(-1, j, k)] = buff[j * z_block + k];
                }
            }
        } else {
            // ранее ничего не было - берем граничные условия
            for (int j = 0; j < y_block; ++j) {
                for (int k = 0; k < z_block; ++k) {
                    data[_i(-1, j, k)] = u_left;
                }
            }
        }

        // принимаем данные со след блока по игреку
        if (jb + 1 < y_grid) {
            MPI_Recv(buff, x_block * z_block, MPI_DOUBLE, _ib(ib, jb + 1, kb), _ib(ib, jb + 1, kb), MPI_COMM_WORLD, &status);
            for (int i = 0; i < x_block; ++i) {
                for (int k = 0; k < z_block; ++k) {
                    data[_i(i, y_block, k)] = buff[i * z_block + k];
                }
            }
        } else {
            // дальше ничего нет - берем граничные условия
            for (int i = 0; i < x_block; ++i) {
                for (int k = 0; k < z_block; ++k) {
                    data[_i(i, y_block, k)] = u_back;
                }
            }
        }

        // принимаем данные с пред блока по игреку
        if (jb - 1 >= 0) {
            MPI_Recv(buff, x_block * z_block, MPI_DOUBLE, _ib(ib, jb - 1, kb), _ib(ib, jb - 1, kb), MPI_COMM_WORLD, &status);
            for (int i = 0; i < x_block; ++i) {
                for (int k = 0; k < z_block; ++k) {
                    data[_i(i, -1, k)] = buff[i * z_block + k];
                }
            }
        } else {
            // ранее ничего не было - берем граничные условия
            for (int i = 0; i < x_block; ++i) {
                for (int k = 0; k < z_block; ++k) {
                    data[_i(i, -1, k)] = u_front;
                }
            }
        }

        // принимаем данные со след блока по зету
        if (kb + 1 < z_grid) {
            MPI_Recv(buff, x_block * y_block, MPI_DOUBLE, _ib(ib, jb, kb + 1), _ib(ib, jb, kb + 1), MPI_COMM_WORLD, &status);
            for (int i = 0; i < x_block; ++i) {
                for (int j = 0; j < y_block; ++j) {
                    data[_i(i, j, z_block)] = buff[i * y_block + j];
                }
            }
        } else {
            // дальше ничего нет - берем граничные условия
            for (int i = 0; i < x_block; ++i) {
                for (int j = 0; j < y_block; ++j) {
                    data[_i(i, j, z_block)] = u_up;
                }
            }
        }

        // принимаем данные с пред блока по зету
        if (kb - 1 >= 0) {
            MPI_Recv(buff, x_block * y_block, MPI_DOUBLE, _ib(ib, jb, kb - 1), _ib(ib, jb, kb - 1), MPI_COMM_WORLD, &status);
            for (int i = 0; i < x_block; ++i) {
                for (int j = 0; j < y_block; ++j) {
                    data[_i(i, j, -1)] = buff[i * y_block + j];
                }
            }
        } else {
            // ранее ничего не было - берем граничные условия
            for (int i = 0; i < x_block; ++i) {
                for (int j = 0; j < y_block; ++j) {
                    data[_i(i, j, -1)] = u_down;
                }
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        // считаем новое значение функции
        double cur_max_diff = 0.0; // одновременно будем смотреть максимальное изменение в каждом блоке
        for (int i = 0; i < x_block; ++i) {
            for (int j = 0; j < y_block; ++j) {
                for (int k = 0; k < z_block; ++k) {
                    double num = (
                        (data[_i(i + 1, j, k)] + data[_i(i - 1, j, k)]) / (hx * hx)
                        + (data[_i(i, j + 1, k)] + data[_i(i, j - 1, k)]) / (hy * hy)
                        + (data[_i(i, j, k + 1)] + data[_i(i, j, k - 1)]) / (hz * hz)
                        
                    );
                    double denom = (
                        2 * (1.0 / (hx * hx) + 1.0 / (hy * hy) + 1.0 / (hz * hz))
                    );
                    next[_i(i, j, k)] = num / denom;
                    if (fabs(next[_i(i, j, k)] - data[_i(i, j, k)]) > cur_max_diff) {
                        cur_max_diff = fabs(next[_i(i, j, k)] - data[_i(i, j, k)]);
                    }
                }
            }
        }

        double* temp = data;
        data = next;
        next = temp;

        all_max_diffs[id] = cur_max_diff;
        MPI_Barrier(MPI_COMM_WORLD);

        // собираем инфу о диффах по всем блокам
        MPI_Allgather(&cur_max_diff, 1, MPI_DOUBLE, all_max_diffs, 1, MPI_DOUBLE, MPI_COMM_WORLD);
        
        double cur_global_max_diff = 0;
        for (int i = 0; i < numproc; ++i) {
            if (all_max_diffs[i] > cur_global_max_diff) {
                cur_global_max_diff = all_max_diffs[i];
            }
        }
        if (cur_global_max_diff < eps) {
            converge = true; // заканчиваем алгоритм
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);

    // закончили вычисления, формируем ответ
    if (id != 0) {
        // отправляем результаты вычислений в главный процесс
        for (int k = 0; k < z_block; ++k) {
            for (int j = 0; j < y_block; ++j) {
                for (int i = 0; i < x_block; ++i) {
                    buff[i] = data[_i(i, j, k)];
                }
                MPI_Send(buff, x_block, MPI_DOUBLE, 0, k * y_block + j, MPI_COMM_WORLD);
            }
        }
    } else {
        // мы в главном процессе, читаем что нам прислали и пишем это в файл
        std::ofstream output_file(output_filename);
        output_file << std::scientific << std::setprecision(6);

        // итерируемся по всем блокам и по всем ячейкам блоков
        for (int kb = 0; kb < z_grid; ++kb) {
            for (int k = 0; k < z_block; ++k) {
                for (int jb = 0; jb < y_grid; ++jb) {
                    for (int j = 0; j < y_block; ++j) {
                        for (int ib = 0; ib < x_grid; ++ib) {
                            
                            // считываем в буфер значения на слое
                            if (_ib(ib, jb, kb) == 0) {
                                for (int i = 0; i < x_block; ++i) {
                                    buff[i] = data[_i(i, j, k)];
                                }
                            } else {
                                MPI_Recv(buff, x_block, MPI_DOUBLE, _ib(ib, jb, kb), k * y_block + j, MPI_COMM_WORLD, &status);
                            }

                            for (int i = 0; i < x_block; ++i) {
                                output_file << buff[i] << " ";
                            }
                            if (ib + 1 == x_grid) {
                                output_file << "\n";
                            }
                        }
                    }
                }
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Buffer_detach(buffer, &buffer_size);
    MPI_Finalize();	// заканчиваем работу с MPI

    // освобождаем всю память
    free(data);
    free(next);
    free(buff);
    free(buffer);
    free(all_max_diffs);

    return 0;
}
