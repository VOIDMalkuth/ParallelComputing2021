#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>
#include <pthread.h>
#include <mpi.h>

#define SCATTER_A_ID (50)
#define SCATTER_B_ID (51)
#define SHIFT_A_ID (60)
#define SHIFT_B_ID (61)
#define REDUCE_C_ID (70)

#define SCATTER_TYPE_A (0)
#define SCATTER_TYPE_B (1)

int myrank, numprocs;

// 分发矩阵，根据sndOrRecv是不是0号判断是分发还是接收
// 0号进程：isMaster为1，发送数据，matBuf是要分发的矩阵，buf是自己的缓存
// 其余进程：isMaster为0，接收数据，matBuf无效，buf是要接收数据存放的缓存
// type: A阵是1，B阵是2
void scatter_matrix(int isMaster, int type, int row, int col, double *buf, int rootNumprocs, int tag, double *matBuf) {
    // 计算最大行数和最大列数
    int maxrows = (row + rootNumprocs - 1) / rootNumprocs;
    int maxcols = (col + rootNumprocs - 1) / rootNumprocs;
    if (isMaster) {
        double *tmpBuf = (double *)malloc(sizeof(double) * maxrows * maxcols);
        int i = 0, j = 0, k = 0, l = 0;
        for (i = 0; i < rootNumprocs; i++) {
            for (j = 0; j < rootNumprocs; j++) {
                // 处理第(i,j)=i*rootNumprocs+j个数据块
                for (k = 0; k < maxrows; k++) {
                    for (l = 0; l < maxcols; l++) {
                        // (i, j)的初始第(k, l)个元素
                        // 其实是总的(i * maxrows + k, j * maxcols + l)个
                        // 此处i*maxrow是因为竖着第i块前有i个maxrow行的块，j同理
                        double value = 0;
                        // 自动填0
                        if (i * maxrows + k < row && j * maxcols + l < col) {
                            value = matBuf[(i * maxrows + k) * col + j * maxcols + l];
                        }
                        tmpBuf[k*maxcols + l] = value;
                    }
                }
                // 如果是自己的，直接拷贝过去
                if (i ==0 && j == 0) {
                    for (k = 0; k < maxrows * maxcols; k++) {
                        buf[k] = tmpBuf[k];
                    }
                } else {
                    // 否则，MPI发过去
                    int target = -1;
                    // 发送同时完成初始的位移
                    if (type == SCATTER_TYPE_A) {
                        // A阵需要第i行左移i列
                        target = i * rootNumprocs + (j - i + rootNumprocs) % rootNumprocs;
                        // printf("A: send (%d, %d) to (%d, %d)\n", i, j, i, (j - i + rootNumprocs) % rootNumprocs);
                    } else {
                        // B阵需要第j列上移j列
                        target = ((i - j + rootNumprocs) % rootNumprocs) * rootNumprocs + j;
                        // printf("B: send (%d, %d) to (%d, %d)\n", i, j, (i - j + rootNumprocs) % rootNumprocs, j);
                    }
                    MPI_Send(tmpBuf, maxrows * maxcols, MPI_DOUBLE, target, tag, MPI_COMM_WORLD);
                }
            }
        }
        free(tmpBuf);
    } else {
        MPI_Status status;
        MPI_Recv(buf, maxrows * maxcols, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD, &status);
    }
}

void simpleMatMul(double *bufA, double *bufB, double *bufRes, int n1, int n2, int n3) {
    int i, j, k;
    for (i = 0; i < n1; i++) {
        for (j = 0; j < n3; j++) {
            double res = 0;
            for (k = 0; k < n2; k++) {
                res += bufA[i * n2 + k] * bufB[k * n3 + j];
            }
            bufRes[i * n3 + j] = res;
        }
    }
}

void cannon(double *bufA, double *bufB, double *bufC, double *tmpBufA, double *tmpBufB, double *tmpBufC, int n1, int n2, int n3) {
    int rootNumprocs = sqrt(numprocs);
    // 确定各个缓存的大小
    // 如果不能整除，就要按最大块向上取整确定大小
    int maxrows_a = (n1 + rootNumprocs - 1) / rootNumprocs;
    int maxcols_a = (n2 + rootNumprocs - 1) / rootNumprocs;
    int maxrows_b = maxcols_a;
    int maxcols_b = (n3 + rootNumprocs - 1) / rootNumprocs;
    int procBlockRow = myrank / rootNumprocs;
    int procBlockCol = myrank % rootNumprocs;

    int i, j;
    for (i = 0; i < maxrows_a*maxcols_b; i++) {
        bufC[i] = 0;
    }

    for (i = 0; i < rootNumprocs; i++) {
        // 每轮计算开始前同步一下
        MPI_Barrier(MPI_COMM_WORLD);
        // 第i轮
        simpleMatMul(bufA, bufB, tmpBufC, maxrows_a, maxcols_a, maxcols_b);
        for (j = 0; j < maxrows_a*maxcols_b; j++) {
            bufC[j] += tmpBufC[j];
        }
        // 开始循环移位块
        // 先向右移
        // 避免死锁，第0行进程先接再发，其他先发再接
        MPI_Request req[2];
        MPI_Status status[2];
        int sendTo = procBlockRow * rootNumprocs + (procBlockCol - 1 + rootNumprocs) % rootNumprocs;
        int recvFrom = procBlockRow * rootNumprocs + (procBlockCol + 1) % rootNumprocs;
        if (procBlockCol == 0) {
            MPI_Send(bufA, maxrows_a*maxcols_a, MPI_DOUBLE, sendTo, SHIFT_A_ID, MPI_COMM_WORLD);
            MPI_Recv(tmpBufA, maxrows_a*maxcols_a, MPI_DOUBLE, recvFrom, SHIFT_A_ID, MPI_COMM_WORLD, &status[0]);
        } else {
            MPI_Recv(tmpBufA, maxrows_a*maxcols_a, MPI_DOUBLE, recvFrom, SHIFT_A_ID, MPI_COMM_WORLD, &status[0]);
            MPI_Send(bufA, maxrows_a*maxcols_a, MPI_DOUBLE, sendTo, SHIFT_A_ID, MPI_COMM_WORLD);
        }
        // 更改buf和tmpBuf的位置
        double *tempPtr;
        tempPtr = bufA;
        bufA = tmpBufA;
        tmpBufA = tempPtr;
        MPI_Barrier(MPI_COMM_WORLD);
        // 再向上移位
        sendTo = ((procBlockRow - 1 + rootNumprocs) % rootNumprocs) * rootNumprocs + procBlockCol;
        recvFrom = ((procBlockRow + 1) % rootNumprocs) * rootNumprocs + procBlockCol;
        if (procBlockRow == 0) {
            MPI_Recv(tmpBufB, maxrows_b*maxcols_b, MPI_DOUBLE, recvFrom, SHIFT_B_ID, MPI_COMM_WORLD, &status[1]);
            MPI_Send(bufB, maxrows_b*maxcols_b, MPI_DOUBLE, sendTo, SHIFT_B_ID, MPI_COMM_WORLD);
        } else {
            MPI_Send(bufB, maxrows_b*maxcols_b, MPI_DOUBLE, sendTo, SHIFT_B_ID, MPI_COMM_WORLD);
            MPI_Recv(tmpBufB, maxrows_b*maxcols_b, MPI_DOUBLE, recvFrom, SHIFT_B_ID, MPI_COMM_WORLD, &status[1]);
        }
        // 更改buf和tmpBuf的位置
        tempPtr = bufB;
        bufB = tmpBufB;
        tmpBufB = tempPtr;
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

double* readMatFile(char *filename, int *row, int *col) {
    FILE* fh;
    // 打开文件
    if(!(fh = fopen(filename, "r"))) {
        printf("Can't open file %s\n", filename);
        exit(-1);
    }

    struct stat fstat;
    int n1, n2, fsize;
    char* fstream;

    // 获取文件大小
    stat(filename, &fstat);
    fsize = fstat.st_size;
    // 分配存储整个文件的空间
    fstream = (char *)malloc(fsize);
    // 将整个矩阵读入内存
    int r = fread(fstream, sizeof(char), fsize, fh);
    if (r < fsize) {
        exit(-1);
    }
    // 获取矩阵的形状参数
    n1 = ((int*)fstream)[0];
    n2 = ((int*)fstream)[1];
    // 验证矩阵形状：行数列数均大于0
    if (n1 <=0 || n2 <=0) {
        printf("Matrix size error, %dx%d\n", n1, n2);
        exit(-1);
    } 
    // 验证真实形状和矩阵形状之间的对应关系
    if (fsize < (sizeof(int)*2 + sizeof(double)*n1*n2)) {
        printf("Actual size mismatches with stated size\n");
        exit(-1);
    }
    
    double* A = (double*)(fstream + sizeof(int)*2);
    double *matrix = (double *)malloc(sizeof(double)*n1*n2);

    // printf("       ---- read %s: %d*%d Matrix -----\n", filename, n1, n2);
    int i, j;
    for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
            matrix[i * n2 + j] = *(A+i*n2+j);
        }
    }

    *row = n1;
    *col = n2;

    free(fstream);
    fclose(fh);
    return matrix;
}

// 将矩阵打印到屏幕上并存储矩阵文件
// filename：要存储的文件名称
// m：矩阵的行数
// n：矩阵的列数
// matrix：矩阵数组
int writeMatFile(char *filename, int m, int n, double *matrix) {
    double* a;

    // 分配缓存空间，存储m*n个double变量和2个int变量
    int bufsize = sizeof(int)*2 + sizeof(double)*m*n;

    a = (double*) malloc(bufsize);

    // 存储矩阵形状
    ((int*)a)[0] = m;
    ((int*)a)[1] = n;

    double *ptr = (double*)((int*)a + 2);

    srand48(time(NULL)); // Use time as a seed
    int i, j;
    printf("Result matrix: %d * %d\n", m, n);
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            // 输出矩阵数据，并且将其写入缓存中
            // printf("%.6lf  ", matrix[i * n + j]);
            *(ptr + i * n + j) = matrix[i * n + j];
        }
        // printf("\n");
    }

    // 尝试打开要写入的矩阵文件
    FILE* file;
    if(!(file = fopen(filename, "w"))) {
        printf("Can't open file %s\n", filename);
    }
    // 将整个缓存写入文件
    fwrite(a, sizeof(char), bufsize, file);
    // 关闭文件
    fclose(file);
    free(a);

    return 0;
}


int main(int argc, char** argv) {
    double *A = NULL, *B = NULL, *C = NULL;
    double *bufA = NULL, *bufB = NULL, *bufC = NULL;
    double *tmpBufA = NULL, *tmpBufB = NULL, *tmpBufC = NULL;
    // 初始化MPI
    // 获取进程总数和自身id
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    // 获取进程总数，必须是完全平方数
    int rootNumprocs = sqrt(numprocs);
    if (numprocs != rootNumprocs * rootNumprocs) {
        printf("Processor number must be a square!\n");
        exit(-1);
    }

    // On proc 0, preprocess the command line, read in files for A, B and
    // put their sizes in dim[].
    int dim[3];
    if (myrank == 0) {
        // 读取矩阵，并且存入维数
        A = readMatFile("matA.bin", &dim[0], &dim[1]);
        B = readMatFile("matB.bin", &dim[1], &dim[2]);
        C = (double *)malloc(sizeof(double) * dim[0] * dim[2]);
    }

    // 使用广播方式将维数发送到所有节点
    MPI_Bcast(dim, 3, MPI_INT, 0, MPI_COMM_WORLD);

    int n1 = dim[0];
    int n2 = dim[1];
    int n3 = dim[2];

    // Allocate memories for A, B, C, bufA and bufB.
    // Suppose an m*n matrix is 2D block-distributed on a rootp*rootp processor grid.
    // If rootp doesn't divide m or n, then submatrixes won't have the same size.
    // Because we will shift A, B, so we allocate memories according to the max
    // rows and cols of A and B.
    
    // bufA和bufB是为了在交换矩阵的时候保存数据使用，从A发送，用bufA接收
    // 如果不能整除，就要按最大块向上取整确定大小
    int maxrows_a = (n1 + rootNumprocs - 1) / rootNumprocs;
    int maxcols_a = (n2 + rootNumprocs - 1) / rootNumprocs;
    int maxrows_b = maxcols_a;
    int maxcols_b = (n3 + rootNumprocs - 1) / rootNumprocs;

    // 分配各种缓存空间
    bufA = (double *)malloc(sizeof(double) * maxrows_a * maxcols_a);
    tmpBufA = (double *)malloc(sizeof(double) * maxrows_a * maxcols_a);
    bufB = (double *)malloc(sizeof(double) * maxrows_b * maxcols_b);
    tmpBufB = (double *)malloc(sizeof(double) * maxrows_b * maxcols_b);
    bufC = (double *)malloc(sizeof(double) * maxrows_a * maxcols_b);
    tmpBufC = (double *)malloc(sizeof(double) * maxrows_a * maxcols_b);

    if (myrank == 0) {
        // 分发矩阵
        scatter_matrix(1, SCATTER_TYPE_A, n1, n2, bufA, rootNumprocs, SCATTER_A_ID, A);
        scatter_matrix(1, SCATTER_TYPE_B, n2, n3, bufB, rootNumprocs, SCATTER_B_ID, B);
    } else {
        // 接收0号进程来的矩阵
        scatter_matrix(0, SCATTER_TYPE_A, n1, n2, bufA, rootNumprocs, SCATTER_A_ID, NULL);
        scatter_matrix(0, SCATTER_TYPE_B, n2, n3, bufB, rootNumprocs, SCATTER_B_ID, NULL);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed_time = MPI_Wtime();
    // 每个程序计算他自己的那一块
    cannon(bufA, bufB, bufC, tmpBufA, tmpBufB, tmpBufC, n1, n2, n3);

    // 接收并综合数据
    MPI_Status status;
    if (myrank == 0) {
        int i, j, k, l;
        for (i = 0; i < rootNumprocs; i++) {
            for (j = 0; j < rootNumprocs; j++) {
                // 处理(i, j)块
                if (i != 0 || j != 0){
                    // 读取第(i, j)进程发送的数据
                    MPI_Recv(tmpBufC, maxrows_a * maxcols_b, MPI_DOUBLE, i * rootNumprocs + j, REDUCE_C_ID, MPI_COMM_WORLD, &status);
                } else {
                    // 0, 0也就是主进程自己，就把数据直接复制过去
                    for (k = 0; k < maxrows_a * maxcols_b; k++) {
                        tmpBufC[k] = bufC[k];
                    }
                }
                // (i, j)块现在已经在tmpBufC里了
                // 将其放在C中合适位置处
                for (k = 0; k < maxrows_a; k++) {
                    for (l = 0; l < maxcols_b; l++) {
                        // (i, j)的第(k, l)个元素
                        // 其实是总的(i * maxrows_a + k, j * maxcols_b + l)个
                        // 此处i*maxrow_a是因为竖着第i块前有i个maxrow_a行的块，j同理
                        if (i * maxrows_a + k < n1 && j * maxcols_b + l < n3) {
                            C[(i * maxrows_a + k) * n3 + j * maxcols_b + l] = tmpBufC[k*maxcols_b + l];
                        }
                    }
                }
            }
        }
    } else {
        // 其余进程：只需要把C发给0号进程就行
        MPI_Send(bufC, maxrows_a * maxcols_b, MPI_DOUBLE, 0, REDUCE_C_ID, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    elapsed_time = MPI_Wtime() - elapsed_time;

    if (myrank == 0) {
        writeMatFile("matC.bin", n1, n3, C);
        printf("Cannon algrithm: multiply a %dx%d with a %dx%d, use %.2lf(s)\n", n1, n2, n2, n3, elapsed_time);
        free(A);
        free(B);
        free(C);
    }
    free(bufA);
    free(bufB);
    free(bufC);
    free(tmpBufA);
    free(tmpBufB);
    free(tmpBufC);

    MPI_Finalize();

    return 0;
}

