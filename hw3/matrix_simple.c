#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>
#include <pthread.h>
#include <mpi.h>

#define M (50)
#define N (60)
#define P (70)

float A[M][N], B[N][P], C[M][P];

// 用来给进程传递参数的结构体
struct threadArg {
    // 进程id
    int tid;
    // B矩阵指针
    float *B;
    // 要相乘的A的一行
    float *A_row;
    // 存放结果的C的一行
    float *C_row;
    // 进程个数
    int numthreads;
};

// 实际进行计算的线程
void* worker(void* arg) {
    int i, j;
    struct threadArg* myarg = (struct threadArg*)arg;

    // 分配方式为循环
    // A_row*B第1列-进程1 A_row*B第2列-进程2 ... A_row*B第n列-进程n A_row*B第n+1列-进程1 A_row*B第n+2列-进程2

    for (i = myarg->tid; i < P; i += myarg->numthreads) {
        // malloc出来的空间数据是不确定的，因此需要先清空
        myarg->C_row[i] = 0.0;
        for (j = 0; j < N; j++) {
            // B的第k列的第i个元素乘以A对应的第k个元素，存在C这一行的[i]里
            myarg->C_row[i] += myarg->A_row[j] * B[j][i];
        }
    }

    return NULL;
}

int readMatFile(char *filename, float *matrix) {
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
    fread(fstream, sizeof(char), fsize, fh);
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

    printf("       ---- %s: %d*%d Matrix -----\n", filename, n1, n2);
    int i, j;
    for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
            matrix[i * n2 + j] = *(A+i*n2+j);
        }
    }

    free(fstream);
    fclose(fh);
    return 0;
}

// 将矩阵打印到屏幕上并存储矩阵文件
// filename：要存储的文件名称
// m：矩阵的行数
// n：矩阵的列数
// matrix：矩阵数组
int writeMatFile(char *filename, int m, int n, float *matrix) {
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
            printf("%.4f  ", matrix[i * n + j]);
            *(ptr + i * n + j) = matrix[i * n + j];
        }
        printf("\n");
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

    return 0;
}

int main(int argc, char *argv[]) {
    int myid, numprocs;
    // 存储状态的结构体，通过他获取tag id
    MPI_Status status;
    // 初始化MPI环境
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    if(!myid) {
        // 是0号进程，负责分发调度，读取矩阵文件A和B
        readMatFile("matA.bin", A[0]);
        readMatFile("matB.bin", B[0]);
    }

    // MPI Bcast广播发送消息
    // int MPI_Bcast( void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm )
    // 将root的buffer广播到comm域
    // 每个进程都有收和发的操作，因此需要将bcast代码放在所有程序都能运行到的地方
    MPI_Bcast(B[0], N*P, MPI_FLOAT, 0, MPI_COMM_WORLD);

    int i, j;
    if (!myid) {
        // 主进程，确定第一轮发送的进程数，取计算MPI进程数目和矩阵行数的较小者
        j = (numprocs - 1) < M ? (numprocs - 1) : M;
        for (i = 1; i < numprocs; i++) {
            if (i <= j) {
                // 向第i个进程发送i-1行运算
                MPI_Send(A[i-1], N, MPI_FLOAT, i, 99, MPI_COMM_WORLD);
            } else { // !这里本来有一个会导致死锁的错误，因为进程数大于A矩阵行数时
                     // !其他进程会一直尝试等待收消息
                     // !现在直接向这些进程发送结束消息
                MPI_Send(&j, 0, MPI_INT, i, 0, MPI_COMM_WORLD);
            }
        }
        // 记录已经发送的矩阵行数
        int numsend = j;

        for (i = 1; i <= M; i++) {
            // 循环接受每一行
            int sender = (i - 1) % (numprocs - 1) + 1;
            MPI_Recv(C[i - 1], P, MPI_FLOAT, sender, 100, MPI_COMM_WORLD, &status);
            // 分配给他的一行算完了
            // 如果还有剩下的行就分给sender继续运算
            // 第一行1，第二行2，... ，第n行n
            // 第1行算完时，第n+1行分给他
            if (numsend < M) {
                // !这里有一个会导致计算错误的问题
                // !应该把第n+1行分给sender而不是第i行
                MPI_Send(A[numsend], N, MPI_FLOAT, sender, 99, MPI_COMM_WORLD);
                numsend++;
            } else {
                // 发送结束信号
                MPI_Send(&j, 0, MPI_INT, sender, 0, MPI_COMM_WORLD);
            }
        }

        // 这里已经收到了矩阵的每一行
        writeMatFile("matC.bin", M, P, C[0]);
    } else {
        // 设置进程数量，设置为CPU核数以最大化速度
        int numthreads = get_nprocs();
        // 给每个进程分配存储进程号的空间
        pthread_t* tids = (pthread_t*)malloc(numthreads * sizeof(pthread_t));
        // 要处理的A的一行
        float* A_row = (float*)malloc(N * sizeof(float));
        // C的一行，用来存储结果
        float* C_row = (float*)malloc(P * sizeof(float));
        // 传给进程的参数
        struct threadArg *targs = (struct threadArg *)malloc(numthreads * sizeof(struct threadArg));
        for (i = 0; i < numthreads; i++) {
            // 初始化进程ID，后续依据进程ID和总进程个数划分任务
            targs[i].tid = i;
            targs[i].B = B[0];
            targs[i].A_row = A_row;
            targs[i].C_row = C_row;
            // 总进程个数，根据进程id和总进程个数划分任务
            targs[i].numthreads = numthreads;
        }

        while(1) {
            // 只要没收到结束信号就继续运行
            // MPI_ANY_TAG接受任意tag的信号
            MPI_Recv(A_row, N, MPI_FLOAT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            // tag为0代表结束信号
            if (status.MPI_TAG == 0) {
                // 结束处理
                break;
            }

            for (i = 0; i < numthreads; i++) {
                // 创建进程，并且传入参数
                pthread_create(&tids[i], NULL, worker, &targs[i]);
            }

            for (i = 0; i < numthreads; i++) {
                // 等待全部进程结束
                pthread_join(tids[i], NULL);
            }

            // 将矩阵数据发回给主进程
            MPI_Send(C_row, P, MPI_FLOAT, 0, 100, MPI_COMM_WORLD);
        }
    }
    // 结束MPI环境
    MPI_Finalize();

    return 0;
}