#include <stdio.h>
#include "mpi.h"

// 循环次数，确保计数准确
#define ROUND 5000

int main(int argv, char *argc[]) {
    int size, myId;
    // 初始化MPI
    MPI_Init(&argv, &argc);
    // 获取总结点数
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // 获取本节点id
    MPI_Comm_rank(MPI_COMM_WORLD, &myId);
    MPI_Status status;

    // 要传输的数据，每传一次加一
    int msg = 1005;

    int i = 0;
    // 同步各线程，确保第一次计时准确
    MPI_Barrier(MPI_COMM_WORLD);

    double globalStartTime = MPI_Wtime();
    for (i = 0; i < ROUND; i++) {
        if (myId == 0) {
            // 记录开始时间
            double startTime = 0;
            if (i == 0) {
                startTime = MPI_Wtime();
                printf("[%d]: Sending msg %d to %d\n", myId, msg, (myId + 1) % size);
            }

            // 向第一个进程发送数据（1号）
            MPI_Send(&msg, 1, MPI_INT, (myId + 1) % size, 123, MPI_COMM_WORLD);
            // 等最后一个进程向自己发送数据
            // 最后一个进程号=进程数-1
            MPI_Recv(&msg, 1, MPI_INT, size - 1, 123, MPI_COMM_WORLD, &status);

            if (i == 0) {
                printf("[%d]: Got msg %d from %d\n", myId, msg, size - 1);
                // 记录结束时间并输出
                double endTime = MPI_Wtime();
                printf("First Transmission takes %lf s for %d nodes\n", endTime - startTime, size);
            }
        } else {
            // 从前一个结点接收数据
            // 前一个结点id：myId - 1
            MPI_Recv(&msg, 1, MPI_INT, myId - 1, 123, MPI_COMM_WORLD, &status);
            
            // 输出信息
            if (i == 0) {
                printf("[%d]: Got msg %d from %d\n", myId, msg, myId - 1);
            }

            // 更新数据
            msg = msg + 1;
            
            // 输出信息
            if (i == 0) {
                printf("[%d]: Sending msg %d to %d\n", myId, msg, (myId + 1) % size);
            }

            // 向后一个进程传输数据
            // 除了最后一个进程，每个进程的后一个进程id = 当前id + 1
            // 最后一个进程的目的id是0
            // 可以统一用(myId + 1) % size表示
            MPI_Send(&msg, 1, MPI_INT, (myId + 1) % size, 123, MPI_COMM_WORLD);
        }
        // 同步各线程，方便每一轮计时
        MPI_Barrier(MPI_COMM_WORLD);
    }

    double globalEndTime = MPI_Wtime();
    if (myId == 0) {
        // 输出循环总时间和每轮次时间
        printf("Summary: Transmission takes %lf s for %d nodes and %d rounds, averages to %lf s per round.\n", globalEndTime - globalStartTime, size, ROUND, (globalEndTime - globalStartTime)/ROUND);
    }

    // 结束程序
    MPI_Finalize();
    
    return 0;
}