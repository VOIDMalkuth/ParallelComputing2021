#include <stdio.h>
#include <stdlib.h>

void scatter_matrix(int isMaster, int row, int col, double *buf, int rootNumprocs, int tag, double *matBuf) {
    // 计算最大行数和最大列数
    int maxrows = (row + rootNumprocs - 1) / rootNumprocs;
    int maxcols = (col + rootNumprocs - 1) / rootNumprocs;
    if (isMaster) {
        double *tmpBuf = (double *)malloc(sizeof(double) * maxrows * maxcols);
        int i = 0, j = 0, k = 0, l = 0;
        for (i = 0; i < rootNumprocs; i++) {
            for (j = 0; j < rootNumprocs; j++) {
                // 处理分发给(i,j)=i*rootNumprocs+j号进程的数据块
                printf("-------------------------------------");
                printf("Handling block for (%d, %d)\n", i , j);
                for (int k = 0; k < maxrows; k++) {
                    for (int l = 0; l < maxcols; l++) {
                        // (i, j)的初始第(k, l)个元素
                        // 其实是总的(i * maxrows + k, j * maxcols + l)个
                        int value = 0;
                        // 自动填0
                        if (i * maxrows + k < row && j * maxcols + l < col) {
                            value = matBuf[(i * maxrows + k) * col + j * maxcols + l];
                        }
                        printf("%d ", value);
                    }
                    printf("\n");
                }
            }
        }
    } else {
        ;
    }
}

int main(void) {
    int n1 = 10;
    int n2 = 10;
    int rootNumProcs = 3;
    double *a = malloc(n1*n2*sizeof(double));
    int k = 0;
    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n2; j++) {
            printf("%d ", k);
            a[i * n2 + j] = k++;
        }
        printf("\n");
    }

    scatter_matrix(1, n1, n2, NULL, rootNumProcs, 0, a);
}