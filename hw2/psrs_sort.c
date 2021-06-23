#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// 定义不同的MPI消息类型，以便区分通信过程中的信息
#define INIT_TYPE 10
#define ALLTOONE_TYPE 100
#define ONETOALL_TYPE 200
#define MULTI_TYPE 300
#define RESULT_TYPE 400
#define RESULT_LEN 500
#define MULTI_LEN 600

// 分块数
int Spt;
// 数据总数，必须是核心数的倍数才正常工作
long DataSize;
// 存放数据和结果的数组（结果可能在任意一个中）
int *arr,*arr1;
// 本地生成的数组长度
int mylength;
// 数据分割的位置
int *index;
// 临时数据
int *temp1;

main(int argc,char* argv[])
{
    long BaseNum = 1;
    int PlusNum;
    int MyID;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&MyID);

    PlusNum=60;
    // 数据总数，必须是核心数的倍数才正常工作
    DataSize = BaseNum*PlusNum;

    if (MyID==0)
        printf("The DataSize is : %lu\n",PlusNum);
    Psrs_Main();

    if (MyID==0)
        printf("\n");

    MPI_Finalize();
}


Psrs_Main( )
{
    // 各变量具体含义在第一次遇到时说明
    int i,j;
    int MyID,SumID;
    int n,c1,c2,c3,c4,k,l;
    FILE * fp;
    int ready;
    // 在非阻塞接收中用来暂存接收句柄和状态结构体
    MPI_Status status[32*32*2];
    MPI_Request request[32*32*2];

    MPI_Comm_rank(MPI_COMM_WORLD,&MyID);
    MPI_Comm_size(MPI_COMM_WORLD,&SumID);

    // 要分的段数
    Spt = SumID-1;

	/*初始化参数*/
    // arr arr1是两个长为datasize的数组
    arr = (int *)malloc(2*DataSize*sizeof(int));
    if (arr==0) merror("malloc memory for arr error!");
    arr1 = &arr[DataSize];

    if (SumID>1)
    {
        // 用于存放自身选取的p-1个代表元素，0号处理器用它来排序其他处理器的代表元素
        // p台处理器，每台p-1元素，故空间为sizeof(int)*p*(p-1)
        // 也用来存放段的长度
        temp1 = (int *)malloc(sizeof(int)*SumID*Spt);
        if (temp1==0) merror("malloc memory for temp1 error!");
        // 存放本机段的开始结束点
        index = (int *)malloc(sizeof(int)*2*SumID);
        if (index==0) merror("malloc memory for index error!");
    }
    // 同步
    MPI_Barrier( MPI_COMM_WORLD);
    
    // 自身长度就是总数据量/处理器数，因此只有可以整除时最后得到的数据量才是期望的
    mylength = DataSize / SumID;
    // 用自身ID作为随机数种子，确保每次生成的数据相同
    srand(MyID);

    printf("This is node %d \n",MyID);
    printf("On node %d the input data is:\n",MyID);
    for (i=0;i<mylength;i++)
    {
        // 生成随机数据作为待排序对象
        arr[i] = (int)rand();
        printf("%d : ",arr[i]);
    }
    printf("\n");

	/*每个处理器将自己的n/P个数据用串行快速排序(Quicksort)，得到一个排好序的序列，对应于算法13.5步骤（1）*/
    MPI_Barrier( MPI_COMM_WORLD);
    quicksort(arr,0,mylength - 1);
    MPI_Barrier( MPI_COMM_WORLD);

	/*每个处理器从排好序的序列中选取第w，2w，3w，…，(P-1)w个共P-1个数据作为代表元素，其中w=n/P*P，对应于算法13.5步骤（2）*/
    if (SumID>1)
    {
        MPI_Barrier(MPI_COMM_WORLD);

        // 本算法中每个子序列取p-1个代表元素
        // n/p*p就是为了取整，所以这里直接有w = n
        n = (int)(mylength/(Spt+1));
        for (i=0;i<Spt;i++)
            temp1[i] = arr[(i+1)*n-1];

        MPI_Barrier(MPI_COMM_WORLD);

        if (MyID==0)
        {
			/*每个处理器将选好的代表元素送到处理器P0中，对应于算法13.5步骤（3） */
            // MPI_Irecv非阻塞接受，可以一个没接收完的时候继续接受新的
            // int MPI_Irecv(
            //     void *buf,
            //     int count,
            //     MPI_Datatype datatype,
            //     int source,
            //     int tag,
            //     MPI_Comm comm,
            //     MPI_Request *request)
            // 多出来request用来保存接受结构体，在稍后等待过程中判断是否全部接受完使用
            
            j = 0;
            for (i=1;i<SumID;i++)
                // buf：从i*(p - 1)位置开始接受，不会重叠，接收后index内存布局
                // [0处理器p-1个元素, 1处理器p-1个元素, ...]
                // 按byte传输
                // ALLTOONE_TYPE+i：为了区分不同处理器传过来的数据
                MPI_Irecv(&temp1[i*Spt], sizeof(int)*Spt, MPI_CHAR, i,ALLTOONE_TYPE+i, MPI_COMM_WORLD, &request[j++]);
            // 等待所有数据传送结束
            MPI_Waitall(SumID-1, request, status);

			/* 处理器P0将上一步送来的P段有序的数据序列做P路归并，再选择排序后的第P-1，2(P-1)，…，(P-1)(P-1)个共P-1个主元，，对应于算法13.5步骤（3）*/
            // 同步1：0处理器接收完成
            MPI_Barrier(MPI_COMM_WORLD);
            // 简单方法：直接快排，排完也能达到有序的效果
            quicksort(temp1,0,SumID*Spt-1);
            // 同步2：排序完成
            MPI_Barrier( MPI_COMM_WORLD);

            // i从1到p-1
            for (i=1;i<Spt+1;i++)
                // 数组从0开始，第i+1个元素是第i*(p - 1)个
                // 可能会覆盖，但被覆盖的都已经被处理过了
                temp1[i] = temp1[i*Spt-1];
			/*处理器P0将这P-1个主元播送到所有处理器中，对应于算法13.5步骤（4）*/
            MPI_Bcast(temp1, sizeof(int)*(1+Spt), MPI_CHAR, 0, MPI_COMM_WORLD);
            // 同步3：所有都接收完成
            MPI_Barrier(MPI_COMM_WORLD);
        }
        else
        {
            // 本地选出的代表元素发送到0
            MPI_Send(temp1,sizeof(int)*Spt,MPI_CHAR,0,ALLTOONE_TYPE+MyID, MPI_COMM_WORLD);
            // 同步1：0处理器接收完成
            MPI_Barrier( MPI_COMM_WORLD);
            // 同步2：排序完成
            MPI_Barrier( MPI_COMM_WORLD);
            // 接受传回来的p-1个主元
            MPI_Bcast(temp1, sizeof(int)*(1+Spt), MPI_CHAR, 0, MPI_COMM_WORLD);
            // 同步3：所有都接收完成
            MPI_Barrier(MPI_COMM_WORLD);
        }

		/*每个处理器根据上步送来的P-1个主元把自己的n/P个数据分成P段，记为处理器Pi的第j+1段，其中i=0,…,P-1，j=0,…,P-1，对应于算法13.5步骤（5）*/
        n = mylength;
        index[0] = 0;
        i = 1;
        // 处理本地长度为0的段-本地所有元素均大于该主元
        while ((arr[0]>=temp1[i])&&(i<SumID))
        {
            // 把该主元对应的上一段结束的ID和下一段开始的ID都设为0
            index[2*i-1] = 0;
            index[2*i] = 0;
            i++;
        }
        // 如果本地所有元素均大于最大主元
        // 直接将所有元素放在最后一段
        if (i==SumID) index[2*i-1] = n;
        // 开始正常处理
        c1 = 0;
        while (i<SumID)
        {
            // c1：左端
            // c4：目标值
            c4 = temp1[i];
            // c3：右端
            c3 = n;
            // c2：二分位置
            c2 = (int)((c1+c3)/2);
            // 二分法查找下一段应该到什么地方结束
            while ((arr[c2]!=c4)&&(c1<c3))
            {
                if (arr[c2]>c4)
                {
                    // 大于目标值，右侧左移
                    c3 = c2-1;
                    // ?info：为了防止c1=1, c2=1, c3=2情况 
                    if (c3 < c1) {
                        c3 = c1;
                    }
                    c2 = (int)((c1+c3)/2);
                }
                else
                {
                    // 小于目标值，左侧右移
                    c1 = c2+1;
                    if (c3 < c1) {
                        c1 = c3;
                    }
                    c2 = (int)((c1+c3)/2);
                }
            }
            // 两种情况，一种是arr[c2]==c4，另一种是c1>=c3
            // 只要arr[c2]还小于等于c4就一直增加c2
            // arr[c1]必定小于c4，随着左侧移动c2一定会逐渐逼近直至到达c1
            while ((arr[c2]<=c4)&&(c2<n)) c2++;
            if (c2==n)
            {
                // 本段要到最后一个了
                index[2*i-1] = n;
                for (k=i;k<SumID;k++)
                {
                    // 其他段都设成0-0
                    index[2*k] = 0;
                    index[2*k+1] = 0;
                }
                i = SumID;
            }
            else
            {
                // 上一段结束是c2
                // 下一段开始也是c2
                index[2*i] = c2;
                index[2*i-1] = c2;
            }
            // 新的二分从c2开始
            c1 = c2;
            c2 = (int)((c1+c3)/2);
            i++;
        }
        if (i==SumID) index[2*i-1] = n;
        
        // 同步4：各处理器分段完成
        MPI_Barrier( MPI_COMM_WORLD);

		/*每个处理器送它的第i+1段给处理器Pi，从而使得第i个处理器含有所有处理器的第i段数据(i=0,…,P-1)，，对应于算法13.5步骤（6）*/

        j = 0;
        for (i=0;i<SumID;i++)
        {
            // 对每台机器进行处理
            if (i==MyID)
            {
                // 自身机器：对应发送数据
                // i从0开始
                // temp1[i]存的就是第p=(i+1)段的长度
                // 本机要存temp1[i]。就不用通信了
                temp1[i] = index[2*i+1]-index[2*i];
                for (n = 0; n < SumID; n++) {
                    // 向不是本机的每台机器i（对应需要i+1段）发送本机第i+1段的长度
                    if (n != MyID) {
                        k = index[2*n+1]-index[2*n];
                        MPI_Send(&k, sizeof(int), MPI_CHAR, n, MULTI_LEN+MyID,MPI_COMM_WORLD);
                    }
                }
            }
            else
            {
                // 从第i台机器获取第i台机器上的i+1段长度，存入temp1[i]
                MPI_Recv(&temp1[i], sizeof(int), MPI_CHAR, i,MULTI_LEN+i, MPI_COMM_WORLD, &status[j++]);
            }
        }
        // 同步5：长度发送接收完成
        MPI_Barrier(MPI_COMM_WORLD);

        j = 0;
        k = 0;
        l = 0;

        // 按机器同步处理。每台机器都从1-SumID进行处理
        // 处理id i时，第i台机器向其他各台机器发送自己对应他们那台机器的一段
        // 其他各台机器接收并存放到当前位置，是按顺序的
        for (i=0;i<SumID;i++)
        {
            // 同步7：
            MPI_Barrier(MPI_COMM_WORLD);

            if (i==MyID)
            {
                // 本地，只要复制过去就行
                for (n=index[2*i];n<index[2*i+1];n++)
                    arr1[k++] = arr[n];
            }

            // 同步8：
            MPI_Barrier(MPI_COMM_WORLD);

            if (i==MyID)
            {
                // 向其他每台机器发送属于他们的段
                for (n=0;n<SumID;n++) {
                    if (n!=MyID) {
                        MPI_Send(&arr[index[2*n]], sizeof(int)*(index[2*n+1]-index[2*n]),MPI_CHAR, n, MULTI_TYPE+MyID, MPI_COMM_WORLD);
                    }
                }
            }
            else
            {
                l = temp1[i];
                MPI_Recv(&arr1[k], l*sizeof(int), MPI_CHAR, i ,MULTI_TYPE+i, MPI_COMM_WORLD, &status[j++]);
                k=k+l;
            }
            // 同步9
            MPI_Barrier(MPI_COMM_WORLD);
        }
        mylength = k;
        // 同步10
        MPI_Barrier(MPI_COMM_WORLD);

		/*每个处理器再通过P路归并排序将上一步的到的数据排序；从而这n个数据便是有序的，，对应于算法13.5步骤（7） */
        k = 0;
        multimerge(arr1,temp1,arr,&k,SumID);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Status statusx;
    int msg = 0; // 保证串行输出
    if (MyID != 0) {
        MPI_Recv(&msg, 1, MPI_INT, MyID - 1, 90, MPI_COMM_WORLD, &statusx);
    }

    printf("On node %d the sorted data is : \n",MyID);
    for (i=0;i<mylength;i++) {
        // !注意这里曾经有一个bug
        // 因为使用了递归，多路归并排序的结果可能在arr1里，也可能在arr里
        // 需要根据递归层数k判断
        if (SumID > 1) {
            printf("%d : ",(k%2 == 0) ? arr1[i] : arr[i]);
        } else {
            printf("%d : ", arr[i]);
        }
    }
    printf("\n");

    // 保证串行输出
    if (MyID != SumID - 1) {
        MPI_Send(&msg, 1, MPI_INT, (MyID + 1) % SumID, 90, MPI_COMM_WORLD);
    }
}


/*输出错误信息*/
merror(char* ch)
{
    printf("%s\n",ch);
    exit(1);
}


/*串行快速排序算法*/
// 和正常的快速排序算法一样，在此不再赘述
quicksort(int *datas,int bb,int ee)
{
    int tt,i,j;
    tt = datas[bb];
    i = bb;
    j = ee;

    if (i<j)
    {
        // 分区
        while(i<j)
        {
            while ((i<j)&&(tt<=datas[j])) j--;
            if (i<j)
            {
                datas[i] = datas[j];
                i++;
                while ((i<j)&&(tt>datas[i])) i++;
                if (i<j)
                {
                    datas[j] = datas[i];
                    j--;
                    if (i==j) datas[i] = tt;
                }
                else datas[j] = tt;
            } else datas[i] = tt;
        }
        // 递归快排
        quicksort(datas,bb,i-1);
        quicksort(datas,i+1,ee);
    }
}


/*串行多路归并算法*/
// multimerge(arr1,temp1,arr,&k,SumID);
// 二分合并。先合并相邻的两个，然后继续合并完的结果
multimerge(int *data1,int *ind,int *data,int *iter,int SumID)
{
    int i,j,n;

    j = 0;
    // 除去是0的段的数目
    for (i=0;i<SumID;i++)
        if (ind[i]>0)
    {
        ind[j++] = ind[i];
        if (j<i+1) ind[i] = 0;
    }

    if ( j>1 )
    {
        n = 0;
        // [1,2,3,4,5]
        // 合并12 34 5
        // 得到[12, 34, 5]
        // 再次递归归并
        for (i=0;i<j,i+1<j;i=i+2)
        {
            merge(&(data1[n]),ind[i],ind[i+1],&(data[n]));
            ind[i] += ind[i+1];
            ind[i+1] = 0;
            n += ind[i];
        }
        if (j%2==1 )
            for (i=0;i<ind[j-1];i++) data[n++]=data1[n];
        (*iter)++;
        multimerge(data,ind,data1,iter,SumID);
    }
}

// 正常的二路归并算法，结果放在data2里，不再赘述
merge(int *data1,int s1,int s2,int *data2)
{
    int i,l,m;

    l = 0;
    m = s1;
    for (i=0;i<s1+s2;i++)
    {
        if (l==s1)
            data2[i]=data1[m++];
        else
        if (m==s2+s1)
            data2[i]=data1[l++];
        else
        if (data1[l]>data1[m])
            data2[i]=data1[m++];
        else
            data2[i]=data1[l++];
    }
}
