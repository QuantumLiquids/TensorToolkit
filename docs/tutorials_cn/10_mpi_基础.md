# 张量 MPI 操作基础

## 第0课：MPI前置知识 (可跳过)

> **已熟悉MPI？** 直接跳转到 [核心 MPI API](#核心-mpi-api)

### 什么是MPI

MPI (Message Passing Interface) 是分布式并行计算的标准接口。

### 基本概念

- **进程 (Process)**：独立的计算单元，有自己的内存空间
- **Rank**：进程的唯一标识符，从0开始
- **通信器 (Communicator)**：进程组，默认使用 `MPI_COMM_WORLD`，亦可定制化优化通信。
- **消息传递**：进程间通过显式发送/接收数据通信

### 典型工作流程

```
1. 所有进程并行启动
2. 每个进程执行相同代码
3. 根据rank分配不同任务  
4. 进程间通信交换数据协同工作
5. 同步结束
```

### 更多学习资源

- [MPI Tutorial](https://mpitutorial.com/)
- [Introduction to MPI Programming](https://computing.llnl.gov/tutorials/mpi/)
- [MPI: A Message-Passing Interface Standard](https://www.mpi-forum.org/docs/)

---

## 本库提供的 TOP LEVEL MPI API

TensorToolkit 为分布式张量计算提供三个核心 MPI 操作：

### 1. 常量定义

```cpp
namespace qlten::hp_numeric {
    const size_t kMPIMasterRank = 0;        // 主进程等级
    const size_t kMPIMaxChunkSize = ((size_t)1 << 31) - 1; // 每块最大字节数 = INT_MAX
}
```

**分块传输？** 为不同 MPI 实现与硬件平台的消息大小限制提供保护。（更深入的说明见开发者文档）

### 2. 张量通信

#### 发送/接收模式
```cpp
// 发送方进程
QLTensor<QLTEN_Double, U1QN> tensor = /* 你的张量 */;
tensor.MPI_Send(dest_rank, tag, MPI_COMM_WORLD);

// 接收方进程  
QLTensor<QLTEN_Double, U1QN> recv_tensor;
recv_tensor.MPI_Recv(source_rank, tag, MPI_COMM_WORLD);
```

#### 广播模式
```cpp
QLTensor<QLTEN_Double, U1QN> tensor;

if (rank == static_cast<int>(qlten::hp_numeric::kMPIMasterRank)) {
    tensor = /* 初始化张量 */;
}

// 所有进程都执行这一行
tensor.MPI_Bcast(qlten::hp_numeric::kMPIMasterRank, MPI_COMM_WORLD);
// 现在所有进程都有了这个张量
```


## 3. 分布式 SVD

使用 Master-Slave 模式：

```cpp
#include "qlten/qlten.h"
#include <mpi.h>
using namespace qlten;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;
    int rank;
    MPI_Comm_rank(comm, &rank);

    if (rank == static_cast<int>(qlten::hp_numeric::kMPIMasterRank)) {  // Master process
        // 创建并初始化张量
        QLTensor<QLTEN_Double, U1QN> tensor;
        // tensor = /* 初始化 */;

        // SVD 参数
        QLTensor<QLTEN_Double, U1QN> u, vt;
        QLTensor<QLTEN_Double, U1QN> s;
        double actual_trunc_err;
        size_t D;

        // 执行分布式 SVD (Master)
        MPISVDMaster(&tensor,
                     2,                    // svd_ldims
                     U1QN(0),             // left_div 
                     1e-10,               // trunc_err
                     1,                   // Dmin
                     100,                 // Dmax  
                     &u, &s, &vt,         // output tensors
                     &actual_trunc_err, &D,
                     comm);

    } else {  // Slave processes
        // 执行分布式 SVD (Slave)。 
        // Slave 进程不需要有任何的数据输入输出，但是会协助完成计算。
        MPISVDSlave<QLTEN_Double>(comm);
    }

    MPI_Finalize();
    return 0;
}
```


## 错误处理

所有 MPI 操作通过 `HANDLE_MPI_ERROR` 宏自动处理错误。出错时程序会终止并提供诊断信息。

## 常见问题
为什么没有 Contraction 的 MPI 版本？
通信时间可能会超过计算时间。请参考进阶版本在更全局情况下实现。