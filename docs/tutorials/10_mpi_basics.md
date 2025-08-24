# MPI Basics for Tensor Operations

## Lesson 0: MPI Prerequisites (Skippable)

> **Already familiar with MPI?** Jump to [Essential MPI APIs](#essential-mpi-apis)

### What is MPI

MPI (Message Passing Interface) is the standard interface for distributed parallel computing.

### Basic Concepts

- **Process**: Independent computing unit with its own memory space
- **Rank**: Unique identifier for each process, starting from 0
- **Communicator**: Process group, default is `MPI_COMM_WORLD`, can be customized for communication optimization
- **Message Passing**: Processes communicate via explicit send/receive operations

### Typical Workflow

```
1. All processes start in parallel
2. Each process executes the same code
3. Different tasks assigned based on rank
4. Inter-process communication for data exchange
5. Synchronized termination
```

### Additional Learning Resources

- [MPI Tutorial](https://mpitutorial.com/)
- [Introduction to MPI Programming](https://computing.llnl.gov/tutorials/mpi/)
- [MPI: A Message-Passing Interface Standard](https://www.mpi-forum.org/docs/)

---

## Essential MPI APIs

TensorToolkit provides three core MPI operations for distributed tensor computing:

### 1. Constants Definition

```cpp
namespace qlten::hp_numeric {
    const size_t kMPIMasterRank = 0;        // Master process rank
    const size_t kMPIMaxChunkSize = 2^31-1; // Max bytes per chunk = INT_MAX
}
```

**Chunking?** Provides protection for MPI message transmission across different MPI implementations and hardware platforms. (This should probably be documented in developer docs)

### 2. Tensor Communication

#### Send/Receive Pattern
```cpp
// Sender process
QLTensor<QLTEN_Double, U1QN> tensor = /* your tensor */;
tensor.MPI_Send(dest_rank, tag, MPI_COMM_WORLD);

// Receiver process  
QLTensor<QLTEN_Double, U1QN> recv_tensor;
recv_tensor.MPI_Recv(source_rank, tag, MPI_COMM_WORLD);
```

#### Broadcast Pattern
```cpp
QLTensor<QLTEN_Double, U1QN> tensor;

if (rank == qlten::hp_numeric::kMPIMasterRank) {
    tensor = /* initialize tensor */;
}

// All processes execute this
tensor.MPI_Bcast(qlten::hp_numeric::kMPIMasterRank, MPI_COMM_WORLD);
// Now all processes have the tensor
```


### 3. Distributed SVD

Using Master-Slave pattern:

```cpp
#include "qlten/qlten.h" // include everything
using namespace qlten;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank;
    MPI_Comm_rank(comm, &rank);
    
    if (rank == qlten::hp_numeric::kMPIMasterRank) {  // Master process
        // Create and initialize tensor
        QLTensor<QLTEN_Double, U1QN> tensor;
        // tensor = /* your tensor initialization */;
        
        // SVD parameters
        QLTensor<QLTEN_Double, U1QN> u, vt;
        QLTensor<QLTEN_Double, U1QN> s;
        double actual_trunc_err;
        size_t D;
        
        // Perform distributed SVD (Master)
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
        // Perform distributed SVD (Slave).
        // Slave processes don't need any data input/output, but assist in computation.
        MPISVDSlave<QLTEN_Double>(comm);
    }
    
    MPI_Finalize();
    return 0;
}
```


## Error Handling

All MPI operations use automatic error handling via `HANDLE_MPI_ERROR` macro. On error, the program terminates with diagnostic information.

## FAQ
Why is there no MPI version of Contraction?
The communication time would probably exceed the computation time. Please refer to the advanced version for implementation in more global scenarios.
