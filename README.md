# Algebraic Multigrid Solver (AmgX) Library

AmgX is a GPU accelerated core solver library that speeds up computationally intense linear solver portion of simulations. The library includes a flexible solver composition system that allows a user to easily construct complex nested solvers and preconditioners. The library is well suited for implicit unstructured methods.
The AmgX library offers optimized methods for massive parallelism, the flexibility to choose how the solvers are constructed, and is accessible through a simple C API that abstracts the parallelism and scale across a single or multiple GPUs using user provided MPI.

This is the source of the [AMGX library](https://developer.nvidia.com/amgx) on the NVIDIA Registered Developer Program portal.

Key features of the library include:
* fp32, fp64 and mixed precision solve
* Complex datatype support (currently limited)
* Scalar or coupled block systems
* Distributed solvers using provided MPI
* Flexible configuration allows for nested solvers, smoothers and preconditioners
* Classical (Ruge-Steuben) and Unsmoothed Aggregation algebraic multigrid
* Krylov methods: CG, BiCGSTAB, GMRES, etc. with optional preconditioning
* Various smoother: Jacobi, Gauss-Seidel, Incomplete LU, Chebyshev Polynomial, etc.
* A lot of exposed parameters for algorithms via solver configuration in JSON format
* Modular structure for easy implementation of your own methods
* Linux and Windows support

Check out these case studies and white papers:
  * [AmgX: Multi-Grid Accelerated Linear Solvers for Industrial Applications](http://devblogs.nvidia.com/parallelforall/amgx-multi-grid-accelerated-linear-solvers-industrial-applications/)
  * [AmgX V1.0: Enabling Reservoir Simulation with Classical AMG](http://devblogs.nvidia.com/parallelforall/amgx-v1-0-enabling-reservoir-simulation-with-classical-amg/)
  * [
AmgX: A Library for GPU Accelerated Algebraic Multigrid and Preconditioned Iterative Methods](https://research.nvidia.com/publication/amgx-library-gpu-accelerated-algebraic-multigrid-and-preconditioned-iterative-methods)

## Table of Contents

* [Quickstart](#quickstart)
  * [Building AMGX](#building)
  * [Running examples](#running)
* [Further reading](#further-reading)
  * [Plugins and bindings](#bindings)

## <a name="quickstart"></a> Quickstart

Here are the instructions on how to build library and run an example solver on the matrix in the [Matrix Market](http://math.nist.gov/MatrixMarket/) format file. By default provided examples use vector of ones as RHS of the linear system and vector of zeros as initial solution. In order to provide you own values for RHS and initial solution edit the examples.

### Dependencies and requirements

In order to build project you would need [CMake](https://cmake.org/) and [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit
). If you want to try distributed version of AMGX library you will also need MPI implementation, such as [OpenMPI](https://www.open-mpi.org/) for Linux or [MPICH](https://www.mpich.org/downloads/) for Windows. You will need compiler with c++11 support (for example GCC 4.8 or MSVC 14.0).
You also need NVIDIA GPU with Compute Capability >=3.0, check to see if your GPU supports this [here](https://developer.nvidia.com/cuda-gpus).

### <a name="cloning"></a> Cloning / Pulling

In order to pull all necessary dependencies, AmgX must be cloned using the `--recursive` option, i.e.:

`git clone --recursive git@github.com:nvidia/amgx.git`

If you want to update a copy of the repository which was cloned without --recursive, you can use:

`git submodule update --init --recursive`

### <a name="building"></a> Building
Typical build commands from the project root:

```bash
mkdir build
cd build
cmake ../
make -j16 all
```
Therer are few custom CMake flags that you could use:
- CUDA_ARCH: List of virtual architectures values that in the CMakeLists file is translated to the corresponding nvcc flags. For example:
```bash
cmake ....  -DCUDA_ARCH="35 52 60" ....
```
- CMAKE_NO_MPI: Boolean value. If True then non-MPI (single GPU) build will be forced. Results in smaller sized library which could be run on systems without MPI installed. If not specified then MPI build would be enabled if FindMPI script found any MPI installation.
- AMGX_NO_RPATH: Boolean value. By default CMake adds -rpath flags to binaries. Setting this flag to True tell CMake to not do that - useful for controlling execution environment.
- MKL_ROOT_DIR and MAGMA_ROOT_DIR: string values. MAGMA/MKL functionality is used to accelerate some of the AMGX eigensolvers. Those solvers will return error 'not supported' if AMGX was not build with MKL/MAGMA support.

The build system now enables CUDA as a language, and employs FindCUDAToolkit and FindMPI,
so refer to those scripts from your CMake installation for module-specific flags.

When building with the NVIDIA HPC SDK, please use CMake >= 3.22,
and GCC for C/CXX compilation, e.g.

```
cmake \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUDA_ARCH="80" ..
```

Artifacts of the build are shared and static libraries (libamgxsh.so or amgxsh.dll
and libamgx.a or amgx.lib) and few binaries from 'examples' directory that give you
examples of using various AMGX C API. MPI examples are built only if MPI build was
enabled.

### <a name="running"></a> Running examples

Sample input matrix [matrix.mtx](examples/matrix.mtx) is in the examples directory. Sample AMGX solvers configurations are located in the [src/configs](src/configs) directory in the root folder. Make sure that examples are able to find AMGX shared library - by default _-rpath_ flag is used for binaries, but you might specify path manually in the environment variable: _LD_LIBRARY_PATH_ for Linux and _PATH_ for Windows.

#### Running single GPU example from the build directory:

```bash
> examples/amgx_capi -m ../examples/matrix.mtx -c ../src/configs/FGMRES_AGGREGATION.json
AMGX version 2.0.0-public-build125
Built on Oct  7 2017, 04:51:11
Compiled with CUDA Runtime 9.0, using CUDA driver 9.0
Warning: No mode specified, using dDDI by default.
Reading data...
RHS vector was not found. Using RHS b=[1,…,1]^T
Solution vector was not found. Setting initial solution to x=[0,…,0]^T
Finished reading
AMG Grid:
         Number of Levels: 1
            LVL         ROWS               NNZ    SPRSTY       Mem (GB)
         --------------------------------------------------------------
           0(D)           12                61     0.424       8.75e-07
         --------------------------------------------------------------
         Grid Complexity: 1
         Operator Complexity: 1
         Total Memory Usage: 8.75443e-07 GB
         --------------------------------------------------------------
           iter      Mem Usage (GB)       residual           rate
         --------------------------------------------------------------
            Ini            0.403564   3.464102e+00
              0            0.403564   1.619840e-14         0.0000
         --------------------------------------------------------------
         Total Iterations: 1
         Avg Convergence Rate:               0.0000
         Final Residual:           1.619840e-14
         Total Reduction in Residual:      4.676075e-15
         Maximum Memory Usage:                0.404 GB
         --------------------------------------------------------------
Total Time: 0.00169123
    setup: 0.00100198 s
    solve: 0.000689248 s
    solve(per iteration): 0.000689248 s
```
#### Running multi GPU example from the build directory:

```bash
> mpirun -n 2 examples/amgx_mpi_capi.exe -m ../examples/matrix.mtx -c ../src/configs/FGMRES_AGGREGATION.json
Process 0 selecting device 0
Process 1 selecting device 0
AMGX version 2.0.0-public-build125
Built on Oct  7 2017, 04:51:11
Compiled with CUDA Runtime 9.0, using CUDA driver 9.0
Warning: No mode specified, using dDDI by default.
Warning: No mode specified, using dDDI by default.
Cannot read file as JSON object, trying as AMGX config
Converting config string to current config version
Parsing configuration string: exception_handling=1 ;
Using Normal MPI (Hostbuffer) communicator...
Reading matrix dimensions in file: ../examples/matrix.mtx
Reading data...
RHS vector was not found. Using RHS b=[1,…,1]^T
Solution vector was not found. Setting initial solution to x=[0,…,0]^T
Finished reading
Using Normal MPI (Hostbuffer) communicator...
Using Normal MPI (Hostbuffer) communicator...
Using Normal MPI (Hostbuffer) communicator...
AMG Grid:
         Number of Levels: 1
            LVL         ROWS               NNZ    SPRSTY       Mem (GB)
         --------------------------------------------------------------
           0(D)           12                61     0.424        1.1e-06
         --------------------------------------------------------------
         Grid Complexity: 1
         Operator Complexity: 1
         Total Memory Usage: 1.09896e-06 GB
         --------------------------------------------------------------
           iter      Mem Usage (GB)       residual           rate
         --------------------------------------------------------------
            Ini             0.79834   3.464102e+00
              0             0.79834   3.166381e+00         0.9141
              1              0.7983   3.046277e+00         0.9621
              2              0.7983   2.804132e+00         0.9205
              3              0.7983   2.596292e+00         0.9259
              4              0.7983   2.593806e+00         0.9990
              5              0.7983   3.124839e-01         0.1205
              6              0.7983   5.373423e-02         0.1720
              7              0.7983   9.795357e-04         0.0182
              8              0.7983   1.651436e-13         0.0000
         --------------------------------------------------------------
         Total Iterations: 9
         Avg Convergence Rate:               0.0331
         Final Residual:           1.651436e-13
         Total Reduction in Residual:      4.767284e-14
         Maximum Memory Usage:                0.798 GB
         --------------------------------------------------------------
Total Time: 0.0170917
    setup: 0.00145344 s
    solve: 0.0156382 s
    solve(per iteration): 0.00173758 s
```

### <a name="testing"></a> Testing the library

AmgX is automatically tested using the infrastructure in the `ci/` directory,
see the [`README.md`](./ci/README.md) for more information.

## <a name="further-reading"></a> Further reading

### <a name="bindings"></a> Plugins and bindings to other software
User @shwina built python bindings to AMGX, check out following repository: https://github.com/shwina/pyamgx.

User @piyueh provided link to their work on PETSc wrapper plugins for AMGX: https://github.com/barbagroup/AmgXWrapper.

Julia bindings to AMGX are available at: https://github.com/JuliaGPU/AMGX.jl.

See [API reference doc](doc/AMGX_Reference.pdf) for detailed description of the interface. In the next few weeks we will be providing more information and details on the project such as:
  * Plans on the project development and priorities
  * Issues
  * Information on contributing
  * Information on solver configurations
  * Information on the code and algorithms
