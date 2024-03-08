# Parallel Multigrid Monte Carlo (MGMC)
ParMGMC is a C++17 implementation of the Multigrid Monte Carlo method to sample from high-dimensional Gaussian distributions on distributed memory machines.

## Installation and usage
ParMGMC is a header-only library and as such requires no installation. If you want to use the library, simply copy the folder `parmgmc` inside the `include` directory somewhere on your computer. ParMGMC has the following dependencies:
- An MPI installation (e.g., [OpenMPI](https://www.open-mpi.org/) or [MPICH](https://www.mpich.org/))
- [PETSc](https://petsc.org/)  (tested with version 3.20, anything >= 3.19 should work, some features currently require a custom build, see [below](#enabling-the-parallel-cholesky-sampler))

### Using ParMGMC in a CMake project
Inside another CMake project, you can use CMake's `FetchContent` to download and configure ParMGMC automatically. Simply add the following to your `CMakeLists.txt`
```
include(FetchContent)

FetchContent_Declare(
  ParMGMC
  GIT_REPOSITORY https://github.com/nilsfriess/ParMGMC.git
  GIT_TAG main
)
FetchContent_MakeAvailable(ParMGMC)

add_executable(main main.cc)
target_link_libraries(main PRIVATE parmgmc)
```

### Running the tests and examples
To compile and run the tests and/or the examples, you need CMake. Clone the repository and configure the build with
```bash
$ git clone https://github.com/nilsfriess/ParMGMC.git
$ cd ParMGMC
$ mkdir build && cd build
$ cmake .. -DCMAKE_PREFIX_PATH=/path/to/petsc
```
To specify a custom compiler (e.g., a MPI compiler wrapper) add `-DCMAKE_CXX_COMPILER=mpic++`. To add custom compiler flags, add, e.g., `-DCMAKE_CXX_FLAGS="-O3 -march=native"`.
If the CMake configuration finished successfully, compile the tests using 
```bash
$ make tests
```
Then execute 
```bash
$ ./tests/tests [seq]
```
to run the sequential tests and
```bash
$ mpirun -np 4 ./tests/tests [mpi]
```
to run the parallel test suite.

## Running the MFEM examples
Optionally, one can pass `-Dmfem_DIR=/path/to/mfem_install/lib/cmake/mfem` during CMake configuration to build the examples that use [MFEM](https://mfem.org/). Note that this requires a MFEM installation built with MPI, METIS, and PETSc support enabled (refer to the MFEM documentation for details). Configure the project as described above and then run 
```bash
$ make examples
```
to build the examples.

## Enabling the parallel Cholesky sampler
The MGMC sampler can use a parallel Cholesky sampler on the coarsest level using the PETSc interface of Intel's oneAPI Math Kernel Library (MKL). On Ubuntu, the required packages can be installed using the following commands (for details, see [Intel's documentation](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html)):
``` bash
$ wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
$ echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
$ sudo apt update
$ sudo apt install intel-oneapi-mkl intel-oneapi-mkl-devel

```
This also currently requires PETSc to be built from source using a custom fork. Run 
```bash
$ git clone -b cpardiso_fw_bw_solve https://gitlab.com/nilsfriess/petsc.git petsc
```
to clone PETSc and configure and build it with
```bash
$ cd petsc
$ ./configure --with-fortran-bindings=0 --with-blas-lapack-dir=/opt/intel/oneapi/mkl/latest/lib --with-mkl_cpardiso 
$ make PETSC_DIR=... PETSC_ARCH=... all
```
The values for `PETSC_DIR` and `PETSC_ARCH` are printed at the end of a successful `./configure` run. If PETSc was built with `C/Pardiso` enabled, the Cholesky sampler will automatically be enabled. Note that it might be necessary on some systems to preload an executable that uses the parallel Cholesky sampler with the library `libmkl_blacs_intelmpi_lp64.so` (or `libmkl_blacs_openmpi_lp64.so` if using OpenMPI). If you are getting errors of the form `Intel MKL FATAL ERROR: Cannot load symbol MKLMPI_Get_wrappers`, e.g., when running the tests, then try running instead
```bash
$ LD_PRELOAD=/opt/intel/oneapi/mkl/latest/lib/libmkl_blacs_intelmpi_lp64.so mpirun -np 4 ./tests/tests [mpi]
```
