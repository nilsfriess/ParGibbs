# Examples

This directory contains some examples (that also function as the test suite for the ParMGMC library, see below). Currently, there are the following examples:

| Example | Description                                                                                                                                              |
|---------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| ex1.c   | Sample from Gaussian random fields with Matern covariance using PETSc's KSP interface and custom "preconditioners" (=samplers). Works with stand-alone Gibbs samplers and MGMC. |
| ex2.c   | Sample from Gaussian random fields with Matern covariance using the MS (Matern Sampler) interface. Currently does not test anything, but only checks if everything compiles and runs without error. |

# Test suite

The examples in this directory also function as the test suite. It uses the [LLVM Integration Tester](https://llvm.org/docs/CommandGuide/lit.html) (`lit`) to detect and run the tests and simple `PetscCheck`s to test actual assertions. The samplers are always tested as a whole; there are no unit tests. To build the examples/ tests, first install the library to some directory as described in the [README](/README.md). The `lit` executable can be installed via `pip`/`pipx` by running

```bash
$ pipx install lit
```

Next, configure the everything by running

```bash
$ cd examples
$ mkdir build && cd build
$ cmake .. -DCMAKE_PREFIX_PATH="/path/to/petsc;/path/to/parmgmc"
```

Finally, execute the examples/ test suite by running

```bash
$ make check
```
This runs all examples both sequentially and in parallel. They can also be executed separately using `make check-seq` and `make check-par`, respectively.

### How this works
Each file contains comments of the form
```c
// RUN: %cc %s -o %t %flags && %mpirun -np %NP %t ...
```
at the top. The `lit` executable automatically parses all files in the `examples` directory and substitutes the variables in the comments to generate commands that are then executed.
