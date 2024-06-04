# Tests

The test suite uses the [LLVM Integration Tester](https://llvm.org/docs/CommandGuide/lit.html) (`lit`) to detect and run the tests and simple `PetscCheck`s to test actual assertions. All tests are full examples that test the samplers as a whole; there are no unit tests. To configure the tests, first install the library to some directory as described in the [README](/README.md). The `lit` executable can be installed via `pip`/`pipx` by running

```bash
$ pipx install lit
```

Next, configure the tests by running

```bash
$ cd test
$ mkdir build && cd build
$ cmake .. -DCMAKE_PREFIX_PATH="/path/to/petsc;/path/to/parmgmc"
```

Finally, execute the test suite by running

```bash
$ make check
```
This runs both the sequential tests and the parallel tests. They can also be executed separately using `make check-seq` and `make check-par`, respectively.

### How this works
Each test file contains comments of the form
```c
// RUN: %cc %s -o %t %flags && %mpirun -np %NP %t ...
```
at the top. The `lit` executable automatically parses all files in the `tests` directory and substitutes the variables in the comments to generate commands that are then executed.
