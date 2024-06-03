# Tests

The test suite uses the [LLVM Integration Tester](https://llvm.org/docs/CommandGuide/lit.html) (`lit`) to detect and run the tests and [doctest](https://github.com/doctest/doctest) to test actual assertions. To configure the tests, first install the library to some directory as described in the [README](/README.md). The `lit` executable can be installed via `pip`/`pipx` by running

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
make check
```
