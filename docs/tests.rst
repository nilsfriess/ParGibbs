Running the test suite
======================

The test suite consist of the files in the directory ``examples``. It uses the `LLVM Integration Tester <https://llvm.org/docs/CommandGuide/lit.html>`_ (``lit``) to detect and run the tests. To build the examples/ tests, first install the library to some directory as described in the :ref:`installation` section. The `lit` executable can be installed via `pip`/`pipx` by running

.. code-block:: console

  > pipx install lit

Next, configure the everything by running

.. code-block:: console

  > cd examples
  > mkdir build && cd build
  > cmake .. -DCMAKE_PREFIX_PATH="/path/to/petsc;/path/to/parmgmc"

Finally, execute the examples/ test suite by running

.. code-block:: console

  > make check

This runs all tests both sequentially and in parallel. They can also be executed separately using ``make check-seq`` and ``make check-par``, respectively.


How this works
-----------------------
Each file contains comments of the form
.. code-block:: c

  // RUN: %cc %s -o %t %flags && %mpirun -np %NP %t ...

at the top. The ``lit`` executable automatically parses all files in the ``examples`` directory and substitutes the variables in the comments to generate commands that are then executed.

