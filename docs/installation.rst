#############
Installation
#############

.. toctree::
   :maxdepth: 4
   :hidden:

   self
   tests
   pybind

The library depends on PETSc configured with MUMPS and Intel C/Pardiso enabled. C/Pardiso requires the `Intel Math Kernel Library (MKL) <https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html>`_. Further, an MPI installation is required. All dependencies except for the Intel MKL can be installed using PETSc's ``./configure`` script. After downloading PETSc, run

.. code-block:: console
		
   > ./configure --download-mpich      \
                 --download-mumps      \
                 --download-scalapack  \
                 --with-mkl_cpardiso   \
                 --with-mkl_pardiso    \
                 --with-blas-lapack-dir=/opt/intel/oneapi/mkl/latest/lib
   
Note that the path to the Intel MKL might differ depending on the platform.

Building the library and examples
---------------------------------
CMake is used to configure and build the library and examples. If all dependencies are installed as described above, run

.. code-block:: console

   > git clone https://github.com/nilsfriess/ParMGMC.git
   > cd ParMGMC
   > mkdir build && cd build
   > cmake .. -DCMAKE_PREFIX_PATH=/path/to/petsc
   > make

To specify a custom compiler (e.g., a MPI compiler wrapper) add ``-DCMAKE_C_COMPILER=mpicc`` and/or ``-DCMAKE_CXX_COMPILER=mpicxx`` (the library itself is written in C but can also be used from C++).

Installing the library
---------------------------------
To install the library to some specific location, pass ``-DCMAKE_INSTALL_PREFIX=/path/to/install`` during CMake configuration. Then run ``make install`` to copy the compiled library and headers to the specified directory. This also generates a pkg-config file that can be used to simplify using this library in other projects. If the environment variable ``PKG_CONFIG_PATH`` contains both the path to ``parmgmc.pc`` (located in ``/path/to/install/lib/pkgconfig``) and the path to PETSc's pkg-config file (located in ``/path/to/petsc/lib/pkgconfig``), then a program using ParMGMC can be compiled with

.. code-block:: console

   > gcc main.c -o main $(pkg-config --cflags --libs petsc parmgmc)
