# Python bindings

The library has some (very experimental) support for usage from Python with bindings to the C++ library generated using `pybind11`. Building the Python bindings (which will generate a Python module `pymgmc`) can be enabled by passing `-DPARMGMC_BUILD_PYTHON_BINDINGS=True` during CMake configuration.

## Setup and requirements
We use `petsc4py` to access PETSc types from Python. Therefore, `petsc4py` has to be available (already during CMake configuration). When `PARMGMC_BUILD_PYTHON_BINDINGS` is set to true during CMake configuration, a make target `pymgmc` is added that builds the Python module in form of a shared library with a name similar to `pymgmc.cpython-311-x86_64-linux-gnu.so` that can be imported as 
```python
>>> import pymgmc
```

## A basic example
Below is a simple Python script that demonstrates the usage.

```python
from petsc4py import PETSc
import pymgmc

import numpy as np
import matplotlib.pyplot as plt

# Coarsest grid in the hierarchy is 65 x 65, we refine two times
n_coarse = 65
n_levels = 3

da = PETSc.DMDA().create([n_coarse, n_coarse], stencil_width=1, comm=PETSc.COMM_WORLD)
da_hierarchy = pymgmc.DMHierarchy(da, n_levels)

A = da_hierarchy.getFine().createMat()

# Assemble A using some external function
assemble(A)

# Wrap the matrix in linear operator
op = pymgmc.LinearOperator(A)

# Create Vecs that hold the sample and the target mean
sample, mean = A.createVecs()
mean.set(3)

# Create the "right-hand side" vector used in the sampler
rhs = sample.duplicate()
A.mult(mean, rhs)

# Create sampler and draw some burn-in samples
sampler_mgmc = pymgmc.MGMCSampler(op, da_hierarchy)

n_burnin = 100
for i in range(n_burnin):
    sampler_mgmc.sample(rhs, sample)
	
# Draw another sample and visualise it using matplotlib
sampler_mgmc.sample(b, x)

n, _ = da_hierarchy.getFine().getSizes()
plt.imshow(np.reshape(sample.getArray(), (n,n)))
plt.colorbar()
plt.show()
```
