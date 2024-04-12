from petsc4py import PETSc
import numpy as np
import emcee
import pymgmc
import matplotlib.pyplot as plt

def construct_mat_and_hierarchy(n_coarse, n_levels, diag_noise=1e-4):
    """ Setup the grid hierarchy. We provide the coarse grid and number of total levels. """
    da = PETSc.DMDA().create([n_coarse, n_coarse], stencil_width=1, comm=PETSc.COMM_WORLD)
    da_hierarchy = pymgmc.DMHierarchy(da, n_levels)

    """ Assemble the precision matrix for a simple GMRF model """
    A = da_hierarchy.getFine().createMat()
    
    def index_to_grid(r):
        """Convert a row number into a grid point."""
        return (r // n, r % n)
    
    n,_ = da_hierarchy.getFine().getSizes()    
    rstart, rend = A.getOwnershipRange()
    for row in range(rstart, rend):
        i, j = index_to_grid(row)
        k = 0
        if i > 0:
            column = row - n
            A[row, column] = -1.0
            k += 1
        if i < n - 1:
            column = row + n
            A[row, column] = -1.0
            k += 1
        if j > 0:
            column = row - 1
            A[row, column] = -1.0
            k += 1
        if j < n - 1:
            column = row + 1
            A[row, column] = -1.0
            k += 1
            
        A[row, row] = k + diag_noise
    
    A.assemblyBegin()
    A.assemblyEnd()

    return A, da_hierarchy

def compute_IACT(op, da_hierarchy, sampler_name, n_burnin, n_samples, qoi):
    # Create vectors for the target mean, the sample, and the "right-hand side" used in the samplers
    sample, mean = op.getMat().createVecs()
    mean.set(0)
    
    rhs = sample.duplicate()
    op.getMat().mult(mean, rhs)

    if sampler_name == "gibbs":
        sampler = pymgmc.GibbsSampler(op, omega=1.999,
                                      sweepType=pymgmc.GibbsSweepType.Forward)
    elif sampler_name == "mgmc":        
        sampler = pymgmc.MGMCSampler(op, da_hierarchy,
                                     coarseSampler=pymgmc.CoarseSamplerType.Cholesky,
                                     cycle=pymgmc.CycleType.V,
                                     smoothingSteps=2)
    else:
        raise Exception("Unknown sampler")

    # Perform burnin
    for i in range(n_burnin):
        sampler.sample(rhs, sample)

    # Compute actual samples
    samples = np.zeros(n_samples)
    for i in range(n_samples):
        sampler.sample(rhs, sample)
        samples[i] = qoi(sample)

    return emcee.autocorr.function_1d(samples)

n_burnin        = 1000
n_samples_gibbs = [20000, 20000, 20000, 50000]
n_samples_mgmc  = [10000] * len(n_samples_gibbs)
diag_noises     = [0.1, 0.01, 0.001, 0.0001]

n_coarse, n_levels = 3, 2

def qoi(v):
    vsize = v.getSize()
    middle = vsize // 2
    return v.getValue(middle)
    

gibbs_res = np.zeros((len(diag_noises), max(n_samples_gibbs)))
mgmc_res  = np.zeros((len(diag_noises), max(n_samples_mgmc)))

fine_sizes = []

for i, diag_noise in enumerate(diag_noises):
    A, da_hierarchy = construct_mat_and_hierarchy(n_coarse, n_levels, diag_noise)
    n,_ = da_hierarchy.getFine().getSizes()
    fine_sizes.append(n)

    op = pymgmc.LinearOperator(A)

    autocorr_gibbs = compute_IACT(op, da_hierarchy, "gibbs", n_burnin, n_samples_gibbs[i], qoi)
    autocorr_mgmc  = compute_IACT(op, da_hierarchy, "mgmc", n_burnin, n_samples_mgmc[i], qoi)

    gibbs_res[i,:n_samples_gibbs[i]] = autocorr_gibbs
    mgmc_res[i,:n_samples_mgmc[i]]  = autocorr_mgmc

linestyles = ['-', '--', '-.', ':']
assert(len(linestyles) >= len(diag_noises))

plot_range = 50
    
for i in range(len(diag_noises)):    
    plt.plot(gibbs_res[i,0:plot_range], color='C0', linestyle=linestyles[i],
             label=f"Gibbs ($\eta = {diag_noises[i]}$)")
    plt.plot(mgmc_res[i,0:plot_range], color='C1', linestyle=linestyles[i],
             label=f"MGMC ($\eta = {diag_noises[i]}$)")

plt.legend()
plt.title(f"Lagged autorcorrelation function (Fine grid {fine_sizes[0]}x{fine_sizes[0]})")
plt.xlabel("Lag")
plt.ylabel("Normalised autocorr.")
plt.savefig("autocorr.png")
