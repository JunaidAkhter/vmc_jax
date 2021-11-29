import sys
# Find jVMC package
sys.path.append("/Users/akhter/githesis-/jvmc/vmc_jax")

import jax
from jax.config import config
config.update("jax_enable_x64", True)

import jax.random as random
import jax.numpy as jnp
import numpy as np

import jVMC



import  functools
from typing import Any, Callable, Sequence, Optional
import flax
from flax import linen as nn
from flax import optim
from jax import lax
from functools import partial

import jVMC.nets.initializers as init
import jVMC.global_defs as global_defs

import time





# DMRG energies produced with the TeNPy library https://github.com/tenpy/tenpy
#DMRG_energies = {"10": -1.0545844370449059, "20": -1.0900383739, "100": -1.1194665474274852}

L = 128 # system size
g = -0.7 # strength of external field

# Set up hamiltonian for open boundary conditions

tik = time.perf_counter()
hamiltonian = jVMC.operator.BranchFreeOperator()
for l in range(L - 1):
    hamiltonian.add(jVMC.operator.scal_opstr(-1., (jVMC.operator.Sz(l), jVMC.operator.Sz(l + 1))))
    hamiltonian.add(jVMC.operator.scal_opstr(g, (jVMC.operator.Sx(l), )))
hamiltonian.add(jVMC.operator.scal_opstr(g, (jVMC.operator.Sx(L - 1), )))
tok = time.perf_counter()
print("   == Total time for setting up hamiltonian step: %fs\n" % (tok - tik))

iterations = 1
rng = 0
time_step = 1e-2

h = L
net_init = jVMC.nets.CpxRBM(numHidden = h, bias = False)
params = net_init.init(jax.random.PRNGKey(1),jnp.zeros((L,), dtype=global_defs.tCpx)) # the "dtype" here is not so important
print("Shape of the model", jax.tree_map(np.shape, params))
print("total number of parameters:", L * L)


psi = jVMC.vqs.NQS(net_init, seed=rng)  # Variational wave function
#Checking the dhape of the mpo and the values of the initialized parameters

# Set up sampler. This uses the ncon function is two times in the follwing sampler
tic = time.perf_counter()
sampler = jVMC.sampler.MCSampler(psi, (L,), random.PRNGKey(4321), updateProposer=jVMC.sampler.propose_spin_flip_Z2,
                                 numChains=100, sweepSteps=L,
                                 numSamples=30000, thermalizationSweeps=25)
toc = time.perf_counter()
print("   == Total time for sampling step: %fs\n" % (toc - tic))


# Set up TDVP
tic = time.perf_counter()
tdvpEquation = jVMC.util.tdvp.TDVP(sampler, rhsPrefactor=1.,
                                   svdTol=1e-8, diagonalShift=10, makeReal='real')
toc = time.perf_counter()
#print("   == Total time for setting up TDVP step: %fs\n" % (toc - tic))


tik = time.perf_counter()    
stepper = jVMC.util.stepper.Euler(timeStep=time_step)  # ODE integrator
tok = time.perf_counter()
#print("   == Total time for setting up stepper step: %fs\n" % (toc - tic))


res = []
for n in range(iterations):
    tic = time.perf_counter()
    dp, _ = stepper.step(0, tdvpEquation, psi.get_parameters(), hamiltonian=hamiltonian, psi=psi, numSamples=None)
    toc = time.perf_counter()
    print("   == Total time for stepper.step step: %fs\n" % (toc - tic))
    
    tic = time.perf_counter()
    psi.set_parameters(dp)
    toc = time.perf_counter()
 #   print("   == Total time for set_parameters iteration step: %fs\n" % (toc - tic))
    #print(n, jax.numpy.real(tdvpEquation.ElocMean0) / L, tdvpEquation.ElocVar0 / L)

    res.append([jax.numpy.real(tdvpEquation.ElocMean0) /L])



