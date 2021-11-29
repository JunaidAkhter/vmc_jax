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

L = 12 # system size
g = -0.7 # strength of external field

# Set up hamiltonian for open boundary conditions

tic_whole = time.perf_counter()

tic = time.perf_counter()
hamiltonian = jVMC.operator.BranchFreeOperator()
for l in range(L - 1):
    hamiltonian.add(jVMC.operator.scal_opstr(-1., (jVMC.operator.Sz(l), jVMC.operator.Sz(l + 1))))
    hamiltonian.add(jVMC.operator.scal_opstr(g, (jVMC.operator.Sx(l), )))
hamiltonian.add(jVMC.operator.scal_opstr(g, (jVMC.operator.Sx(L - 1), )))
toc = time.perf_counter()

print("   == Total time for hamiltonian step: %fs\n" % (toc - tic))

def simulate(rng, iterations, h):
    # Initialize net
    net = net_init
    psi = jVMC.vqs.NQS(net, seed=rng)  # Variational wave function

    # Set up sampler
    
    tic = time.perf_counter()
    sampler = jVMC.sampler.MCSampler(psi, (L,), random.PRNGKey(4321), updateProposer=jVMC.sampler.propose_spin_flip_Z2,
                                 numChains=100, sweepSteps=L,
                                 numSamples=30000, thermalizationSweeps=25)
    toc = time.perf_counter()

    print("   == Total time for sampler step: %fs\n" % (toc - tic))

    # Set up TDVP

    tdvpEquation = jVMC.util.tdvp.TDVP(sampler, rhsPrefactor=1.,
                                   svdTol=1e-8, diagonalShift=10, makeReal='real')


    stepper = jVMC.util.stepper.Euler(timeStep=1e-2)  # ODE integrator



    res = []
    for n in range(iterations):
        tic = time.perf_counter()
        dp, _ = stepper.step(0, tdvpEquation, psi.get_parameters(), hamiltonian=hamiltonian, psi=psi, numSamples=None)
        psi.set_parameters(dp)

        #print(n, jax.numpy.real(tdvpEquation.ElocMean0) / L, tdvpEquation.ElocVar0 / L)
        toc = time.perf_counter()
        
        print("   == Total time for tdvp step: %fs\n" % (toc - tic))
        
        res.append([jax.numpy.real(tdvpEquation.ElocMean0) / L])

    return np.array(res)


#iterations = 2500
#rng_list = [0, 1, 2, 3, 4, 5,  6, 7, 8, 9, 10]

iterations = 1
rng_list = [0]

E_0_aarray = np.zeros((iterations, len(rng_list)))#an empty two dimensional array corresponding to the D and "rng".
h = 12
net_init = jVMC.nets.CpxRBM(numHidden = h, bias = False)
#Checking the dhape of the mpo and the values of the initialized parameters
params = net_init.init(jax.random.PRNGKey(1),jnp.zeros((L,), dtype=global_defs.tCpx)) # the "dtype" here is not so important
print("Shape of the model", jax.tree_map(np.shape, params))
#print("parameters:", params)

for j,rng in enumerate(rng_list):
    print("rng:", rng)
    res = simulate(rng, iterations, h=h)
    E_0 = res + 1.0660513358196495#this energy is for 12 spins
    #adding the energy values obtained to the first entry of the row
    #print("length", len(E_0))
    E_0_aarray[:, j] = E_0[:, 0]
    #print("final_energy:", E_0[-1])

tok_whole = time.perf_counter()

print("   == Total time for whole script: %fs\n" % (tok_whole - tic_whole))

#print("E_array", E_0_aarray)

#np.savetxt('cpxrbm_12_avg_h36_gs', E_0_aarray, header='Data for rlmpo with h = 36 for 11 different initializations')