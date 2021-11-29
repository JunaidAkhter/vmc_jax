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
hamiltonian = jVMC.operator.BranchFreeOperator()
for l in range(L - 1):
    hamiltonian.add(jVMC.operator.scal_opstr(-1., (jVMC.operator.Sz(l), jVMC.operator.Sz(l + 1))))
    hamiltonian.add(jVMC.operator.scal_opstr(g, (jVMC.operator.Sx(l), )))
hamiltonian.add(jVMC.operator.scal_opstr(g, (jVMC.operator.Sx(L - 1), )))


def simulate(rng, iterations, h, t_step):
    net = net_init
    psi = jVMC.vqs.NQS(net, seed=rng)  # Variational wave function


    # Set up sampler
    sampler = jVMC.sampler.MCSampler(psi, (L,), random.PRNGKey(4321), updateProposer=jVMC.sampler.propose_spin_flip_Z2,
                                 numChains=100, sweepSteps=L,
                                 numSamples=30000, thermalizationSweeps=25)
    

    # Set up TDVP
    tdvpEquation = jVMC.util.tdvp.TDVP(sampler, rhsPrefactor=1.,
                                   svdTol=1e-8, diagonalShift=10, makeReal='real')

    stepper = jVMC.util.stepper.Euler(timeStep=t_step)  # ODE integrator


    res = []
    for n in range(iterations):
        dp, _ = stepper.step(0, tdvpEquation, psi.get_parameters(), hamiltonian=hamiltonian, psi=psi, numSamples=None)
        psi.set_parameters(dp)

        res.append([jax.numpy.real(tdvpEquation.ElocMean0) / L])

    return np.array(res)



iterations = 2500
rng_list = [0,1,2,3,4]
time_step = 30e-2
E_0_aarray = np.zeros((iterations, len(rng_list)))#an empty two dimensional array corresponding to the D and "rng".

h = 36
net_init = jVMC.nets.CpxRBM(numHidden = h, bias = False)

for j,rng in enumerate(rng_list):
    #print("rng:", rng)
    res = simulate(rng, iterations, h=h, t_step = time_step)
    E_0 = res + 1.0660513358196495#this energy is for 36 spins
    #adding the energy values obtained to the first entry of the row
    E_0_aarray[:, j] = E_0[:, 0]


np.savetxt('cpxrbm_12_avg_h36_sr_3330t', E_0_aarray, header='Data for rlmpo with h = 36 for 5 initializations')
