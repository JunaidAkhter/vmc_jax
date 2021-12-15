import sys
# Find jVMC package
#sys.path.append("/Users/akhter/githesis-/jvmc/vmc_jax")
sys.path.append("/Users/akhter/thesis/vmc_jax")


import jax
from jax.config import config
config.update("jax_enable_x64", True)

import jax.random as random
import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_flatten, tree_unflatten
import jVMC

import tensornetwork as tn
tn.set_default_backend("jax")

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

L = 16 # system size
g = -0.7 # strength of external field

# Set up hamiltonian for open boundary conditions
hamiltonian = jVMC.operator.BranchFreeOperator()
for l in range(L - 1):
    hamiltonian.add(jVMC.operator.scal_opstr(-1., (jVMC.operator.Sz(l), jVMC.operator.Sz(l + 1))))
    hamiltonian.add(jVMC.operator.scal_opstr(g, (jVMC.operator.Sx(l), )))
hamiltonian.add(jVMC.operator.scal_opstr(g, (jVMC.operator.Sx(L - 1), )))

def svd(dp,shape, rank=L):

    """Takes in the concatenated matrix and spits out the copressed one"""
        
    #getting the real and the complex parts of the matrix
    real_matrix = jnp.reshape(dp[:L*h], (L,h)) 
    complex_matrix = jnp.reshape(dp[L*h:], (L,h))
    print("real_matrix", real_matrix, "complex_matrix:", complex_matrix)
    #creating the W matrix from the real and the complex parts 
    matrix = jax.lax.complex(real_matrix, complex_matrix)
    print("matrix:", matrix)
    #Now that we have the matrix we can svd it and reject some of the singular values. 
    tensor1 = jnp.reshape(matrix, shape)
    print("tensor1_shape and atype:", tensor1.shape, type(tensor1))
    #reshaping the matrix in a tensor of given shape e.g. a four legged tensor
    node = tn.Node(tensor1)
    #now we perform the svd of the node keeping the left two and the right two legs as they are 
    u, vh, _ = tn.split_node(node, left_edges=[node[0], node[1]], right_edges=[node[2],node[3]], max_singular_values=r)
    print("shape of u:", u.shape, "shape of vh:", vh.shape)
    node_contracted = (u @ vh).tensor
    matrix_returned = jnp.reshape(node_contracted, (matrix.shape))
    print("shape of matrix_returned:", matrix_returned.shape)
    return matrix_returned
        

def simulate(rng, iterations, rank, t_step):
    net = net_init
    psi = jVMC.vqs.NQS(net, seed=rng)  # Variational wave function


    # Set up sampler
    #tic = time.perf_counter()
    sampler = jVMC.sampler.MCSampler(psi, (L,), random.PRNGKey(4321), updateProposer=jVMC.sampler.propose_spin_flip_Z2,
                                 numChains=100, sweepSteps=L,
                                 numSamples=30000, thermalizationSweeps=25)
    #toc = time.perf_counter()
    
    #print("   == Total time for sampling step: %fs\n" % (toc - tic))

    # Set up TDVP
    tdvpEquation = jVMC.util.tdvp.TDVP(sampler, rhsPrefactor=1.,
                                   svdTol=1e-8, diagonalShift=10, makeReal='real')

    stepper = jVMC.util.stepper.Euler(timeStep=t_step)  # ODE integrator


    res = []
    
    for n in range(iterations):
        dp, _ = stepper.step(0, tdvpEquation, psi.get_parameters(), hamiltonian=hamiltonian, psi=psi, numSamples=None)
        print("dp_inserted", dp)
        dp = svd(dp, (4,4,2,2), rank = r)
        
        dp = jnp.concatenate([p.ravel() for p in tree_flatten(dp)[0]])
        dp = jnp.concatenate([dp.real, dp.imag])
        print("dp_returned", dp)
        psi.set_parameters(dp)

        print(n, jax.numpy.real(tdvpEquation.ElocMean0) / L, tdvpEquation.ElocVar0 / L)

        res.append([jax.numpy.real(tdvpEquation.ElocMean0) / L])
        np.savetxt('dp', dp) 
    return np.array(res)


#iterations = 2500
#rng_list = [0, 1, 2, 3, 4, 5,  6, 7, 8, 9, 10]

iterations = 2
rng_list = [0, 1]
time_step = 12e-2 
h = L
net_init = jVMC.nets.CpxRBM(numHidden = h, bias = False)

#rank_list = jnp.arange(L/2, L+1)
rank_list = [8,9]
results = []
for j,rng in enumerate(rng_list):
    
    E_0_aarray = np.zeros((iterations, len(rng_list)))#an empty two dimensional array corresponding to the D and "rng".

    for r in rank_list:
        
        #print("rng:", rng)
        res = simulate(rng, iterations, rank=r, t_step = time_step)
        E_0 = res + 1.0660513358196495#this energy is for 16 spins
        #adding the energy values obtained to the first entry of the row
        #print("length", len(E_0))
        E_0_aarray[:, j] = E_0[:, 0]
        #print("final_energy:", E_0[-1])
    
    results.apend(E_0_aarray)

#print("E_array", E_0_aarray)

np.savetxt('cpxrbm_16_h16_sr_12t', np.array(results), header='Data for CpxRBM with h = 16 for 1 initializations')
