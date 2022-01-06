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

from jVMC.util import OutputManager

tik = time.perf_counter() 

# DMRG energies produced with the TeNPy library https://github.com/tenpy/tenpy
#DMRG_energies = {"10": -1.0545844370449059, "20": -1.0900383739, "100": -1.1194665474274852}

import pickle

L = 16 # system size

g = -0.7
h = L
outp = OutputManager("output.hdf5", append=False)
with open('parameters_rbm.pkl', 'rb') as f:
    params = pickle.load(f) 


print("len of parameters:", len(params))

def uncatenate(dp):
    """Transforms the one dimensional vector into a matrix"""
    #getting the real and the complex parts of the matrix
    real_matrix = jnp.reshape(dp[:L*h], (L,h)) 
    complex_matrix = jnp.reshape(dp[L*h:], (L,h))
    #print("real_matrix", real_matrix, "complex_matrix:", complex_matrix)
    #creating the W matrix from the real and the complex parts 
    matrix = jax.lax.complex(real_matrix, complex_matrix)
   
    return matrix

def svd(matrix,shape, rank):

    """Takes in the concatenated matrix and spits out the copressed one"""
           #Now that we have the matrix we can svd it and reject some of the singular values. 
    tensor1 = jnp.reshape(matrix, shape)
    print("tensor1_shape and atype:", tensor1.shape, type(tensor1))
    #reshaping the matrix in a tensor of given shape e.g. a four legged tensor
    node = tn.Node(tensor1)
    #now we perform the svd of the node keeping the left two and the right two legs as they are 
    print("Rank for compression:", r)
    u, vh, _ = tn.split_node(node, left_edges=[node[0], node[1]], right_edges=[node[2],node[3]], max_singular_values=r)
    print("shape of u:", u.shape, "shape of vh:", vh.shape)
    node_contracted = (u @ vh).tensor
    matrix_returned = jnp.reshape(node_contracted, (matrix.shape))
    print("shape of matrix_returned:", matrix_returned.shape)

    print("Truncation error:", jnp.linalg.norm(matrix-matrix_returned)/jnp.linalg.norm(matrix))
    return matrix_returned, jnp.linalg.norm(matrix-matrix_returned)/jnp.linalg.norm(matrix)
        
def concatenate(matrix):
    """Transforms the matrix into a vector"""
    matrix = jnp.concatenate([p.ravel() for p in tree_flatten(matrix)[0]])
    matrix = jnp.concatenate([matrix.real, matrix.imag])

    return matrix


# Set up hamiltonian for open boundary conditions
hamiltonian = jVMC.operator.BranchFreeOperator()
for l in range(L - 1):
    hamiltonian.add(jVMC.operator.scal_opstr(-1., (jVMC.operator.Sz(l), jVMC.operator.Sz(l + 1))))
    hamiltonian.add(jVMC.operator.scal_opstr(g, (jVMC.operator.Sx(l), )))
hamiltonian.add(jVMC.operator.scal_opstr(g, (jVMC.operator.Sx(L - 1), )))


r_max = 4
ranks = np.arange(1, r_max+1) 
shape = (2,2,8,8)
rng = 0

E_array = np.zeros((len(ranks), len(params)))

Error_array = np.zeros((len(ranks), len(params)))
for i, r in enumerate(ranks):
    print("rank in loop:", r)

    for j, p in enumerate(params):

        print("paramsi shape in loop:", p.shape)
        p_matrix = uncatenate(p)

        p_postsvd, error = svd(p_matrix, shape, rank = r)

        #saving the svd error
        Error_array[i, j] = error

        p_vector = concatenate(p_postsvd)

        net= jVMC.nets.CpxRBM(numHidden = h, bias = False)


        psi = jVMC.vqs.NQS(net,batchSize=10000, seed=rng)  # Variational wave function

        #psi1 = psi
        # Set up sampler
        #tic = time.perf_counter()
        sampler = jVMC.sampler.MCSampler(psi, (L,), random.PRNGKey(4321), updateProposer=jVMC.sampler.propose_spin_flip_Z2,
                                     numChains=300, sweepSteps=L,
                                     numSamples=20000, thermalizationSweeps=25)

        # Set up TDVP
        tdvpEquation = jVMC.util.tdvp.TDVP(sampler, rhsPrefactor=1.,
                                       svdTol=1e-8, diagonalShift=10, makeReal='real')

        stepper = jVMC.util.stepper.Euler(timeStep=20e-2)  # ODE integrator


        psi.set_parameters(p_vector)
        #psi1.set_parameters(params_uncat)

        dp, _ = stepper.step(0, tdvpEquation, psi.get_parameters(), hamiltonian=hamiltonian, psi=psi, numSamples=None, outp = outp)

        energy = jax.numpy.real(tdvpEquation.ElocMean0) / L

        E_array[i, j] = energy +  1.0809205861288604 



import pickle
with open('Energy_post_svd.pkl', 'wb') as f:
    pickle.dump(E_array, f)

with open('Svd_error.pkl', 'wb') as f:
    pickle.dump(Error_array, f)


print("Relataive energy:",E_array)
print("SVD error:", Error_array)
