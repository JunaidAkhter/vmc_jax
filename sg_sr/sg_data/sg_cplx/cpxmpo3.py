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






import tensornetwork as tn
tn.set_default_backend("jax")


import jVMC.nets.initializers as init
import jVMC.global_defs as global_defs


import time
start = time.time()


# DMRG energies produced with the TeNPy library https://github.com/tenpy/tenpy
#DMRG_energies = {"10": -1.0545844370449059, "20": -1.0900383739, "100": -1.1194665474274852}

L = 12 # system size
g = -0.7 # strength of external field



#FIRST WE DEFINE AN MPO LAYER. THE CALCULATION DONE HERE IS FOR TWO NODES BECAUSE THE INITIALIZATION
# WE DEFINE HERE IS FOR TWO NODES.
class MPO(nn.Module):
    """MPO with "n" nodes
    Acts on: 
        x: Input data vector of any shape without the batch number.
        batching will be taken care by vmap function.
    Arguments:
        num_nodes: the number of nodes that we want.
        inp_dims: list containing input dimensions for each node.
        oup_dim: Output dimension (same for every node).
        D: Bond dimension
    Returns:
        An n dimensional array
    Note: One must know the dimension of "x"(without the batch)
        before the mpo acts on "x" and choose the number of nodes
        and the input dimensions so that the product of the input 
        dimensions of MPO is same as the total dimensionality of "x"
        """  
    num_nodes: int 
    inp_dims: Sequence[int]
    oup_dims: Sequence[int] 
    D: int  
        
    @nn.compact
    def __call__(self, x):
        n = self.num_nodes
        inp_dms = self.inp_dims
        oup_dms = self.oup_dims
        D = self.D
        #print("Input_dimension:", inp_dms)
        x = x.reshape(inp_dms) #reshaping to feed to mpo
        #print("reshaped_x:", x.shape)
        nodes = [] #empty list in which we will store nodes(which are basically just arrays) 
        legs = [] #empty list in which we are going to store the sequences of contractions 
        nodes.append(x) # adding the data as the first node to the list
        legs.append([ i for i in range(1,n+1)]) # naming list for input legs from the data
        #print('n:', n, 'input_dimensions:', inp_dms, 'output_dimensions:', oup_dm, 'D:', D)
        
        #x = 0.05047557
        #new = partial(init.init1, var = x)
        k_a = k_b = 1
        for i, dm in enumerate(inp_dms):
            if i == 0:
                #print('i:', i, 'dm:', dm)

                #calculating the variance for this node 
                #var_a1 = (n_i2* k_b**2)/(k_a*4*(n_i1**2)*n_D*(n_j1**3))**(1/6)

                var_a1 = ((inp_dms[i+1]*k_b**2)/(k_a*4*(inp_dms[i]**2)*D*(oup_dms[i]**3)))**(1/3)
                
                #var_a1 = ((inp_dms[1]*k_b**2)/(k_a*4*(inp_dms[0]**2)*D*(oup_dms[0]**3)))**(1/3)
                print("var_a1:", var_a1)
                nodes.append(self.param('a'+str(i), partial(init.init1, var = var_a1), (oup_dms[i],dm,D))) # include the node name later
                legs.append([-1,1,n+1])
            elif i == n-1:
                #print('i:', i, 'dm:', dm)
                #calculating the variance for this node 
                #np.sqrt(k_a*n_i1*n_j1/(k_b*n_i2*n_j2))*std_A

                var_al = (k_a*inp_dms[i-1]*oup_dms[i-1])/(k_b*inp_dms[i]*oup_dms[i])*var_a1
                print("var_al:", var_al)
                nodes.append(self.param('a'+str(i), partial(init.init1, var = var_al), (dm,oup_dms[i],D)))
                legs.append([n, -n, 2*n-1])

            else:
                #print('i:', i, 'dm:', dm)
                nodes.append(self.param('a'+str(i), init.cplx_init, (dm,D,oup_dms[i],D)))
                legs.append([i+1, n+2, -(i+1), n+1])
       
        # creating the bias which we need to add at the end
        #bias = self.param('bias', self.kernel_init, [oup_dm]*n)
       
        result = tn.ncon(nodes, legs)  # bias must be added here if the above line in ucommented. 
        result = result 
    
        return result

# This class defines the network structure of a complex RBM
class MyNet(flax.linen.Module):
    num_nodes: int 
    inp_dims: Sequence[int] 
    oup_dims: Sequence[int] 
    D: int 

    @flax.linen.compact
    def __call__(self, s):


        # introducing the mpo layer
        def apply_mpo(single_config):
            return MPO(num_nodes = self.num_nodes, inp_dims = self.inp_dims, \
                                    oup_dims = self.oup_dims, D = self.D)(single_config)

        return jnp.sum(jnp.log(jnp.cosh(apply_mpo(2 * s - 1))))



# Set up hamiltonian for open boundary conditions
hamiltonian = jVMC.operator.BranchFreeOperator()
for l in range(L - 1):
    hamiltonian.add(jVMC.operator.scal_opstr(-1., (jVMC.operator.Sz(l), jVMC.operator.Sz(l + 1))))
    hamiltonian.add(jVMC.operator.scal_opstr(g, (jVMC.operator.Sx(l), )))
hamiltonian.add(jVMC.operator.scal_opstr(g, (jVMC.operator.Sx(L - 1), )))


def simulate(rng, iterations, D):

    net = MyNet(num_nodes = 2, inp_dims = (4,3), oup_dims = (6,6), D = D) # D = 08 in reality
    psi = jVMC.vqs.NQS(net, seed=rng)  # Variational wave function
    #Checking the dhape of the mpo and the values of the initialized parameters
    params = net.init(jax.random.PRNGKey(1),jnp.zeros((L,), dtype=global_defs.tCpx)) # the "dtype" here is not so important
    print("Shape of the model", jax.tree_map(np.shape, params))
    #print("parameters:", params)


    # Set up sampler
    sampler = jVMC.sampler.MCSampler(psi, (L,), random.PRNGKey(4321), updateProposer=jVMC.sampler.propose_spin_flip_Z2,
                                 numChains=100, sweepSteps=L,
                                 numSamples=30000, thermalizationSweeps=25)

    # Set up TDVP
    tdvpEquation = jVMC.util.tdvp.TDVP(sampler, rhsPrefactor=1.,
                                   svdTol=1e-8, diagonalShift=50, makeReal='real')

    stepper = jVMC.util.stepper.Euler(timeStep=1e-2)  # ODE integrator



    res = []
    for n in range(iterations):
        dp, _ = stepper.step(0, tdvpEquation, psi.get_parameters(), hamiltonian=hamiltonian, psi=psi, numSamples=None)
        psi.set_parameters(dp)

        #print(n, jax.numpy.real(tdvpEquation.ElocMean0) / L, tdvpEquation.ElocVar0 / L)

        res.append([jax.numpy.real(tdvpEquation.ElocMean0) /L])

    return np.array(res)


iterations = 1500
rng_list = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10]
#iterations = 1
#rng_list = [0, 1]

E_0_aarray = np.zeros((iterations, len(rng_list)))#an empty two dimensional array corresponding to the D and "rng".



for j,rng in enumerate(rng_list):
    print("rng:", rng)
    res = simulate(rng, iterations, D=8)
    E_0 = res + 1.0660513358196495#this energy is for 12 spins
    #adding the energy values obtained to the first entry of the row
    #print("length", len(E_0))
    E_0_aarray[:, j] = E_0[:, 0]
    #print("final_energy:", E_0[-1])

#print("E_array", E_0_aarray)


print("E_array", E_0_aarray)


np.savetxt('cpxmpo_12_avg_d8_43_gs', E_0_aarray, header='Data for rlmpo with D = 8 for 10 different initializations')






