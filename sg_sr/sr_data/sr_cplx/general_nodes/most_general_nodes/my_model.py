import sys
# Find jVMC package
#sys.path.append("/p/home/jusers/akhter1/jureca/project/jvmc/vmc_jax")
sys.path.append("/Users/akhter/thesis/vmc_jax")



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

# DMRG energies produced with the TeNPy library https://github.com/tenpy/tenpy
#DMRG_energies = {"10": -1.0545844370449059, "20": -1.0900383739, "100": -1.1194665474274852}

L = 32 # system size
g = -0.7 # strength of external field

tik = time.perf_counter()
#FIRST WE DEFINE AN MPO LAYER

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
    bond_dims: Sequence[int]
        
    @nn.compact
    def __call__(self, x):
        n = self.num_nodes
        inp_dms = self.inp_dims
        oup_dms = self.oup_dims
        bond_dms = self.bond_dims
        #print("Input_dimension:", inp_dms)
        x = x.reshape(inp_dms) #reshaping to feed to mpo
        #print("reshaped_x:", x.shape)
        nodes = [] #empty list in which we will store nodes(which are basically just arrays) 
        legs = [] #empty list in which we are going to store the sequences of contractions 
        nodes.append(x) # adding the data as the first node to the list
        legs.append([ i for i in range(1,n+1)]) # naming list for input legs from the data
        #print('n:', n, 'input_dimensions:', inp_dms, 'output_dimensions:', oup_dm, 'D:', D)
        
        #D = bond_dms[0]
        #a = 1/(inp_dms[1]*inp_dms[2]*oup_dms[0]*(D**3)*oup_dms[1]**2 * oup_dms[2]**2)
        #b = 1/(inp_dms[0]*inp_dms[2]*oup_dms[0]**2*(D**2)*oup_dms[1] * oup_dms[2]**2)
        #c = 1/(inp_dms[1]*inp_dms[0]*oup_dms[2]*(D**3)*oup_dms[1]**2 * oup_dms[0]**2)
        #d = (a*b*c)**(1/5)
       
        n_d = sum(list(bond_dms))/len(bond_dms)
        v = 1
        def var_calc(i, v):
            """This function calculates the variance a particular node"""
            numerator = jnp.product(jnp.array(inp_dms))*v/(inp_dms[i]**(2*n-1))
            
            if i ==0 or i == n-1:
                denomenator = n_d
            else:
                denomenator = n_d**(2*n)
            return 1/(oup_dms[i])*(numerator/denomenator)**(1/(2*n-1))

        for i, dm in enumerate(inp_dms):
            if i == 0:
                #print('i:', i, 'dm:', dm)
                #print("var:",d**2/a, "var_func:", var_calc(i,v) )
                nodes.append(self.param('a'+str(i), partial(init.cplx_init1, var = var_calc(i, v)), (oup_dms[i],dm,bond_dms[i]))) # include the node name later
               
                legs.append([-1,1,n+1])
            elif i == n-1:
                #print('i:', i, 'dm:', dm)
                #print("var:",d**2/c, "var_func:", var_calc(i,v) )
                nodes.append(self.param('a'+str(i), partial(init.cplx_init1, var = var_calc(i,v)), (dm,oup_dms[i],bond_dms[i-1])))
                legs.append([n, -n, 2*n-1])

            else:
                #print('i:', i, 'dm:', dm)
                #print("var:",d**2/b, "var_func:", var_calc(i,v) )
                nodes.append(self.param('a'+str(i),  partial(init.cplx_init1, var = var_calc(i,v)), (dm,bond_dms[i],oup_dms[i],bond_dms[i-1])))
                legs.append([i+1, n+(i+1), -(i+1), n+(i)])
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
    bond_dims:Sequence[int] 

    @flax.linen.compact
    def __call__(self, s):


        # introducing the mpo layer
        def apply_mpo(single_config):
            return MPO(num_nodes = self.num_nodes, inp_dims = self.inp_dims, \
                                    oup_dims = self.oup_dims, bond_dims = self.bond_dims)(single_config)

        return jnp.sum(jnp.log(jnp.cosh(apply_mpo(2 * s - 1))))



# Set up hamiltonian for open boundary conditions
hamiltonian = jVMC.operator.BranchFreeOperator()
for l in range(L - 1):
    hamiltonian.add(jVMC.operator.scal_opstr(-1., (jVMC.operator.Sz(l), jVMC.operator.Sz(l + 1))))
    hamiltonian.add(jVMC.operator.scal_opstr(g, (jVMC.operator.Sx(l), )))
hamiltonian.add(jVMC.operator.scal_opstr(g, (jVMC.operator.Sx(L - 1), )))


def simulate(rng, iterations, t_step = 1e-2): #Original t = 1e-2

    net = net_init

    psi = jVMC.vqs.NQS(net,batchSize=10000, seed=rng)  # Variational wave function
    #Checking the dhape of the mpo and the values of the initialized parameters

    # Set up sampler
    sampler = jVMC.sampler.MCSampler(psi, (L,), random.PRNGKey(4321), updateProposer=jVMC.sampler.propose_spin_flip_Z2,
                                 numChains=100, sweepSteps=L,
                                 numSamples=50000, thermalizationSweeps=25)

    # Set up TDVP
    tdvpEquation = jVMC.util.tdvp.TDVP(sampler, rhsPrefactor=1.,
                                   svdTol=1e-8, diagonalShift=50, makeReal='real')

    stepper = jVMC.util.stepper.Euler(timeStep=t_step)  # ODE integrator



    res = []
    for n in range(iterations):
        dp, _ = stepper.step(0, tdvpEquation, psi.get_parameters(), hamiltonian=hamiltonian, psi=psi, numSamples=None)
        psi.set_parameters(dp)
        res.append([jax.numpy.real(tdvpEquation.ElocMean0) /L])

    return np.array(res)


#CREATING DATA

iterations = 10
#rng_list = [0,1,2,3,4,5,6,7,8,9]
#iterations = 1
rng_list = [0]
time_step = 80e-2
E_0_aarray = np.zeros((iterations, len(rng_list)))#an empty two dimensional array corresponding to the D and "rng".

#Printing the shape of parameters
net_init = MyNet(num_nodes = 2, inp_dims = (2,16), oup_dims = (2,16), bond_dims = (4,)) 
params = net_init.init(jax.random.PRNGKey(1),jnp.zeros((L,), dtype=global_defs.tCpx)) # the "dtype" here is not so important
print("Shape of the model", jax.tree_map(np.shape, params))


for j,rng in enumerate(rng_list):
    #print("rng:", rng)
    res = simulate(rng, iterations, t_step = time_step)
    E_0 = res + 1.1038222181006074 #this energy is for 32 spins
    #adding the energy values obtained to the first entry of the row
    #print("length", len(E_0))
    E_0_aarray[:, j] = E_0[:, 0]

tok = time.perf_counter()

print("   == Total time for running the script: %fs\n" % (tok - tik))


np.savetxt('cpxmpo_32_avg_d4_216_sr', E_0_aarray, header='Data for cpxmpo with D = 4 for 10 different initializations')

