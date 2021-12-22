import jax
from jax.config import config
config.update("jax_enable_x64", True)
import flax
#from flax import nn
import flax.linen as nn
import jax.numpy as jnp

import jVMC.global_defs as global_defs
import jVMC.nets.activation_functions as act_funs

from functools import partial

import jVMC.nets.initializers

import tensornetwork as tn
tn.set_default_backend("jax")

from typing import Any, Callable, Sequence, Optional
import jVMC.nets.initializers as init
import jVMC.global_defs as global_defs


class CpxRBM(nn.Module):
    """Restricted Boltzmann machine with complex parameters.

    Arguments:

        * ``s``: Computational basis configuration.
        * ``numHidden``: Number of hidden units.
        * ``bias``: ``Boolean`` indicating whether to use bias.

    Returns:
        Complex wave-function coefficient
    """
    numHidden: int = 2
    bias: bool = False

    @nn.compact
    def __call__(self, s):

        layer = nn.Dense(self.numHidden, use_bias=self.bias, dtype=global_defs.tCpx,
                         kernel_init=jVMC.nets.initializers.cplx_init,
                         bias_init=partial(jax.nn.initializers.zeros, dtype=global_defs.tCpx))

        return jnp.sum(jnp.log(jnp.cosh(layer(2 * s.ravel() - 1))))

# ** end class CpxRBM


class RBM(nn.Module):
    """Restricted Boltzmann machine with real parameters.

    Args:

        * ``s``: Computational basis configuration.
        * ``numHidden``: Number of hidden units.
        * ``bias``: ``Boolean`` indicating whether to use bias.

    Returns:
        Wave function coefficient
    """
    numHidden: int = 2
    bias: bool = False

    @nn.compact
    def __call__(self, s):

        layer = nn.Dense(self.numHidden, use_bias=self.bias, dtype=global_defs.tReal,
                         kernel_init=jax.nn.initializers.lecun_normal(dtype=global_defs.tReal),
                         bias_init=partial(jax.nn.initializers.zeros, dtype=global_defs.tReal))

        return jnp.sum(jnp.log(jnp.cosh(layer(2 * s - 1))))

# ** end class RBM


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


