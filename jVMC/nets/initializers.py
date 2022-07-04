from typing import NewType
import jax
from jax.config import config
config.update("jax_enable_x64", True)
import flax
from flax import nn
import jax.numpy as jnp
import jVMC.global_defs as global_defs
from functools import partial 

def rl_init(rng, shape):

    #unif = jax.nn.initializers.uniform()
    
    #unif = jax.nn.initializers.uniform(scale = 0.1)
    unif = jax.nn.initializers.normal(stddev = 0.155)
    return unif(rng, shape, dtype=global_defs.tReal)

def cplx_init(rng, shape):
    rng1, rng2 = jax.random.split(rng)
    #unif = jax.nn.initializers.uniform()
    unif = jax.nn.initializers.uniform(scale = 0.01)
    #unif = jax.nn.initializers.normal(stddev = 0.01)
    return unif(rng1, shape, dtype=global_defs.tReal) + 1.j * unif(rng2, shape, dtype=global_defs.tReal)

def rl_init1(rng, shape, var):

    """This function iniliatizes the real parameters with given variance, "var" """
    
    #unif = jax.nn.initializers.uniform(scale = 0.1)
    norm = jax.nn.initializers.normal(stddev = jnp.sqrt(var))
    return norm(rng, shape, dtype=global_defs.tReal)


def cplx_init1(rng, shape, var):
    """This function initializes the real and the complex part of the complex initialization
    by using the fact that Var[Z] = Var[Real(Z)] + Var[Img(Z)]"""
    
    rng1, rng2 = jax.random.split(rng)
    sigma_1_r = jnp.sqrt(var/2)
    sigma_1_i = sigma_1_r
    norm1 = jax.nn.initializers.normal(stddev = sigma_1_r)
    norm2 = jax.nn.initializers.normal(stddev = sigma_1_i) 

    return norm1(rng1, shape, dtype=global_defs.tReal) + 1.j * norm2(rng2, shape, dtype=global_defs.tReal)




def cplx_variance_scaling(rng, shape):
    rng1, rng2 = jax.random.split(rng)
    unif = jax.nn.initializers.uniform(scale=1.)
    elems = 1
    for k in shape[:-2]:
        elems *= k
    w = jax.numpy.sqrt((shape[-1] + shape[-2]) * elems)
    return (1. / w) * unif(rng1, shape, dtype=global_defs.tReal) * jax.numpy.exp(1.j * 3.141593 * unif(rng2, shape, dtype=global_defs.tReal))
