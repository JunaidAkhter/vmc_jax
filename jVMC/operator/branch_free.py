import jax
from jax import jit, vmap, grad
import jax.numpy as jnp
import numpy as np

import sys
# Find jVMC package
sys.path.append(sys.path[0] + "/../..")

import jVMC.global_defs as global_defs
from . import Operator

import functools

opDtype = global_defs.tCpx

# Common operators


def Id(idx=0, lDim=2):
    """Returns an identity operator

    Args:

    * ``idx``: Index of the local Hilbert space.
    * ``lDim``: Dimension of local Hilbert space.

    Returns:
        Dictionary defining an identity operator

    """

    return {'idx': idx, 'map': jnp.array([j for j in range(lDim)], dtype=np.int32),
            'matEls': jnp.array([1. for j in range(lDim)], dtype=opDtype), 'diag': True}


def Sx(idx):
    """Returns a :math:`\hat\sigma^x` Pauli operator

    Args:

    * ``idx``: Index of the local Hilbert space.

    Returns:
        Dictionary defining :math:`\hat\sigma^x` Pauli operator

    """

    return {'idx': idx, 'map': jnp.array([1, 0], dtype=np.int32), 'matEls': jnp.array([1.0, 1.0], dtype=opDtype), 'diag': False}


def Sy(idx):
    """Returns a :math:`\hat\sigma^x` Pauli operator

    Args:

    * ``idx``: Index of the local Hilbert space.

    Returns:
        Dictionary defining :math:`\hat\sigma^x` Pauli operator

    """

    return {'idx': idx, 'map': jnp.array([1, 0], dtype=np.int32), 'matEls': jnp.array([1.j, -1.j], dtype=opDtype), 'diag': False}


def Sz(idx):
    """Returns a :math:`\hat\sigma^z` Pauli operator

    Args:

    * ``idx``: Index of the local Hilbert space.

    Returns:
        Dictionary defining :math:`\hat\sigma^z` Pauli operator

    """

    return {'idx': idx, 'map': jnp.array([0, 1], dtype=np.int32), 'matEls': jnp.array([-1.0, 1.0], dtype=opDtype), 'diag': True}


def Sp(idx):
    """Returns a :math:`S^+` spin-1/2 ladder operator

    Args:

    * ``idx``: Index of the local Hilbert space.

    Returns:
        Dictionary defining :math:`S^+` ladder operator

    """

    return {'idx': idx, 'map': jnp.array([1, 0], dtype=np.int32), 'matEls': jnp.array([1.0, 0.0], dtype=opDtype), 'diag': False}


def Sm(idx):
    """Returns a :math:`S^-` spin-1/2 ladder operator

    Args:

    * ``idx``: Index of the local Hilbert space.

    Returns:
        Dictionary defining :math:`S^-` ladder operator

    """

    return {'idx': idx, 'map': jnp.array([0, 0], dtype=np.int32), 'matEls': jnp.array([0.0, 1.0], dtype=opDtype), 'diag': False}


import copy


def scal_opstr(a, op):
    """Add prefactor to operator string

    Arguments:
        * ``a``: Scalar prefactor.
        * ``op``: Operator string.

    Returns:
        Rescaled operator string. Effectively, the matrix elements of the first element of \
        the operator string are multiplied by ``a``.

    """

    newOp = [copy.deepcopy(o) for o in op]
    newOp[0]['matEls'] = a * newOp[0]['matEls']
    return tuple(newOp)


class BranchFreeOperator(Operator):
    """This class provides functionality to compute operator matrix elements

    Initializer arguments:

        * ``lDim``: Dimension of local Hilbert space.
    """

    def __init__(self, lDim=2):
        """Initialize ``Operator``.

        Arguments:
            * ``lDim``: Dimension of local Hilbert space.
        """
        self.ops = []
        self.lDim = lDim

        super().__init__()

    def add(self, opDescr):
        """Add another operator to the operator

        Arguments:
            * ``opDescr``: Operator string to be added to the operator.

        """

        self.ops.append(opDescr)
        self.compiled = False

    def compile(self):
        """Compiles a operator mapping function from the given operator strings.

        """

        self.idx = []
        self.map = []
        self.matEls = []
        self.diag = []
        self.maxOpStrLength = 0
        for op in self.ops:
            if len(op) > self.maxOpStrLength:
                self.maxOpStrLength = len(op)
        IdOp = Id(lDim=self.lDim)
        o = 0
        for op in self.ops:
            self.idx.append([])
            self.map.append([])
            self.matEls.append([])
            isDiagonal = True
            for k in range(self.maxOpStrLength):
                if k < len(op):
                    if not op[k]['diag']:
                        isDiagonal = False
                    self.idx[o].append(op[k]['idx'])
                    self.map[o].append(op[k]['map'])
                    self.matEls[o].append(op[k]['matEls'])
                else:
                    self.idx[o].append(IdOp['idx'])
                    self.map[o].append(IdOp['map'])
                    self.matEls[o].append(IdOp['matEls'])

            if isDiagonal:
                self.diag.append(o)
            o = o + 1

        self.idxC = jnp.array(self.idx, dtype=np.int32)
        self.mapC = jnp.array(self.map, dtype=np.int32)
        self.matElsC = jnp.array(self.matEls, dtype=opDtype)
        self.diag = jnp.array(self.diag, dtype=np.int32)

        return functools.partial(self._get_s_primes, idxC=self.idxC, mapC=self.mapC, matElsC=self.matElsC, diag=self.diag)

    def _get_s_primes(self, s, idxC, mapC, matElsC, diag):

        numOps = idxC.shape[0]
        matEl = jnp.ones(numOps, dtype=matElsC.dtype)
        sp = jnp.array([s] * numOps)

        def apply_fun(config, configMatEl, idx, sMap, matEls):

            configShape = config.shape
            config = config.ravel()
            configMatEl = configMatEl * matEls[config[idx]]
            config = config.at[idx].set(sMap[config[idx]])

            return config.reshape(configShape), configMatEl

        def apply_multi(config, configMatEl, opIdx, opMap, opMatEls):

            for idx, mp, me in zip(opIdx, opMap, opMatEls):
                config, configMatEl = apply_fun(config, configMatEl, idx, mp, me)

            return config, configMatEl

        # vmap over operators
        sp, matEl = vmap(apply_multi, in_axes=(0, 0, 0, 0, 0))(sp, matEl, idxC, mapC, matElsC)
        if len(diag) > 1:
            matEl = matEl.at[diag[0]].set(jnp.sum(matEl[diag], axis=0))
            matEl = matEl.at[diag[1:]].set(jnp.zeros((diag.shape[0] - 1,), dtype=matElsC.dtype))

        return sp, matEl

