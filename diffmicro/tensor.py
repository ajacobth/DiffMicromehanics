import jax.numpy as jnp
from jax import lax

class Tensor:
    """
    Tensor class to help with basic arithmetic operations on tensor space using JAX.
    """
    def __init__(self):
        self.e1 = jnp.array([1., 0., 0.])
        self.e2 = jnp.array([0., 1., 0.])
        self.e3 = jnp.array([0., 0., 1.])

        # Construct Mandel basis using JAX operations
        mandel_bases = [
            self._diade(self.e1, self.e1),
            self._diade(self.e2, self.e2),
            self._diade(self.e3, self.e3),
            (jnp.sqrt(2)/2) * (self._diade(self.e2, self.e3) + self._diade(self.e3, self.e2)),
            (jnp.sqrt(2)/2) * (self._diade(self.e1, self.e3) + self._diade(self.e3, self.e1)),
            (jnp.sqrt(2)/2) * (self._diade(self.e1, self.e2) + self._diade(self.e2, self.e1))
        ]
        self.B = jnp.stack(mandel_bases, axis=2)

        # Construct Voigt basis
        voigt_bases = [
            self._diade(self.e1, self.e1),
            self._diade(self.e2, self.e2),
            self._diade(self.e3, self.e3),
            self._diade(self.e2, self.e3) + self._diade(self.e3, self.e2),
            self._diade(self.e1, self.e3) + self._diade(self.e3, self.e1),
            self._diade(self.e1, self.e2) + self._diade(self.e2, self.e1)
        ]
        self.B_voigt = jnp.stack(voigt_bases, axis=2)

    def _diade(self, di, dj):
        return jnp.einsum('i,j->ij', di, dj)

    def _diade4(self, bi, bj):
        return jnp.einsum('ij,kl->ijkl', bi, bj)

    def tensor_product(self, tensor_a, tensor_b):
        return jnp.einsum('ij,jk->ik', tensor_a, tensor_b)

    def matrix2voigt(self, matrix):
        return jnp.array([
            matrix[0, 0], matrix[1, 1], matrix[2, 2],
            matrix[1, 2], matrix[0, 2], matrix[0, 1]
        ])

    def matrix2mandel(self, matrix):
        b = jnp.sqrt(2)
        return jnp.array([
            matrix[0, 0], matrix[1, 1], matrix[2, 2],
            b * matrix[1, 2], b * matrix[0, 2], b * matrix[0, 1]
        ])

    def _tensor2matrix(self, tensor, representation):
        assert representation in ('voigt', 'mandel'), "Invalid representation"
        
        c = 1.0 if representation == 'voigt' else jnp.sqrt(2)
        b = 1.0 if representation == 'voigt' else jnp.sqrt(2)
        
        return jnp.array([
            [tensor[0,0,0,0], tensor[0,0,1,1], tensor[0,0,2,2],
             b*tensor[0,0,1,2], b*tensor[0,0,0,2], b*tensor[0,0,0,1]],
            [tensor[1,1,0,0], tensor[1,1,1,1], tensor[1,1,2,2],
             b*tensor[1,1,1,2], b*tensor[1,1,0,2], b*tensor[1,1,0,1]],
            [tensor[2,2,0,0], tensor[2,2,1,1], tensor[2,2,2,2],
             b*tensor[2,2,1,2], b*tensor[2,2,0,2], b*tensor[2,2,0,1]],
            [c*tensor[1,2,0,0], c*tensor[1,2,1,1], c*tensor[1,2,2,2],
             b*c*tensor[1,2,1,2], b*c*tensor[1,2,0,2], b*c*tensor[1,2,0,1]],
            [c*tensor[0,2,0,0], c*tensor[0,2,1,1], c*tensor[0,2,2,2],
             b*c*tensor[0,2,1,2], b*c*tensor[0,2,0,2], b*c*tensor[0,2,0,1]],
            [c*tensor[0,1,0,0], c*tensor[0,1,1,1], c*tensor[0,1,2,2],
             b*c*tensor[0,1,1,2], b*c*tensor[0,1,0,2], b*c*tensor[0,1,0,1]]
        ])

    def mandel2tensor(self, mandel):
        tensor = jnp.zeros((3, 3, 3, 3))
        for i in range(6):
            for j in range(6):
                tensor += mandel[i,j] * self._diade4(self.B[:,:,i], self.B[:,:,j])
        return tensor

    def voigt2tensor(self, voigt):
        tensor = jnp.zeros((3, 3, 3, 3))
        for i in range(6):
            for j in range(6):
                tensor += voigt[i,j] * self._diade4(self.B_voigt[:,:,i], self.B_voigt[:,:,j])
        return tensor

