import jax.numpy as jnp
from jax import lax

class Tensor:
    """Tensor operations with JAX support for Mandel/Voigt notation conversions
    
    Attributes:
        B: Mandel basis tensors (3,3,6) with orthonormal properties
        B_voigt: Voigt basis tensors (3,3,6) with engineering scaling
    """
    
    def __init__(self):
        """Initialize basis vectors and construct Mandel/Voigt basis tensors"""
        # Orthonormal basis vectors for first-order tensors
        self.e1 = jnp.array([1., 0., 0.])
        self.e2 = jnp.array([0., 1., 0.])
        self.e3 = jnp.array([0., 0., 1.])

        # Construct Mandel basis with orthonormal scaling
        mandel_bases = [
            self._diade(self.e1, self.e1),  # 𝕀₁₁
            self._diade(self.e2, self.e2),  # 𝕀₂₂
            self._diade(self.e3, self.e3),  # 𝕀₃₃
            (jnp.sqrt(2)/2)*(self._diade(self.e2, self.e3) + self._diade(self.e3, self.e2)),  # 𝕀₂₃
            (jnp.sqrt(2)/2)*(self._diade(self.e1, self.e3) + self._diade(self.e3, self.e1)),  # 𝕀₁₃
            (jnp.sqrt(2)/2)*(self._diade(self.e1, self.e2) + self._diade(self.e2, self.e1))   # 𝕀₁₂
        ]
        self.B = jnp.stack(mandel_bases, axis=2)

        # Construct Voigt basis without scaling
        voigt_bases = [
            self._diade(self.e1, self.e1),
            self._diade(self.e2, self.e2),
            self._diade(self.e3, self.e3),
            self._diade(self.e2, self.e3) + self._diade(self.e3, self.e2),  # Unsymmetrized
            self._diade(self.e1, self.e3) + self._diade(self.e3, self.e1),
            self._diade(self.e1, self.e2) + self._diade(self.e2, self.e1)
        ]
        self.B_voigt = jnp.stack(voigt_bases, axis=2)

    def _diade(self, di, dj):
        """Compute dyadic product of two vectors (2nd order tensor)
        
        Args:
            di: First vector (3,)
            dj: Second vector (3,)
            
        Returns:
            (3,3) second-order tensor: di ⊗ dj
        """
        return jnp.einsum('i,j->ij', di, dj)

    def _diade4(self, bi, bj):
        """Compute fourth-order tensor from two second-order basis tensors
        
        Args:
            bi: First basis tensor (3,3)
            bj: Second basis tensor (3,3)
            
        Returns: 
            (3,3,3,3) fourth-order tensor: bi ⊗ bj
        """
        return jnp.einsum('ij,kl->ijkl', bi, bj)

    def tensor_product(self, tensor_a, tensor_b):
        """Compute tensor product in reduced notation (matrix multiplication)
        
        Args:
            tensor_a: (6,6) tensor in reduced notation
            tensor_b: (6,6) tensor in reduced notation
            
        Returns:
            (6,6) product tensor in same notation
        """
        return jnp.einsum('ij,jk->ik', tensor_a, tensor_b)

    def matrix2voigt(self, matrix):
        """Convert 2nd order tensor to Voigt vector notation
        
        Args:
            matrix: (3,3) symmetric tensor
            
        Returns:
            (6,) vector: [σ₁₁, σ₂₂, σ₃₃, σ₂₃, σ₁₃, σ₁₂]
        """
        return jnp.array([
            matrix[0, 0], matrix[1, 1], matrix[2, 2],
            matrix[1, 2], matrix[0, 2], matrix[0, 1]
        ])

    def matrix2mandel(self, matrix):
        """Convert 2nd order tensor to Mandel vector notation with √2 scaling
        
        Args:
            matrix: (3,3) symmetric tensor
            
        Returns:
            (6,) vector: [σ₁₁, σ₂₂, σ₃₃, √2σ₂₃, √2σ₁₃, √2σ₁₂]
        """
        b = jnp.sqrt(2)
        return jnp.array([
            matrix[0, 0], matrix[1, 1], matrix[2, 2],
            b * matrix[1, 2], b * matrix[0, 2], b * matrix[0, 1]
        ])

    def _tensor2matrix(self, tensor, representation):
        """Convert 4th order tensor to reduced notation matrix
        
        Args:
            tensor: (3,3,3,3) fourth-order tensor
            representation: 'voigt' or 'mandel' scaling
            
        Returns:
            (6,6) matrix in specified notation
        """
        assert representation in ('voigt', 'mandel'), "Invalid representation"
        
        # Scaling factors based on notation
        c = 1.0 if representation == 'voigt' else jnp.sqrt(2)
        b = 1.0 if representation == 'voigt' else jnp.sqrt(2)
        
        # Construct matrix representation
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

    def tensor2mandel(self, tensor):
        """Convert 4th order tensor to Mandel notation matrix"""
        return self._tensor2matrix(tensor, 'mandel')

    def tensor2voigt(self, tensor):
        """Convert 4th order tensor to Voigt notation matrix"""
        return self._tensor2matrix(tensor, 'voigt')

    def mandel2tensor(self, mandel):
        """Reconstruct 4th order tensor from Mandel notation
        
        Args:
            mandel: (6,6) matrix in Mandel notation
            
        Returns:
            (3,3,3,3) tensor: ∑ₖₗ mandel[k,l] * Bₖ ⊗ Bₗ
        """
        def body_fn(i, tensor):
            def inner_fn(j, t):
                return t + mandel[i,j] * self._diade4(self.B[:,:,i], self.B[:,:,j])
            return lax.fori_loop(0, 6, inner_fn, tensor)
        return lax.fori_loop(0, 6, body_fn, jnp.zeros((3,3,3,3)))

    def voigt2tensor(self, voigt):
        """Reconstruct 4th order tensor from Voigt notation
        
        Args:
            voigt: (6,6) matrix in Voigt notation
            
        Returns:
            (3,3,3,3) tensor: ∑ₖₗ voigt[k,l] * B_voigtₖ ⊗ B_voigtₗ
        """
        def body_fn(i, tensor):
            def inner_fn(j, t):
                return t + voigt[i,j] * self._diade4(self.B_voigt[:,:,i], self.B_voigt[:,:,j])
            return lax.fori_loop(0, 6, inner_fn, tensor)
        return lax.fori_loop(0, 6, body_fn, jnp.zeros((3,3,3,3)))

    def mandel2voigt(self, mandel):
        """Convert Mandel matrix to Voigt matrix via tensor reconstruction"""
        return self.tensor2voigt(self.mandel2tensor(mandel))

    def voigt2mandel(self, voigt):
        """Convert Voigt matrix to Mandel matrix via tensor reconstruction"""
        return self.tensor2mandel(self.voigt2tensor(voigt))
