import jax.numpy as jnp
from .tensor import Tensor


class Elasticity(Tensor):
    """
    Elasticity class to express generic elastic stiffness tensors using JAX.
    The class inherits from the JAX-compatible Tensor class.

    Attributes
    ----------
    stiffness3333 : jax.Array of shape (3, 3, 3, 3)
        Stiffness values in the regular tensor notation in Pa.
    stiffness66 : jax.Array of shape (6, 6)
        Stiffness values in the normalized Voigt notation in Pa.
    """

    def __init__(self):
        super().__init__()
        self.stiffness3333 = jnp.zeros((3, 3, 3, 3))
        self.stiffness66 = jnp.zeros((6, 6))


class TransverseIsotropy(Elasticity):
    """
    Transverse Isotropy class with JAX operations for stiffness tensors.

    Parameters/Attributes remain the same as original but using JAX arrays
    """

    def __init__(self, E1, E2, G12, G23, nu12):
        super().__init__()
        self.E1 = E1
        self.E2 = E2
        self.G12 = G12
        self.G23 = G23
        self.nu12 = nu12
        self.nu21 = self.E2 / self.E1 * self.nu12
        self.nu23 = self.E2 / (2 * self.G23) - 1
        self._get_stiffness()

    def _get_stiffness(self):
        """Calculate stiffness parameters using JAX operations"""
        C1111 = (1 - self.nu23) / (1 - self.nu23 - 2 * self.nu12 * self.nu21) * self.E1
        lam = (
            (self.nu12 * self.nu21 + self.nu23)
            / (1 - self.nu23 - 2 * self.nu12 * self.nu21)
            / (1 + self.nu23)
            * self.E2
        )
        
        # Construct stiffness matrix with JAX array operations
        self.stiffness66 = jnp.array([
            [
                C1111,
                2 * self.nu12 * (lam + self.G23),
                2 * self.nu12 * (lam + self.G23),
                0,
                0,
                0,
            ],
            [2 * self.nu12 * (lam + self.G23), lam + 2 * self.G23, lam, 0, 0, 0],
            [2 * self.nu12 * (lam + self.G23), lam, lam + 2 * self.G23, 0, 0, 0],
            [0, 0, 0, 2 * self.G23, 0, 0],
            [0, 0, 0, 0, 2 * self.G12, 0],
            [0, 0, 0, 0, 0, 2 * self.G12],
        ])
        
        self.stiffness3333 = self.mandel2tensor(self.stiffness66)


class Isotropy(TransverseIsotropy):
    """
    Isotropy class with JAX operations for isotropic materials

    Parameters/Attributes remain the same with JAX compatibility
    """

    def __init__(self, E, nu):
        self.E = E
        self.nu = nu
        self.lam = self._get_lambda()
        self.mu = self._get_mu()
        super().__init__(self.E, self.E, self.mu, self.mu, self.nu)

    def _get_lambda(self):
        """Calculate first Lamé constant using JAX math"""
        return self.nu / (1 - 2 * self.nu) * 1 / (1 + self.nu) * self.E

    def _get_mu(self):
        """Calculate shear modulus (second Lamé constant) with JAX"""
        return 1 / 2 * 1 / (1 + self.nu) * self.E
