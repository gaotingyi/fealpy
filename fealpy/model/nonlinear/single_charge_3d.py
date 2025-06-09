from typing import Sequence
from ...decorator import cartesian
from ...backend import backend_manager as bm
from ...backend import TensorLike

class SingleCharge3D:
    """
    3D Single Charge Density Coupled Problem in spherical shell domain:
    
        Domain: Ω = {(x,y,z) | r1 < r < r2}, where r = sqrt(x^2 + y^2 + z^2)

        PDE system:
            -Δφ - ρ = f1
             ∇ρ ⋅ ∇φ - ρ² = f2

        Exact solution:
            φ(x,y,z) = U0 * log(r / r2) / log(r1 / r2)
            ρ(x,y,z) = c1 / sqrt(k3^2 + r^2)

        Dirichlet boundary conditions:
            φ(r1) = U0, φ(r2) = 0
            ρ(r1) = c1 / sqrt(k3^2 + r1^2)
            ρ(r2) = c1 / sqrt(k3^2 + r2^2)
    """

    def geo_dimension(self) -> int:
        """Return the geometric dimension of the domain."""
        return 3

    def domain(self) -> Sequence[float]:
        """Return radial domain range [r1, r2] for spherical shell."""
        return [0.05, 0.5]  # r1, r2

    # Constants
    r1 = 0.05
    r2 = 0.5
    m =  0.4
    delta = 1
    rho0 = 565.9759
    U0 = 200.0
    E_on = 33.7 *m *delta *(1+0.24/bm.sqrt(bm.array(100*r1*delta)))*100
    print(E_on)
    eps0 = 1.0
    c1 = bm.sqrt(r1*E_on*eps0*rho0)
    print(c1)
    k2 = bm.sqrt(r1*E_on*eps0/rho0)
    print(k2)
    k3 = bm.sqrt(k2**2-r1**2)
    print(k3)

    @cartesian
    def solution_phi(self, p: TensorLike) -> TensorLike:
        """Analytical solution φ(x,y,z)."""
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        r = bm.sqrt(x**2 + y**2 + z**2)
        val = self.U0 * bm.log(bm.array(r / self.r2)) / bm.log(bm.array(self.r1 / self.r2))
        return val

    @cartesian
    def solution_rho(self, p: TensorLike) -> TensorLike:
        """Analytical solution ρ(x,y,z)."""
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        r = bm.sqrt(x**2 + y**2 + z**2)
        val = self.c1 / bm.sqrt(self.k3**2 + r**2)
        return val

    @cartesian
    def source_f1(self, p: TensorLike) -> TensorLike:
        """Compute source term f1 = -Δφ - ρ."""
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        r = bm.sqrt(x**2 + y**2 + z**2)
        lap_phi = -self.U0 / (r**2 * bm.log(bm.array(self.r1 / self.r2)))
        rho = self.c1 / bm.sqrt(self.k3**2 + r**2)
        return lap_phi - rho

    @cartesian
    def source_f2(self, p: TensorLike) -> TensorLike:
        """Compute source term f2 = ∇ρ⋅∇φ - ρ²."""
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        r = bm.sqrt(x**2 + y**2 + z**2)
        val1 = -(self.c1*self.U0)/((self.k3**2+r**2)**(3/2)*bm.log(bm.array(self.r1/self.r2)))
        val2 = - self.c1**2/(self.k3**2+r**2)
        return val1+val2
    @cartesian
    def gradient_phi(self, p: TensorLike) -> TensorLike:
        """
        Compute gradient of φ(x, y, z).
        
        ∇φ = U0 / (r² log(r1/r2)) · (x, y, z)
        """
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        r = bm.sqrt(x**2 + y**2 + z**2)
        factor = self.U0 / (r**2 * bm.log(bm.array(self.r1 / self.r2)))

        val = bm.zeros_like(p, dtype=bm.float64)
        val[..., 0] = factor * x
        val[..., 1] = factor * y
        val[..., 2] = factor * z
        return val

    @cartesian
    def gradient_rho(self, p: TensorLike) -> TensorLike:
        """
        Compute gradient of ρ(x, y, z).

        ∇ρ = -c1 / (k3² + r²)^{3/2} · (x, y, z)
        """
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        r = bm.sqrt(x**2 + y**2 + z**2)
        denom = (self.k3**2 + r**2)**1.5

        factor = -self.c1 / denom
        val = bm.zeros_like(p, dtype=bm.float64)
        val[..., 0] = factor * x
        val[..., 1] = factor * y
        val[..., 2] = factor * z
        return val

#    @cartesian
#    def rho_squared(self, p: TensorLike) -> TensorLike:
#        """
#        Compute ρ² with a regularization term:
#
#            ρ² = (c₁ / √(k₃² + r²) + ε)²
#        """
#        x, y, z = p[..., 0], p[..., 1], p[..., 2]
#        r = bm.sqrt(x**2 + y**2 + z**2)
#        epsilon = 1e-4  # Small regularization to avoid division by zero
#        denom = bm.sqrt(self.k3**2 + r**2)
#        val = (self.c1 / denom + epsilon)**2
#        return val
#
#
#    @cartesian
#    def grad_dot_grad(self, p: TensorLike) -> TensorLike:
#        """
#        Compute dot product of ∇ρ and ∇φ:
#        
#            ∇ρ · ∇φ = Σ_i ∂ρ/∂x_i · ∂φ/∂x_i
#        """
#        x, y, z = p[..., 0], p[..., 1], p[..., 2]
#        r = bm.sqrt(x**2 + y**2 + z**2)
#        log_term = bm.log(self.r1 / self.r2)
#        denom_rho = (self.k3**2 + r**2)**1.5
#        denom_phi = r**2 * log_term
#        factor = -self.c1 * self.U0 / (denom_rho * denom_phi)
#        val = factor * (x**2 + y**2 + z**2)  
#        return val
    @cartesian
    def init_phi(self, p: TensorLike) -> TensorLike:
        flag = self.is_dirichlet_boundary(p)
        val = self.solution_phi(p) + 0.0001
        val[flag] = self.solution_phi(p)[flag]
        return val
    @cartesian
    def init_rho(self, p: TensorLike) -> TensorLike:
        flag = self.is_dirichlet_boundary(p)
        val = self.solution_rho(p) + 0.0001
        val[flag] = self.solution_rho(p)[flag]
        return val



    @cartesian
    def dirichlet_phi(self, p: TensorLike) -> TensorLike:
        """Dirichlet boundary condition for φ."""
        return self.solution_phi(p)

    @cartesian
    def dirichlet_rho(self, p: TensorLike) -> TensorLike:
        """Dirichlet boundary condition for ρ."""
        return self.solution_rho(p)
    @cartesian
    def dirichlet_zero(self, p: TensorLike) -> TensorLike:
        """Dirichlet boundary condition for ρ."""
        return self.solution_rho(p)*0


    @cartesian
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        """Check whether a point is on the spherical shell boundary (r ≈ r1 or r2)."""
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        r = bm.sqrt(x**2 + y**2 + z**2)
        atol = 1e-6
        return (bm.abs(r - self.r1) < atol) | (bm.abs(r - self.r2) < atol)

    @cartesian
    def coeff1(self, p):

        return 1

    @cartesian
    def coeff2(self, p):

        return -1
