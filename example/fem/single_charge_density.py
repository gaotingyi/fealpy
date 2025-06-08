from fealpy.backend import backend_manager as bm
from fealpy.mesh import TetrahedronMesh
#from fealpy.pde.singlechargedensity import SingleChargeDensity
from fealpy.fem import GradSourceIntegrator, ScalarSourceIntegrator 
from fealpy.fem import BilinearForm, LinearForm
from fealpy.fem import ScalarMassIntegrator, ScalarDiffusionIntegrator
from fealpy.fem.scalar_nonlinear_mass_diffusion_integrator import ScalarNonlinearMassAndDiffusionIntegrator
#from fealpy.fem import 
from fealpy.functionspace import LagrangeFESpace
from fealpy.decorator import cartesian, barycentric
from fealpy.model import PDEDataManager


backend = 'pytorch'                                                             
device = 'cpu'                                                                 
bm.set_backend(backend)                                                         
bm.set_default_device(device)    
def coeff1(p, **args):
    return pde.coeff1(p)

def coeff2(p, **args):
    return pde.coeff2(p)


n = 1
p = 1
mesh = TetrahedronMesh.from_spherical_shell(r1=0.05,r2=0.5,device=device)
space = LagrangeFESpace(mesh, p=p)
pde = PDEDataManager('nonlinear').get_example('single')
# 刚度矩阵
bform = BilinearForm(space)
integrator1 = ScalarDiffusionIntegrator(coef=1, q=p+1)
bform.add_integrator(integrator1)
D = bform.assembly()
# 质量矩阵
bform = BilinearForm(space)
integrator2 = ScalarMassIntegrator(coef=1, q=p+1)
bform.add_integrator(integrator2)
M = bform.assembly()
# 右端项
lform = LinearForm(space)
integrator3 = ScalarSourceIntegrator(pde.source_f1, q=p+1)
lform.add_integrator(integrator3)
f1 = lform.assembly()

lform = LinearForm(space)
integrator4 = ScalarSourceIntegrator(pde.source_f2, q=p+1)
lform.add_integrator(integrator4)
f2 = lform.assembly()

phi0 = space.function()
rho0 = space.function()
ipoints = space.interpolation_points()
phi0[:] = pde.init_phi(ipoints)
rho0[:] = pde.init_rho(ipoints)
coeff1.phi_h = phi0
coeff2.rho_h = rho0
bform = BilinearForm(space)
integrator5 = ScalarNonlinearMassAndDiffusionIntegrator(coef1=coeff1,coef2=coeff2, grad_var=0, q=p+3)
bform.add_integrator(integrator5)
A, F = bform.assembly()

bform = BilinearForm(space)
integrator6 = ScalarNonlinearMassAndDiffusionIntegrator(coef1=coeff1,coef2=coeff2, grad_var=1, q=p+3)
bform.add_integrator(integrator5)
A, F = bform.assembly()


