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
from fealpy.fem import LinearBlockForm, BlockForm, NonlinearForm
from fealpy.fem import DirichletBC

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
integrator1 = ScalarDiffusionIntegrator(coef=1, q=p+3)
bform.add_integrator(integrator1)
D = bform.assembly()
# 质量矩阵
bform = BilinearForm(space)
integrator2 = ScalarMassIntegrator(coef=-1, q=p+3)
bform.add_integrator(integrator2)
M = bform.assembly()

# 右端项
lform = LinearForm(space)
integrator3 = ScalarSourceIntegrator(pde.source_f1, q=p+3)
lform.add_integrator(integrator3)
f1 = lform.assembly()

lform = LinearForm(space)
integrator4 = ScalarSourceIntegrator(pde.source_f2, q=p+3)
lform.add_integrator(integrator4)
f2 = lform.assembly()

phi0 = space.interpolate(pde.init_phi)
rho0 = space.interpolate(pde.init_rho)

'''
f3 = D@phi0[:]
f4 = M@rho0[:]
f1 = f3-f4-f1
coeff1.phi_h = phi0
coeff2.rho_h = rho0
bform = NonlinearForm(space)
integrator5 = ScalarNonlinearMassAndDiffusionIntegrator(coef1=coeff1,coef2=coeff2, grad_var=0, q=p+3)
bform.add_integrator(integrator5)
A1, F1 = bform.assembly()

bform = NonlinearForm(space)
integrator6 = ScalarNonlinearMassAndDiffusionIntegrator(coef1=coeff1,coef2=coeff2, grad_var=1, q=p+3)
bform.add_integrator(integrator5)
A2, F2 = bform.assembly()
f2 = F1 - F2 - f2
C = BlockForm([[D, M],[A1, A2]])
C = C.assembly()

B = LinearBlockForm([f1,f2])
B = B.assembly()
A, b = BC.apply(C, B)
'''
BC = DirichletBC((space, space), gd=(pde.dirichlet_zero, pde.dirichlet_zero), 
                      threshold=(None, None), method='interp')
gdof = space.number_of_global_dofs()

for i in range(30):
    
    f3 = D@phi0[:]
    f4 = M@rho0[:]
    f5 = -(f3-f4-f1)
    coeff1.phi_h = phi0
    coeff2.rho_h = rho0
    bform = NonlinearForm(space)
    integrator5 = ScalarNonlinearMassAndDiffusionIntegrator(coef1=coeff1,coef2=coeff2, grad_var=0, q=p+3)
    bform.add_integrator(integrator5)
    A1, F1 = bform.assembly()

    bform = NonlinearForm(space)
    integrator6 = ScalarNonlinearMassAndDiffusionIntegrator(coef1=coeff1,coef2=coeff2, grad_var=1, q=p+3)
    bform.add_integrator(integrator6)
    A2, F2 = bform.assembly()
    f6 = F2  + f2
    C = BlockForm([[D, M],[A1, A2]])
    C = C.assembly()

    B = LinearBlockForm([f5,f6])
    B = B.assembly()

    A, b = BC.apply(C, B)

    phi0[:] += b[:gdof]
    rho0[:] += b[gdof:]
    print(bm.max(bm.abs(b[:gdof])))
    print(bm.max(bm.abs(b[gdof:])))


rhoso = space.interpolate(pde.solution_rho)
phiso = space.interpolate(pde.solution_phi)
phierror = mesh.error(pde.solution_phi, phi0.value)
rhoerror = mesh.error(pde.solution_rho, rho0.value)
gphierror = mesh.error(pde.gradient_phi, phi0.grad_value)
grhoerror = mesh.error(pde.gradient_rho, rho0.grad_value)

print(phierror)
print(rhoerror)
print(gphierror)
print(grhoerror)
