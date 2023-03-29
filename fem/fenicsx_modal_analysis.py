# REF: https://fenicsproject.discourse.group/t/modal-analysis-using-dolfin-x/7349

# Imports
import numpy as np
import ufl
import sys, slepc4py
slepc4py.init(sys.argv)

from dolfinx import fem
from dolfinx.fem import (Constant, dirichletbc, Function, VectorFunctionSpace,
        locate_dofs_topological)
from dolfinx.mesh import (create_box, meshtags, locate_entities, CellType,
        locate_entities_boundary)
from dolfinx.io import XDMFFile
from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc

# Define computational domain
L = np.array([5, 0.6, 0.4])
N = [25, 3, 2]
mesh = create_box(MPI.COMM_WORLD, [np.array([0,0,0]), L], N,
        cell_type=CellType.hexahedron)
        
# Material constants
E, nu = (2e11), (0.3)  
rho = (7850) 
mu = Constant(mesh, E/2./(1+nu))
lambda_ = Constant(mesh, E*nu/(1+nu)/(1-2*nu))

# Convenience functions
def epsilon(u):
    return ufl.sym(ufl.grad(u))
def sigma(u):
    # return lambda_ * ufl.nabla_div(u) * ufl.Identity(u.geometric_dimension()) + 2*mu*epsilon(u)
    return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2*mu*epsilon(u)

# Define function space, trial and test functions
V = VectorFunctionSpace(mesh, ("CG", 1))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Define Dirichlet boundary condition
def clamped_boundary(x):
    return np.isclose(x[0], 0)

fdim = mesh.topology.dim - 1
boundary_facets = locate_entities_boundary(mesh, fdim, clamped_boundary)

u_D = Function(V)
u_D.interpolate(lambda x: np.zeros_like(x))
bc = dirichletbc(u_D, locate_dofs_topological(V, fdim, boundary_facets))

# Define variational form
k_form = ufl.inner(sigma(u),epsilon(v))*ufl.dx
m_form = rho*ufl.inner(u,v)*ufl.dx

# Assemble stiffness and mass matrices
#
# Using the "diagonal" kwarg ensures that Dirichlet BC modes will not be among
# the lowest-frequency modes of the beam. 
K = fem.petsc.assemble_matrix(fem.form(k_form), bcs=[bc], diagonal=62831)
M = fem.petsc.assemble_matrix(fem.form(m_form), bcs=[bc], diagonal=1/62831)
K.assemble()
M.assemble()

# Create and configure eigenvalue solver
N_eig = 6
eigensolver = SLEPc.EPS().create(MPI.COMM_WORLD)
eigensolver.setDimensions(N_eig)
eigensolver.setProblemType(SLEPc.EPS.ProblemType.GHEP)
st = SLEPc.ST().create(MPI.COMM_WORLD)
st.setType(SLEPc.ST.Type.SINVERT)
st.setShift(0.1)
st.setFromOptions()
eigensolver.setST(st)
eigensolver.setOperators(K, M)
eigensolver.setFromOptions()

# Compute eigenvalue-eigenvector pairs
eigensolver.solve()
evs = eigensolver.getConverged()
vr, vi = K.getVecs()
u_output = Function(V)
u_output.name = "Eigenvector"
print( "Number of converged eigenpairs %d" % evs )
if evs > 0:
    with XDMFFile(MPI.COMM_WORLD, "eigenvectors.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        for i in range (min(N_eig, evs)):
            l = eigensolver.getEigenpair(i, vr, vi)
            freq = np.sqrt(l.real)/2/np.pi
            print(f"Mode {i}: {freq} Hz")
            u_output.x.array[:] = vr
            xdmf.write_function(u_output, i)