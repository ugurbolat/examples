#chatgpt asked for the sample function from matlab
#didn't test it yet

from fenics import *
import numpy as np

# Define the domain and the mesh
mesh = UnitCubeMesh(8, 8, 8)

# Define the function space
V = FunctionSpace(mesh, "Lagrange", 1)

# Define the boundary conditions
top = CompiledSubDomain("near(x[2], 1.0) && on_boundary")
bot = CompiledSubDomain("near(x[2], 0.0) && on_boundary")
non_cut = CompiledSubDomain("!(near(x[0], 0.5) && near(x[1], 0.5)) && on_boundary")
cut = CompiledSubDomain("(near(x[0], 0.5) && near(x[1], 0.5)) && on_boundary")

bc_top = DirichletBC(V, Constant(ambient_temperature), top)
bc_bot = DirichletBC(V, Constant(ambient_temperature), bot)
bc_non_cut = DirichletBC(V, Constant(ambient_temperature), non_cut)
bc_cut_1 = DirichletBC(V, Constant(laser_temperature), cut)

bcs = [bc_top, bc_bot, bc_non_cut, bc_cut_1]

# Define the initial conditions
u_0 = Constant(ambient_temperature)

# Define the test and trial functions
u = TrialFunction(V)
v = TestFunction(V)

# Define the thermal properties
k = Constant(conductivity)
rho = Constant(density)
c = Constant(heat)

# Define the time step and the end time
dt = Constant(timestep/substeps)
T = timestep

# Define the weak form of the heat equation
F = (rho*c*dot(u-u_0, v) + dt*k*dot(grad(u), grad(v)))*dx

# Define the solution function
u = Function(V)

# Time-stepping loop
for i in range(substeps):
    # Assemble and solve the linear system
    solve(F == 0, u, bcs)
    # Update the initial condition
    u_0.assign(u)
