# check out the link for packages to install: https://github.com/FEniCS/dolfinx
# see conda env section

# mesh generation
from mpi4py import MPI
from dolfinx import mesh
domain = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8, mesh.CellType.triangle)

tdim = domain.topology.dim
fdim = tdim - 1

# generate mesh with rectangle cell type
  


# TODO re-meshing (w/o learning): the result should be coaser mesh


# visualization of mesh
from dolfinx import plot
import pyvista
topology, cell_types, geometry = plot.create_vtk_mesh(domain, tdim)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)


pyvista.set_jupyter_backend("pythreejs")

plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True)
plotter.view_xy()
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    pyvista.start_xvfb()
    figure = plotter.screenshot("fundamentals_mesh.png")
