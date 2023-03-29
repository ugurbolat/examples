
# %%

# Define the frequency (1 Hz)
frequency = 1

# # Define a sinusoidal function with random phase
# def sinusoidal(t, phase):
#     return np.sin(2 * np.pi * frequency * t + phase)


# %%

# generate a timeseries mesh with pyvista where each node has random node atrribute value sinusoidal distribution
# with same frequency but with different phase

import pyvista as pv
import numpy as np
# pv.set_jupyter_backend("pythreejs")
import os

import sys
import numpy as np
import torch
from torch_geometric.utils import from_networkx


# trick to run on remote server
if os.uname()[1] == "quasim":
    os.environ['DISPLAY'] = ':1.0'

np.random.seed(42)
torch.manual_seed(42)

# Define the number of time steps
n_timestep = 100
time = np.linspace(0, 1, n_timestep)


# Define the grid of points
x = np.linspace(0, 10, 50)
y = np.linspace(0, 10, 50)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

r = np.sqrt(X**1.5 + Y**1.5)
z = np.sin(r)

# Create a structured grid from the points
grid = pv.StructuredGrid(X, Y, Z)

# Triangulate the structured grid to generate a triangular mesh
mesh = grid.triangulate()
n_points = mesh.points.shape[0]

print("n_points: ", n_points)
print("n_cells: ", mesh.n_cells)

# Generate sinusodial data distribution for each node
sin_node_data = np.linspace(0, 2 * np.pi, 100)


# # calculating the values of the node attributes based on a sinusoidal distribution. 
# # The grid.points array contains the coordinates of all nodes in the mesh, 
# # and we're using each of the x, y, and z components of the node coordinates 
# # as inputs to the sinusoidal function.
# node_data = np.sin(grid.points[:, 0] * sinusoidal(time, 0)[:, None] * 2 +
#                   grid.points[:, 1] * sinusoidal(time, 0)[:, None] * 3 +
#                   grid.points[:, 2] * sinusoidal(time, 0)[:, None] * 4)

# TODO create a sinusodial temperature going from left to right or top to bottom of the mesh to test message passing

# find points that are on the left boundary of the mesh
left_boundary = np.where(mesh.points[:,0] == mesh.points[:,0].min())[0]
print("left_boundary: ", left_boundary)
# print coordinates of left boundary points
# print("left_boundary coordinates: ", mesh.points[left_boundary,:])


# node_data = np.zeros((n_timestep, n_points))
# for i in range(n_points):
#     phase = np.random.uniform(0, 2 * np.pi)
#     node_data[:,i] = sinusoidal(time, phase)
#     # nx.set_node_attributes(G, {node: attribute_values.tolist()}, f'node_{node}')

phases = np.linspace(0, 2 * np.pi, n_timestep + 1)

node_data = np.zeros((n_timestep, n_points))
# pts = mesh.points.copy()

random_phase_init = np.random.uniform(0, 2 * np.pi)

for i, phase in enumerate(phases[:n_timestep]):
    z = np.sin(r + phase + random_phase_init)
    node_data[i, :] = z.ravel()
    

mesh.point_data["sin_data"] = node_data[0,:]
mesh.point_data_to_cell_data()


plotter = pv.Plotter(notebook=False, off_screen=True)
plotter.open_movie("random_sinusoidal_graph.mp4")
# plotter.set_background("white")
plotter.add_mesh(
    mesh,
    scalars="sin_data", 
    cmap="fire",
    lighting=False,
    show_edges=True,
    clim=[0, 1],
    scalar_bar_args={"title": "Random Sinusodial Temperature"},
)

plotter.show(auto_close=False)  # only necessary for an off-screen movie

# write first frame w/ gt temperature
plotter.write_frame()

nframe = node_data.shape[0]
for i in range(nframe):
    # mesh.cell_data["data"] = temp[i,:]
    mesh.point_data["sin_data"] = node_data[i,:]
    # mesh.point_data["sin_data"][left_boundary] = 1
    mesh.point_data_to_cell_data()
    # Write a frame. This triggers a render.
    plotter.write_frame()

plotter.close()


# sys.exit()

import networkx as nx

# Create a NetworkX graph
basegraph = nx.Graph()

# convert pyvista mesh to networkx graph

# Add nodes to the graph
# for i, point in enumerate(mesh.points):
    # basegraph.add_node(i, pos=point, node_data=mesh.point_data["sin_data"][i])
    # basegraph.add_node(i, pos=point)
basegraph.add_nodes_from(range(0, mesh.points.shape[0]))

cells_matrix = mesh.cells.reshape(-1, 4)
# remove first column of cells matrix
cells_matrix = cells_matrix[:,1:]

# Add edges to the graph
for cell in cells_matrix:
    basegraph.add_edge(cell[0], cell[1])
    basegraph.add_edge(cell[1], cell[2])
    basegraph.add_edge(cell[2], cell[0])

node_coord_attributes = {}

for i in basegraph.nodes:
    i = i - 1
    node_coord_attributes[i + 1] = {
        # "convection": node_has_convection[i], # {0, 1} = {T,F}
        "worldX": mesh.points[i, 0],
        "worldY": mesh.points[i, 1],
        "worldZ": mesh.points[i, 2],
    }
    
nx.set_node_attributes(basegraph, node_coord_attributes)

edge_attributes = {}
for e in basegraph.edges:
    node1 = mesh.points[e[0] - 1, :]
    node2 = mesh.points[e[1] - 1, :]
    delta = node1 - node2
    euclidean_dist = np.sqrt(np.sum(delta ** 2))
    edge_attributes[e] = {"deltaX": delta[0], "deltaY": delta[1], "deltaZ": delta[2], "deltaAbs": euclidean_dist}
nx.set_edge_attributes(basegraph, edge_attributes)


node_data = node_data.T

sim_data_list = []
for t in range(n_timestep-1):
    # step = t // substeps + 1
    graph_at_timestep = basegraph.copy()
    timestep_attributes = {}
    for i in basegraph.nodes:
        i = i - 1
        timestep_attributes[i + 1] = {
            "initial_temperature": node_data[i, t], # [300, 500]
        }

    labels = np.zeros((len(basegraph.nodes), 1))
    labels[:, 0] = node_data[:, t + 1]
    # labels[:, 1:] = flux[:, t + 1, :]
    labels = torch.Tensor(labels)
    nx.set_node_attributes(graph_at_timestep, timestep_attributes)
    pyg_data = from_networkx(graph_at_timestep, group_node_attrs=all, group_edge_attrs=all)
    pyg_data.y = labels
    pyg_data.y = pyg_data.y.float()
    pyg_data.x = pyg_data.x.float()
    pyg_data.edge_attr = pyg_data.edge_attr.float()
    sim_data_list.append(pyg_data)

print(sim_data_list[0])

import matplotlib.pyplot as plt
# Plot the sinusoidal attributes of a few nodes
for i in range(10):
    plt.plot(time, node_data[:,i])

plt.show()

# # Visualize the mesh
# mesh.plot(show_edges=True)


# %%




