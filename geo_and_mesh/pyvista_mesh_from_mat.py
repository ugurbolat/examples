# %%
import scipy.io 
import pyvista as pv
import numpy as np 
import os

# TODO read_2d_mesh_from_matfile after refactoring move to src_quasim_common/quasim_common/datasets/LaserCuttingSyntheticDataset.py
def read_mat_file_2d_mesh(filename):
    # mat_object = scipy.io.loadmat(filename)
    mat_object = scipy.io.loadmat(filename)

    # TODO adapt to pdeobj_mesh_nodes, pdeobj_mesh_elements, etc. 
    nodes = mat_object["nodes"]
    elements = mat_object["elements"] - 1 # matlab indices starts w/ 1 :/
    temp = mat_object["temperature"]
    
    # node_arr = np.array(node_list)
    # convert 2D nodes to 3D nodes by append 0 column to z-axis
    col = np.full((len(nodes),1), 0)
    nodes3D = np.append(nodes, col, axis=1)

    # REF: https://docs.pyvista.org/examples/00-load/create-poly.html
    # expected face data structure: [element_type, element_id_1, ..., element_id_n, element_type, element_id_1, ..., element_id_n]
    # where in our case element_type is triangle so element_type is 3 and there is 3 element_id
    # 1. append element_type column to first col to element_ids
    # 2. unroll 2d array to 1d
    element_type = np.full((elements.shape[0],1), 3)
    faces = np.hstack(np.append(element_type, elements, axis=1))

    #temperature 
    faces_temp = []

    for i in range(temp.shape[1]): 
        faces_t = []
        t = temp[:, i]
        for j in range(elements.shape[0]): #for each face 
            e = elements[j, :]
            avg = (t[e[0]-1]+t[e[1]-1]+t[e[2]-1])/3
            faces_t.append(avg)
        faces_temp.append(faces_t)
    temperature = np.array(faces_temp)

    return nodes3D, faces, temperature

# TODO adapt to pdeobj_mesh_nodes, pdeobj_mesh_elements, etc.
def visualize_heat_transfer_2d(matfile, is_export=False):
    nodes, faces, temp = read_mat_file_2d_mesh(matfile)

    # if os.uname()[1] == "quasim":
    #     os.environ['DISPLAY'] = ':1.0'

    mesh = pv.PolyData(nodes, faces = faces)
    mesh.cell_data["temperature"] = temp[0,:]

    plotter = pv.Plotter(notebook=False, off_screen=True)
    plotter.add_mesh(
        mesh,
        scalars="temperature", 
        cmap="fire",
        lighting=False,
        show_edges=True,
        scalar_bar_args={"title": "Temperature"},
    )

    # mkdir output if not exist
    if not os.path.exists("output"):
        os.makedirs("output")
    nframe = temp.shape[0]
    plotter.open_movie("output/simplot_2d.mp4")
    for i in range(nframe):
        mesh.cell_data["temperature"] = temp[i,:]
        # Write a frame. This triggers a render.
        if is_export:
            plotter.write_frame()
    
    plotter.close()
    
    
visualize_heat_transfer_2d("sample_data/sample_fem_prediction_2d.mat", is_export=True)
