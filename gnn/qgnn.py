

n_layers = 4 if test else 4


# %%
device_backend = "default"

# how many qubits we use to encode per feature
# for ex: we can encode one feature to 4 dim. hilbert w/ 2 qubits
n_qubits_per_feature = 1  # NOTE if you wanna change it, first complete u_decoding()

# the size of the feature vector, for ex: currently we have only temperature as feature
# TODO get it from dataset's graph node feature size
n_features = 4

# maximum number of edges for the worst case node that has the highest connectivity
# for ex: in a mesh with triangle element, we will have 6 edges 
# 7 for the first two simulations
# 10 for the first 4 simulations
n_neighbors_worst_case = 7 if test else 7

# number of qubits need for the worst case
n_qubits_for_worst_case = n_neighbors_worst_case * \
    n_qubits_per_feature + n_qubits_per_feature

# number of steps of message passing
mp_steps = 1 if test else 2

n_qubits_per_node = n_qubits_per_feature * n_features

from pennylane import numpy as npq


# %%
def u_encoding_classical_to_quantum(
    rel_temp: npq.ndarray,
    graph: nx.Graph,
    node: int,
):

    nodes = [node]
    node_neighbours = list(graph.neighbors(node))
    nodes += node_neighbours
    n_qubits_for_current_node = len(node_neighbours) + 1
    # NOTE shared params
    # n_qubits_for_worst_case = 10
    
    # if n_qubits_for_current_node > 7:
    #     print(n_qubits_for_current_node)
    #     print('yo! interesting node')

    for node, wire in zip(nodes, range(n_qubits_for_current_node)):
        qml.RY(npq.pi*rel_temp[node], wires=wire)
    # disabling unused qubits
    if n_qubits_for_current_node < n_qubits_for_worst_case:
        for wire in range(n_qubits_for_current_node, n_qubits_for_worst_case):
            qml.RY(0, wires=wire)


def u_enconding_feature_layer(
    layer_idx: int,
    node_thetas: npq.ndarray,
    graph: nx.Graph,
    node: int
):

    node_neighbors = list(graph.neighbors(node))
    n_neighbors = len(node_neighbors)
    n_qubits_for_current_node = n_neighbors + 1
    # NOTE shared params
    # n_qubits_for_worst_case = 10

    # TODO instead of rotation always around x, we perform rotation around the axis of x,y,z in alternating fashion
    if n_features == 1 and n_qubits_per_feature == 1:
        for wire in range(n_qubits_for_current_node):
            if layer_idx%3 == 0:
                qml.RX(node_thetas[0], wires=wire)
            if layer_idx%3 == 1:
                qml.RY(node_thetas[0], wires=wire)
            if layer_idx%3 == 2:
                qml.RZ(node_thetas[0], wires=wire)
            # qml.RX(node_thetas[0], wires=wire) # sharing thetas instead of node_thetas[wire]
            # TODO
            # qml.Kerr(node_thetas[0], wires=wire)
        # disabling unused qubits
        if n_qubits_for_current_node < n_qubits_for_worst_case:
            for wire in range(n_qubits_for_current_node, n_qubits_for_worst_case):
                # qml.RX(0, wires=wire)
                if layer_idx%3 == 0:
                    qml.RX(0, wires=wire)
                if layer_idx%3 == 1:
                    qml.RY(0, wires=wire)
                if layer_idx%3 == 2:
                    qml.RZ(0, wires=wire)
    else:
        #TODO generalization to multi-qubit per feature/multi-qubit for feature vector
        asd = []


def u_message_passing_layer(
    edge_thetas: npq.ndarray,
    graph: nx.Graph,
    node: int,
):

    node_neighbors = list(graph.neighbors(node))
    n_neighbors = len(node_neighbors)
    n_qubits_for_current_node = n_neighbors + 1
    # NOTE shared params
    # n_qubits_for_worst_case = 10
    edges = [(0, x + 1) for x in range(len(node_neighbors))]

    if n_features == 1 and n_qubits_per_feature == 1:
        # ZZ rotations
        for edge in edges:
            wire = edge[1]
            qml.CNOT(wires=[0, wire])
            qml.RZ(edge_thetas[0], wires=wire) # shared params instead of edge_thetas[wire - 1]
            qml.CNOT(wires=[0, wire])
        # disable unused rotations
        if n_qubits_for_current_node < n_qubits_for_worst_case:
            for edge in range(n_qubits_for_current_node, n_qubits_for_worst_case):
                wire = edge
                qml.CNOT(wires=[0, wire])
                qml.RZ(0, wires=wire)
                qml.CNOT(wires=[0, wire])
        
        # RX rotations
        for wire in range(n_qubits_for_current_node):
            # qml.RX(2*npq.pi * npq.random.rand(), wires=wire)
            qml.RX(edge_thetas[1], wires=wire)
            # qml.RX(node_thetas[0], wires=wire) # sharing thetas instead of node_thetas[wire]
            # TODO
            # qml.Kerr(node_thetas[0], wires=wire)
        # disabling unused qubits
        if n_qubits_for_current_node < n_qubits_for_worst_case:
            for wire in range(n_qubits_for_current_node, n_qubits_for_worst_case):
                qml.RX(0, wires=wire)
        
        
    else:
        # TODO generalization to multi-qubit per feature/multi-qubit for feature vector
        asd=[]


def u_decoding_layer(
    layer_idx: int,
        node_thetas: npq.ndarray,
        graph: nx.Graph,
        node: int,
):

    node_neighbors = list(graph.neighbors(node))
    n_neighbors = len(node_neighbors)
    n_qubits_for_current_node = n_neighbors + 1
    
    if n_features == 1 and n_qubits_per_feature == 1:
        for wire in range(n_features*n_qubits_per_feature):
            if layer_idx%3 == 0:
                qml.RX(node_thetas[0], wires=wire)
            if layer_idx%3 == 1:
                qml.RY(node_thetas[0], wires=wire)
            if layer_idx%3 == 2:
                qml.RZ(node_thetas[0], wires=wire)
            # qml.RX(node_thetas[0], wires=wire)
    else:
        # TODO generalization to multi-qubit per feature/multi-qubit for feature vector
        asd=[]


# %%


def generate_initial_thetas_at_node(
    # node: int,
    # graph: nx.Graph,
    n_qubits_for_worst_case: int, 
    n_features: int,
    n_qubits_per_feature: int,
    n_layers: int,
    init_zero: bool = False,
) -> Tuple[npq.ndarray, npq.ndarray, npq.ndarray]:
    """
    The first n_quantum columns are node parameters and the others edge parameters
    """
    # NOTE shared params
    # n_neighbors = len(list(graph.neighbors(node)))

    n_qubits_per_node = n_qubits_per_feature * n_features
    npq.random.seed(42)
    if init_zero:
        # thetas = 2*npq.pi*npq.zeros((n_layers, 2*n_neighbors + 1)) # NOTE that this is NOT optimized
        thetas_encoder = 2*npq.pi * \
            npq.zeros((n_layers, n_qubits_for_worst_case))
        thetas_message_passing = 2*npq.pi * \
            npq.zeros((n_layers, n_neighbors_worst_case))
        thetas_decoder = 2*npq.pi*npq.zeros((n_layers, n_qubits_per_node))
    else:
        # NOTE shared enconder but not node update/mess. passing
        # thetas = 2*npq.pi*npq.random.rand(n_layers, 2*n_neighbors + 1, requires_grad=True)
        if n_features == 1 and n_qubits_per_feature == 1:
            # just parameterized rotations case (nothing to entangle)
            thetas_encoder = 2*npq.pi * \
                npq.random.rand(n_layers, 1,
                                requires_grad=True)
            thetas_message_passing = 2*npq.pi * \
                npq.random.rand(n_layers, 2,
                                requires_grad=True)
            thetas_decoder = 2*npq.pi * \
                npq.random.rand(n_layers, 1, requires_grad=True)
        else: # multi-qubit generalization
            # rotate and entangle case
            thetas_encoder = 2*npq.pi * \
                npq.random.rand(n_layers, 
                                # (first part) + (second part) -> (rotation) + (entangled rotation)
                                n_features*n_qubits_per_feature+(n_features-1)*n_qubits_per_feature,
                                requires_grad=True)
            # TODO generalization to multi-qubit per feature/multi-qubit for feature vector
            thetas_message_passing = 2*npq.pi * \
                npq.random.rand(n_layers, n_features*n_qubits_per_feature*2,
                                requires_grad=True)
            # TODO generalization to multi-qubit per feature/multi-qubit for feature vector
            # only decoding and reading out central node
            thetas_decoder = 2*npq.pi * \
                npq.random.rand(n_layers, n_qubits_per_node, requires_grad=True)       
            
    return thetas_encoder, thetas_message_passing, thetas_decoder


# %% 

# NOTE shared params
# thetas_init = generate_initial_thetas(nodes, my_graph, n_layers)
thetas_encoder, thetas_message_passing, thetas_decoder = generate_initial_thetas_at_node(n_qubits_for_worst_case, n_features, n_qubits_per_feature, n_layers)

# print(thetas_init.shape)

# %%%

# %%

if device_backend == "qiskit":
    dev = qml.device("qiskit.aer", wires=n_qubits_for_worst_case)
else:
    dev = qml.device("default.qubit", wires=n_qubits_for_worst_case)
    # TODO throwing this error: pennylane.wires.WireError: Wire with label 4 not found in <Wires = [0, 1, 2, 3]>.
    #dev = qml.device("lightning.qubit", wires=n_qubits_for_current_node)
@qml.qnode(dev, diff_method="adjoint")
def circuit(
    # thetas : npq.ndarray,
    thetas_encoder : npq.ndarray, 
    thetas_message_passing : npq.ndarray, 
    thetas_decoder : npq.ndarray,
    **kwargs
):     
    # if isinstance(thetas, tuple):
    #     thetas = thetas[0]
    graph = kwargs["graph"]
    node = kwargs["node"]
    rel_temp = kwargs["rel_temp"]

    u_encoding_classical_to_quantum(rel_temp, graph, node)

    # # testing shared params if qnode is ok w/ it
    # theta_flat = all_thetas.flatten()
    # for i in range(len(theta_flat)):
    #     qml.RX(theta_flat[i], wires=0)
    # for i in range(len(theta_flat)):
    #     qml.RX(theta_flat[i], wires=0)

    # TODO generalize multi-qubit
    # thetas_encoder = thetas[:,0].reshape(len(thetas),1)
    # enconding features to high dim. space
    for layer in range(n_layers):
        u_enconding_feature_layer(layer,
                                    thetas_encoder[layer, :],
                                    graph, node)

    # thetas_message_passing = thetas[:,1].reshape(len(thetas),1)
    
    # message passing
    for _ in range(mp_steps):
        # NOTE sharing params w=
        for layer in range(n_layers):
            u_message_passing_layer(thetas_message_passing[layer, :],
                                    graph, node)

    # TODO generaize multi-qubit

    # thetas_decoder = thetas[:,2].reshape(len(thetas),1)

    for layer in range(n_layers):
        u_decoding_layer(
            layer,
            thetas_decoder[layer, :],
                                        graph, node)
        
    # measurement
    return qml.expval(qml.PauliZ(0))




    output = circuit(
        # thetas,
        thetas_encoder, thetas_message_passing, thetas_decoder,
                     graph=graph,
                     rel_temp=rel_temp,
                     node=node)
    
    if draw_circut_during_training:
        qc_drawer = qml.draw(circuit, max_length=1000)
        print(qc_drawer(
        # thetas,
        thetas_encoder, thetas_message_passing, thetas_decoder,
                    graph=graph,
                    rel_temp=rel_temp,
                    node=node))
        print("\n")

    # unnormalize
    new_z_avg = output
    new_rel_temp = npq.arccos(new_z_avg)/npq.pi
