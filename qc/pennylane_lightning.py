import pennylane as qml
from timeit import default_timer as timer

# To set the number of threads used when executing this script,
# export the OMP_NUM_THREADS environment variable.

# Choose number of qubits (wires) and circuit layers
wires = 20
layers = 3

# Set number of runs for timing averaging
num_runs = 5

# Instantiate CPU (lightning.qubit) or GPU (lightning.gpu) device
dev = qml.device('lightning.gpu', wires=wires)


# Create QNode of device and circuit
@qml.qnode(dev, diff_method="adjoint")
def circuit(parameters):
    qml.StronglyEntanglingLayers(weights=parameters, wires=range(wires))
    return [qml.expval(qml.PauliZ(i)) for i in range(wires)]

# Set trainable parameters for calculating circuit Jacobian
shape = qml.StronglyEntanglingLayers.shape(n_layers=layers, n_wires=wires)
weights = qml.numpy.random.random(size=shape)

# Run, calculate the quantum circuit Jacobian and average the timing results
timing = []
for t in range(num_runs):
    start = timer()
    jac = qml.jacobian(circuit)(weights)
    end = timer()
    timing.append(end - start)

print(qml.numpy.mean(timing))
