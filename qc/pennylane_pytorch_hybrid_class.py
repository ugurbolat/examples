import numpy as np
from sklearn.datasets import make_moons
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import pennylane as qml
import sys
from time import perf_counter

class Model(nn.Module):
    def __init__(self, dev, diff_method="backprop", torch_device="cpu"):
        super().__init__()

        self.cnet_in = self.cnet()
        self.qcircuit = qml.qnode(dev, interface="torch", 
                                  diff_method=diff_method)(self.qnode)
        
        weight_shape = {"weights":(2,)}
        self.qlayer = qml.qnn.TorchLayer(self.qcircuit, weight_shape)
        self.cnet_out = self.cnet()

    def cnet(self):
        layers = [nn.Linear(2,10, device=torch_device), nn.ReLU(True), nn.Linear(10,2, device=torch_device), nn.Tanh()]
        return nn.Sequential(*layers)   

    def qnode(self, inputs, weights):
        # Data encoding:
        for x in range(len(inputs)):
            qml.Hadamard(x)
            qml.RZ(2.0 * inputs[x], wires=x)
        # Trainable part:
        qml.CNOT(wires=[0,1])
        qml.RY(weights[0], wires=0)
        qml.RY(weights[1], wires=1)
        return [qml.expval(qml.PauliZ(wires=0)), qml.expval(qml.PauliZ(wires=1))]

    def forward(self, x):
        x1 = self.cnet_in(x)
        x2 = self.qlayer(x1)
        x_output = self.cnet_out(x2)
        return x_output

def train(X, y_hot, dev_name, diff_method, torch_device, n_qubits=2):
    
    dev = qml.device(dev_name, wires=n_qubits, shots=None)
    model  = Model(dev, diff_method, torch_device)
    
    # Train the model
    opt = torch.optim.SGD(model.parameters(), lr=0.2)
    loss = torch.nn.L1Loss()

    X = torch.tensor(X, requires_grad=False, device=torch_device).float()
    y_hot = y_hot.float().to(torch_device)

    batch_size = 5
    batches = 200 // batch_size

    data_loader = torch.utils.data.DataLoader(
        list(zip(X, y_hot)), batch_size=batch_size, shuffle=True, drop_last=True
    )

    epochs = 26

    for epoch in range(epochs):

        running_loss = 0

        for xs, ys in data_loader:
            opt.zero_grad()

            loss_evaluated = loss(model(xs), ys)
            loss_evaluated.backward()

            opt.step()

            running_loss += loss_evaluated

        avg_loss = running_loss / batches
        print("Average loss over epoch {}: {:.4f}".format(epoch + 1, avg_loss))

    y_pred = model(X)
    predictions = torch.argmax(y_pred, axis=1).detach().numpy()

    correct = [1 if p == p_true else 0 for p, p_true in zip(predictions, y)]
    accuracy = sum(correct) / len(correct)
    print(f"Accuracy: {accuracy * 100}%")

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    X, y = make_moons(n_samples=200, noise=0.1)
    # torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    y_ = torch.unsqueeze(torch.tensor(y), 1)  # used for one-hot encoded labels
    y_hot = torch.scatter(torch.zeros((200, 2)), 1, y_, 1)
    begin_time = perf_counter()
    # train(X, y_hot, str(sys.argv[1]), str(sys.argv[2]))
    # X = torch.tensor(X, device="cuda", requires_grad=False).float()
    # max number of qubits for cuda is 24 (takes around 13gb of memory)
    # max number of qubits for cpu is 26 (takes around 47gb of memory)
    n_qubits = 26 
    torch_device = "cpu"
    train(X, y_hot, "default.qubit", "backprop", torch_device, n_qubits)
    end_time = perf_counter()
    runtime = end_time-begin_time
    print(f'Runtime: {runtime:.2e} s or {(runtime/60):.2e} min.')