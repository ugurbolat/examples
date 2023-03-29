# %%

from IPython.display import clear_output
import cv2
import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

import pennylane as qml
from pennylane import numpy as npq
from pennylane.optimize import AdamOptimizer

from pennylane.templates import RandomLayers, CVNeuralNetLayers

from prettytable import PrettyTable
import sys

# if output directory does not exist, create it
import os
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


torch.manual_seed(0)
np.random.seed(0)
npq.random.seed(0)

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
    

# # Generating toy data
# x_data = np.linspace(0, 2 * np.pi, 1000)
# y_data = np.sin(x_data)

# # Converting data to PyTorch tensors
# x = torch.tensor(x_data, dtype=torch.float32).unsqueeze(1)
# y = torch.tensor(y_data, dtype=torch.float32).unsqueeze(1)

# dataset parameters
num_samples_in_sin = 20
num_samples_in_dataset = 50
# noise = [0.01, 0.01]
noise = [0.0, 0.01]
batch_size = 32
train_size = int(0.8 * num_samples_in_dataset)
test_size = num_samples_in_dataset - train_size

# model parameters
model_input_size = num_samples_in_sin
hidden_size = 16
model_output_size = num_samples_in_sin

# training parameters
epochs = 2000
# epochs = 2
is_plot_all_test_samples = False

epochs_qnn = 200

dev = "default.qubit"

class TwoSinusoidalDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples_in_sin, num_samples, noise=[0.01, 0.0001]):
        self.num_samples = num_samples
        self.num_samples_in_sin = num_samples_in_sin
        self.noise = noise
        self.x_gt = torch.linspace(0, 2 * np.pi, self.num_samples_in_sin)
        self.y_gt = torch.sin(self.x_gt) + torch.sin(self.x_gt * 5)
        self.x_gt = torch.linspace(0, 2 * np.pi, self.num_samples_in_sin) / (2 * np.pi)
        self.y = torch.zeros((self.num_samples, self.num_samples_in_sin))
        self.x = torch.zeros((self.num_samples, self.num_samples_in_sin))
        for i in range(self.num_samples):
            self.y[i, :] = self.y_gt
            # add noise to y
            self.y[i, :] += torch.randn(self.num_samples_in_sin) * self.noise[1]
            self.x[i, :] = self.x_gt + torch.randn(self.num_samples_in_sin) * self.noise[0]
            
    def __getitem__(self, index):
        return self.x[index,:].view(-1,1), self.y[index,:].view(-1,1)

    def __len__(self):
        return self.num_samples

class OneSinusoidalDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples_in_sin, num_samples, noise=[0.01, 0.0001]):
        self.num_samples = num_samples
        self.num_samples_in_sin = num_samples_in_sin
        self.noise = noise
        self.x_gt = torch.linspace(0, 2 * np.pi, self.num_samples_in_sin)
        self.y_gt = torch.sin(self.x_gt)
        self.x_gt = torch.linspace(0, 2 * np.pi, self.num_samples_in_sin) / (2 * np.pi)
        self.y = torch.zeros((self.num_samples, self.num_samples_in_sin))
        self.x = torch.zeros((self.num_samples, self.num_samples_in_sin))
        for i in range(self.num_samples):
            self.y[i, :] = self.y_gt
            # add noise to y
            self.y[i, :] += torch.randn(self.num_samples_in_sin) * self.noise[1]
            self.x[i, :] = self.x_gt + torch.randn(self.num_samples_in_sin) * self.noise[0]
            
    def __getitem__(self, index):
        return self.x[index,:].view(-1,1), self.y[index,:].view(-1,1)

    def __len__(self):
        return self.num_samples


# dataset = TwoSinusoidalDataset(num_samples_in_sin, num_samples_in_dataset, noise)
dataset = OneSinusoidalDataset(num_samples_in_sin, num_samples_in_dataset, noise)

# print the shape of the first sample x and y
# print("dataset 0 ", dataset[0])
print("x shape: ", dataset[0][0].shape)
print("y shape: ", dataset[0][1].shape)

# print("dataset 1 ", dataset[1])
print("x shape: ", dataset[1][0].shape)
print("y shape: ", dataset[1][1].shape)


# # plot the first 2 sin curves
sin_x = dataset[0][0]
sin_y = dataset[0][1]
plt.plot(sin_x, sin_y)
plt.title("Temperature at node i")
# axis labels
plt.xlabel("time")
plt.ylabel("temperature")
# draw circle at each data point
plt.scatter(sin_x, sin_y)
plt.show()
sin_x = dataset[1][0]
sin_y = dataset[1][1]
plt.plot(sin_x, sin_y, label="sin 1")
plt.show()

sys.exit()

# split the dataset into train and test 80/20
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# train and test dataloader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Defining the neural network model
class Net(torch.nn.Module):
    def __init__(self, input_size=1000, hidden_size=16, output_size=1000):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(1, hidden_size)
        self.hidden_mid = torch.nn.Linear(hidden_size, hidden_size)
        # self.hidden_mid_2 = torch.nn.Linear(hidden_size, hidden_size)
        # self.hidden_mid_3 = torch.nn.Linear(hidden_size, hidden_size)

        self.output = torch.nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(1)
            
        # without non-linearity
        x = self.hidden(x) 
        x = self.hidden_mid(x)
        
        # x = torch.nn.functional.relu(self.hidden(x))
        # x = torch.nn.functional.relu(self.hidden_mid(x))
        # x = torch.nn.functional.relu(self.hidden_mid_2(x))
        # x = torch.nn.functional.relu(self.hidden_mid_3(x))
        x = self.output(x)
        return x


# %%%%%%%%%%%%%%


# TODO
# num_qubits = 4

# dev = qml.device('default.qubit', wires=4)

# @qml.qnode(dev,interface="torch",diff_method="backprop")
# def vqc_embedding(inputs,weights):
#     # num_qubits = weights.shape[1]

#     qml.AngleEmbedding(features=inputs, wires=range(num_qubits), rotation='Y')

#     qml.BasicEntanglerLayers(weights=weights, wires=range(num_qubits), rotation=qml.RX)
    
#     results = []
#     for i in range(num_qubits):
#         results.append(qml.expval(qml.PauliZ(i)))
#     return results

# def normalizeData(data,columns,min_values, max_values):
#     ind = 0
#     for i in columns:
#         data.x[:,i] = (data.x[:,i]-min_values[ind])/(max_values[ind]-min_values[ind])*3.141592653589
#         ind += 1


# class VQC(nn.Module):
#     def __init__(self, 
#                  num_qubits=4,
#                  num_output_features=1, 
#                  quantum_layers=2, 
#                  norm_parameters=None,
#                  ):
#         super().__init__()
#         # self.conv1 = convolution_layer(num_node_features, num_node_features*2)
#         # self.conv2 = convolution_layer(num_node_features*2, num_node_features*4)

#         # self.graph_classification = graph_classification
        
#         # self.decoder = nn.Linear(num_node_features*4,num_output_features)
#         # self.norm_parameters = norm_parameters

#         weight_shapes = {"weights": (quantum_layers,len(norm_parameters[0]))}

#         self.quantum_embedding = qml.qnn.TorchLayer(vqc_embedding, weight_shapes)

#     def forward(self, data):
#         normalizeData(data,self.norm_parameters[0],self.norm_parameters[1], self.norm_parameters[2])
#         data.x[:,self.norm_parameters[0]] = self.quantum_embedding(data.x[:,self.norm_parameters[0]])
#         # x, edge_index, batch = data.x, data.edge_index, data.batch
#         # x = self.conv1(x, edge_index)
#         # x = F.relu(x)
#         # x = self.conv2(x, edge_index)
#         # x = F.relu(x)
#         # if self.graph_classification:
#         #     x = global_mean_pool(x,batch)
#         # x = F.dropout(x, p=0.2, training=self.training)
#         # x = self.decoder(x)
#         return x
    
#     def reset_parameters(self):
#         self.quantum_embedding.reset_parameters()
#         # self.decoder.reset_parameters()
#         # self.conv1.reset_parameters()
#         # self.conv2.reset_parameters()


# TODO
# class Hybrid(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, dev, diff_method="backprop", torch_device="cpu"):
#         super().__init__()

#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         # self.diff_method = diff_method
#         self.torch_device = torch_device

#         self.cnet_in = self.cnet_in_layer()
#         self.qcircuit = qml.qnode(dev, interface="torch", 
#                                   diff_method=diff_method)(self.qnode)
        
#         weight_shape = {"weights":(2,)}
#         self.qlayer = qml.qnn.TorchLayer(self.qcircuit, weight_shape)
#         self.cnet_out = self.cnet_out_layer()

#     def cnet_in_layer(self):
#         layers = [nn.Linear(self.input_size,self.hidden_size, device=self.torch_device), 
#                   nn.ReLU(True), 
#                 #   nn.Linear(hidden_size,output_size, device=torch_device), # TODO
#                 nn.Linear(self.hidden_size,2, device=self.torch_device), 
#                   nn.Tanh()]
#         return nn.Sequential(*layers)   
    
#     def cnet_out_layer(self):
#         layers = [nn.Linear(2,self.hidden_size, device=self.torch_device), 
#                   nn.ReLU(True), 
#                 #   nn.Linear(hidden_size,output_size, device=torch_device), # TODO
#                 nn.Linear(self.hidden_size,self.output_size, device=self.torch_device), 
#                   nn.Tanh()]
#         return nn.Sequential(*layers)  
    
#     def qnode(self, inputs, weights):
#         # Data encoding:
#         for x in range(len(inputs)):
#             qml.Hadamard(x)
#             qml.RZ(2.0 * inputs[x], wires=x)
#         # Trainable part:
#         qml.CNOT(wires=[0,1])
#         qml.RY(weights[0], wires=0)
#         qml.RY(weights[1], wires=1)
#         return [qml.expval(qml.PauliZ(wires=0)), qml.expval(qml.PauliZ(wires=1))]

#     def forward(self, x):
#         x1 = self.cnet_in(x)
#         x2 = self.qlayer(x1)
#         x_output = self.cnet_out(x2)
#         return x_output
    

# %%%%%%%%%%%%%
# Instantiating the model
# model = VQC(num_qubits, 1, 2)) # TODO
model = Net(model_input_size, hidden_size, model_output_size)
# model = Hybrid(model_input_size, hidden_size, model_output_size, 
#                dev, 
#                diff_method="backprop", 
#                torch_device="cpu")

# Move the model to GPU
if torch.cuda.is_available():
    model = model.to("cuda")
    
# model param summary
# pytorch_total_params = sum(p.numel() for p in model.parameters())
# print(f"Total number of parameters: {pytorch_total_params}")
# pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"Total number of trainable parameters: {pytorch_total_trainable_params}")

count_parameters(model=model)


# Defining the loss function and the optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%









fourcc_nn = cv2.VideoWriter_fourcc(*'mp4v') # Use the 'mp4v' codec for MP4 file
video_nn = cv2.VideoWriter("output/sin_nn.mp4", fourcc_nn, 1.0, (640, 480)) # Create VideoWriter object
# video_nn= cv2.VideoWriter('sin_nn.avi', 
#                          cv2.VideoWriter_fourcc(*'MJPG'),
#                          10, (int(1920/2), int(1080/2)))

# Initializing best loss and best model state
best_loss = float("inf")
best_state_dict = model.state_dict()
epoch_loss_list = []
test_loss_list = []


# Training the model
for epoch in tqdm(range(epochs), desc='Training Progress'):
    # train
    epoch_loss = 0
    epoch_length = 0
    for x, y in train_loader:
        epoch_length += 1
        # Move data to GPU
        if torch.cuda.is_available():
            x = x.to("cuda")
            y = y.to("cuda")
            
        y_pred = model(x)
        loss = criterion(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    epoch_loss /= epoch_length
    # tqdm.write("Epoch: {}, Train Loss: {:.4f}".format(epoch+1, epoch_loss))
    
    # validation
    test_loss = 0
    test_loss_length = 0
    y_pred_list = []
    x_list = []
    with torch.no_grad():
        for x, y in test_loader:
            test_loss_length += 1
            # Move data to GPU
            if torch.cuda.is_available():
                x = x.to("cuda")
                y = y.to("cuda")
            y_pred = model(x)
            test_loss += criterion(y_pred, y).item()
            
    test_loss /= test_loss_length
    # tqdm.write("Epoch: {}, Test Loss: {:.4f}".format(epoch+1, test_loss))
    
    # clear_output(wait=True)
    # clear_output(wait=True)

    # Updating the best loss and model state
    if test_loss < best_loss:
        best_loss = test_loss
        best_state_dict = model.state_dict()
                
    if epoch % 10 == 0:
        epoch_loss_list.append(epoch_loss)
        test_loss_list.append(test_loss)
        
        
        # make a plot video
        with torch.no_grad():
            x_test_0 = dataset[0][0]
            y_test_0 = dataset[0][1]
            x_gt = dataset.x_gt
            y_gt = dataset.y_gt
            # x_test_0 = xs[0,:]
            # y_test_0 = ys[0,:]
            if torch.cuda.is_available():
                x_test_0 = x_test_0.to("cuda")
                y_test_0 = y_test_0.to("cuda")
            y_pred_0 = model(x_test_0)

            if torch.cuda.is_available():
                x_test_0 = x_test_0.cpu()
                y_pred_0 = y_pred_0.cpu().detach().numpy().flatten()
                y_test_0 = y_test_0.cpu()
                x_gt = x_gt.cpu()
                y_gt = y_gt.cpu()
            else:
                y_pred_0 = y_pred_0.detach().numpy().flatten()

            # Plotting the results
            # plt.plot(x_test_0, y_test_0, label='Noisy')
            plt.plot(x_gt, y_gt, label='Ground Truth')
            plt.plot(x_test_0, y_pred_0, label='Predicted')
            
            plt.title("Prediction vs True {}th epoch".format(epoch))
            plt.legend()
            # plt.show()
        
            # Save the plot to a temporary image
            plt.savefig("output/temp.png")
            plt.close()

            # Read the image and write it to the video
            frame = cv2.imread("output/temp.png")
            video_nn.write(frame)

video_nn.release()

# Loading the best model state
model.load_state_dict(best_state_dict)


x_test_0 = dataset[0][0]
y_test_0 = dataset[0][1]
x_gt = dataset.x_gt
y_gt = dataset.y_gt
# x_test_0 = xs[0,:]
# y_test_0 = ys[0,:]
if torch.cuda.is_available():
    x_test_0 = x_test_0.to("cuda")
    y_test_0 = y_test_0.to("cuda")
y_pred_0 = model(x_test_0)

if torch.cuda.is_available():
    x_test_0 = x_test_0.cpu()
    y_pred_0 = y_pred_0.cpu().detach().numpy().flatten()
    y_test_0 = y_test_0.cpu()
    x_gt = x_gt.cpu()
    y_gt = y_gt.cpu()
else:
    y_pred_0 = y_pred_0.detach().numpy().flatten()

# Plotting the results
# plt.plot(x_test_0, y_test_0, label='Noisy')
plt.plot(x_gt, y_gt, label='Ground Truth')
plt.plot(x_test_0, y_pred_0, label='Predicted')
plt.title("Prediction vs True")
plt.legend()
plt.savefig("output/nn_prediction.png")
plt.show()

# Testing loop

with torch.no_grad():
    best_test_loss = 0
    y_pred_list = []
    x_list = []
    i = 0
    for x, y in test_loader:
        i += 1
        # Move data to GPU
        if torch.cuda.is_available():
            x = x.to("cuda")
            y = y.to("cuda")
        y_pred = model(x)
        best_test_loss += criterion(y_pred, y).item()
        # Plotting the results
        
        # move back to cpu for plotting
        if torch.cuda.is_available():
            x = x.cpu()
            y = y.cpu()
            y_pred = y_pred.cpu()
            
        if is_plot_all_test_samples:
            plt.plot(x.numpy().flatten(), y.numpy().flatten(), label='True')
            plt.plot(x.numpy().flatten(), y_pred.numpy().flatten(), label='Predicted')
            plt.title("Prediction vs True {}th sample".format(i))
            plt.legend()
            plt.savefig("output/nn_prediction_{}.png".format(i))
            plt.show()
    best_test_loss /= len(test_loader)
    print("Best Test Loss:", best_test_loss)

# Plotting the loss curve
# skip first 1 values
epoch_loss_list = epoch_loss_list[1:]
test_loss_list = test_loss_list[1:]
plt.plot(epoch_loss_list, label="Train Loss")
plt.plot(test_loss_list, label="Test Loss")
# y limit
# plt.ylim(0, 0.3)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss curve")
plt.legend()
plt.savefig("output/nn_loss.png")
plt.show()



# %%%%%%%%%%%%%


shapes = CVNeuralNetLayers.shape(n_layers=2, n_wires=2)
weights = [np.random.random(shape) for shape in shapes]

def cv_circuit():
  CVNeuralNetLayers(*weights, wires=[0, 1])
  return qml.expval(qml.PauliZ(0))





# %%%%%%%%

num_qubits = 6

######  QNN
dev = qml.device("default.qubit", wires=num_qubits)
# dev = qml.device("strawberryfields.fock", wires=1, cutoff_dim=10)

# def layer_multi_qubit(v):
    
#     qml.RX(v[0], wires=0)
#     qml.RX(v[1], wires=1)
#     qml.RX(v[2], wires=2)
#     qml.RX(v[3], wires=3)
    
#     qml.CNOT(wires=[0, 1])
#     qml.RZ(v[4], wires=1)
#     qml.CNOT(wires=[0, 1])
    
#     qml.CNOT(wires=[1, 2])
#     qml.RZ(v[5], wires=2)
#     qml.CNOT(wires=[1, 2])

#     qml.CNOT(wires=[2, 3])
#     qml.RZ(v[6], wires=3)
#     qml.CNOT(wires=[2, 3])
    
#     qml.CNOT(wires=[3, 0])
#     qml.RZ(v[7], wires=0)
#     qml.CNOT(wires=[3, 0])


def layer_multi_qubit(v):
    
    qml.RX(v[0], wires=0)
    qml.RX(v[1], wires=1)
    qml.RX(v[2], wires=2)
    qml.RX(v[3], wires=3)
    qml.RX(v[4], wires=3)
    qml.RX(v[5], wires=3)
    
    qml.CNOT(wires=[0, 1])
    qml.RZ(v[6], wires=1)
    qml.CNOT(wires=[0, 1])
    
    qml.CNOT(wires=[1, 2])
    qml.RZ(v[7], wires=2)
    qml.CNOT(wires=[1, 2])

    qml.CNOT(wires=[2, 3])
    qml.RZ(v[8], wires=3)
    qml.CNOT(wires=[2, 3])
    
    qml.CNOT(wires=[3, 4])
    qml.RZ(v[9], wires=0)
    qml.CNOT(wires=[3, 4])
    
    qml.CNOT(wires=[4, 5])
    qml.RZ(v[10], wires=0)
    qml.CNOT(wires=[4, 5])
 
    qml.CNOT(wires=[5, 0])
    qml.RZ(v[11], wires=0)
    qml.CNOT(wires=[5, 0])
       

# @qml.qnode(dev, diff_method="adjoint")
@qml.qnode(dev)
def quantum_neural_net_multi_qubit(var, x):
    # Encode input x into quantum state
    qml.RY(x, wires=0)
    qml.RY(x, wires=1)
    qml.RY(x, wires=2)
    qml.RY(x, wires=3)
    qml.RY(x, wires=4)
    qml.RY(x, wires=5)

    # "layer" subcircuits
    for v in var:
        layer_multi_qubit(v)

    # return qml.expval(qml.X(0))
    # return qml.expval(qml.PauliZ(0))
    
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(3) @ qml.PauliZ(4) @ qml.PauliZ(5))
    
    # y_pred = 0
    # for j in range(4):
    #     y_pred = y_pred + qml.expval(qml.PauliZ(j))
    
    # return y_pred
    # return (qml.expval(qml.PauliX(0)) + qml.expval(qml.PauliX(1)) + qml.expval(qml.PauliX(2)) + qml.expval(qml.PauliX(3))) / 4
    # return (qml.expval(qml.PauliX(0)).value + qml.expval(qml.PauliX(1)).value + qml.expval(qml.PauliX(2)).value + qml.expval(qml.PauliX(3)).value) / 4

    # return [qml.expval(qml.PauliZ(j)) for j in range(4)]

# %%

######  QNN
# dev = qml.device("default.qubit", wires=1)
dev = qml.device("strawberryfields.fock", wires=1, cutoff_dim=10)

# dev = qml.device("strawberryfields.fock", wires=1, cutoff_dim=10)

def layer(v):
    # Matrix multiplication of input layer
    qml.Rotation(v[0], wires=0)
    qml.Squeezing(v[1], 0.0, wires=0)
    qml.Rotation(v[2], wires=0)

    # Bias
    qml.Displacement(v[3], 0.0, wires=0)

    # Element-wise nonlinear transformation
    qml.Kerr(v[4], wires=0)
    


# @qml.qnode(dev, diff_method="adjoint")
@qml.qnode(dev)
def quantum_neural_net(var, x):
    # Encode input x into quantum state
    qml.Displacement(x, 0.0, wires=0)

    # "layer" subcircuits
    for v in var:
        layer(v)

    return qml.expval(qml.X(0))




def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2

    loss = loss / len(labels)
    return loss

def cost(var, features, labels):
    # preds = [quantum_neural_net(var, x) for x in features]
    # preds = [quantum_neural_net_multi_qubit(var, x) for x in features]
    preds = cv_circuit()
    # take avera of the preds list
    # preds = npq.sum(preds, axis=0)
    
    return square_loss(labels, preds)

num_layers = 4
# var_init = 0.05 * npq.random.randn(num_layers, 5, requires_grad=True)
# var_init = 0.05 * npq.random.randn(num_layers, 8, requires_grad=True)

var_init = 0.05 * npq.random.randn(num_layers, 12, requires_grad=True)

print(var_init)

opt = AdamOptimizer(0.01, beta1=0.9, beta2=0.999)

fourcc_qnn = cv2.VideoWriter_fourcc(*'mp4v') # Use the 'mp4v' codec for MP4 file
video_qnn = cv2.VideoWriter("output/sin_qnn.mp4", fourcc_qnn,1.0, (640, 480)) # Create VideoWriter object

# var = var_init
var = weights
# for it in range(epochs_qnn):
for it in tqdm(range(epochs_qnn), desc='Training Progress'):
    X = npq.array(dataset[0][0].squeeze())
    Y = npq.array(dataset[0][1].squeeze())
    (var, _, _), _cost = opt.step_and_cost(cost, var, X, Y)
    print("Iter: {:5d} | Cost: {:0.7f} ".format(it, _cost))
    
    if it % 2 == 0:
        y_pred = [quantum_neural_net(var, x) for x in X]
        plt.plot(X, Y, label='True')
        plt.plot(X, y_pred, label='Predicted')
        plt.title("Prediction vs True {}th epoch".format(it))
        plt.legend()
        # plt.show()
        
        # Save the plot to a temporary image
        plt.savefig("output/tempq.png")
        plt.close()

        # Read the image and write it to the video
        frame = cv2.imread("output/tempq.png")
        video_qnn.write(frame)

# Release the VideoWriter object
video_qnn.release() 
        
# %%
