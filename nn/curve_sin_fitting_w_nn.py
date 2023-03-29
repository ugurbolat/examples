# %%

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

torch.manual_seed(0)
np.random.seed(0)

# # Generating toy data
# x_data = np.linspace(0, 2 * np.pi, 1000)
# y_data = np.sin(x_data)

# # Converting data to PyTorch tensors
# x = torch.tensor(x_data, dtype=torch.float32).unsqueeze(1)
# y = torch.tensor(y_data, dtype=torch.float32).unsqueeze(1)

# dataset parameters
num_samples_in_sin = 100
num_samples_in_dataset = 50
noise = [0.1, 0.1]
batch_size = 32
train_size = int(0.8 * num_samples_in_dataset)
test_size = num_samples_in_dataset - train_size

# model parameters
model_input_size = num_samples_in_sin
hidden_size = 32
model_output_size = num_samples_in_sin

# training parameters
epochs = 50
is_plot_all_test_samples = False


class TwoSinusoidalDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples_in_sin, num_samples, noise=[0.01, 0.0001]):
        self.num_samples = num_samples
        self.num_samples_in_sin = num_samples_in_sin
        self.noise = noise
        self.x_gt = torch.linspace(0, 2 * np.pi, self.num_samples_in_sin)
        self.y_gt = torch.sin(self.x_gt) + torch.sin(self.x_gt * 5)
        self.y = torch.zeros((self.num_samples, self.num_samples_in_sin))
        self.x = torch.zeros((self.num_samples, self.num_samples_in_sin))
        for i in range(self.num_samples):
            self.y[i, :] = self.y_gt
            # add noise to y
            self.y[i, :] += torch.randn(self.num_samples_in_sin) * self.noise[0]
            self.x[i, :] = self.x_gt + torch.randn(self.num_samples_in_sin) * self.noise[1]
            
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
        self.y = torch.zeros((self.num_samples, self.num_samples_in_sin))
        self.x = torch.zeros((self.num_samples, self.num_samples_in_sin))
        for i in range(self.num_samples):
            self.y[i, :] = self.y_gt
            # add noise to y
            self.y[i, :] += torch.randn(self.num_samples_in_sin) * self.noise[0]
            self.x[i, :] = self.x_gt + torch.randn(self.num_samples_in_sin) * self.noise[1]
            
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
plt.plot(sin_x, sin_y, label="sin 0")
plt.show()
sin_x = dataset[1][0]
sin_y = dataset[1][1]
plt.plot(sin_x, sin_y, label="sin 1")
plt.show()

# %%

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
        x = torch.nn.functional.relu(self.hidden(x))
        x = torch.nn.functional.relu(self.hidden_mid(x))
        # x = torch.nn.functional.relu(self.hidden_mid_2(x))
        # x = torch.nn.functional.relu(self.hidden_mid_3(x))
        x = self.output(x)
        return x


# Instantiating the model
model = Net(model_input_size, hidden_size, model_output_size)

# Move the model to GPU
if torch.cuda.is_available():
    model = model.to("cuda")
    
# model param summary
pytorch_total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {pytorch_total_params}")
pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of trainable parameters: {pytorch_total_trainable_params}")

# Defining the loss function and the optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

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
        
    # Updating the best loss and model state
    if test_loss < best_loss:
        best_loss = test_loss
        best_state_dict = model.state_dict()
                
    if epoch % 10 == 0:
        epoch_loss_list.append(epoch_loss)
        test_loss_list.append(test_loss)

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
plt.show()

# %%
