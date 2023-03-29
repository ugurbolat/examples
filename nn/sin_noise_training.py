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
num_samples_in_sin = 20
num_samples_in_dataset = 50
noise = [0.0, 0.0]
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

sin_y = (sin_y + 1) * 100 + 300

# acummulative noise to demonstrate the rollout training without noise
sin_y_acc_noise = torch.zeros((num_samples_in_sin, 1))
sin_y_acc_noise[0] = sin_y[0]
noise = 3

# plt.plot(sin_x, sin_y, label="Ground Truth")
plt.plot(sin_x, sin_y)

# x-axis label
plt.xlabel('Time')
plt.ylabel('Temperature')

# add circle markers to plot
# scale sin_y which is in range [-1, 1] to [300, 500]

plt.scatter(sin_x, sin_y, s=10)

for j in range(100):
    acc_noise = 0
    for i in range(1, sin_y.shape[0]):
        # add noise to sin_y_acc_noise
        acc_noise += torch.randn(1) * noise
        sin_y_acc_noise[i] = sin_y[i] + acc_noise
        
    # plt.plot(sin_x, sin_y_acc_noise, label="{}. Prediction w/ Acc. Noise".format(j+1))
    plt.plot(sin_x, sin_y_acc_noise)
    # add circle markers to plot
    plt.scatter(sin_x, sin_y_acc_noise, s=10)
        
plt.legend()
plt.show()
# show labels

# sin_x = dataset[1][0]
# sin_y = dataset[1][1]
# plt.plot(sin_x, sin_y, label="sin 1")
# plt.show()

# %%
