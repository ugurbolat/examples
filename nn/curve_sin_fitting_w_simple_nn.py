# %%

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

torch.manual_seed(0)
np.random.seed(0)

# Generating toy data
x_data = np.linspace(0, 2 * np.pi, 1000)
y_data = np.sin(x_data)

# Converting data to PyTorch tensors
x = torch.tensor(x_data, dtype=torch.float32).unsqueeze(1)
y = torch.tensor(y_data, dtype=torch.float32).unsqueeze(1)

num_samples_in_sin = 1000
num_samples_in_dataset = 10
noise = 0.05
batch_size = 1

model_input_size = num_samples_in_sin
hidden_size = 16
model_output_size = num_samples_in_sin

# Defining the neural network model
class Net(torch.nn.Module):
    def __init__(self, input_size=1000, hidden_size=16, output_size=1000):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(1, hidden_size)
        # self.hidden_mid = torch.nn.Linear(16, 16)
        self.output = torch.nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = torch.nn.functional.relu(self.hidden(x))
        # x = torch.nn.functional.relu(self.hidden_mid(x))
        x = self.output(x)
        return x

# Instantiating the model
model = Net(model_input_size, hidden_size, model_output_size)


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
loss_values = []

# Training the model
for epoch in range(2):
    # for x, y in train_loader:
    y_pred = model(x)
    loss = criterion(y_pred, y)
    
    if epoch % 100 == 0:
        loss_values.append(loss.item())
    
    # Updating the best loss and model state
    if loss < best_loss:
        best_loss = loss
        best_state_dict = model.state_dict()
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Loading the best model state
model.load_state_dict(best_state_dict)

# Evaluating the model
x_test = torch.tensor(np.linspace(0, 2 * np.pi, 1000), dtype=torch.float32).unsqueeze(1)
y_test = np.sin(x_test.numpy().flatten())
y_pred = model(x_test).detach().numpy().flatten()

# Plotting the results
plt.plot(x_test.numpy(), y_test, label='True')
plt.plot(x_test.numpy(), y_pred, label='Predicted')
plt.title("Prediction vs True")
plt.legend()
plt.show()


# Plotting the loss curve
# skip first 1 values
loss_values = loss_values[1:]
plt.plot(loss_values)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss curve")
plt.show()

# %%
