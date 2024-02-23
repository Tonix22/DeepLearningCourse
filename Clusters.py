import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Helper function to generate radial data points
def generate_radial_data(inner_radius, outer_radius, inner_points, outer_points):
    # Generate inner circle points
    inner_angles = np.random.rand(inner_points) * 2 * np.pi
    inner_radiuses = np.random.rand(inner_points) * inner_radius
    inner_x = inner_radiuses * np.cos(inner_angles)
    inner_y = inner_radiuses * np.sin(inner_angles)

    # Generate outer ring points
    outer_angles = np.random.rand(outer_points) * 2 * np.pi
    outer_radiuses = inner_radius + (np.random.rand(outer_points) * (outer_radius - inner_radius))
    outer_x = outer_radiuses * np.cos(outer_angles)
    outer_y = outer_radiuses * np.sin(outer_angles)

    # Combine into one dataset correctly
    inner_data = np.vstack((inner_x, inner_y)).T
    outer_data = np.vstack((outer_x, outer_y)).T
    data = np.vstack((inner_data, outer_data))  # Combine inner and outer data
    labels = np.hstack((np.zeros(inner_points, dtype=np.int64), np.ones(outer_points, dtype=np.int64)))

    return torch.tensor(data), torch.tensor(labels)


# Generate data
inner_radius, outer_radius = 0.5, 1.0
inner_points, outer_points = 1000, 1000
data, labels = generate_radial_data(inner_radius, outer_radius, inner_points, outer_points)

# Define the neural network with a non-linear activation function
class RadialNet(nn.Module):
    def __init__(self):
        super(RadialNet, self).__init__()
        self.fc1 = nn.Linear(2, 10)  # 2 input features, 10 hidden units
        self.fc2 = nn.Linear(10, 10) # 10 hidden units, 10 hidden units
        self.fc3 = nn.Linear(10, 2)  # 10 hidden units, 2 output features
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Instantiate the network, loss function, and optimizer
net = RadialNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)

# Train the network
epochs = 100
for epoch in tqdm(range(epochs), desc='Training Epochs'):
    optimizer.zero_grad()
    outputs = net(data.float())
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

# Plot the decision boundary
def plot_decision_boundary():
    x_min, x_max = data[:, 0].min() - 0.1, data[:, 0].max() + 0.1
    y_min, y_max = data[:, 1].min() - 0.1, data[:, 1].max() + 0.1
    xx, yy = torch.meshgrid(torch.linspace(x_min, x_max, 200),
                            torch.linspace(y_min, y_max, 200))
    grid = torch.stack((xx.flatten(), yy.flatten()), dim=1)
    Z = net(grid)
    Z = Z.argmax(1).reshape(xx.shape)
    plt.contourf(xx, yy, Z.detach().numpy(), alpha=0.8)  # Detach and convert to numpy array

# Plot the data points
def plot_data():
    plt.scatter(data[labels == 0, 0].numpy(), data[labels == 0, 1].numpy(), color='blue', label='Inner Cluster')
    plt.scatter(data[labels == 1, 0].numpy(), data[labels == 1, 1].numpy(), color='red', label='Outer Cluster')
    plt.legend()

plot_decision_boundary()
plot_data()
plt.show()
