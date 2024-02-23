import matplotlib.pyplot as plt
import torch
import json
import numpy as np


f      = open('config.json')
data   = json.load(f)
lrate  = data["lr"]
epochs = data["epochs"]
printe = data["printepoch"]

#this is my ground truth
f = lambda x: -torch.sin(3*x)*torch.cos(5*x)

x = torch.linspace(-1,1, 35) #who can explain this?!!!

# Define the neural network
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(1, 64) # 1 input, 64 hidden neurons in layer 1
        self.hidden2 = torch.nn.Linear(64, 128) # 64 input, 128 hidden neurons in layer 2
        self.output = torch.nn.Linear(128, 1) # 128 input, 1 output

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = self.output(x)
        return x


net = Net()
#loss function
criterion = torch.nn.MSELoss()
#optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=lrate)  # Using Adam optimizer

#losses history
loss_hist = []

#iterate over epochs, and epoch is the entire set
for epoch in range(epochs):
    #current loss
    running_loss = 0.0
    #clean gradient
    optimizer.zero_grad()
    # evaluate network
    # unsqueeze remove extra dimension for example dim=(3,1), dim(3)
    outputs = net(x.unsqueeze(1))
    loss = criterion(outputs.squeeze(), f(x))
    #backpropagation
    loss.backward()
    #update steps
    optimizer.step()
    #loss acmulation
    running_loss += loss.item()

    if epoch % printe == 0:
        loss_epoch = loss.detach().numpy()
        loss_hist.append(loss_epoch)
        print("Epoch {}: Loss = {}".format(epoch, loss_epoch))


data   = np.array(loss_hist)
x_axis = np.arange(len(data))

# Create the plot
plt.plot(x_axis, data, 'o-')  # 'o-' means circle markers with lines

# Optionally, add title and labels
plt.title('Loss over time')
plt.xlabel('E Poch')
plt.ylabel('Loss')
# Set the y-axis to logarithmic scale
plt.yscale('log')

# Show the plot
plt.show()

# Plot the actual function and predicted function
x_plot = torch.linspace(-1,1, 100)
actual_y = torch.tensor([f(p) for p in x_plot])
predicted_y = net(x.unsqueeze(1)).squeeze()
plt.plot(x_plot, actual_y, 'g', label='Actual Function')
plt.plot(x, predicted_y.detach().numpy(), 'b', label='Predicted Function')
plt.legend()
plt.show()