import torch
from torch.utils.data import Dataset, DataLoader, random_split

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Example data and labels
data = torch.randn(100, 10)  # 100 samples, 10 features each
labels = torch.randint(0, 2, (100,))  # 100 labels (for binary classification)

# Creating the Dataset
dataset = CustomDataset(data, labels)

# Splitting the dataset into training, validation, and testing
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Creating the DataLoaders for each set
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False, num_workers=2)

# Example usage
# Training loop
for batch_idx, (data, labels) in enumerate(train_loader):
    # Perform your training operations here with the batched data and labels
    pass

# Validation loop
for batch_idx, (data, labels) in enumerate(val_loader):
    # Perform your validation operations here with the batched data and labels
    pass

# Testing loop (after training is complete)
for batch_idx, (data, labels) in enumerate(test_loader):
    # Perform your testing operations here with the batched data and labels
    pass
