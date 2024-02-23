To view your training progress and results in TensorBoard, open a terminal and run the following command:

```bash
tensorboard --logdir=lightning_logs/
```

# PyTorch Lightning Overview

PyTorch Lightning is a high-level framework for PyTorch that simplifies the complexity of training neural networks. It allows for focusing on the research aspect by automating the engineering part. Lightning structures PyTorch code to separate research from engineering, making code more readable, reusable, and scalable.

**Key Benefits:**
- Simplifies complex PyTorch code without adding overhead.
- Supports advanced training features like 16-bit precision, distributed training, and early stopping.
- Facilitates easy experiment replication and sharing.
- Integrates with popular tools like TensorBoard, MLFlow, and Comet.ml for logging and monitoring.

## Creating a Custom DataLoader in PyTorch Lightning

Customizing data loading in PyTorch Lightning involves creating a custom `Dataset` and `DataLoader`. The `Dataset` handles loading, preprocessing, and augmenting data, while the `DataLoader` provides batches of data to the model during training.

### Example: Custom `Dataset` and `DataLoader`

```python
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self):
        # Load and prepare data here
        self.dataset = CustomDataset(...)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

# Usage
data_module = DataModule(batch_size=64)
model = MyLightningModel(...)
trainer = pl.Trainer(...)
trainer.fit(model, datamodule=data_module)
