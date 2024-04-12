import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
# from rich import *

# MARK: Some inits
class CustomDataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)
        self.scaler = StandardScaler()
        self.inputs = self.scaler.fit_transform(data.iloc[:, :-1])  # Standardize the input features
        self.targets = data.iloc[:, -1].values

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        inputs = torch.tensor(self.inputs[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)  # Target column
        return inputs, target

# Load the CSV data into a custom dataset
csv_file = 'training_data.csv'
dataset = CustomDataset(csv_file)

# Split dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create DataLoader instances for training and validation sets
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define the neural network model
class GPNet(nn.Module):
    def __init__(self):
        super(GPNet, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        x = self.fc4(x)
        return x

net = GPNet()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.0005)

loss_train = []
loss_val = []

# MARK: Training starts
num_epochs = 10000  # Increase the number of epochs
best_val_loss = float('inf')
for epoch in range(num_epochs):
    net.train()  # Set model to training mode
    running_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()  # Zeroing the gradients
        outputs = net(inputs)  # Forward propagation
        loss = criterion(outputs, targets)  # Computing the loss
        loss.backward()  # Backward propagation
        optimizer.step()  # Update the parameters
        running_loss += loss.item()
    with open("loss_train/loss_train_" + str(epoch) + ".json", 'w') as file:
        file.write(str(loss_train))
        file.close()

    with open("loss_val/loss_train" + str(epoch) + ".json", 'w') as file:
        file.write(str(loss_val))
        file.close()

    # Validating
    net.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = net(inputs)
            val_loss += criterion(outputs, targets).item()

    train_loss = running_loss / len(train_loader)
    val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}")
    loss_train.append(train_loss)
    loss_val.append(val_loss)
    # Check if validation loss has improved
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        # Save the model
        torch.save(net.state_dict(), 'model.pth')

net.load_state_dict(torch.load('model.pth'))

# Testing
net.eval()
with torch.no_grad():
    inputs = torch.tensor([[10.0, 5, 15, 9.8]], dtype=torch.float32)
    output = net(inputs)
    print(output)

