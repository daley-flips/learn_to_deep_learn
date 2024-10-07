import torch
import torch.nn as nn
import torch.optim as optim
import time

# Check for MPS support
if torch.backends.mps.is_available():
    print('GPUUUU WOOO')

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f'Using device: {device}')

# Define a much larger neural network
class HugeNN(nn.Module):
    def __init__(self):
        super(HugeNN, self).__init__()
        self.fc1 = nn.Linear(784, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, 128)
        self.fc7 = nn.Linear(128, 64)
        self.fc8 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = torch.relu(self.fc7(x))
        x = self.fc8(x)
        return x

# Function to train the model and time it
def train_model(device, epochs=5):
    model = HugeNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Generate random data (even larger batch size and input)
    inputs = torch.randn(100000, 784).to(device)
    labels = torch.randint(0, 10, (100000,)).to(device)

    start_time = time.time()

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    end_time = time.time()
    print(f"Training on {device} took: {end_time - start_time:.4f} seconds")

# Train and time on CPU
print("Training on CPU...")
train_model(torch.device('cpu'))

# Train and time on MPS (if available)
if torch.backends.mps.is_available():
    print("Training on MPS...")
    train_model(torch.device('mps'))
