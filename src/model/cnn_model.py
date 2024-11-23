import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, num_classes=7):
        super(Net, self).__init__()
        
        # Define the model structure
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # Input shape: (batch, 3, 28, 28)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)

        self.conv6 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(256)

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.2)
        
        # Placeholder for fully connected layers (fc1 input size will be set dynamically)
        self.fc1 = None
        self.bn8 = None
        self.fc2 = nn.Linear(256, 128)
        self.bn9 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn10 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 32)
        self.bn11 = nn.BatchNorm1d(32)
        self.fc5 = nn.Linear(32, num_classes)

    def forward(self, x):
        # Convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool(x)

        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = self.pool(x)

        # Dynamically calculate fc1 input size
        if self.fc1 is None:
            input_features = x.view(x.size(0), -1).size(1)
            self.fc1 = nn.Linear(input_features, 256).to(x.device)
            self.bn8 = nn.BatchNorm1d(256).to(x.device)

        x = self.flatten(x)
        x = self.dropout(x)
        
        # Fully connected layers
        x = F.relu(self.bn8(self.fc1(x)))
        x = F.relu(self.bn9(self.fc2(x)))
        x = F.relu(self.bn10(self.fc3(x)))
        x = F.relu(self.bn11(self.fc4(x)))
        x = F.softmax(self.fc5(x), dim=1)
        return x

if __name__ == "__main__":
    model = Net(num_classes=7)
    print(model)