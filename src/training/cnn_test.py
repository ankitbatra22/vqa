import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

import sys
import os

# Add the src directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Now you can import the modules
from model.cnn_model import Net
from data.Ham10000 import Ham10000, data_loader
from training.cnn_train import train

def test(model, test_loader, device):
    model.eval()
    loss = 0
    correct = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss += nn.CrossEntropyLoss()(outputs, targets).item()  # Use `.item()` to get scalar value
            predictions = outputs.argmax(dim=1, keepdim=True)
            correct += predictions.eq(targets.view_as(predictions)).sum().item()

    avg_loss = loss / len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f"Test Loss: {avg_loss:.6f}, Test Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy


def main():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the dataset
    loaders = data_loader()
    train_loader = loaders['train']
    test_loader = loaders['val']  # Use validation set as the test set here

    # Initialize the model
    model = Net(num_classes=7).to(device)

    # Define optimizer and training parameters
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 100

    # Training and testing loop
    for epoch in range(1, num_epochs + 1):
        print(f"Starting Epoch {epoch}/{num_epochs}")
        train(model, train_loader, optimizer, device, epoch)
        test(model, test_loader, device)  # Pass test_loader explicitly

    # Save the model after training
    torch.save(model.state_dict(), "./model_checkpoint.pth")
    print("Model checkpoint saved to ./model_checkpoint.pth")


if __name__ == "__main__":
    main()