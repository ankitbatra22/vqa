import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.autograd import Variable
import sys
import os

# Add the src directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Now you can import the modules
from data.Ham10000 import Ham10000, data_loader

def train(model, train_loader, optimizer, device, epoch):
    model.train()
    train_accuracy = 0.0
    train_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Move data to the specified device
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        loss.backward()
        optimizer.step()

        # Update loss and accuracy
        train_loss += loss.cpu().data * inputs.size(0)
        _, prediction = torch.max(outputs.data, 1)
        train_accuracy += int(torch.sum(prediction == targets.data))

    # Normalize loss and accuracy
    train_loss = train_loss / len(train_loader.dataset)
    train_accuracy = train_accuracy / len(train_loader.dataset)

    print(f"Epoch {epoch}: Train Loss: {train_loss:.6f}, Train Accuracy: {train_accuracy:.2%}")



    #     if batch_idx % 100 == 0:
    #         print('Epoch: {} {}/{} Training loss: {:.6f}'.format(
    #             epoch,
    #             batch_idx * len(inputs),
    #             len(train_loader.dataset),
    #             loss))

    # print('Training loss: {:.6f}'.format(total_loss / len(train_loader.dataset) * len(inputs)))
