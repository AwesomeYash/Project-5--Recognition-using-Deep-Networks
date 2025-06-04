'''
Code by Francis Jacob Kalliath 
Project 5: Recognition using Deep Networks
'''

import torch
import torchvision
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os  # Added for directory creation

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

# defining the hyper-parameters
N_EPOCHS = 5
LEARNING_RATE = 0.01
MOMENTUM = 0.5
LOG_INTERVAL = 10
batch_size = 64

#Class to build the network
class BasicNetwork(nn.Module):
    def __init__(self,  num_layers, conv_filter_size, dropout_rate):
        super(BasicNetwork, self).__init__()
        self.input_size = 28 
        self.num_layers = num_layers
        self.conv1 = nn.Conv2d(1, 10, kernel_size=conv_filter_size, padding='same')
        self.conv2 = nn.Conv2d(10, 20, kernel_size=conv_filter_size, padding='same')
        self.conv = nn.Conv2d(20, 20, kernel_size=conv_filter_size, padding='same')
        self.conv2_drop = nn.Dropout2d(dropout_rate)
        self.fc1 = nn.Linear(self.get_fc1_input_size(), 50)
        self.fc2 = nn.Linear(50, 10)

    def get_fc1_input_size(self):
        fc1_size = self.input_size / 2
        fc1_size = fc1_size / 2
        fc1_size = fc1_size * fc1_size * 20
        return int(fc1_size)
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        for i in range(self.num_layers):
            x = F.relu(self.conv(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, 1)

# Train Functions definition
def train(epoch, model, optimizer, train_loader, train_losses, train_counter, batch_size):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:  # Changed from hard-coded 2 to LOG_INTERVAL
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * batch_size) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(model.state_dict(), 'results/model.pth')
            torch.save(optimizer.state_dict(), 'results/optimizer.pth')

# Test Functions definition
def test(model, test_loader, test_losses):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)  # Fixed: append to the list
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)  # Return accuracy

# Function to plot the graphs
def plot_curve(train_counter, train_losses, test_counter, test_losses, filename):
    plt.figure()  # Create a new figure for each plot
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.savefig(filename)
    plt.close()  # Close the figure to free memory

# Testing and Training initializations
def experiment(num_epochs, batch_size, num_layers, conv_filter_size, dropout_rate, filename):
    # Following the tutorial, initializing the train_loader and test_loader
    train_data_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/files/', train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                ])), 
                                batch_size=batch_size, shuffle=True)

    test_data_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/files/', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                ])),
                                batch_size=batch_size, shuffle=True)
    
    # Network Variables
    network = BasicNetwork(num_layers, conv_filter_size, dropout_rate)
    optimizer = optim.SGD(network.parameters(), lr=LEARNING_RATE,
                          momentum=MOMENTUM)
    
    # Training and Testing Arrays
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(train_data_loader.dataset) for i in range(num_epochs + 1)]

    # Training the model
    initial_accuracy = test(network, test_data_loader, test_losses)
    for epoch in range(1, num_epochs + 1):
        train(epoch, network, optimizer, train_data_loader, train_losses, train_counter, batch_size)
        test(network, test_data_loader, test_losses)

    # Plotting graphs
    plot_curve(train_counter, train_losses, test_counter, test_losses, filename)
    
    return initial_accuracy  # Return accuracy for tracking

# Main function
def main():
    # Dictionary to store results
    results = []
    
    # Loops with different hyper-parameters
    for num_epochs in [3, 5]:
        for batch_size in [128, 256]:
            for num_layers in [2, 4]:
                for conv_filter_size in [3, 5]:
                    for dropout_rate in [0.3, 0.5]:
                        filename = f'curve_{num_epochs}_{batch_size}_{num_layers}_{conv_filter_size}_{dropout_rate}.png'
                        print(f"\nTraining: Layers={num_layers}, Filter Size={conv_filter_size}, Dropout={dropout_rate}, Batch={batch_size}, Epochs={num_epochs}")
                        try:
                            accuracy = experiment(num_epochs, batch_size, num_layers, conv_filter_size, dropout_rate, filename)
                            results.append({
                                'epochs': num_epochs,
                                'batch_size': batch_size,
                                'layers': num_layers,
                                'filter_size': conv_filter_size,
                                'dropout_rate': dropout_rate,
                                'accuracy': accuracy
                            })
                        except Exception as e:
                            print(f"Error in experiment: {e}")
    
    # Print results summary
    print("\nExperiment Results Summary:")
    for result in results:
        print(f"Layers={result['layers']}, Filter={result['filter_size']}, Dropout={result['dropout_rate']}, " +
              f"Batch={result['batch_size']}, Epochs={result['epochs']}, Accuracy={result['accuracy']:.2f}%")

# Calling Main Function
if __name__ == "__main__":
    main()