"""
Name: Priyanshu Ranka (NUID: 002305396)
Semester: Spring 2025
Subject / Proffessor: PRCV / Prof. Bruce Maxwell
Project 5: Task 1 A,B,C,D
Description: The first task is to build and train a network to do digit recognition using the MNIST data base, then save the network to a file 
so it can be re-used for the later tasks. 
"""

# Importing Libraries
import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys

# Training and Testing Variables as Global Variables
train_losses = [] 
train_counter = []
test_losses = []

"""
TASK 1-B: Building the Network

Now let's go ahead and build our network. We'll use two 2-D convolutional layers followed by two fully-connected (or linear) layers. As activation function 
we'll choose rectified linear units (ReLUs in short) and as a means of regularization we'll use two dropout layers. In PyTorch a nice way to build a network 
is by creating a new class for the network we wish to build. Let's import a few submodules here for more readable code.
"""

# Neural Network class definition
class Net(nn.Module):
    def __init__(self, dropout_rate=0.3):  # Dropout rate can be adjusted (0.05 to 0.5)
        super(Net, self).__init__()
        
        # Convolutional layer with 10 filters of size 5x5
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        # Convolutional layer with 20 filters of size 5x5
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        # Dropout layer with chosen rate
        self.dropout = nn.Dropout2d(p=dropout_rate)
        # Fully connected layer with 50 nodes
        self.fc1 = nn.Linear(in_features=320, out_features=50)
        # Final fully connected layer with 10 nodes (for classification)
        self.fc2 = nn.Linear(in_features=50, out_features=10)

    def forward(self, x):

        # First conv layer → ReLU → Max Pool
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        # Second conv layer → ReLU → Dropout → Max Pool
        x = F.relu(F.max_pool2d(self.dropout(self.conv2(x)), kernel_size=2))
        # Flatten the feature maps
        x = x.view(-1, 320)  # 20 feature maps of size 4x4 → 320 total features
        # Fully connected layer with ReLU
        x = F.relu(self.fc1(x))
        # Final fully connected layer with log_softmax activation
        x = F.log_softmax(self.fc2(x), dim=1)

        return x

# Definition of Train fucntion
def train(network, train_loader, epoch, optimizer, log_interval):
    # Changing network-mode to training
    network.train()
    # Ensure the directory exists
    save_dir = "results"  # Use a relative path instead of "/results"
    os.makedirs(save_dir, exist_ok=True)

    # Local variables 
    local_train_losses, local_train_counter = [], []

    # Main train loop
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            
            # Storing all the test losses into the global array for final plot
            local_train_losses.append(loss.item())
            local_train_counter.append((batch_idx * 64) + ((epoch-1) * len(train_loader.dataset)))
            
            #Task D: Save the model and optimizer states
            # Save model and optimizer states in the 'results' directory
            torch.save(network.state_dict(), os.path.join(save_dir, "model.pth"))
            torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pth"))

    return local_train_losses, local_train_counter

# Defining the test function
def test(network, test_loader):
    network.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    # Storing all the test losses into the global array for final plot
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return test_losses

# Main Function
def main(argv):
    
    # Defining the parameters as global variables
    n_epochs = 5
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10
    random_speed =1

    # This is to make sure that the results are reproducible (down)
    torch.backends.cudnn.enabled = False    
    torch.manual_seed(random_speed)

    """
    # We need DataLoaders for the dataset. This is where TorchVision comes into play.

    The values 0.1307 and 0.3081 used for the Normalize() transformation below are the global mean and standard deviation of the MNIST dataset
    """

    # Following the tutorial, initializing the train_loader and test_loader
    train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/files/', train=True, download=True,
                                transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize((0.1307,), (0.3081,))])), 
                                batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/files/', train=False, download=True,
                                transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
                                batch_size=batch_size_test, shuffle=True)


    """ TASK 1-A
    # Let's visualize the data 
    """
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    print(example_data.shape)
    """
    OUTPUT - torch.Size([1000, 1, 28, 28])
    # This means we have 1000 examples of 28x28 pixels in grayscale (i.e. no rgb channels, hence the one)
    """
    # Plotting the data
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()

    """
    TASK 1-C AND 1-D:

    """
    # Instantiate the model
    network = Net(dropout_rate=0.3)  # You can adjust the dropout rate as needed
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
    test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

    # Run the training loop and test loop
    test(network, test_loader)
    for epoch in range(1, n_epochs + 1):
        train_losses_epoch, train_counter_epoch = train(network, train_loader, epoch, optimizer, log_interval)
        train_losses.extend(train_losses_epoch)
        train_counter.extend(train_counter_epoch)
        test(network, test_loader)

    # Plotting the training and test losses - TASK 1-C
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()

# Calling Main function
if __name__ == "__main__":
    main(sys.argv)