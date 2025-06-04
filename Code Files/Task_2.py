"""
Name: Priyanshu Ranka (NUID: 002305396)
Semester: Spring 2025  
Professor: Prof. Bruce Maxwell
Project 5: Task 2
Description: This file is used to load the trained model and print the size and weights of the first convolution layer.
"""
# Import the required libraries
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os

# Define the network (must match the trained model)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Plotting fucntion for Task A
def plot_taskA(network):
    # TASK 2 A: Plot the weights of the first convolution layer
    # Plot the weights of the first convolution layer
    fig = plt.figure("Plot of Weights of the First Convolution Layer")
    for i in range(10):
        plt.subplot(3, 4, i+1)
        plt.tight_layout()
        plt.imshow(network.conv1.weight[i,0].detach().numpy(), interpolation='none')
        plt.title("Filter {}".format(i+1))
        plt.xticks([])
        plt.yticks([])
    plt.show()    

# Plotting function for Task B
def plot_taskB(network, test_dataset):
    # TASK 2 B: Apply these weights to an image and visualize the output
    # Turoff the gradient calculation
    with torch.no_grad():
        # Get the first image from the test dataset
        image = test_dataset[0][0]
        image = image.unsqueeze(0)

        # Apply the weights of the first convolution layer to the image
        output = F.conv2d(image, network.conv1.weight)

        # Create a 5x4 grid (10 filters → 5 rows, 2 filters per row)
        fig, axes = plt.subplots(5, 4, figsize=(10, 12))
        for i in range(10):
            row = i % 5  # 5 rows
            col = (i // 5) * 2  # 2 sets of filter-image pairs

            # Plot the filter (weight matrix)
            ax1 = axes[row, col]
            ax1.imshow(network.conv1.weight[i,0].detach().numpy(), cmap='gray')
            ax1.set_title(f"Filter {i+1}")
            ax1.set_xticks([])
            ax1.set_yticks([])

            # Plot the filtered image
            ax2 = axes[row, col + 1]
            ax2.imshow(output[0,i].detach().numpy(), cmap='gray')
            ax2.set_title(f"Filtered Image {i+1}")
            ax2.set_xticks([])
            ax2.set_yticks([])

        plt.tight_layout()
        plt.show()

# Main function 
def main(argv):
    # Load the test dataset (MNIST)
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    
    # Load the trained model
    network = Net()
    model_path = "C:\\Users\\yashr\\Desktop\\NEU\\Semester 2\\PRCV\\Projects\\Project_5\\results\\model.pth"

    # Check if the model file exists - DEBUGGING
    if os.path.exists(model_path):
        network.load_state_dict(torch.load(model_path))
        network.eval()  # Set to evaluation mode
        print("Model loaded successfully!")
    else:
        print("Model file not found!")
        exit()

    # Print the size of the first convolution layer
    print("The size of the First Convolution Layer is :", network.conv1.weight.size())
    print("\n")

    # Print weights of each layer
    for i in range(10):
        print("Weights of the First Convolution Layer for the filter ", i, " are :")
        print(network.conv1.weight[i].detach().numpy())
        print("\n")
    
    plot_taskA(network)
    plot_taskB(network, test_dataset)

if __name__ == "__main__":
    main(sys.argv)