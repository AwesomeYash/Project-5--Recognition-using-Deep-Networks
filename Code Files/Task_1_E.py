"""
Name: Priyanshu Ranka (NUID: 002305396)
Semester: Spring 2025
Subject / Proffessor: PRCV / Prof. Bruce Maxwell
Project 5: Task 1 F
Description: This code reads the network and runs the model on the first 10 examples in the test set. The program also plot the first 9 digits 
as a 3x3 grid with the prediction for each example above it.
"""

# Importing Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Define the network
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

def print_predictions(predictions, labels, outputs, images):

    # Print the results
    for i in range(10):
        output_values = outputs[i].detach().numpy()
        formatted_output = ["{:.2f}".format(value) for value in output_values]
        max_index = np.argmax(output_values)  # Index of max output value
        
        print(f"Image {i+1}:")
        print("Network Output:", formatted_output)
        print(f"Index of Max Output: {max_index}, Predicted Label: {predictions[i].item()}, Correct Label: {labels[i].item()}")
        print("-" * 75)

    # Plot the first 9 test images in a 3x3 grid
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    fig.suptitle("Predictions on Test Images")

    for i, ax in enumerate(axes.flat):
        if i < 9:
            image = images[i][0].numpy()
            ax.imshow(image, cmap="gray")
            ax.set_title(f"Pred: {predictions[i].item()}")
            ax.axis("off")
    plt.show()

def main(argv):
    # Load the test dataset (MNIST)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.Compose([transforms.ToTensor()]), download=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False)

    # Load the trained model
    network = Net()
    model_path = "C:\\Users\\yashr\\Desktop\\NEU\\Semester 2\\PRCV\\Projects\\Project_5\\results\\model.pth"

    # DEBUGGING
    if os.path.exists(model_path):
        network.load_state_dict(torch.load(model_path))
        network.eval()  # Set to evaluation mode
        print("Model loaded successfully!")
    else:
        print("Model file not found!")
        exit()

    # Get the first 10 test samples
    data_iter = iter(test_loader)
    images, labels = next(data_iter)

    # Run the model on the test samples
    outputs = network(images)
    predictions = torch.argmax(outputs, dim=1)

    print_predictions(predictions, labels, outputs, images)

if __name__ == "__main__":
    main(sys.argv)