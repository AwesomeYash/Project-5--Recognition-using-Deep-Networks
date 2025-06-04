"""
Name: Priyanshu Ranka (NUID: 002305396)
Semester: Spring 2025
Subject / Proffessor: PRCV / Prof. Bruce Maxwell
Project 5: Task 1 F
Description: This file is used to predict the digits from the written test images using the trained model.
"""
    
# Importing the required libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import os
import matplotlib.pyplot as plt
import sys

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

# Function to preprocess and predict
def predict_digit(image_path, network, transform):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
    if img is None:
        print(f"Error: Could not read {image_path}")
        return None, None
    
    # Invert colors if needed (MNIST is white digits on black background)
    img = cv2.bitwise_not(img)      # if np.mean(img) > 127 else img

    # Convert image to tensor
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    # Run through the network
    with torch.no_grad():
        output = network(img_tensor)
        prediction = output.argmax(dim=1).item()  # Get the predicted class

    return img, prediction

def plot_function(network, transform):

    # Get all test images
    test_folder = "C:\\Users\\yashr\\Desktop\\NEU\\Semester 2\\PRCV\\Projects\\Project_5\\Paint_Test_Images"
    image_files = [f for f in os.listdir(test_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()  # Sort for consistency

    # Plot results
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))  # 2 rows, 5 columns

    for i, image_file in enumerate(image_files[:10]):  # Process only 10 images
        img_path = os.path.join(test_folder, image_file)
        processed_img, predicted_label = predict_digit(img_path, network, transform)

        if processed_img is not None:
            row, col = divmod(i, 5)
            axes[row, col].imshow(processed_img, cmap='gray')
            axes[row, col].set_title(f"Pred: {predicted_label}")
            axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()

def main(argv):
    # Load the trained model
    model_path = "C:\\Users\\yashr\\Desktop\\NEU\\Semester 2\\PRCV\\Projects\\Project_5\\results\\model.pth"
    network = Net()
    network.load_state_dict(torch.load(model_path))
    network.eval()     # Set model to evaluation mode

    # Define transformation (match MNIST format)
    transform = transforms.Compose([
        transforms.ToPILImage(), transforms.Grayscale(num_output_channels=1),  # Convert to grayscale if not already
        transforms.Resize((28, 28)), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))  # Normalize using MNIST dataset mean & std
    ])

    # calling plot function
    plot_function(network, transform)

if __name__ == "__main__":
    main(sys.argv)