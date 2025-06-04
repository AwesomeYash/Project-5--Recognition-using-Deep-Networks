"""
Name: Priyanshu Ranka (NUID: 002305396)
Semester: Spring 2025  
Professor: Prof. Bruce Maxwell
Project 5: Task 3
Description: 
"""

# Importing Libraries
import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from PIL import Image
import sys
import numpy as np

# Define the MNIST network structure (same as Task 1)
class Net(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.fc1 = nn.Linear(in_features=320, out_features=50)
        self.fc2 = nn.Linear(in_features=50, out_features=10)  # Original output for MNIST (10 digits)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.dropout(self.conv2(x)), kernel_size=2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # Removed log_softmax to allow changing the final layer
        return x

# Greek data transform class
class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        #x = torchvision.transforms.functional.affine(x, 0, (0, 0), 36/128, 0)
        #x = torchvision.transforms.functional.center_crop(x, (28, 28))
        # # Resize to 28x28 instead of cropping
        x = torchvision.transforms.functional.resize(x, (28, 28))
        
        return torchvision.transforms.functional.invert(x)

# Function to visualize some examples from the dataset
def visualize_greek_examples(greek_train):
    examples = enumerate(greek_train)
    batch_idx, (example_data, example_targets) = next(examples)
    
    # Map class indices to Greek letter names
    class_names = ['Alpha', 'Beta', 'Gamma']
    
    fig = plt.figure(figsize=(10, 6))
    for i in range(min(6, len(example_data))):
        plt.subplot(2, 3, i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title(f"{class_names[example_targets[i]]}")
        plt.xticks([])
        plt.yticks([])
    plt.show()

# Function to visualize results after training
def visualize_model_predictions(network, greek_train):
    network.eval()
    examples = enumerate(greek_train)
    batch_idx, (example_data, example_targets) = next(examples)
    
    class_names = ['Alpha', 'Beta', 'Gamma']
    
    # Get predictions
    with torch.no_grad():
        outputs = network(example_data)
        predictions = outputs.argmax(dim = 1)
    
    # Plot images with both ground truth and predictions
    fig = plt.figure(figsize=(12, 8))
    for i in range(min(6, len(example_data))):
        plt.subplot(2, 3, i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        correct = predictions[i] == example_targets[i]
        color = "green" if correct else "red"
        plt.title(f"True: {class_names[example_targets[i]]}\nPred: {class_names[predictions[i]]}", 
                 color=color)
        plt.xticks([])
        plt.yticks([])
    plt.show()

# Load the pre-trained MNIST model and modify it for Greek letters
def prepare_transfer_learning_model(save_dir):
    # Create a new instance of the network
    network = Net(dropout_rate=0.3)
    
    # Load the pre-trained weights from the MNIST model
    model_path = os.path.join(save_dir, "model.pth")
    if os.path.exists(model_path):
        network.load_state_dict(torch.load(model_path))
        print("Loaded pre-trained MNIST model")
    else:
        raise FileNotFoundError(f"No model found at {model_path}. Please run Task 1 first.")
    
    # Freeze all parameters in the network
    for param in network.parameters():
        param.requires_grad = False
    
    # Replace the last layer with a new Linear layer for 3 classes (alpha, beta, gamma)
    network.fc2 = nn.Linear(in_features=50, out_features=3)
    
    # Verify which parameters are trainable
    print("\nTrainable parameters:")
    for name, param in network.named_parameters():
        if param.requires_grad:
            print(f"  {name}")
    
    return network

# Training function
def train(network, optimizer, epoch, greek_train, log_interval):
    network.train()
    correct = 0
    total = 0
    
    # main training loop
    for batch_idx, (data, target) in enumerate(greek_train):
        optimizer.zero_grad()
        output = network(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += len(data)
        
        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(greek_train.dataset)} '
                  f'({100. * batch_idx / len(greek_train):.0f}%)]\tLoss: {loss.item():.6f} '
                  f'Accuracy: {100. * correct / total:.1f}%')

# Testing function
def test(network, greek_train):
    network.eval()
    test_loss = 0
    correct = 0
    class_correct = [0, 0, 0]
    class_total = [0, 0, 0]
    class_names = ['Alpha', 'Beta', 'Gamma']
    
    with torch.no_grad():
        for data, target in greek_train:  # Using the same data for testing in this example
            output = network(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            # Per-class accuracy
            for i in range(len(target)):
                label = target[i]
                class_correct[label] += (pred[i] == label).item()
                class_total[label] += 1
    
    test_loss /= len(greek_train.dataset)
    accuracy = 100. * correct / len(greek_train.dataset)
    """
    DEBUGGING
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(greek_train.dataset)} ({accuracy:.1f}%)')
    
    # Print per-class accuracy
    print('\nPer-class accuracy:')
    for i in range(3):
        class_acc = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        print(f'{class_names[i]}: {class_acc:.1f}% ({class_correct[i]}/{class_total[i]})')
    """
    return accuracy

# Function to predict a single custom image
def predict_custom_image(network, image_path):
    # Load and transform the image
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        GreekTransform(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Get prediction
    network.eval()
    with torch.no_grad():
        output = network(image_tensor)
        prediction = output.argmax(dim=1).item()
        confidence = F.softmax(output, dim=1)[0][prediction].item() * 100
    
    class_names = ['Alpha', 'Beta', 'Gamma']
    return class_names[prediction], confidence

# Main function definition
def main():
    # Parameters
    n_epochs = 30  # Adjustable based on training performance
    batch_size_train = 5
    learning_rate = 0.001
    momentum = 0.5
    log_interval = 1

    # Set random seed for reproducibility
    torch.backends.cudnn.enabled = False
    torch.manual_seed(1)
    # Define paths
    save_dir = "results"  # Path where model.pth is saved
    training_set_path = "greek_train"  # Path to greek letters dataset

    # Create DataLoader for the Greek dataset
    greek_train = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(training_set_path, transform=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor(), GreekTransform(),
                                        torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
                                        batch_size=batch_size_train, shuffle=True)

    # Visualize example images from the Greek dataset
    print("Visualizing examples from the Greek dataset...")
    visualize_greek_examples(greek_train)
    
    # Prepare the transfer learning model
    print("\nPreparing transfer learning model...")
    network = prepare_transfer_learning_model(save_dir)
    
    # Define optimizer (only for the new layer)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, network.parameters()), 
                          lr=learning_rate, momentum=momentum)
    
    # Training loop
    print("\nStarting training...")
    accuracies = []
    for epoch in range(1, n_epochs + 1):
        train(network, optimizer, epoch, greek_train, log_interval)
        accuracy = test(network, greek_train)
        accuracies.append(accuracy)
    
    # Visualizing the model predictions
    visualize_model_predictions(network, greek_train)

    # Save the model
    transfer_model_path = os.path.join(save_dir, "greek_model.pth")
    torch.save(network.state_dict(), transfer_model_path)
    print(f"\nModel saved to {transfer_model_path}")
    
    # Plot training accuracy over epochs
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_epochs + 1), accuracies, marker='o')
    plt.title('Training Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.show()
    
# Main execution
if __name__ == "__main__":
    main()