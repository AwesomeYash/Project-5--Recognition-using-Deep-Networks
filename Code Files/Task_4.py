"""
Name: Priyanshu Ranka (NUID: 002305396)
Semester: Spring 2025  
Professor: Prof. Bruce Maxwell
Project 5: Task 4
"""

# Importing Libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import pandas as pd
import os

# Define a configurable CNN model
class DynamicCNN(nn.Module):
    def __init__(self, num_conv_layers, filter_size):
        super(DynamicCNN, self).__init__()
        layers = []
        in_channels = 1  # Fashion-MNIST is grayscale (1 channel)
        
        # Calculate proper padding to maintain spatial dimensions
        padding = filter_size // 2
        
        for i in range(num_conv_layers):
            out_channels = 32 * (i + 1)  # Increase filters per layer
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=filter_size, padding=padding))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels  # Update input channels
        
        self.conv = nn.Sequential(*layers)
        
        # Automatically determine FC layer input size
        sample_input = torch.rand(1, 1, 28, 28)  # Dummy input
        conv_output_size = self._get_conv_output_size(sample_input)
        
        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, 128),
            nn.ReLU(),
            nn.Linear(128, 10)  # 10 classes in Fashion-MNIST
        )

    def _get_conv_output_size(self, x):
        """Pass a dummy tensor through conv layers to get the flattened size"""
        with torch.no_grad():
            try:
                x = self.conv(x)
                return x.view(1, -1).size(1)  # Flatten and get size
            except RuntimeError as e:
                print(f"Error in determining conv output size: {e}")
                raise

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten before FC layer
        x = self.fc(x)
        return x

# Train and evaluate function
def train_and_evaluate(num_conv_layers, filter_size, batch_size, num_epochs, device):
    transform = transforms.Compose([transforms.ToTensor()])
    train_loader = torch.utils.data.DataLoader(torchvision.datasets.FashionMNIST(root='./dataFashion', train=True, download=True, 
                                                                                transform=transform), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torchvision.datasets.FashionMNIST(root='./dataFashion', train=False, download=True, 
                                                                                transform=transform), batch_size=batch_size, shuffle=False)
    # Move model to CPU
    model = DynamicCNN(num_conv_layers, filter_size).to(device)  
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Save model after training
    model_path = f'saved_models/model_layers_{num_conv_layers}_filter_{filter_size}_batch_{batch_size}_epochs_{num_epochs}.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    start_time = time.time()
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)  # Move data to CPU
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            # Print statistics every 100 mini-batches
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}')
                running_loss = 0.0
    
    train_time = time.time() - start_time

    # Save model after training
    model_path = f'saved_models/model_layers_{num_conv_layers}_filter_{filter_size}_batch_{batch_size}_epochs_{num_epochs}.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Evaluate model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to CPU
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")
    return train_time, accuracy, model

# Main function
def main():
    # Create directory for saved models if it doesn't exist
    os.makedirs('saved_models', exist_ok=True)
    # Use CPU
    device = torch.device("cpu")

    # Total = 4 x 1 x 4 x 4 = 64 Network variations
    conv_layers = [2, 3, 4, 5]
    filter_sizes = [3]  
    batch_sizes = [32, 64, 128, 256]
    epochs_list = [5, 10, 15, 20]  

    # Run experiments
    results = []
    current_batch_size = None
    last_model = None
    last_config = None
    try:
        for num_layers in conv_layers:
            for batch_size in batch_sizes:
                
                # Check if batch size has changed to save model
                if current_batch_size != batch_size and current_batch_size is not None and last_model is not None:
                    print(f"Batch size changed from {current_batch_size} to {batch_size}")
                    # Save the last trained model from previous batch size
                    batch_change_model_path = f'saved_models/batch_change_from_{current_batch_size}_model_layers_{last_config[0]}_filter_{last_config[1]}_epochs_{last_config[2]}.pth'
                    torch.save(last_model.state_dict(), batch_change_model_path)
                    print(f"Model saved at batch size change: {batch_change_model_path}")
                
                current_batch_size = batch_size
                
                for filter_size in filter_sizes:
                    for epochs in epochs_list:
                        print(f"\nTraining: Layers={num_layers}, Filter={filter_size}, Batch={batch_size}, Epochs={epochs}")
                        try:
                            train_time, accuracy, model = train_and_evaluate(num_layers, filter_size, batch_size, epochs, device)
                            results.append([num_layers, filter_size, batch_size, epochs, train_time, accuracy])
                            
                            # Store this model as the last model for this batch size
                            last_model = model
                            last_config = (num_layers, filter_size, epochs)
                            
                        except RuntimeError as e:
                            print(f"Error during training: {e}")
                            print(f"Skipping configuration: Layers={num_layers}, Filter={filter_size}, Batch={batch_size}, Epochs={epochs}")
                            results.append([num_layers, filter_size, batch_size, epochs, -1, -1])

    except KeyboardInterrupt:
        print("Experiment interrupted by user. Saving partial results...")

    # Save results
    df = pd.DataFrame(results, columns=["Conv Layers", "Filter Size", "Batch Size", "Epochs", "Training Time", "Accuracy"])
    df.to_csv("fashion_mnist_experiment_results.csv", index=False)
    print("All experiments completed. Results saved to 'fashion_mnist_experiment_results.csv'")

if __name__ == "__main__":
    main()