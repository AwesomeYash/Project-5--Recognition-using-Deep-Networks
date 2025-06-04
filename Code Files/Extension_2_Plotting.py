""" 
Name: Priyanshu Ranka (NUID: 002305396)
Semester: Spring 2025
Subject / Professor: PRCV / Prof. Bruce Maxwell
Project 5: Task 4 Plots
Description: Plotting for Task 4 
"""

# Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
import sys

# Definition of Main function
def main(argv):
    # Load data from CSV file
    df = pd.read_csv('mnist_experiment_results.csv')
    
    # Calculate Efficiency
    df['Efficiency'] = df['Accuracy'] / df['Training Time']
    
    # Create a model name column for better plotting
    df['Model'] = df.apply(lambda row: f"L{int(row['Conv Layers'])}_F{int(row['Filter Size'])}_BS{int(row['Batch Size'])}_E{int(row['Epochs'])}", axis=1)
    
    # Plot Accuracy
    plt.figure(figsize=(14, 6))
    plt.bar(df['Model'], df['Accuracy'], color='blue')
    plt.title('Accuracy by Model Configuration', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('E2_accuracy_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot Training Time
    plt.figure(figsize=(14, 6))
    plt.bar(df['Model'], df['Training Time'], color='orange')
    plt.title('Training Time by Model Configuration', fontsize=14)
    plt.ylabel('Training Time (seconds)', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('E2_training_time_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot Efficiency
    plt.figure(figsize=(14, 6))
    plt.bar(df['Model'], df['Efficiency'], color='green')
    plt.title('Efficiency (Accuracy/Training Time) by Model Configuration', fontsize=14)
    plt.ylabel('Efficiency', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('E2_efficiency_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create a table with the top 5 models by accuracy
    print("\nTop 5 Models by Accuracy:")
    print(df.sort_values('Accuracy', ascending=False).head(5)[['Model', 'Accuracy', 'Training Time', 'Efficiency']])
    
    # Create a table with the top 5 models by efficiency
    print("\nTop 5 Models by Efficiency:")
    print(df.sort_values('Efficiency', ascending=False).head(5)[['Model', 'Accuracy', 'Training Time', 'Efficiency']])

if __name__ == "__main__":
    main(sys.argv)