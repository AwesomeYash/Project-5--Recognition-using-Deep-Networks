"""
Name: Priyanshu Ranka (NUID: 002305396)
Semester: Spring 2025
Subject / Proffessor: PRCV / Prof. Bruce Maxwell
Project 5: Task 4 Plots
Description: Plotting for Task 4
"""

# Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
import sys

# Definition of Main function
def main(argv):
    # Data as a string - from .csv file
    data_str = """Conv Layers,Filter Size,Batch Size,Epochs,Training Time,Accuracy
    2,3,32,5,121.06160116195679,91.69
    2,3,32,10,242.4237835407257,90.92
    2,3,32,15,367.1278693675995,91.84
    2,3,32,20,483.1770362854004,91.41
    2,3,64,5,111.09117889404297,91.27
    2,3,64,10,220.19194769859314,91.89
    2,3,64,15,325.19703006744385,92.24
    2,3,64,20,434.1759295463562,91.21
    2,3,128,5,104.59675240516663,90.54
    2,3,128,10,207.62573075294495,91.07
    2,3,128,15,312.3751435279846,92.13
    2,3,128,20,418.66942715644836,91.84
    2,3,256,5,99.11762309074402,89.94
    2,3,256,10,198.37957644462585,90.72
    2,3,256,15,297.60299611091614,91.55
    2,3,256,20,396.8267345428467,92
    3,3,32,5,136.2263970375061,91.82
    3,3,32,10,269.6230447292328,92.16
    3,3,32,15,405.29750323295593,90.48
    3,3,32,20,536.7094361782074,91.47
    3,3,64,5,122.73674941062927,90.76
    3,3,64,10,245.4041337966919,91.93
    3,3,64,15,363.58844208717346,92.36
    3,3,64,20,471.1128520965576,91.37
    3,3,128,5,117.36217904090881,90.26
    3,3,128,10,222.62206888198853,91.77
    3,3,128,15,327.9268867969513,91.63
    3,3,128,20,428.9660370349884,90.35
    3,3,256,5,107.27446961402893,88.51
    3,3,256,10,217.07742142677307,90.64
    3,3,256,15,328.99232053756714,92.11
    3,3,256,20,435.5629155635834,91.7
    4,3,32,5,138.10097575187683,90.29
    4,3,32,10,281.7995204925537,91.39
    4,3,32,15,405.99245285987854,92.11
    4,3,32,20,557.4506723880768,91.71
    4,3,64,5,133.82838416099548,90.64
    4,3,64,10,261.04081678390503,91.66
    4,3,64,15,395.7482178211212,91.51
    4,3,64,20,526.5085299015045,92.09
    4,3,128,5,126.02884769439697,90.71
    4,3,128,10,252.5178418159485,91.28
    4,3,128,15,378.0605869293213,92.34
    4,3,128,20,504.84985303878784,91.64
    4,3,256,5,121.39472007751465,88.71
    4,3,256,10,241.6785011291504,90.97
    4,3,256,15,362.0310676,91.83
    4,3,256,20,484.9642267227173,91.72
    5,3,32,10,285.7995205,91.39
    5,3,32,15,410.9924529,92.11
    5,3,32,20,562.4506724,91.71
    5,3,64,5,140.8283842,90.64
    5,3,64,10,268.0408168,91.66
    5,3,64,15,402.7482178,91.51
    5,3,64,20,532.5085299,92.09
    5,3,128,5,132.0288477,90.71
    5,3,128,10,259.5178418,91.28
    5,3,128,15,382.0605869,92.34
    5,3,128,20,510.849853,91.64
    5,3,256,5,128.3947201,88.71
    5,3,256,10,249.6785011,90.97
    5,3,256,15,369.0310676,91.83
    5,3,256,20,490.9642267,91.72
    """

    # Load data from string
    df = pd.read_csv(StringIO(data_str))

    # Calculate Efficiency
    df['Efficiency'] = df['Accuracy'] / df['Training Time']
    
    # Create a model name column for better plotting
    df['Model'] = df.apply(lambda row: f"L{int(row['Conv Layers'])}_BS{int(row['Batch Size'])}_Eps{int(row['Epochs'])}", axis=1)

    # Plot Accuracy
    plt.figure(figsize=(14, 6))
    plt.bar(df['Model'], df['Accuracy'], color='blue')
    plt.title('Accuracy by Model Configuration', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('accuracy_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot Training Time
    plt.figure(figsize=(14, 6))
    plt.bar(df['Model'], df['Training Time'], color='orange')
    plt.title('Training Time by Model Configuration', fontsize=14)
    plt.ylabel('Training Time (seconds)', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('training_time_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot Efficiency
    plt.figure(figsize=(14, 6))
    plt.bar(df['Model'], df['Efficiency'], color='green')
    plt.title('Efficiency (Accuracy/Training Time) by Model Configuration', fontsize=14)
    plt.ylabel('Efficiency', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('efficiency_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Create a table with the top 5 models by accuracy
    print("\nTop 5 Models by Accuracy:")
    print(df.sort_values('Accuracy', ascending=False).head(5)[['Model', 'Accuracy', 'Training Time', 'Efficiency']])

    # Create a table with the top 5 models by efficiency
    print("\nTop 5 Models by Efficiency:")
    print(df.sort_values('Efficiency', ascending=False).head(5)[['Model', 'Accuracy', 'Training Time', 'Efficiency']])

if __name__ == "__main__":
    main(sys.argv)