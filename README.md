# 🧠 Deep Learning for Digit and Character Recognition

**Author**: Priyanshu Ranka  
**NUID**: 002305396  
**Course**: Pattern Recognition and Computer Vision (PRCV)  
**Professor**: Prof. Bruce Maxwell  
**Semester**: Spring 2025  
**OS**: Windows 11  
**Editor**: VS Code  
**Time Travel Days Used**: 3 of 8 🚀

---
## 📌 Project Overview

This project explores **deep learning for image classification**, focusing on both digit and character recognition. Using the **MNIST dataset** and transfer learning on **Greek letters**, the project covers model design, training, visualization, adaptation, and experimentation.

The entire project is organized across **five main tasks**, each building on the previous:

---
## 🧠 Tasks Breakdown

### 🔹 Task 1: Train a CNN on MNIST
- Build and train a convolutional neural network (CNN) for digit recognition.
- Emphasizes modular code using classes/functions.
- Saves the trained model for reuse.

### 🔹 Task 2: Network Visualization
- Visualize the learned filters from the first convolutional layer.
- Analyze filter effects on input images to interpret the network’s inner workings.

### 🔹 Task 3: Transfer Learning to Greek Letters
- Modify and reuse the pre-trained MNIST model to classify Greek characters.
- Retrain the output layer and evaluate performance on new datasets.

### 🔹 Task 4: Automated Experimentation
- Automate tests with varying architectures, hyperparameters, and datasets.
- Analyze and plot performance to identify optimal configurations.

---

## 📁 Folder Structure

Project_5/

├── Code Files/

│ ├── Task_1_ABCD/

│ ├── Task_1_E/

│ ├── Task_1_F/

│ ├── Task_2/

│ ├── Task_3/

│ ├── Task_4/

│ ├── Task_4_Plotting/

│ ├── Extension_1/

│ ├── Extension_2/

│ └── Extension_2_Plotting/

├── CSV Files/

│ ├── fashion_mnist_experiment_results.csv

│ └── mnist_experiment_results.csv

├── Datasets/

│ ├── data/ # MNIST

│ ├── dataFashion/ # Fashion MNIST

│ ├── Greek_train/

│ ├── Paint_test_images/

│ └── Written_test_images/

├── Models/ # Saved model checkpoints

├── Outputs/ # Performance metrics, logs, visual results

├── PRCV_Project_5_Report.pdf # Final report

└── README.md # This file

---
## 🛠️ Requirements

Install dependencies using pip:

```bash
pip install torch torchvision matplotlib numpy pandas opencv-python
```

---
## ▶️ How to Run
Clone the repository.

Ensure all required datasets and .csv files are extracted into the corresponding folders.

Navigate to the specific task directory.

Run the appropriate script directly using Python:
```bash
python task_script.py
```
📌 Note: For Task 4 and Extensions, make sure CSV files are in the same folder as the code.

---
## 📊 Outputs
Accuracy plots

Filter visualizations

Feature maps

Transfer learning results

Experiment summaries in CSV format

---
## 🔗 Project Resources

- [📂 Outputs Folder](https://northeastern-my.sharepoint.com/:f:/g/personal/ranka_pr_northeastern_edu/EvZCFr95vyZHv6nNxy2BWYsBR9aHtED5cVeYv9j-Vt4nLA?e=44QEsq)  
- [📁 Models Folder](https://northeastern-my.sharepoint.com/:f:/g/personal/ranka_pr_northeastern_edu/EsoEcx2mPXpGq4vcgTME-xgBK8k-1Rs0d1uoSNK34Ontaw?e=nAQvWB)  
- [🗃️ Datasets Folder](https://northeastern-my.sharepoint.com/:f:/g/personal/ranka_pr_northeastern_edu/EpUryZ6QClFJu_l6hjENcx4Blq3u_d9GVoxSlapKWSer6A?e=WD9BXv)


---
## 💡 Concepts Covered
Convolutional Neural Networks (CNNs)

Feature visualization

Transfer learning

Model evaluation

Hyperparameter optimization

Experiment tracking with CSV logging

---
## 📬 Contact
Feel free to connect with me on LinkedIn to discuss deep learning, computer vision, or collaborations.

---
## 📖 License
This project was developed as part of coursework for PRCV.

Use is limited to educational and research purposes.
