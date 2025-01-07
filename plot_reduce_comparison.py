import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Open a file where you stored the pickled data
DATASET = 'cifar' #'fashionmnist' - 'cifar'
start_from = 1
with open(f'Mean_accuracies_qcnn_reduce_{DATASET}_final.pkl', 'rb') as f:
   accuracies_qcnn = pickle.load(f)[start_from:]
with open(f'Std_accuracies_qcnn_reduce_{DATASET}_final.pkl', 'rb') as f:
   std_accuracies_qcnn = pickle.load(f)[start_from:]
with open(f'Mean_accuracies_cnn_reduce_{DATASET}_final.pkl', 'rb') as f:
   accuracies_cnn = pickle.load(f)[start_from:]
with open(f'Std_accuracies_cnn_reduce_{DATASET}_final.pkl', 'rb') as f:
   std_accuracies_cnn = pickle.load(f)[start_from:]


dataset_percentages = [0.1, 1, 5, 10] + list(range(20, 110, 20))
dataset_percentages = dataset_percentages[start_from:]
xticks_labels = [f"{percentage}%" for percentage in dataset_percentages]  # Add percentages to x labels

# Plotting the accuracy score against the dataset percentage
plt.figure(figsize=(10, 6))  # Adjust the figure size

# Plotting with fancier options
plt.plot(dataset_percentages, accuracies_qcnn, marker='o', linestyle='-', color='b', label='QuanvNN')
plt.plot(dataset_percentages, accuracies_cnn, marker='o', linestyle='-', color='r', label='CNN')

plt.fill_between(dataset_percentages, np.array(accuracies_qcnn) - np.array(std_accuracies_qcnn),
                 np.array(accuracies_qcnn) + np.array(std_accuracies_qcnn), color='b', alpha=0.2)
plt.fill_between(dataset_percentages, np.array(accuracies_cnn) - np.array(std_accuracies_cnn),
                 np.array(accuracies_cnn) + np.array(std_accuracies_cnn), color='r', alpha=0.2)

plt.xlabel('Percentage of Training data', fontsize=20)  # Increase font size for x label
plt.ylabel('Accuracy', fontsize=20)  # Increase font size for y label
plt.xticks(dataset_percentages, xticks_labels, rotation=45, fontsize=15)  # Increase font size and add percentages to x ticks
plt.yticks(fontsize=15)  # Increase font size for y ticks
plt.legend(fontsize=15)  # Increase font size for legend

if DATASET == 'fashionmnist': 
    DATASET = 'Fashion MNIST'
else: 
    DATASET = 'CIFAR'

plt.title(f'Accuracy vs. Training Data Percentage - {DATASET}', fontsize=20)  # Increase font size for title
plt.ylim(0.7, 0.85)
plt.grid(True)
plt.tight_layout()
# Add vertical lines at each xtick position
#for x in dataset_percentages:
#    plt.axvline(x, color='gray', linestyle='--', linewidth=0.5)

# Customize grid lines
#plt.grid(axis='y', linestyle='--', alpha=0.7)  # Only vertical grid lines with dashed style and alpha blending

plt.show()
