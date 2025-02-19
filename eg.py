import matplotlib.pyplot as plt
import numpy as np

# Data (replace with your actual data)
genes = np.arange(1, 31)  # Number of genes (1 to 30)
naive_bayes = [0.42, 0.47, 0.48, 0.53, 0.52, 0.54, 0.56, 0.55, 0.53, 0.51, 0.52, 0.55, 0.57, 0.59, 0.61, 0.58, 0.59, 0.60, 0.61, 0.62, 0.61, 0.60, 0.61, 0.62, 0.63, 0.62, 0.61, 0.60, 0.61, 0.63]
decision_tree = [0.53, 0.55, 0.58, 0.62, 0.64, 0.59, 0.61, 0.66, 0.65, 0.63, 0.56, 0.54, 0.55, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72, 0.71, 0.70, 0.69, 0.70, 0.71, 0.73]
neural_network = [0.58, 0.61, 0.52, 0.54, 0.63, 0.64, 0.55, 0.53, 0.59, 0.61, 0.63, 0.52, 0.51, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72, 0.71, 0.70, 0.69, 0.70, 0.71, 0.73]
random_forest = [0.50, 0.65, 0.70, 0.72, 0.75, 0.73, 0.71, 0.74, 0.76, 0.75, 0.73, 0.70, 0.72, 0.74, 0.76, 0.75, 0.73, 0.70, 0.72, 0.74, 0.76, 0.75, 0.73, 0.70, 0.72, 0.74, 0.76, 0.75, 0.73, 0.77]
knn_2 = [0.49, 0.50, 0.52, 0.55, 0.58, 0.62, 0.65, 0.68, 0.66, 0.63, 0.60, 0.58, 0.61, 0.63, 0.65, 0.63, 0.61, 0.63, 0.65, 0.63, 0.61, 0.63, 0.65, 0.63, 0.61, 0.63, 0.65, 0.63, 0.61, 0.63]
knn_3 = [0.52, 0.53, 0.55, 0.58, 0.60, 0.63, 0.66, 0.69, 0.67, 0.64, 0.61, 0.59, 0.62, 0.64, 0.66, 0.64, 0.62, 0.64, 0.66, 0.64, 0.62, 0.64, 0.66, 0.64, 0.62, 0.64, 0.66, 0.64, 0.62, 0.64]
knn_4 = [0.50, 0.51, 0.53, 0.56, 0.59, 0.61, 0.64, 0.67, 0.65, 0.62, 0.59, 0.57, 0.60, 0.62, 0.64, 0.62, 0.60, 0.62, 0.64, 0.62, 0.60, 0.62, 0.64, 0.62, 0.60, 0.62, 0.64, 0.62, 0.60, 0.62]

# Plotting
plt.figure(figsize=(10, 6))  # Adjust figure size for better visualization

plt.plot(genes, naive_bayes, label='Naive Bayes', marker='o', markersize=4)
plt.plot(genes, decision_tree, label='Decision Tree', marker='o', markersize=4)
plt.plot(genes, neural_network, label='Neural Network', marker='o', markersize=4)
plt.plot(genes, random_forest, label='Random Forest', marker='o', markersize=4)
plt.plot(genes, knn_2, label='KNN (k=2)', marker='o', markersize=4)
plt.plot(genes, knn_3, label='KNN (k=3)', marker='o', markersize=4)
plt.plot(genes, knn_4, label='KNN (k=4)', marker='o', markersize=4)

plt.xlabel('Number of Genes')
plt.ylabel('Accuracy')
plt.title('Model Accuracy vs Number of Genes')
plt.xlim(0, 31)  # Set x-axis limits to match your graph
plt.ylim(0.44, 0.78)  # Set y-axis limits to match your graph
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Place legend outside the plot
plt.grid(True, linestyle='--', alpha=0.7)  # Add a subtle grid
plt.xticks(np.arange(0, 31, 5))  # Set x-axis ticks for better readability

plt.tight_layout()  # Adjust layout to prevent labels from overlapping
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Data (replace with your actual data)
naive_bayes = [0.42, 0.47, 0.48, 0.53, 0.52, 0.54, 0.56, 0.55, 0.53, 0.51, 0.52, 0.55, 0.57, 0.59, 0.61, 0.58, 0.59, 0.60, 0.61, 0.62, 0.61, 0.60, 0.61, 0.62, 0.63, 0.62, 0.61, 0.60, 0.61, 0.63]
decision_tree = [0.53, 0.55, 0.58, 0.62, 0.64, 0.59, 0.61, 0.66, 0.65, 0.63, 0.56, 0.54, 0.55, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72, 0.71, 0.70, 0.69, 0.70, 0.71, 0.73]
neural_network = [0.58, 0.61, 0.52, 0.54, 0.63, 0.64, 0.55, 0.53, 0.59, 0.61, 0.63, 0.52, 0.51, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72, 0.71, 0.70, 0.69, 0.70, 0.71, 0.73]
random_forest = [0.50, 0.65, 0.70, 0.72, 0.75, 0.73, 0.71, 0.74, 0.76, 0.75, 0.73, 0.70, 0.72, 0.74, 0.76, 0.75, 0.73, 0.70, 0.72, 0.74, 0.76, 0.75, 0.73, 0.70, 0.72, 0.74, 0.76, 0.75, 0.73, 0.77]
knn_2 = [0.49, 0.50, 0.52, 0.55, 0.58, 0.62, 0.65, 0.68, 0.66, 0.63, 0.60, 0.58, 0.61, 0.63, 0.65, 0.63, 0.61, 0.63, 0.65, 0.63, 0.61, 0.63, 0.65, 0.63, 0.61, 0.63, 0.65, 0.63, 0.61, 0.63]
knn_3 = [0.52, 0.53, 0.55, 0.58, 0.60, 0.63, 0.66, 0.69, 0.67, 0.64, 0.61, 0.59, 0.62, 0.64, 0.66, 0.64, 0.62, 0.64, 0.66, 0.64, 0.62, 0.64, 0.66, 0.64, 0.62, 0.64, 0.66, 0.64, 0.62, 0.64]
knn_4 = [0.50, 0.51, 0.53, 0.56, 0.59, 0.61, 0.64, 0.67, 0.65, 0.62, 0.59, 0.57, 0.60, 0.62, 0.64, 0.62, 0.60, 0.62, 0.64, 0.62, 0.60, 0.62, 0.64, 0.62, 0.60, 0.62, 0.64, 0.62, 0.60, 0.62]

data = [naive_bayes, decision_tree, neural_network, random_forest, knn_2, knn_3, knn_4]
labels = ['Naive Bayes', 'Decision Tree', 'Neural Network', 'Random Forest', 'KNN (k=2)', 'KNN (k=3)', 'KNN (k=4)']

plt.figure(figsize=(12, 6))  # Adjust figure size for better visualization

# Create boxplot with customizations
boxplot_parts = plt.boxplot(data, labels=labels, patch_artist=True)

# Customize colors
colors = ['#ADD8E6', '#90EE90', '#FFFFE0', '#FFDAB9', '#D3D3D3', '#FFA07A', '#AFEEEE']  # Example colors
for i, box in enumerate(boxplot_parts['boxes']):
    box.set(facecolor=colors[i], edgecolor='black')  # Set facecolor and edgecolor
for median in boxplot_parts['medians']:
    median.set(color='black', linewidth=2)  # Customize median line
for whisker in boxplot_parts['whiskers']:
    whisker.set(color='gray', linestyle='--')  # Customize whiskers
for cap in boxplot_parts['caps']:
    cap.set(color='gray')  # Customize caps

plt.title('Distribution of Model Performances')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for readability
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add a subtle grid
plt.tight_layout()  # Adjust layout to prevent labels from overlapping
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Correct the data: Make sure all lists have the same length (11 in this case)
data = {
    'Naive Bayes': [0.420, 0.466, 0.480, 0.585, 0.525, 0.553, 0.538, 0.583, 0.597, 0.611, 0.75],
    'Decision Tree': [0.535, 0.521, 0.556, 0.564, 0.662, 0.515, 0.562, 0.577, 0.624, 0.506, 0.70],
    'Neural Network': [0.582, 0.476, 0.534, 0.639, 0.509, 0.496, 0.521, 0.521, 0.466, 0.549, 0.65],
    'Random Forest': [0.712, 0.726, 0.758, 0.727, 0.726, 0.756, 0.756, 0.741, 0.715, 0.770, 0.60],
    'KNN (k=2)': [0.495, 0.537, 0.564, 0.755, 0.727, 0.698, 0.712, 0.569, 0.611, 0.569, 0.55],
    'KNN (k=3)': [0.497, 0.638, 0.638, 0.682, 0.652, 0.655, 0.711, 0.610, 0.624, 0.624, 0.50],
    'KNN (k=4)': [0.527, 0.668, 0.697, 0.710, 0.696, 0.640, 0.625, 0.596, 0.609, 0.609, 0.45]
}

genes = [2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 75]  # Number of genes (must match the length of the data lists)

df = pd.DataFrame(data, index=genes)

# ... (rest of the plotting code remains exactly the same as the previous correct version)
plt.figure(figsize=(12, 7))
ax = sns.heatmap(df, cmap='viridis', annot=True, fmt=".3f", linewidths=.5, cbar_kws={'label': 'Accuracy'})
# ... (rest of the code for labels, title, etc.)

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

# Your data (replace with your actual data if different)
best_accuracies_lists = {  # Keep the lists of accuracies here
    'Naive Bayes': [0.420, 0.466, 0.480, 0.585, 0.525, 0.553, 0.538, 0.583, 0.597, 0.611, 0.75],
    'Decision Tree': [0.535, 0.521, 0.556, 0.564, 0.662, 0.515, 0.562, 0.577, 0.624, 0.506, 0.70],
    'Neural Network': [0.582, 0.476, 0.534, 0.639, 0.509, 0.496, 0.521, 0.521, 0.466, 0.549, 0.65],
    'Random Forest': [0.712, 0.726, 0.758, 0.727, 0.726, 0.756, 0.756, 0.741, 0.715, 0.770, 0.60],
    'KNN (k=2)': [0.495, 0.537, 0.564, 0.755, 0.727, 0.698, 0.712, 0.569, 0.611, 0.569, 0.55],
    'KNN (k=3)': [0.497, 0.638, 0.638, 0.682, 0.652, 0.655, 0.711, 0.610, 0.624, 0.624, 0.50],
    'KNN (k=4)': [0.527, 0.668, 0.697, 0.710, 0.696, 0.640, 0.625, 0.596, 0.609, 0.609, 0.45]
}

# Find the best accuracy for each model (the MAX value in the list)
best_accuracies = {}  # This dictionary will store the max accuracies
for model, accuracies in best_accuracies_lists.items():
    best_accuracies[model] = max(accuracies) # Get the maximum accuracy

# Prepare data for the pie chart
labels = list(best_accuracies.keys())
sizes = list(best_accuracies.values())

# Create the pie chart
plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightcoral', 'lightgreen', 'lightyellow', 'lightpink', 'lightcyan', 'lightgray'])
plt.title('Best Accuracy Achieved by Each Model', fontsize=14, fontweight='bold')
plt.axis('equal')

plt.show()