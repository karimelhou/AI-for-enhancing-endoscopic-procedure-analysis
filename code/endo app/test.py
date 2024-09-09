import matplotlib.pyplot as plt

# Accuracy values
train_accuracy = 0.9806
test_accuracy = 0.9806

# Create bar plot
plt.figure(figsize=(10, 6))
accuracies = [train_accuracy, test_accuracy]
labels = ['Training', 'Test']
colors = ['#3498db', '#e74c3c']

plt.bar(labels, accuracies, color=colors)

# Customize the plot
plt.title('EfficientNetB2 Model Accuracy', fontsize=16)
plt.ylabel('Accuracy', fontsize=14)
plt.ylim(0, 1)  # Set y-axis limits from 0 to 1

# Add value labels on top of each bar
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontsize=12)

# Add a horizontal line at y=1 for reference
plt.axhline(y=1, color='gray', linestyle='--', alpha=0.7)

# Remove top and right spines
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Add grid lines for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()