import matplotlib.pyplot as plt

# Model names
models = ['SVM', 'Random Forest', 'KNN']

# Scores from your results
accuracy = [0.82, 0.88, 0.49]
f1_score = [0.81, 0.87, 0.44]

# Create line plot
plt.figure()
plt.plot(models, accuracy, marker='o', label='Accuracy')
plt.plot(models, f1_score, marker='o', label='Macro F1-score')

# Labels and title
plt.xlabel('Machine Learning Models')
plt.ylabel('Score')
plt.title('Performance Comparison of Machine Learning Models')
plt.legend()
plt.grid(True)

# Show plot
plt.show()
