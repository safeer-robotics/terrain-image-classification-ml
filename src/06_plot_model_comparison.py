import matplotlib.pyplot as plt

models = ["SVM", "Random Forest", "KNN"]
accuracy = [0.8166, 0.8769, 0.4874]
f1_macro = [0.8151, 0.8748, 0.4369]

x = range(len(models))

plt.figure()
plt.bar(x, accuracy, label="Accuracy")
plt.xticks(x, models)
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Model Performance Comparison (Accuracy)")
plt.legend()
plt.tight_layout()
plt.savefig("model_accuracy_comparison.png", dpi=200)
plt.show()

plt.figure()
plt.bar(x, f1_macro, label="Macro F1-score")
plt.xticks(x, models)
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Model Performance Comparison (Macro F1-score)")
plt.legend()
plt.tight_layout()
plt.savefig("model_f1_comparison.png", dpi=200)
plt.show()

print("Saved: model_accuracy_comparison.png and model_f1_comparison.png")
