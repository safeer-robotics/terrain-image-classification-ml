import pandas as pd

results = {
    "Model": ["SVM", "Random Forest", "KNN"],
    "Accuracy": [0.8166, 0.8769, 0.4874],
    "Macro F1-score": [0.8151, 0.8748, 0.4369]
}

df = pd.DataFrame(results)
print(df)

# Save for report
df.to_csv("model_comparison.csv", index=False)
print("\nSaved: model_comparison.csv")
