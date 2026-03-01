
# Terrain Image Classification Using Machine Learning

This is my Machine Learning coursework project. In this project, I built and compared multiple classical machine learning models to classify terrain images. The goal was to extract meaningful features from images and evaluate which model performs best for terrain classification.

I followed a complete workflow: data preparation → feature extraction → model training → evaluation → comparison.

## Project Overview

The objective of this project was to classify terrain images into different categories using supervised machine learning algorithms.

I performed:

- Data preprocessing and feature extraction  
- Training multiple machine learning models  
- Evaluating performance using accuracy and macro F1 score  
- Comparing model performance using visual plots and confusion matrices  

This project helped me understand how different machine learning algorithms behave on the same dataset.

## Dataset

The dataset consists of labeled terrain images.

Steps performed:

- Loaded and inspected dataset  
- Checked class balance  
- Extracted image features  
- Split data into training and testing sets  
- Scaled features using StandardScaler  

## Models Used

I trained and compared the following models:

- Support Vector Machine  
- Random Forest Classifier  
- K Nearest Neighbours  

Each model was trained on the same dataset to ensure fair comparison.

## Evaluation Metrics

I evaluated the models using:

- Accuracy score  
- Macro F1 score  
- Confusion matrix  
- Model comparison plots  

I generated visual comparison graphs for:

- Accuracy comparison  
- F1 score comparison  
- Confusion matrices for each model  

## Technologies Used

- Python  
- NumPy  
- Pandas  
- Scikit learn  
- Matplotlib  
- Seaborn  

## Repository Structure

- 01_extract_features.py → Feature extraction script  
- 02_train_svm.py → SVM model training  
- 03_train_rf.py → Random Forest training  
- 04_train_knn.py → KNN training  
- 05_model_comparison.py → Model performance comparison  
- 06_plot_model_comparison.py → Accuracy and F1 visualization  
- 07_plot_confusion_matrices.py → Confusion matrix generation  
- 08_line_graph_comparison.py → Performance line graph  

- Dataset/ → Image dataset  
- PNG files → Generated result graphs  
- model_comparison.csv → Performance results  

Note: Trained model files are not included because of file size. You can regenerate them by running the training scripts.

## How to Run

1. Install required libraries:

pip install numpy pandas scikit-learn matplotlib seaborn

2. Run feature extraction:

python 01_extract_features.py

3. Train models:

python 02_train_svm.py  
python 03_train_rf.py  
python 04_train_knn.py  

4. Generate comparison results and plots:

python 05_model_comparison.py  
python 06_plot_model_comparison.py  
python 07_plot_confusion_matrices.py  
python 08_line_graph_comparison.py  

## What I Learned

- Feature extraction from image datasets  
- Training and comparing multiple machine learning models  
- Understanding bias and variance  
- Evaluating performance using multiple metrics  
- Visualising model performance clearly  
- Organising ML projects in a structured workflow  

## Author

Safeer Ahmed  
Email: safeerahmed5471@gmail.com  
