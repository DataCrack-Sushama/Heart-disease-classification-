# Heart Disease Classification Using Machine Learning

This repository contains a project focused on predicting heart disease using various machine learning algorithms. The dataset used is publicly available and consists of patient health records with multiple attributes.



##  Project Overview

The objective of this project is to develop and compare the performance of multiple machine learning models to classify the presence of heart disease.

###  Algorithms Implemented:
- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Support Vector Machine (SVM)  
- Naive Bayes  
- Decision Tree Classifier  
- Random Forest Classifier  


##  Dataset

The dataset includes medical attributes of patients such as:
- Age  
- Gender  
- Chest Pain Type  
- Resting Blood Pressure  
- Serum Cholesterol  
- Fasting Blood Sugar  
- Resting ECG Results  
- Maximum Heart Rate  
- Exercise-Induced Angina  
- ST Depression  
- Slope of the Peak Exercise ST Segment  
- Number of Major Vessels  
- Thalassemia  
- Target Variable: Presence of Heart Disease (0 or 1)  


##  Steps Involved

1. **Data Preprocessing**  
   - Handling missing values  
   - Encoding categorical variables  
   - Scaling numerical features  

2. **Exploratory Data Analysis (EDA)**  
   - Visualizations to analyze relationships between features and target  

3. **Model Training**  
   - Training multiple machine learning models on the preprocessed data  

4. **Model Evaluation**  
   - Accuracy, Precision, Recall, and F1-Score  
   - Confusion Matrix  
   - ROC Curve  

---

## Results

| Model                   | Accuracy (%) |
|------------------------|--------------|
| Logistic Regression     | 86.89%           |
| K-Nearest Neighbors     | 88.52%           |
| Support Vector Machine  | 88.52%           |
| Naive Bayes             | 86.89%           |
| Decision Tree Classifier| 78.69%           |
| Random Forest Classifier| 88.52%           |


## Requirements

- Python 3.x  
- Libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`

Install dependencies using:
pip install -r requirements.txt

Usage
Clone the repository:
git clone https://github.com/DataCrack-Sushama/Heart-disease-classification.git

Navigate to the directory:
cd Heart-disease-classification

Install dependencies:
pip install -r requirements.txt

Run the notebook:
jupyter notebook heart_disease_classification.ipynb

Conclusion
This project highlights how different machine learning algorithms perform in predicting heart disease. The evaluation results help in selecting the best model based on the dataset characteristics. It demonstrates the importance of preprocessing and model evaluation in medical data analysis.
