# Disease Prediction from Medical Data

## CodeAlpha Machine Learning Internship - Task 4

## Project Overview

This project predicts whether a breast tumor is **Malignant** or **Benign** using structured medical data and machine learning classification algorithms.

The project uses the Breast Cancer Wisconsin Diagnostic dataset available through Scikit-learn.

---

## Objective

To predict the possibility of disease using medical measurements and classification models.

---

## Dataset

The dataset contains 569 patient records and 30 numerical medical features.

Target classes:

- Malignant
- Benign

The dataset includes measurements such as:

- Mean radius
- Mean texture
- Mean perimeter
- Mean area
- Mean smoothness
- Compactness
- Concavity
- Symmetry
- Fractal dimension

---

## Algorithms Used

- Logistic Regression
- Support Vector Machine
- Decision Tree
- Random Forest

---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix
- Classification Report

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib
- Jupyter Notebook

---

## Project Structure

```text
CodeAlpha_DiseasePrediction/
│
├── dataset.csv
├── disease_prediction.ipynb
├── disease_prediction.py
├── README.md
├── requirements.txt
├── Project_Report.pdf
│
├── models/
│   ├── disease_prediction_model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
│
└── images/
    ├── accuracy_comparison.png
    ├── confusion_matrix.png
    ├── feature_importance.png
    ├── roc_curve.png
    ├── correlation_heatmap.png
    └── dataset_distribution.png
```

---

## How to Run

### Install requirements

```bash
pip install -r requirements.txt
```

### Run the notebook

Open `disease_prediction.ipynb` in Google Colab or Jupyter Notebook and run all cells.

### Run the Python script

```bash
python disease_prediction.py
```

---

## Outputs

The project generates:

- Accuracy comparison graph
- Confusion matrix
- ROC curve
- Feature importance graph
- Correlation heatmap
- Diagnosis distribution chart
- Saved machine learning model

---

## Future Scope

- Add more real-world patient data
- Use XGBoost and LightGBM
- Perform hyperparameter tuning
- Deploy the model with Streamlit
- Build a user-friendly disease prediction interface

---

## Author

**Sandesh Solagi**

Machine Learning Intern  
CodeAlpha Internship Program

---

## Disclaimer

This project is for educational purposes only and is not intended for clinical diagnosis.

---

## License

This project is developed for educational and internship purposes.
