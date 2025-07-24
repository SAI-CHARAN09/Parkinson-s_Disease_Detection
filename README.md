# 🧠 Parkinson's Disease Detection using Machine Learning

This repository presents a machine learning approach for the early detection of Parkinson’s Disease using biomedical voice features. The project demonstrates end-to-end steps including data preprocessing, model training, evaluation, and comparison across various classification algorithms.

---

## 📌 Problem Statement

Parkinson's Disease is a progressive neurological disorder that impacts movement and speech. Early diagnosis can significantly improve patient outcomes. This project leverages supervised learning techniques to classify individuals as healthy or affected based on voice-related biomedical measurements.

---

## 📂 Dataset

- **Source**: [UCI Machine Learning Repository – Parkinson's Dataset](https://archive.ics.uci.edu/ml/datasets/parkinsons)
- **Samples**: 195  
- **Features**: 23 + 1 target label (`status`)  
- **Target Column**:  
  - `1` = Parkinson’s Positive  
  - `0` = Healthy

---

## 🧪 Project Workflow

### 1. 📥 Import Libraries
- `NumPy`, `Pandas`, `scikit-learn` for data handling and model building

### 2. 📊 Data Exploration & Preprocessing
- Loaded dataset using `pandas`
- Validated data integrity (no nulls)
- Explored structure with `.info()` and `.describe()`
- Extracted features (`X`) and label (`y`)
- Standardized features using `StandardScaler`

### 3. 🧾 Train-Test Split
- Split dataset into 65% training and 35% testing
- Used `train_test_split` from `sklearn.model_selection`

### 4. 🤖 Model Training & Evaluation

| Model                     | Training Accuracy | Test Accuracy |
|---------------------------|-------------------|---------------|
| Support Vector Machine    | 92.06%            | 88.41%        |
| Logistic Regression       | 87.30%            | 76.81%        |
| K-Nearest Neighbors (KNN) | 96.83%            | 91.30%        |
| Decision Tree Classifier  | 100%              | 76.81%        |

### 5. 📈 Metrics Used
- Accuracy
- Confusion Matrix
- Precision, Recall, F1-Score

---

## ✅ Key Findings

- **KNN** outperformed other models with **91.30%** test accuracy.
- **SVM** showed robust generalization with **88.41%** accuracy.
- **Decision Tree** overfitted with perfect training accuracy but dropped in testing.
- **Logistic Regression** was consistent but less precise on the healthy class (`0`).

---

## 🚀 How to Run This Project

1. Clone this repository or open in [Google Colab](https://colab.research.google.com/drive/1xHwTxqrgo1xNmmBAKCN4GAquHnk3gLq0)
2. Ensure the dataset `parkinsons data.csv` is available in your working directory.
3. Run all cells sequentially to see preprocessing, training, and evaluation in action.

---

## 🛠️ Technologies Used

- **Python 3.9+**
- `pandas`
- `numpy`
- `scikit-learn` (SVM, KNN, Logistic Regression, Decision Tree, preprocessing, metrics)

---

## 🔮 Future Enhancements

- Apply PCA or other dimensionality reduction techniques
- Explore ensemble methods like **Random Forest**, **XGBoost**, or **Gradient Boosting**
- Integrate k-fold cross-validation for robust performance estimation

---

## 👤 Author

Name: Vennu Sai Charan

Machine Learning Enthusiast | Open to Collaboration  

📧 Email: [vennusaicharan09@gmail.com]

🔗 LinkedIn Profile: [https://www.linkedin.com/in/vennu-sai-charan-361039285/]

---

## 📄 License

This project is intended for educational and research purposes. For commercial use, please contact the author.
