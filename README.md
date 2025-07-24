Parkinson's Disease Detection 🧠
This project aims to detect Parkinson's Disease using machine learning models trained on biomedical voice measurements. The goal is to support early diagnosis through data-driven analysis and classification.

📌 Project Overview
Parkinson's Disease is a progressive neurological disorder that affects movement and speech. Early detection is crucial for effective management. This notebook leverages supervised machine learning to classify individuals as either healthy or affected based on vocal features.

📂 Dataset
Source: UCI Machine Learning Repository – Parkinson's Dataset

Instances: 195

Features: 22 (including biomedical voice measurements and one target column status)

Feature	Description
MDVP columns	Various voice frequency measurements
PPE, DFA	Measures of signal noise and complexity
status	Target label (1 = Parkinson's, 0 = Healthy)

⚙️ Workflow
Data Preprocessing

Removed irrelevant columns (name)

Normalized features using StandardScaler

Exploratory Data Analysis (EDA)

Visualized class distribution

Checked for correlations among features

Model Training

Logistic Regression

Accuracy score calculated on both training and testing sets

Prediction System

Developed a real-time prediction function to test with new inputs

📊 Results
Model Used: Logistic Regression

Training Accuracy: ~85%

Testing Accuracy: ~84%

Performs well on small-scale data with minimal overfitting.

🚀 How to Use
Run the notebook in Google Colab or Jupyter:

bash
Copy
Edit
pip install pandas numpy scikit-learn matplotlib seaborn
python
Copy
Edit
# Upload the notebook and run all cells sequentially
To test with your own data, use the input array structure specified in the prediction cell.

📎 Dependencies
Python 3.x

NumPy

Pandas

scikit-learn

Matplotlib

Seaborn

📬 Contact
For questions or contributions:

Charan Vennu
Email: [vennusaicharan09@gmail.com]

LinkedIn: [https://www.linkedin.com/in/vennu-sai-charan-361039285/]

