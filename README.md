# Diabetes Prediction System



## 📋 Table of Contents
- [Overview](#overview)
- [Demo](#demo)
- [Dataset Description](#dataset-description)
- [Project Architecture](#project-architecture)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Model Performance](#model-performance)
- [Technical Implementation](#technical-implementation)
- [Installation Guide](#installation-guide)
- [Usage Instructions](#usage-instructions)
- [Feature Importance Analysis](#feature-importance-analysis)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [References](#references)

## 📊 Overview

The Diabetes Prediction System is a web application that helps predict the likelihood of diabetes in patients based on various health parameters. This project uses machine learning to analyze patient data and provide a binary classification (diabetic or non-diabetic) to assist in early diagnosis and preventive healthcare.

### Problem Statement

Diabetes is a growing health concern worldwide, with many cases remaining undiagnosed until complications arise. Early detection is crucial for effective management and prevention of serious complications. Traditional diagnostic methods often require patients to exhibit symptoms or undergo comprehensive blood tests, which may delay detection.

### Solution Approach

This project implements a machine learning-based solution that:
- Analyzes several health metrics that are commonly measured in routine check-ups
- Trains a logistic regression model on historical patient data
- Enables real-time predictions through a user-friendly web interface
- Helps identify potential diabetic patients who might require further medical examination

## 🎬 Demo



The web application allows users to input health parameters such as:
- Number of pregnancies (for female patients)
- Glucose levels
- Blood pressure
- Skin thickness
- Insulin levels
- BMI (Body Mass Index)
- Diabetes Pedigree Function (a measure of diabetes family history)
- Age

The system then returns a prediction of either "Diabetic" or "Non-Diabetic" based on these inputs.

## 📑 Dataset Description

The project uses the Pima Indians Diabetes Dataset, which contains medical information for female patients of Pima Indian heritage. This dataset is widely used for binary classification tasks in machine learning.

### Dataset Features

| Feature | Description | Unit | Range in Dataset |
|---------|-------------|------|-----------------|
| **Pregnancies** | Number of times pregnant | Count | 0-17 |
| **Glucose** | Plasma glucose concentration at 2 hours in an oral glucose tolerance test | mg/dL | 0-199 |
| **BloodPressure** | Diastolic blood pressure | mm Hg | 0-122 |
| **SkinThickness** | Triceps skin fold thickness | mm | 0-99 |
| **Insulin** | 2-Hour serum insulin | μU/ml | 0-846 |
| **BMI** | Body mass index (weight in kg/(height in m)²) | kg/m² | 0-67.1 |
| **DiabetesPedigreeFunction** | Diabetes pedigree function (a function that scores likelihood of diabetes based on family history) | Score | 0.078-2.42 |
| **Age** | Age of the patient | Years | 21-81 |
| **Outcome** | Class variable (0: Non-diabetic, 1: Diabetic) | Binary | 0-1 |

### Data Statistics

Below is a statistical summary of the dataset:

```
Number of instances: 768
Number of attributes: 9 (including the target variable)
Missing values: Some features contain zero values that were replaced with mean values during preprocessing
Class distribution: 65.1% Non-diabetic (0), 34.9% Diabetic (1)
```

### Data Visualization

Here's a visualization of the dataset showing the distribution of features:


## 🏗️ Project Architecture

The project follows a client-server architecture with the following components:


### 1. Data Preprocessing and Model Training

This phase includes:
- Data cleaning and preprocessing
- Feature scaling
- Model selection and training
- Model evaluation
- Model serialization (saving trained models)

### 2. Web Application

The web application is built using:
- **Backend**: Flask (Python web framework)
- **Frontend**: HTML/CSS
- **Model Deployment**: Pickle for model serialization/deserialization

### 3. Prediction Flow

The prediction workflow follows these steps:
1. User inputs health parameters through the web form
2. Flask backend receives the input data
3. Data is preprocessed using the saved scaler
4. Preprocessed data is fed to the trained model
5. Model makes a prediction (Diabetic or Non-Diabetic)
6. Result is displayed to the user

## 🔄 Machine Learning Pipeline

The machine learning pipeline consists of several stages:


### 1. Data Exploration and Preparation

The dataset was explored to understand the distributions, correlations, and potential issues:

- **Handling Missing Values**: Zero values in certain features (Glucose, BloodPressure, SkinThickness, Insulin, BMI) were replaced with the mean values of those features, as zeros are physiologically implausible.

- **Feature Analysis**: Exploratory Data Analysis (EDA) was performed to understand the distribution of each feature and its relationship with the target variable.

### 2. Data Preprocessing

Before training, the following preprocessing steps were applied:

- **Data Splitting**: The dataset was split into 75% training and 25% testing sets using `train_test_split` from scikit-learn.

- **Feature Scaling**: Standardization was applied to normalize feature values using `StandardScaler` from scikit-learn. This ensures that all features contribute equally to the model's decision-making process.

### 3. Model Training

Logistic Regression was chosen as the classification algorithm due to its:
- Interpretability
- Good performance on linearly separable data
- Low computational requirements
- Ability to provide probability scores

The model was trained using scikit-learn's `LogisticRegression` class with the following optimizations:

- **Hyperparameter Tuning**: GridSearchCV was used to find the optimal hyperparameters for the logistic regression model.
- **Cross-Validation**: 10-fold cross-validation was implemented to ensure the model's generalizability.

### 4. Model Evaluation

The trained model was evaluated using various metrics:

- **Accuracy**: Measures the overall correctness of the model
- **Precision**: Measures the accuracy of positive predictions
- **Recall**: Measures the ability to find all positive instances
- **F1-score**: Harmonic mean of precision and recall

### 5. Model Serialization

The trained model and the standard scaler were serialized using pickle to be deployed in the web application:
- `standardScalar.pkl`: Contains the fitted scaler
- `modelForPrediction.pkl`: Contains the trained logistic regression model

## 📈 Model Performance

The model achieved the following performance metrics on the test dataset:

| Metric | Value |
|--------|-------|
| Accuracy | 79.69% |
| Precision | 90.00% |
| Recall | 81.82% |
| F1-Score | 85.71% |

### Confusion Matrix

The confusion matrix provides a detailed breakdown of the model's predictions:



```
Confusion Matrix:
[[117,  13],
 [ 26,  36]]
```

Where:
- True Negatives (TN) = 117: Correctly predicted as Non-Diabetic
- False Positives (FP) = 13: Incorrectly predicted as Diabetic
- False Negatives (FN) = 26: Incorrectly predicted as Non-Diabetic
- True Positives (TP) = 36: Correctly predicted as Diabetic

## 💻 Technical Implementation

### Project Structure

```
Diabetes-Prediction/
│
├── Model/
│   ├── standardScalar.pkl        # Saved standard scaler
│   └── modelForPrediction.pkl    # Saved logistic regression model
│
├── Notebook/
│   ├── Logistic_Regression.ipynb # Jupyter notebook with model development
│   └── diabetes.csv              # Dataset
│
├── templates/
│   ├── home.html                 # Form template for data input
│   ├── index.html                # Homepage template
│   └── single_prediction.html    # Template for displaying results
│
├── application.py                # Flask application
├── requirements.txt              # Dependencies
└── README.md                     # Project documentation
```

### Key Components

#### 1. Model Development (Logistic_Regression.ipynb)

The Jupyter notebook contains the entire model development process:

```python
# Key steps in the notebook:

# 1. Data loading and exploration
data = pd.read_csv('diabetes.csv')
data.describe()

# 2. Data preprocessing
# Replacing zeros with mean values
data['BMI'] = data['BMI'].replace(0, data['BMI'].mean())
data['Glucose'] = data['Glucose'].replace(0, data['Glucose'].mean())
data['Insulin'] = data['Insulin'].replace(0, data['Insulin'].mean())
data['SkinThickness'] = data['SkinThickness'].replace(0, data['SkinThickness'].mean())
data['BloodPressure'] = data['BloodPressure'].replace(0, data['BloodPressure'].mean())

# 3. Feature selection and target definition
X = data.drop(columns=['Outcome'])
y = data['Outcome']

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# 5. Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Model training with hyperparameter tuning
parameters = {
    'penalty': ['l1', 'l2'], 
    'C': np.logspace(-3, 3, 7),
    'solver': ['newton-cg', 'lbfgs', 'liblinear'],
}

log_reg = LogisticRegression()
clf = GridSearchCV(log_reg, param_grid=parameters, scoring='accuracy', cv=10)
clf.fit(X_train_scaled, y_train)

# 7. Model evaluation
y_pred = clf.predict(X_test_scaled)
conf_mat = confusion_matrix(y_test, y_pred)
accuracy = (conf_mat[0][0] + conf_mat[1][1]) / (conf_mat[0][0] + conf_mat[0][1] + conf_mat[1][0] + conf_mat[1][1])
precision = conf_mat[0][0] / (conf_mat[0][0] + conf_mat[0][1])
recall = conf_mat[0][0] / (conf_mat[0][0] + conf_mat[1][0])
f1_score = 2 * (recall * precision) / (recall + precision)

# 8. Model serialization
import pickle
file = open('modelForPrediction.pkl', 'wb')
pickle.dump(log_reg, file)
file.close()
```

#### 2. Flask Application (application.py)

The Flask application serves as the backend for the web interface:

```python
from flask import Flask, request, app, render_template
from flask import Response
import pickle
import numpy as np
import pandas as pd

application = Flask(__name__)
app = application

# Load the saved scaler and model
scaler = pickle.load(open("Model/standardScalar.pkl", "rb"))
model = pickle.load(open("Model/modelForPrediction.pkl", "rb"))

# Route for homepage
@app.route("/")
def index():
    return render_template("index.html")

# Route for prediction
@app.route("/predictdata", methods=['GET', 'POST'])
def predict_datapoint():
    result = ""

    if request.method == 'POST':
        # Get the data from the form
        Pregnancies = int(request.form.get("Pregnancies"))
        Glucose = float(request.form.get('Glucose'))
        BloodPressure = float(request.form.get('BloodPressure'))
        SkinThickness = float(request.form.get('SkinThickness'))
        Insulin = float(request.form.get('Insulin'))
        BMI = float(request.form.get('BMI'))
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
        Age = float(request.form.get('Age'))

        # Scale the input data and make prediction
        new_data = scaler.transform([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        predict = model.predict(new_data)

        # Format the result
        if predict[0] == 1:
            result = 'Diabetic'
        else:
            result = 'Non-Diabetic'
            
        return render_template('single_prediction.html', result=result)
    
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0")
```

#### 3. HTML Templates

The application uses three HTML templates:

**home.html** - Contains the form for user input:
```html
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>ML API</title>
</head>
<body>
 <div class="login">
    <h1>Diabetes Prediction</h1>
    <form action="{{ url_for('predict_datapoint')}}" method="post">
      <input type="text" name="Pregnancies" placeholder="Pregnancies" required="required" /><br>
      <input type="text" name="Glucose" placeholder="Glucose" required="required" /><br>
      <input type="text" name="BloodPressure" placeholder="BloodPressure" required="required" /><br>
      <input type="text" name="SkinThickness" placeholder="SkinThickness" required="required" /><br>
      <input type="text" name="Insulin" placeholder="Insulin" required="required" /><br>
      <input type="text" name="BMI" placeholder="BMI" required="required" /><br>
      <input type="text" name="DiabetesPedigreeFunction" placeholder="DiabetesPedigreeFunction" required="required" /><br>
      <input type="text" name="Age" placeholder="Age" required="required" /><br>
      <button type="submit" class="btn btn-primary btn-block btn-large">Predict</button>
    </form>
 </div>
 {{result}}
</body>
</html>
```

**index.html** - The landing page:
```html
<h1>Welcome to the home page</h1>
<button onclick="home.html">Predict</button>
```

**single_prediction.html** - Displays the prediction result:
```html
<h1>Person is: {{result}}</h1>
```

## 🔧 Installation Guide

Follow these steps to set up the Diabetes Prediction System on your local machine:

### Prerequisites

- Python 3.7 or higher
- pip package manager
- Git (optional)

### Step 1: Clone the Repository

```bash
# Clone the repository (if you have Git installed)
git clone https://github.com/Arshnoor-Singh-Sohi/Diabetes-Prediction.git

# Navigate to the project directory
cd Diabetes-Prediction
```

Alternatively, you can download the project as a ZIP file from GitHub and extract it.

### Step 2: Create a Virtual Environment (Recommended)

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

If the requirements.txt file is not available, install the following packages:

```bash
pip install numpy pandas scikit-learn flask matplotlib seaborn
```

### Step 4: Ensure Model Files are Available

Make sure the `Model` directory contains the following files:
- `standardScalar.pkl`
- `modelForPrediction.pkl`

If these files are not available, you can generate them by running the Jupyter notebook:

```bash
jupyter notebook Notebook/Logistic_Regression.ipynb
```

Execute all cells in the notebook to generate the model files.

### Step 5: Run the Application

```bash
python application.py
```

The application will start and be accessible at `http://localhost:5000/` in your web browser.

## 📝 Usage Instructions

### Accessing the Web Interface

1. Open your web browser and navigate to `http://localhost:5000/`
2. Click on the "Predict" button on the homepage to access the prediction form

### Making a Prediction

1. Enter the following patient health metrics in the form:
   - **Pregnancies**: Number of pregnancies (enter 0 for male patients)
   - **Glucose**: Plasma glucose concentration (mg/dL)
   - **BloodPressure**: Diastolic blood pressure (mm Hg)
   - **SkinThickness**: Triceps skin fold thickness (mm)
   - **Insulin**: 2-Hour serum insulin (μU/ml)
   - **BMI**: Body Mass Index (weight in kg/(height in m)²)
   - **DiabetesPedigreeFunction**: Diabetes pedigree function score
   - **Age**: Age in years

2. Click the "Predict" button to submit the form
3. View the prediction result (Diabetic or Non-Diabetic)

### Example Input Values

Here's a sample input for testing the system:

| Feature | Sample Value |
|---------|-------------|
| Pregnancies | 6 |
| Glucose | 148 |
| BloodPressure | 72 |
| SkinThickness | 35 |
| Insulin | 0 |
| BMI | 33.6 |
| DiabetesPedigreeFunction | 0.627 |
| Age | 50 |

Expected prediction: Diabetic

## 🔍 Feature Importance Analysis

Understanding which features have the most significant impact on the prediction can provide valuable insights. The logistic regression model allows us to examine the coefficients to determine feature importance.

### Feature Coefficient Analysis

The following chart shows the relative importance of each feature in the prediction model:



Based on the model, the features with the highest impact on diabetes prediction are:

1. **Glucose Level**: The most influential factor, with high glucose levels strongly correlated with diabetes.
2. **BMI (Body Mass Index)**: Higher BMI values increase the likelihood of diabetes.
3. **Age**: Advanced age is associated with a higher risk of diabetes.
4. **Diabetes Pedigree Function**: Family history plays a significant role in diabetes risk.
5. **Insulin**: Abnormal insulin levels are indicative of potential diabetes.

### Clinical Interpretation

The feature importance aligns with medical knowledge about diabetes risk factors:

- **Glucose**: Elevated blood glucose is a direct indicator of diabetes, representing the body's inability to properly regulate blood sugar levels.
- **BMI**: Obesity (high BMI) is a known risk factor for Type 2 diabetes due to increased insulin resistance.
- **Age**: The risk of Type 2 diabetes increases with age, particularly after 45.
- **Family History**: Genetic factors significantly influence diabetes risk, captured by the Diabetes Pedigree Function.
- **Insulin**: Abnormal insulin levels may indicate the body's inability to properly use insulin or insufficient insulin production.

## 🚀 Future Enhancements

The current implementation provides a solid foundation, but several enhancements could further improve the system:

### Model Improvements

1. **Advanced Algorithms**: Experiment with more complex models like Random Forest, Gradient Boosting, or Neural Networks to potentially improve prediction accuracy.
2. **Ensemble Methods**: Implement ensemble techniques to combine multiple models for better performance.
3. **Feature Engineering**: Create new features or transformations that might capture more complex relationships in the data.

### Application Enhancements

1. **User Authentication**: Add user accounts to allow patients and healthcare providers to securely store and track predictions over time.
2. **Prediction History**: Implement a database to store prediction history for registered users.
3. **Visualization Dashboard**: Create interactive visualizations to help users understand their risk factors.
4. **API Integration**: Develop a RESTful API to allow integration with other healthcare systems.

### UI/UX Improvements

1. **Responsive Design**: Enhance the user interface to be fully responsive for mobile devices.
2. **Interactive Forms**: Add validation and interactive elements to the input form.
3. **Explanation Component**: Provide explanations for why a certain prediction was made, highlighting the most influential factors.
4. **Risk Percentage**: Instead of a binary outcome, provide a risk percentage to indicate the confidence level of the prediction.

### Deployment and Scalability

1. **Containerization**: Package the application using Docker for easier deployment.
2. **Cloud Deployment**: Deploy the application to cloud platforms like AWS, Azure, or Google Cloud for better accessibility.
3. **Load Balancing**: Implement load balancing for handling multiple concurrent users.

## 🤝 Contributing

Contributions to the Diabetes Prediction System are welcome! Here's how you can contribute:

1. **Fork the Repository**: Create your own copy of the project to work on.
2. **Create a Branch**: Make your changes in a new branch.
3. **Submit a Pull Request**: Once you've made your changes, submit a pull request for review.

### Contribution Areas

- **Model Improvement**: Enhance the machine learning model or implement new algorithms.
- **Feature Engineering**: Create new features or improve existing feature preprocessing.
- **UI Enhancement**: Improve the user interface and user experience.
- **Documentation**: Enhance the documentation or add more explanations.
- **Bug Fixes**: Fix any issues or bugs in the existing code.

### Development Guidelines

- Follow PEP 8 style guide for Python code.
- Include comments to explain complex code sections.
- Update documentation to reflect any changes made.
- Add tests for new features or bug fixes.

## 📚 References

1. American Diabetes Association. (2022). *Standards of Medical Care in Diabetes—2022*.

2. World Health Organization. (2020). *Global Report on Diabetes*.

3. Temurtas, H., Yumusak, N., & Temurtas, F. (2009). A comparative study on diabetes disease diagnosis using neural networks. *Expert Systems with Applications*, 36(4), 8610-8615.

4. Smith, J. W., Everhart, J. E., Dickson, W. C., Knowler, W. C., & Johannes, R. S. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. *Proceedings of the Annual Symposium on Computer Application in Medical Care*, 261-265.

5. Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.

6. Flask Web Framework Documentation: https://flask.palletsprojects.com/

7. Pima Indians Diabetes Database: https://www.kaggle.com/uciml/pima-indians-diabetes-database

---

## 👤 Author

**Arshnoor Singh Sohi**

GitHub: [Arshnoor-Singh-Sohi](https://github.com/Arshnoor-Singh-Sohi)

---

*This project was created for educational and research purposes. It should not be used as a substitute for professional medical advice, diagnosis, or treatment.*
