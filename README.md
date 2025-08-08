## Introduction
Health insurance premiums vary widely from person to person due to multiple factors such as age, gender, lifestyle habits,health condition, and geographic location.
Accurately predicting these premiums is critical for insurance firms to establish reasonable and competitive rates and for clients to fully understand the factors influencing their expenses and make appropriate plans.
This project uses machine learning regression techniques to forecast a person's yearly medical insurance premium based on their health-related and demographic data. The model learns the underlying correlations between personal characteristics (such as age, BMI and smoking status) and the premium amount by utilizing past insurance data.
I chose three different regression algorithms Linear Regression,Decision Tree and Random Forest algorithms in this project.
By including these three models, this project can compare simple (Linear Regression), single tree-based (Decision Tree), and ensemble tree-based (Random Forest) methods, allow identify the most accurate and robust approach for insurance premium prediction.

## Goals
The primary goals of this project are:

1) Data Understanding & Preparation – Exploring the dataset, identifying key predictors, and handling missing or inconsistent values.

2) Feature Engineering – Transforming raw variables into machine-learning-friendly formats (e.g., encoding categorical data like “smoker” or “region”).

3) Model Development – Training and evaluating different regression algorithms (e.g., Linear Regression,Decision Tree Regressor Random Forest Regressor) to identify the most accurate predictor.

4) Performance Evaluation – Using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² Score to measure how well the model predicts premiums.

5) Insights & Interpretability – Understanding which factors most influence insurance costs, providing valuable knowledge for policy design and risk assessment.

## Tools
Programming language: Python 
Jupyter Notebook

## Data set
The dataset used in this project contains records of individual health insurance policyholders, along with demographic, lifestyle, and regional information, as well as the annual insurance charges. Each row represents a unique policyholder. 

The columns are:
age – Age of the primary beneficiary (in years).

sex – Gender of the insured individual (male, female).

bmi – Body Mass Index, a measure of body fat based on height and weight (kg/m²).

children – Number of dependent children covered under the insurance plan.

smoker – Smoking status (yes or no), a major factor influencing medical costs.

region – Residential area in the U.S. (northeast, northwest, southeast, southwest).

charges – The actual annual medical insurance premium billed to the individual (in USD) — this is the target variable for prediction.

## STEPS 

### step:1
To begin the project,first import the essential Python libraries( Numpy,Pandas) that will help us handle data and perform computations. To better understand the dataset and identify patterns,import Python’s visualization libraries like Matplotlib,Seaborn.
Next import StandardScaler that standardizes features by removing the mean and scaling to unit variance. Then import regression algorithms from the scikit-learn library. 
To measure how well these machine learning models predict insurance charges, import evaluation metrics like mean_absolute_error,mean_squared_error,r2_score from the scikit-learn library.
### step:2
load the dataset into our Python environment so it can be explored and processed. pd.read_csv() – A Pandas function that reads a CSV file and converts it into a DataFrame named 'data'
### step:3 Understanding Dataset
-> Viewing the first few rows with data.head()
-> Checking column names and data types with data.info()
-> Performing statistical analysis with data.describe()
-> After loading the dataset, it’s important to know how many records (rows) and attributes (columns) it contains using data.shape
### step:4 

