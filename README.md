## Overview
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

AGE – Age of the primary beneficiary (in years).

SEX – Gender of the insured individual (male, female).

BMI – Body Mass Index, a measure of body fat based on height and weight (kg/m²).

CHILDREN – Number of dependent children covered under the insurance plan.

SMOKER – Smoking status (yes or no), a major factor influencing medical costs.

REGION – Residential area in the U.S. (northeast, northwest, southeast, southwest).

CHARGES – The actual annual medical insurance premium billed to the individual (in USD) — this is the target variable for prediction.

## STEPS 

### Import
To begin the project,first import the essential Python libraries( Numpy,Pandas) that will help us handle data and perform computations. To better understand the dataset and identify patterns,import Python’s visualization libraries like Matplotlib,Seaborn.
Next import StandardScaler that standardizes features by removing the mean and scaling to unit variance. Then import regression algorithms from the scikit-learn library. 
To measure how well these machine learning models predict insurance charges, import evaluation metrics like mean_absolute_error,mean_squared_error,r2_score from the scikit-learn library.
### step:2
load the dataset into our Python environment so it can be explored and processed. pd.read_csv() – A Pandas function that reads a CSV file and converts it into a DataFrame named 'data'
### step:3 Understanding Dataset
-> After loading the dataset, it’s important to know how many records (rows) and attributes (columns) it contains using data.shape. This dataset contains 1,338 rows and 7 columns.

-> Viewing the first few rows with data.head()

-> Checking column names and data types with data.info().

-> Performing statistical analysis with data.describe()

### step:4 Missing values 
check dataset has any missing values using data.isnull().sum() . It shows this dataset is complete has no missing values.
### step:5 Duplicates
Check any duplicate values in this dataset using data[data.duplicated()]. In this case it shows only one duplicate record . So remove that using data.drop_duplicates().

## Exploratory Data analysis

### Visualizing Age and BMI Distributions

<img width="1205" height="649" alt="age Bmi distribution" src="https://github.com/user-attachments/assets/848e8df4-6242-4a3e-8bdb-62abba574ea3" />



To understand how key numerical features are distributed in this dataset, I plotted histograms for Age and BMI using Seaborn and Matplotlib.
The figure above shows histograms with Kernel Density Estimation (KDE) curves for Age and BMI of policyholders in the insurance dataset. The histogram reveals that ages in the dataset range roughly from 18 to mid-60s.
##### Age Distribution 
There is a noticeable peak at age 18, indicating many young adults in the dataset.
Beyond age 18, the distribution is relatively uniform, meaning all other ages are fairly evenly represented.
The KDE curve confirms that there is no strong skewness, but the spike at 18 is significant and may influence model training if not considered.

#### BMI Distribution 
BMI values mostly fall between 15 and 50, with the majority clustering between 25 and 35.
The distribution resembles a normal distribution centered around a BMI of ~30, which falls in the overweight range according to WHO classifications.
A small tail to the right suggests the presence of outliers with very high BMI values (>45).

#### Insights and Relevance to the Project
Age is a critical predictor since older individuals generally face higher health risks, potentially increasing insurance charges.

BMI is also an important feature because higher BMI values are associated with obesity-related health conditions, which can increase medical costs.

### Relationship Between BMI and Insurance Charges (with Smoking Status)

<img width="1082" height="544" alt="Insurance price vs BMI" src="https://github.com/user-attachments/assets/014f112f-bcdc-4bb8-85c9-0e010b4134bd" />


The scatter plot above shows how BMI relates to insurance charges, with data points color-coded by smoking status:
Blue points → Non-smokers (Non_Smoker = 0), Orange points → Smokers (Smoker = 1)
#### Clear Separation Between Smokers and Non-Smokers
Smokers have significantly higher insurance charges compared to non-smokers, even when their BMI is similar. Many smokers have charges exceeding $30,000, whereas most non-smokers remain below $20,000.
#### BMI Impact
For both smokers and non-smokers, higher BMI tends to correlate with higher charges.
However, this effect is much more pronounced for smokers—those with both high BMI and smoking habits face the highest charges.
#### Insights and Relevance to Prediction Model:
Smoking status is likely to be one of the most important predictors in the dataset. BMI will also play a significant role, but its impact depends strongly on whether the individual smokes.
This plot suggests potential interaction effects between smoking and BMI, which could be captured by non-linear models like Random Forests or by explicitly adding interaction terms in regression.

### Correlation Analysis of Insurance Dataset

<img width="1198" height="905" alt="Correlation_matrix" src="https://github.com/user-attachments/assets/51d3e667-42a5-41c0-8218-6dc7044d03a0" />


The heatmap above displays the correlation matrix for the numerical variables in the insurance dataset.
#### Strongest Correlation with Charges

Smoker status shows the highest positive correlation with charges (0.79). This confirms that smoking has a major impact on insurance costs, likely due to higher health risks.

Age also has a moderate positive correlation with charges (0.30), suggesting that older individuals tend to incur higher medical costs.

BMI has a weaker positive correlation with charges (0.20), indicating that higher BMI is associated with slightly higher insurance costs.
#### Low Correlation Among Most Variables

Other variables such as sex, children, and region have very low correlations with charges, implying they may not be strong predictors individually.

Some variables (e.g., region and charges) are almost uncorrelated, suggesting minimal direct impact.

### Encoding Categorical Variables
The insurance dataset contains several categorical variables such as sex, smoker, and region. So I mapped the categorical values to numeric values as follows:

data['sex']=data['sex'].map({'female':0,'male':1})

data['smoker']=data['smoker'].map({'no':0,'yes':1})

data['region']=data['region'].map({'southwest':100,'southeast':101,'northwest':102,'northeast':103})

### Feature Scaling
In the dataset, numerical features such as age and bmi are measured on different scales. To standardize these features,I applied StandardScaler() from the sklearn.preprocessing library

scaler= StandardScaler()

data[['age','bmi']]=scaler.fit_transform(data[['age','bmi']])

### Feature and Target Variable Separation
X=data.drop(['charges'],axis=1)

y=data['charges']

### split dataset into training and testing sets.
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42) 

### Model Training
In this phase, three different regression models named Linear Regression, Decision Tree Regressor and Random Forest Regressor were trained on the insurance dataset to predict the insurance charges based on various input features.
Each of these models was fitted to the training dataset to learn the underlying patterns that influence insurance charges. Subsequent evaluation on the test set will help determine which model best predicts the insurance costs.
#### Training Linear Regression model
lr = LinearRegression()

lr.fit(X_train,y_train)
#### Training Decision Tree Regressor model
dtr=DecisionTreeRegressor()

dtr.fit(X_train,y_train)
#### Training Random Forest Regressor model
rf=RandomForestRegressor()

rf.fit(X_train,y_train)
### After training the models, the next step is generating predictions on the unseen test data (X_test) to evaluate each model's performance in estimating insurance charges
#### Linear Regression Predicts
y_lr_pred=lr.predict(X_test)
#### Decision Tree Regressor Predicts
y_dtr_pred=dtr.predict(X_test)
#### Random Forest Regressor Predicts
y_rf_pred=rf.predict(X_test)

### Model Evaluation
To assess the performance of the models in predicting insurance charges, three key evaluation metrics used. They are Mean Absolute Error (MAE),Mean Squared Error (MSE) and R-squared (R²) Score.
#### Evaluating Linear Regression Model
Mean Absolute Error (MAE): 4,186.51

The model’s predicted insurance charges differ from the actual charges by about 4,186. This provides an intuitive measure of prediction error in the same units as the target variable.

Mean Squared Error (MSE): 33,635,210.43

This large value reflects the squared deviations between predictions and actual values. The high magnitude is partly due to the squaring process.

R-squared (R²) Score: 0.78

The model explains 78% of the variance in insurance charges. 

#### Evaluating Decision Tree regressor model
Mean Absolute Error (MAE): 2,883.38

The model’s predictions differ from the actual insurance charges by approximately 2,883. This is an improvement over the Linear Regression model’s MAE (4,186), indicating that the Decision Tree makes more precise predictions in absolute terms.

Mean Squared Error (MSE): 39,898,140.69

This measures the average squared difference between predicted and actual charges. While the MSE is slightly higher than Linear Regression’s (33.6 million)

R-squared (R²) Score: 0.74

The model explains 74% of the variance in insurance charges,slightly less than Linear Regression (0.78). This means that while the Decision Tree captures certain non-linear relationships, it may not generalize as well to unseen data.

#### Evaluating Random Forest Regressor model
MAE score of Random Forest Regressor:2426.68

The model’s predictions differ from the actual insurance charges by about 2,427. This is the lowest MAE among all tested models, indicating that Random Forest consistently makes the most accurate predictions in terms of average deviation.

MSE score of Random Forest Regressor :20495591.85

This value is significantly lower than both the Linear Regression and Decision Tree models, showing that Random Forest not only reduces average errors but also minimizes large deviations between predicted and actual charges.

r2_score of Random Forest Regressor:0.87

The model explains 87% of the variance in insurance charges, which is a notable improvement over Linear Regression (0.78) and Decision Tree (0.74). This indicates strong predictive performance to unseen data.

### Conclusion 
The Random Forest Regressor outperforms both Linear Regression and Decision Tree models across all evaluation metrics. By combining the predictions of multiple decision trees, it captures complex, non-linear relationships in the data while maintaining strong generalization. These results suggest that Random Forest is the most reliable and accurate model for predicting insurance charges in this project.

### 
Based on the model evaluation score I choose RandomForestRegressor model as the best model for predicting charges for a new customer. So I use entire dataset (X,y)for training the model for getting more accuracy for new data

best_model=RandomForestRegressor(n_estimators=100,random_state=42)

best_model.fit(X,y)

Now I apply a new customer data to my best model Random Forest Regressor
new ,unseen customer data is  age=25,	sex=1	bmi=30.20	children=4	smoker=0	region=1

Here model predicts insurance charge for new customer is 26977.5906091


