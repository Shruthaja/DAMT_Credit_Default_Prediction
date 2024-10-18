# Credit Card Default Prediction
Team Members -
1. Mir Mustafa Ali (1002117402)
2. Shruthaja Patali Rao (1002118604)

# Supervised by: Prof. Ramakrishna Prasad Koganti
Course: CSE - 5301

i) Project Overview -
This project aims to predict credit card defaults using customer repayment history and trends. By analyzing prior behavior, banks can identify at-risk customers and provide alternative solutions, thereby minimizing potential losses.

ii) Background -
The project involves data preprocessing, exploratory data analysis (EDA), and the application of XGBoost regression modeling to predict whether a customer will default on their credit card payments.

iii) Key Steps -
1. Data Preprocessing

2. Identified categorical features and merged datasets for target labels.
3. Converted data types and filled missing values.
4. Exploratory Data Analysis (EDA)

5. Analyzed default rates and missing values.
6. Evaluated correlations between variables.
7. Feature Importance -Determined important features affecting predictions using correlation analysis.
8. Model Development - Employed XGBoost for classification, utilizing gradient boosting techniques to improve prediction accuracy.

# Results
i) Achieved an accuracy of 90.448% in predicting customer defaults.
ii) Confusion matrix visualized model performance.
iii) Tools and Libraries
  1. NumPy: For numerical computing.
  2. Pandas: For data manipulation and analysis.
  3. Matplotlib & Seaborn: For data visualization.
  4. XGBoost: For machine learning modeling.
  5. Optuna: For hyperparameter tuning.

# Conclusion

The model effectively predicts credit card defaults, allowing banks to take proactive measures. Recommendations include handling null values and identifying key features to enhance model accuracy.

# Installation
To run this project, ensure you have the following libraries installed:

1. pip install numpy pandas matplotlib seaborn xgboost optuna cudf cupy
2. Works Cited

Referrences - Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 785-794).
Chen, T., et al. (2021). XGBoost. GitHub repository. GitHub , Dataset from the “American Express - Default Prediction” competition: Kaggle
