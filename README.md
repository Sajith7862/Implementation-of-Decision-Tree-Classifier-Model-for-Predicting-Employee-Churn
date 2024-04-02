# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset and preprocess it, including encoding categorical variables.
2. Split the dataset into features (X) and the target variable (y), then split them into training and testing sets.
3. Initialize and train a DecisionTreeClassifier model with entropy criterion on the training data.
4. Evaluate the model's accuracy on the test data and make predictions on new data points.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: MOHAMED HAMEEM SAJITH J
RegisterNumber:  212223240090
*/

import pandas as pd

data = pd.read_csv("/Employee_EX6 (2).csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder()

data["salary"] = le.fit_transform(data["salary"])

data.head()

x = data[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours", "time_spend_company", "Work_accident", "promotion_last_5years", "salary"]]

x.head()

y = data["left"]

from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

from sklearn.tree import DecisionTreeClassifier 
dt = DecisionTreeClassifier(criterion="entropy") 
dt.fit(x_train, y_train) 
y_pred = dt.predict(x_test)

from sklearn import metrics 
accuracy = metrics.accuracy_score(y_test, y_pred)

accuracy

print(accuracy)

dt.predict([[0.5, 0.8, 9, 260, 6, 0, 1, 2]])

```

## Output:

# 1.DATA:
![image](https://github.com/Sajith7862/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/145972360/c1b8848e-1a4a-4ef0-a022-c2ef44bb7404)

# 2.ACCURACY:
![image](https://github.com/Sajith7862/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/145972360/99c7ecd2-6de0-4eae-b31e-45f6fb3293d5)

# 3.PREDICTION:
![image](https://github.com/Sajith7862/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/145972360/4719a40e-b6fb-4c46-a855-f00d2fe82624)





## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
