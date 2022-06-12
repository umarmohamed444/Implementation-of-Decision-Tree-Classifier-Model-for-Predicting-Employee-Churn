# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1.import pandas module and import the required data set.
Find the null values and count them.
Count number of left values.
From sklearn import LabelEncoder to convert string values to numerical values.
From sklearn.model_selection import train_test_split.
Assign the train dataset and test dataset.
From sklearn.tree import DecisionTreeClassifier.
Use criteria as entropy.
From sklearn import metrics.
Find the accuracy of our model and predict the require values. 
 
 
 

## Program:
~~~
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by:E Umar Mohamed 
RegisterNumber:212220040173  
*/
import pandas as pd
data = pd.read_csv("Employee.csv")
data.head()
data.info()

data.isnull().sum()

data["left"].value_counts

from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x= data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]

x.head()
y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 100)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)

y_pred = dt.predict(x_test)
from sklearn import metrics

accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])

~~~

## Output:
![Github Logo](1.png)
![Github Logo](2.png)
![Github Logo](3.png)
![Github Logo](4.png)
![Github Logo](5.png)
![Github Logo](6.png)
![Github Logo](7.png)
![Github Logo](8.png)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
