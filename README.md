# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas
2. Import Decision tree classifier
3. Fit the data in the model
4. Find the accuracy score.

## Program:
```
Developed by: VASUNDRA SRI 
RegisterNumber: 212222230168 
```
```
import pandas as pd
data=pd.read_csv('/content/Employee.csv')
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
plt.figure(figsize=(18,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()
```
## Output:
![Screenshot 2024-04-03 090113](https://github.com/vasundrasriravi/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119393983/51d1760d-c2ca-4c48-8e14-5ab844466be7)

### Accuracy Value:
![Screenshot 2024-04-03 090218](https://github.com/vasundrasriravi/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119393983/e978b30b-eb0d-4cdb-81dd-f650980e507b)

### Predicted Value:
![Screenshot 2024-04-03 090306](https://github.com/vasundrasriravi/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119393983/02802d0c-6238-42ca-a16f-31e48081408d)

### Result Tree:
![Screenshot 2024-04-03 090330](https://github.com/vasundrasriravi/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119393983/98f1d01e-52de-4c46-b910-66e89fe80199)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
