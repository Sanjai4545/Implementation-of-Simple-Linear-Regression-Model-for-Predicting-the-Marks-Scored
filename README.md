# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.
## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
#### 1.Import the standard Libraries. 
#### 2.Set variables for assigning dataset values. 
#### 3.Import linear regression from sklearn. 
#### 4.Assign the points for representing in the graph. 
#### 5.Predict the regression for marks by using the representation of the graph. 
#### 6.Compare the graphs and hence we obtained the linear regression for the given datas.
## Program:
### SANJAI .T (212224240144)
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```
## Output:
![Screenshot 2025-04-30 141000](https://github.com/user-attachments/assets/2b065d2c-a027-456c-9763-6c843b935f05)

![Screenshot 2025-04-30 141154](https://github.com/user-attachments/assets/9813b19f-bd08-4368-bbac-1e0c6e0b9fe1)

![Screenshot 2025-04-30 141452](https://github.com/user-attachments/assets/e9959bee-a826-4ba9-9374-4adf6b47ddf6)

![Screenshot 2025-04-30 141510](https://github.com/user-attachments/assets/488678ce-6cfa-420c-a1a1-a8f112d0d33a)

![Screenshot 2025-04-30 141525](https://github.com/user-attachments/assets/59011f60-a21e-4cef-b728-1943aee76a38)





