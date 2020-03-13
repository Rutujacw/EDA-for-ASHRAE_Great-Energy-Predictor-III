import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("train.csv")
data = data.drop(["timestamp"], axis=1)
print(data.head(5))
x=data.iloc[:,:-1].values
y=data.iloc[:,2].values

#check for missing values
print(data.isnull().sum())

#split the data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#linearregression
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train,y_train)

#pred
y_pred=reg.predict(x_test)
#print(y_pred)
print(y_pred.tail())

prediction = pd.DataFrame(y_pred, columns=['meter_reading']).to_csv('prediction.csv')


