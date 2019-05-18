import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
import numpy as np


#loading dataset
data = pd.read_csv('.\Admission_Predict.csv')

#preparing dataset
train = data.iloc[:,1:8].values
labels = data.iloc[:,8:].values

#80-20 train-test split
x_train, x_test, y_train, y_test = train_test_split(train,labels,test_size=0.2)

#normalizing
x_train = x_train.astype('float64')
x_test = x_test.astype('float64')
y_train = np.reshape(y_train,-1)
y_test =np.reshape(y_test,-1)

x_scale = MinMaxScaler(feature_range=(0,1))
x_train = x_scale.fit_transform(x_train)
x_test = x_scale.fit_transform(x_test)

#Linear Regression
linear = LinearRegression()
linear.fit(x_train,y_train)

pred = linear.predict(x_test)
acc = linear.score(x_test,y_test)
print('Linear Regression Accuarcy:',acc)


#Decision Tree
dtree = DecisionTreeRegressor(random_state = 0)
dtree.fit(x_train, y_train)

pred_dtree = dtree.predict(x_test)
acc = dtree.score(x_test,y_test)
print('Decision Tree Accuracy: ',acc)

#RandomForest
r = RandomForestRegressor(n_estimators=100)
r.fit(x_train,y_train)

pred = r.predict(x_test)
acc = r.score(x_test,y_test)
print('Random Forest regression Accuracy:',acc)

#GradientBoosting
gb = GradientBoostingRegressor()
gb.fit(x_train,y_train)
pred = gb.predict(x_test)
acc = gb.score(x_test,y_test)
print('Gradient Boosting Accuracy: ',acc)



