import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error

df = 0 # pd.read... # data

features = []
want = []

x = df[features]
y = df[want]

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.7,test_size=0.3,random_state=0)

rfr = RandomForestRegressor(n_estimators=100)
rfr.fit(x_train,y_train)

y_pred = rfr.predict(x_test)
cm = confusion_matrix(y_test, y_pred)   

print("*"*25)
print(cm)
print("*"*50)
print(mean_absolute_error(y_pred=y_pred,y_true=y_test))
print("*"*50)