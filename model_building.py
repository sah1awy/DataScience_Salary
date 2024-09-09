# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 15:02:35 2024

@author: Ahmed Mahmoud
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

df = pd.read_csv("D:\\Data_Science_Salary\\DataScience_Salary\\eda_data.csv")

# choose relevant columns

df.columns

df_model = df[['avg_salary','Rating','Size','Type of ownership','Industry', 'Sector','Revenue','num_comp','hourly',
               'employer_provided','job_state', 'same_state', 'age', 'python_yn','spark',
               'aws', 'excel', 'job_simp', 'seniority']]

# get dummy data

df_dum = pd.get_dummies(df_model,dtype=int)

# train test split

from sklearn.model_selection import train_test_split
x = df_dum.drop("avg_salary",axis=1)
y = df_dum.avg_salary.values
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

# multiple linear regression

import statsmodels.api as sm
x_sm = sm.add_constant(x)
model = sm.OLS(y,x_sm)
model.fit().summary()
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score
lr = LinearRegression()
cross_val_score(lr, x_train,y_train,scoring='neg_mean_absolute_error',cv=1)

# lasso regression

ls = Lasso()
np.mean(cross_val_score(ls,x_train,y_train,scoring='neg_mean_absolute_error',cv=3))

alpha = []
error = []
for i in range(1,100):
    alpha.append(i/100)
    lml = Lasso(alpha=(i/100))
    error.append(np.mean(cross_val_score(lml,x_train,y_train,scoring='neg_mean_absolute_error',cv=3)))
    
plt.plot(alpha,error)
err = tuple(zip(alpha,error))
df_error = pd.DataFrame(err,columns=["alpha","error"])
df_error[df_error["error"] == max(df_error["error"])]

# random forest

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
np.mean(cross_val_score(rf,x_train,y_train,scoring='neg_mean_absolute_error',cv=3))

# tune models using GridsearchCV

from sklearn.model_selection import RandomizedSearchCV
params = {"n_estimators":range(10,300,10), 'criterion':("absolute_error","squared_error"),'max_features':['auto','log2','sqrt']}
rs = RandomizedSearchCV(rf, param_distributions=params,scoring='neg_mean_absolute_error',cv=3)
rs.fit(x_train,y_train)

rs.best_score_
rs.best_estimator_

# test ensembles
rf_pred = rs.best_estimator_.predict(x_test)
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test,rf_pred)

# saving model
import pickle
with open('D:\\Data_Science_Salary\\DataScience_Salary\\rf.pkl','wb') as f:
    pickle.dump(rs.best_estimator_,f)


