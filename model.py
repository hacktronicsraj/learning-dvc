# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

column_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
df = pd.read_csv("car.data",names=column_names)

le = LabelEncoder()
for column in df.columns:
    df[column] = le.fit_transform(df[column])

X = df.drop(['safety','class'], axis =1 )
y = df['safety']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)



model = LinearRegression()
model.fit(X_train,y_train)

y_predictions = model.predict(X_test)

mse = mean_squared_error(y_test, y_predictions)
r2 = r2_score(y_test, y_predictions)

mse

r2


