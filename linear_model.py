import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

df = pd.read_csv("Python_Linear_Regres.csv")
dp = ['Address', 'Date', 'Postcode', 'YearBuilt', 'Lattitude', 'Longtitude']
df = df.drop(dp, axis=1)
df[['Propertycount', 'Distance', 'Bedroom2', 'Bathroom', 'Car']] = df[
    ['Propertycount', 'Distance', 'Bedroom2', 'Bathroom', 'Car']].fillna(0)
df['Landsize'] = df['Landsize'].fillna(df.Landsize.mean())
df['BuildingArea'] = df['BuildingArea'].fillna(df.BuildingArea.mean())
df.dropna(inplace=True)
X = df.drop('Price', axis=1)
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
print(X.columns)
ohe = OneHotEncoder()
ohe.fit(X[['Suburb', 'Rooms', 'Type', 'Method', 'SellerG', 'Distance', 'Bedroom2', 'Bathroom', 'Car', 'Landsize',
           'BuildingArea', 'CouncilArea', 'Regionname', 'Propertycount']])
print(ohe.categories_)
column_trans = make_column_transformer(
    (OneHotEncoder(handle_unknown="ignore", categories=ohe.categories), ['Suburb', 'Rooms', 'Type', 'Method', 'SellerG', 'Distance', 'Bedroom2',
                                                 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'CouncilArea',
                                                 'Regionname', 'Propertycount']), remainder='passthrough')
print('------------', column_trans)
reg = LinearRegression()
pipe = make_pipeline(column_trans, reg)
print(pipe.fit(X_train, y_train))
y_pred = pipe.predict(X_test)
print(r2_score(y_test, y_pred))
scores = []

for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=i)
    lr = LinearRegression()
    pipe = make_pipeline(column_trans,lr)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    scores.append(r2_score(y_test, y_pred))
print(scores[np.argmax(scores)])
print(X_test.info())
price = pipe.predict(pd.DataFrame(
    columns=X_test.columns,
    data=np.array(['Airport West', 3, 't', 'SP', 'Barry', 13.5, 3, 2, 1, 293, 102, 'Moonee Valley City Council',
                   'Western Metropolitan', 3464.]).reshape(1, 14)))
# Airport West,37 North St,3,t,825000,SP,Barry,19/11/2016,13.5,3042,3,2,1,293,102,1960,Moonee Valley City Council,-37.72,144.8776,Western Metropolitan,3464
# Suburb,Rooms,Type,Method,SellerG,Distance,Bedroom2,Bathroom,Car,Landsize,BuildingArea,CouncilArea,Regionname,Propertycount
print(price)
