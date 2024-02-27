import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from math import sqrt
from category_encoders.binary import BinaryEncoder


df = pd.read_csv('data/CrabAgePrediction.csv')

bn = BinaryEncoder()
data_category = bn.fit_transform(df.select_dtypes(include=['object'])).astype(int)
data_num = df.select_dtypes(exclude=['object'])
df = pd.concat([data_num, pd.DataFrame(data_category)], axis=1)
pd.set_option('display.max_columns', None)

x_reg=df.drop(['Age'],axis=1)
y_reg=df['Age']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_reg, y_reg, test_size=0.33, shuffle=False)

st.title("Регрессия")
st.header("О датасете:")
st.write("Датасет о крабах, в нем содержатся такие параметры как длина краба, вес и так далее.")
st.write("Необходимо предсказать возраст краба.")
st.header("Полиномиальная регрессия")
("После предобработки данных и разбиение на обучающую и тестовую выборки был обучен и протестирован алгоритм PolinomialRegression cо степенью 1, 2, 3.")


pipeline = Pipeline(
    [
        ("scaler", MinMaxScaler()),
        ("polynomial", PolynomialFeatures(1)),
        ("model", LinearRegression()),
    ]
)

pipeline.fit(x_train, y_train)
y_pred = pipeline.predict(x_test)
st.write('Полиномиальная регрессия со степенью 1:')
st.write('R^2:' ,pipeline.score(x_test, y_test))

pipeline = Pipeline(
    [
        ("scaler", MinMaxScaler()),
        ("polynomial", PolynomialFeatures(2)),
        ("model", LinearRegression()),
    ]
)

pipeline.fit(x_train, y_train)
y_pred = pipeline.predict(x_test)
st.write('Полиномиальная регрессия со степенью 2:')
st.write('R^2:' ,pipeline.score(x_test, y_test))

pipeline = Pipeline(
    [
        ("scaler", MinMaxScaler()),
        ("polynomial", PolynomialFeatures(3)),
        ("model", LinearRegression()),
    ]
)

pipeline.fit(x_train, y_train)
y_pred = pipeline.predict(x_test)
st.write('Полиномиальная регрессия со степенью 3:')
st.write('R^2:' ,pipeline.score(x_test, y_test))

st.write('Лучшим результатом оказался алгоритм со степенью 2')
