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


st.title("Получить предсказание дождя.")

st.header("Lenght")
Lenght = st.number_input("Число:", value=1.4)

st.header("Diameter")
Diameter = st.number_input("Число:", value=1.1)

st.header("Height")
Height = st.number_input("Число:", value=0.4)

st.header("Weight")
Weight = st.number_input("Число:", value=24.6)

st.header("Shucked Weight")
Shucked = st.number_input("Число:", value=12.3)

st.header("Viscera Weight")
Viscera = st.number_input("Число:", value=5.5)

st.header("Shell Weight")
Shell = st.number_input("Число:", value=6.7)

st.header("Sex")
Sex_select = ['F', 'M', 'I']
Sex = st.selectbox("Пол", Sex_select)


df = pd.DataFrame({
                    "Sex": [Sex],
                    'Length': [Lenght],
                    'Diameter': [Diameter],
                    'Height': [Height],
                    'Weight': [Weight],
                    'Shucked Weight': [Shucked],
                    'Viscera Weight': [Viscera],
                    'Shell Weight': [Shell],
                    'Age': [0],
                    })


data_category = bn.transform(df.select_dtypes(include=['object'])).astype(int)
data_num = df.select_dtypes(exclude=['object'])
df = pd.concat([data_num, pd.DataFrame(data_category)], axis=1)
pd.set_option('display.max_columns', None)

x_reg=df.drop(['Age'],axis=1)
y_reg=df['Age']

button_clicked = st.button("Предсказать")

if button_clicked:
    y_pred= pipeline.predict(x_reg)
    st.write(f"{y_pred[0]}")

