import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from category_encoders.binary import BinaryEncoder

df = pd.read_csv('data/weatherAUS.csv')


df.dropna(inplace=True,ignore_index=True)
f = lambda x : str(x)[5:7]
df['Date'] = df['Date'].transform(f)
df['Date'] = df['Date'].astype(int)
f = lambda x : 0 if (x == "No") else 1
df['RainToday'] = df['RainToday'].transform(f)
df['RainToday'] = df['RainToday'].astype(int)

df['RainTomorrow'] = df['RainTomorrow'].transform(f)
df['RainTomorrow'] = df['RainTomorrow'].astype(int)


bn = BinaryEncoder()
data_category = bn.fit_transform(df.select_dtypes(include=['object'])).astype(int)
data_num = df.select_dtypes(exclude=['object'])
df = pd.concat([data_num, pd.DataFrame(data_category)], axis=1)
pd.set_option('display.max_columns', None)

x_class=df.drop(['RainTomorrow'],axis=1)
y_class=df['RainTomorrow']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_class, y_class, test_size=0.33, shuffle=False)


st.title("Классификация")
st.header("О датасете:")
st.write("Датасет об дождях в австралии, в нем содержится множество параметров таких так атмосферное давление в 9 утра, направление ветра и так далее.")
st.write("Необходимо предсказать будет ли дождь завтра.")
st.header("Nearest Centroid classifier")
st.write("После предобработки данных и разбиение на обучающую и тестовую выборки был обучен и протестирован алгоритм Nearest Centroid без подбора гиперпараметров.")

from sklearn.neighbors import NearestCentroid
clf = NearestCentroid()
clf.fit(X_train, y_train)
y_pred= clf.predict(X_test)

st.write('NearestCentroid без гиперпараметров:')
st.write("Accuracy score" ,accuracy_score(y_test, y_pred))

st.write("К сожалению, алгоритм Nearest Centroid имеет всего два гиперпараметра, это метрика и shrink_threshold")
st.write("Значение shrink_threshold по умолчанию оказалось оптимальным, зато изменение метрики помогло немного повысить точность модели")

clf = NearestCentroid(metric = "cityblock")
clf.fit(X_train, y_train)
y_pred= clf.predict(X_test)

st.write('NearestCentroid c метрикой cityblock:')
st.write("Accuracy score" ,accuracy_score(y_test, y_pred))

st.title("Получить предсказание дождя.")

st.header("Date")
Date = st.number_input("Число:", value=2012, min_value=1900, max_value=2100)

st.header("Location")
locations = ['Cobar', 'CoffsHarbour', 'Moree', 'NorfolkIsland', 'Sydney',
       'SydneyAirport', 'WaggaWagga', 'Williamtown', 'Canberra', 'Sale',
       'MelbourneAirport', 'Melbourne', 'Mildura', 'Portland', 'Watsonia',
       'Brisbane', 'Cairns', 'Townsville', 'MountGambier', 'Nuriootpa',
       'Woomera', 'PerthAirport', 'Perth', 'Hobart', 'AliceSprings',
       'Darwin']
Location = st.selectbox("Город", locations)

st.header("MinTemp")
MinTemp = st.number_input("Число:", value=20.9)

st.header("MaxTemp")
MaxTemp = st.number_input("Число:", value=37.8)

st.header("Rainfall")
Rainfall = st.number_input("Число:", value=2)

st.header("Evaporation")
Evaporation = st.number_input("Число:", value=12.8)

st.header("Sunshine")
Sunshine = st.number_input("Число:", value=13.2)

st.header("WindGustDir")
dirs=['SSW', 'S', 'NNE', 'WNW', 'N', 'SE', 'ENE', 'NE', 'E', 'SW', 'W',
       'WSW', 'NNW', 'ESE', 'SSE', 'NW']
WindGustDir = st.selectbox("Направление", dirs)

st.header("WindGustSpeed")
WindGustSpeed = st.number_input("Число:", value=30)

st.header("WindDir9am")
dirs2=['ENE', 'SSE', 'NNE', 'WNW', 'NW', 'N', 'S', 'SE', 'NE', 'W', 'SSW',
       'E', 'NNW', 'ESE', 'WSW', 'SW']
WindDir9am = st.selectbox("Направление", dirs2)

st.header("WindDir3pm")
dirs3=['SW', 'SSE', 'NNW', 'WSW', 'WNW', 'S', 'ENE', 'N', 'SE', 'NNE',
       'NW', 'E', 'ESE', 'NE', 'SSW', 'W']
WindDir3pm = st.selectbox("Направление", dirs3)

st.header("WindSpeed9am")
WindSpeed9am = st.number_input("Число:", value=11)

st.header("WindSpeed3pm")
WindSpeed3pm = st.number_input("Число:", value=7)

st.header("Humidity9am")
Humidity9am = st.number_input("Число:", value=27)

st.header("Humidity3pm")
Humidity3pm = st.number_input("Число:", value=9)

st.header("Pressure9am")
Pressure9am = st.number_input("Число:", value=1012.6)

st.header("Pressure3pm")
Pressure3pm = st.number_input("Число:", value=1010.1)

st.header("Cloud9am")
Cloud9am = st.number_input("Число:", value=0.1)

st.header("Cloud3pm")
Cloud3pm = st.number_input("Число:", value=1)

st.header("Temp9am")
Temp9am = st.number_input("Число:", value=29.8)

st.header("Temp3pm")
Temp3pm = st.number_input("Число:", value=36.4)

st.header("RainToday")
RainToday = st.number_input("Число:", value=0, min_value=0, max_value=1)

data = pd.DataFrame({'Date': [Date],
                    'Location': [Location],
                    'MinTemp': [MinTemp],
                    'MaxTemp': [MaxTemp],
                    'Rainfall': [Rainfall],
                    'Evaporation': [Evaporation],
                    'Sunshine': [Sunshine],
                    'WindGustDir': [WindGustDir],
                    'WindGustSpeed': [WindGustSpeed],
                    'WindDir9am': [WindDir9am],
                    'WindDir3pm': [WindDir3pm],
                    'WindSpeed9am': [WindSpeed9am],
                    'WindSpeed3pm': [WindSpeed3pm],
                    'Humidity9am': [Humidity9am],    
                    'Humidity3pm': [Humidity3pm],   
                    'Pressure9am': [Pressure9am],   
                    'Pressure3pm': [Pressure3pm],   
                    'Cloud9am': [Cloud9am],   
                    'Cloud3pm': [Cloud3pm],   
                    'Temp9am': [Temp9am],   
                    'Temp3pm': [Temp3pm],   
                    'RainToday': [RainToday],  
                    'RainTomorrow': [0]       
                    })


data_category = bn.transform(data.select_dtypes(include=['object'])).astype(int)
data_num = data.select_dtypes(exclude=['object'])
data = pd.concat([data_num, pd.DataFrame(data_category)], axis=1)
pd.set_option('display.max_columns', None)

x_class=data.drop(['RainTomorrow'],axis=1)

button_clicked = st.button("Предсказать")

if button_clicked:
    y_pred= clf.predict(x_class)
    st.write(f"{y_pred[0]}")