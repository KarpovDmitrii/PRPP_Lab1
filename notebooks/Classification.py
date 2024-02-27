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






