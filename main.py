import numpy as np #линейная алгебра
import pandas as pd # обработки и анализа структурированных данных
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#считываем информацию о датасете
data = pd.read_csv("KNNAlgorithmDataset.csv")
# Посмотрим, как устроен датасет
print(data.info())
print(data.describe())
print(data.columns) 

#убираем ненужные столбцы и смотрим, что осталось
data.drop(["id","Unnamed: 32"],axis=1,inplace=True)
data.tail()
# В датасете диагноз указан следующим образом:
# злокачеств. = M  
# доброкачеств. = B  
   
# Разделяем злокач. и доброкачественные
M = data[data.diagnosis == "M"]
B = data[data.diagnosis == "B"]

#строим график, покрасив доброкачественные и злокачественные точки в разные цвета
plt.scatter(M.radius_mean,M.texture_mean,color="red",label="kotu",alpha= 0.3)
plt.scatter(B.radius_mean,B.texture_mean,color="green",label="iyi",alpha= 0.3)
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.legend()
plt.show()

#переводим диагнозы в бинарное отображение
#1 - злокачественное образование
#0 - доброкачественное образование
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
y = data.diagnosis.values #значения диагнозов
x_data = data.drop(["diagnosis"],axis=1) #все остальные значения

#нормализуем датасет 
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))

#разбиваем датасет на train и test
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state=1)

#KNN модель – модель K ближайших соседей (K Nearest Neighbours)
knn = KNeighborsClassifier(n_neighbors = 3)#создаем модель, ищем по 3-ем соседям
knn.fit(x_train,y_train)#обучаем
prediction = knn.predict(x_test)#Предсказываем
print(" {} nn score: {} ".format(3,knn.score(x_test,y_test)))#смотрим точность

#Смотрим точность для разного числа соседей(от 1 до 15). Строим график точности от кол-ва соседей
score_list = []
for each in range(1,15):
    knn2 = KNeighborsClassifier(n_neighbors = each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))
    
plt.plot(range(1,15),score_list)
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.show()
