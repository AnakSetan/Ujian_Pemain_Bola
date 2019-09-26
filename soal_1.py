import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('data.csv')
dfFifa = df[['Name','Age','Overall','Potential']]

#---------------------------Membuat Label direkrut atau tidak sesuai dengan parameter yang ada
dftarget = dfFifa[dfFifa['Age'] <= 25][dfFifa['Overall'] >= 80][dfFifa['Potential'] >= 80]
labeltarget = 'Rekrut'
dftarget['Label'] = labeltarget

dfnontarget = dfFifa.drop(dftarget.index)
labelnontarget = 'Tidak direkrut'
dfnontarget['Label'] = labelnontarget

dflabeled = dftarget.append(dfnontarget, ignore_index = True)
#------------------------------------- Train test split dengan test 10%
print('----------------------------------------------------------------------------------')

x_train, x_test, y_train, y_test = train_test_split(
    dflabeled[['Age', 'Overall', 'Potential']],
    dflabeled['Label'],
    test_size = .1
)
# print(len(x_train))     #16386
# print(len(x_test))      #1821

k = StratifiedKFold(n_splits=100,random_state=None,shuffle=False)
Datapemain = dflabeled[['Age','Overall','Potential']].values

for train_index, test_index in k.split(Datapemain, dflabeled['Label']):
    x_train = Datapemain[train_index]
    y_train = dflabeled['Label'][train_index]

#Define k_value
def k_value():
    k = round((len(x_train)+len(x_test)) ** .5)
    if (k % 2 == 0):
        return k + 1
    else:
        return k
# print(k_value())
print('=====================Prediksi menggunakan Logistic Regression=====================')
print('LR Score : ',round(cross_val_score(LogisticRegression(solver='lbfgs',multi_class='auto'),x_train,y_train).mean()*100),'%')
print('=====================DecisionTreeClassifier=====================')
print('DTC Score : ',round(cross_val_score(DecisionTreeClassifier(),x_train,y_train).mean()*100),'%')
print('=====================KneighborsClassifier=====================')
print('KNN Score : ',round(cross_val_score(KNeighborsClassifier(n_neighbors=k_value()),x_train,y_train).mean()*100),'%')

# Hasil Test :  => LR = 93%
#               => DTC = 89%
#               => KNN = 94%
# Menggunakan model yang akurasi nya paling tinggi, yaitu KNeighborClassifier
model = KNeighborsClassifier(n_neighbors=k_value())
model.fit(x_train,y_train)
dfsoal = pd.DataFrame(
    np.array([
        ['Andik Vermansyah','Madura United FC',27,87,90,'Indonesia'], 
        ['Awan Setho Raharjo','Bhayangkara FC',22,75,83,'Indonesia'],
        ['Bambang Pamungkas','Persija Jakarta',38,85,75,'Indonesia'],
        ['Cristian Gonzales','PSS Sleman',43,90,85,'Indonesia'],
        ['Egy Maulana Vikri','Lechia Gdańsk',18,88,90,'Indonesia'],
        ['Evan Dimas','Barito Putera',24,85,87,'Indonesia'],
        ['Febri Hariyadi','Persib Bandung',23,77,80,'Indonesia'],
        ['Hansamu Yama Pranata','Persebaya Surabaya',24,82,85,'Indonesia'],
        ['Septian David Maulana','PSIS Semarang',22,83,80,'Indonesia'],
        ['Stefano Lilipaly','Bali United',29,88,86,'Indonesia'] 
    ]),
    columns=['Name', 'Club', 'Age', 'Overall', 'Potential', 'Nationality']
)

dfsoal['Potential'] = model.predict(dfsoal[['Age','Overall','Potential']])
print(dfsoal)