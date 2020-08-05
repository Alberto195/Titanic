import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

'''WHAT MATTERS
    sex: more female survivors
    Pclass: more survivors in 1 class less in 2 and least in 3
    SibSp: the more siblings, the less survivability
    Parch: the less parents, the more survivability
    Embarked: more survivors from S and passengers overall
    Fare: the greater the price, more chances of living
    Age: Everyone under 10 have more survivability
    Cabin: it matters, just matters, но не сейчас
    '''

""" Инициализация данных """
trd = pd.read_csv('C:/titanic/train.csv')
tsd = pd.read_csv('C:/titanic/test.csv')
tsl = pd.read_csv('C:/titanic/gender_submission.csv')

""" Заполнение пропущенных данных """
trd['Embarked'].fillna(value='C', inplace=True)
trd['Age'].fillna(value=30.0, inplace=True)

tsd['Embarked'].fillna(value='C', inplace=True)
tsd['Age'].fillna(value=30.0, inplace=True)
tsd['Fare'].fillna(value=720.0, inplace=True)

""" Перевод столбцов со словами в столбцы в цифрами """
embr_train = pd.get_dummies(trd.Embarked, prefix="Emb", drop_first=True)
sex_train = pd.get_dummies(trd.Sex, prefix="Sx", drop_first=True)

embr_test = pd.get_dummies(tsd.Embarked, prefix="Emb", drop_first=True)
sex_test = pd.get_dummies(tsd.Sex, prefix="Sx", drop_first=True)

""" Инициализация лейблов """
train_set_labels = trd['Survived'].values.reshape([891, 1])
test_set_labels = tsl['Survived'].values.reshape([418, 1])

""" Удаление ненужных столбцов """
trd.drop(['Name', 'Ticket', 'PassengerId', 'Survived', 'Sex', 'Embarked', 'Cabin'], axis=1, inplace=True)
tsd.drop(['Name', 'Ticket', 'PassengerId', 'Sex', 'Embarked', 'Cabin'], axis=1, inplace=True)

""" Создание nparray из данных """
testik = np.append(trd, embr_train, axis=1)
train_set = np.append(testik, sex_train, axis=1)

testik = np.append(tsd, embr_test, axis=1)
test_set = np.append(testik, sex_test, axis=1)

""" Перевод все значений в float32 """
train_set = np.asarray(train_set).astype(np.float32)
train_set_labels = np.asarray(train_set_labels).astype(np.float32)

test_set = np.asarray(test_set).astype(np.float32)
test_set_labels = np.asarray(test_set_labels).astype(np.float32)

""" Проверка на наличие упущенных незаполненных данных """
#sns.heatmap(trd.isnull(), cbar=False).set_title("Missing values heatmap train")
#plt.show()

#sns.heatmap(tsd.isnull(), cbar=False).set_title("Missing values heatmap test")
#plt.show()

""" Создание модели """
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(8, ), activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

""" Компиляция модели """
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

""" Тренировка модели """
model.fit(train_set, train_set_labels, epochs=10)

""" Оценка точности модели """
test_loss, test_acc = model.evaluate(test_set,  test_set_labels, verbose=2)
print('\nТочность на проверочных данных:', test_acc)

""" Угадывание выжил или нет """
predictions = model.predict(test_set)

""" Проверка рандомного значения на совпадение """
print(np.argmax(predictions[0]))
print(test_set_labels[0])
