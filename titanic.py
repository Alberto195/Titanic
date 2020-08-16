import pandas as pd
import numpy as np
import tensorflow as tf
import random as rn
import os

os.environ['PYTHONHASHSEED'] = '0'

# Setting the seed for numpy-generated random numbers
np.random.seed(38)

# Setting the seed for python random numbers
rn.seed(1254)

import matplotlib.pyplot as plt
import seaborn as sns

'''WHAT MATTERS
    sex: more female survivors
    Pclass: more survivors in 1 class less in 2 and least in 3
    SibSp: the more siblings, the less survivability
    Parch: the less parents, the more survivability
    Embarked: more survivors from S and passengers overall
    Fare: the greater the price, more chances of living
    Age: Everyone under 10 have more survivability
    Cabin: it matters, just matters but too complicated for this model
    '''

""" Data initialization """
trd = pd.read_csv('C:/titanic/train.csv')
tsd = pd.read_csv('C:/titanic/test.csv')
tsl = pd.read_csv('C:/titanic/gender_submission.csv')


def filling_na(tr):
    """ Filling in missing data """
    tr['Embarked'].fillna(value='C', inplace=True)
    tr['Age'].fillna(value=30.0, inplace=True)
    tr['Fare'].fillna(value=720.0, inplace=True)


def dummies(tr):
    """ Converting words into numbers """
    embr = pd.get_dummies(tr.Embarked, prefix="Emb", drop_first=True)
    sex = pd.get_dummies(tr.Sex, prefix="Sx", drop_first=True)

    return embr, sex


def label_init():
    """ Label init """
    train_set_labels = trd['Survived'].values.reshape([891, 1])
    test_set_labels = tsl['Survived'].values.reshape([418, 1])

    return train_set_labels, test_set_labels


def table_drop():
    """ Deleting unnecessary columns """
    trd.drop(['Name', 'Ticket', 'PassengerId', 'Survived', 'Sex', 'Embarked', 'Cabin'], axis=1, inplace=True)
    tsd.drop(['Name', 'Ticket', 'PassengerId', 'Sex', 'Embarked', 'Cabin'], axis=1, inplace=True)


def conv2arr(tr, embr, sex):
    """ Converting panda into nparray """
    testik = np.append(tr, embr, axis=1)
    sets = np.append(testik, sex, axis=1)

    return sets


def tofloat(sets, set_labels):
    """ Converting int to float """
    sets = np.asarray(sets).astype(np.float32)
    set_labels = np.asarray(set_labels).astype(np.float32)

    return sets, set_labels


def heatmap(tr):
    """ Showing missing data """
    sns.heatmap(tr.isnull(), cbar=False).set_title("Missing values heatmap")
    plt.show()


def createmodel():
    """ Creating a model """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128, input_shape=(8,), activation='tanh'))
    model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(tf.keras.layers.Dense(64, input_shape=(128,), activation='tanh'))
    model.add(tf.keras.layers.Dense(70, input_shape=(64,), activation='tanh'))
    model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(tf.keras.layers.Dense(32, input_shape=(64,), activation='tanh'))
    model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    return model


filling_na(trd)
filling_na(tsd)

embr_train, sex_train = dummies(trd)
embr_test, sex_test = dummies(tsd)

train_set_labels, test_set_labels = label_init()

table_drop()

train_set = conv2arr(trd, embr_train, sex_train)
test_set = conv2arr(tsd, embr_test, sex_test)

train_set, train_set_labels = tofloat(train_set, train_set_labels)
test_set, test_set_labels = tofloat(test_set, test_set_labels)

model = createmodel()

loss = tf.keras.losses.BinaryCrossentropy(
    from_logits=False, label_smoothing=0, reduction="auto", name="binary_crossentropy"
)

model.compile(optimizer='adam',
              loss=loss,
              metrics=['accuracy'])

model.fit(train_set, train_set_labels, epochs=50, shuffle=False)

test_loss, test_acc = model.evaluate(test_set, test_set_labels, verbose=2)
print('\nAccuracy on test set:', test_acc)

predictions = model.predict(test_set)

model.save_weights('first_try.h5')  # always save your weights after training or during training

nin = 0
wrong = []

for i in predictions:
    if np.argmax(i) != test_set_labels[nin]:
        wrong.append(nin)
    nin += 1

