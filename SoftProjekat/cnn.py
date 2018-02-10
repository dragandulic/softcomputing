# -*- coding: utf-8 -*-
"""
Created on Fri Feb 09 02:22:41 2018

@author: dragan
"""

import numpy as np
np.random.seed(123)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten	
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

from matplotlib import pyplot as plt

from keras.datasets import mnist
 
#dimenzije slike da postavim
from keras import backend as K
K.set_image_dim_ordering('th')


def createSaveCnn():
    # Load pre-shuffled MNIST data into train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    #transformisemo nas skup podataka (n, width, height) to (n, depth, width, height).
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255


    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)


    #sekvencijalni model
    model = Sequential()

    #ulazni sloj
    #32-broj filter koji se koristi, 3,3-broj reodva i vrsta u svakom konvolucionom jezgru
    model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(1,28,28)))
    
    #dodajemo slojeve nasem modelu
    model.add(Convolution2D(32, 3, 3, activation='relu'))

    #da smanjimo broj parametara klizanjem filtera 2x2 i uzimanje max vr.
    model.add(MaxPooling2D(pool_size=(2,2)))

    #metoda za regulisanje modela kako bi se smanjilo prekomerno ucenje
    model.add(Dropout(0.25))

    #sloj koji povezuje sve
    #izravnava izlaz iz poslednjeg sloja kako bisnmo dobili vektor koji ce biti ulaz u naredni sloj ( desne_)
    model.add(Flatten())

    #Za DENSE slojeve prvi parametar je izlazna velicina sloja. 
    #Keras automacki odradjuje vezu izmedju slojeva
    model.add(Dense(128, activation='relu'))


    model.add(Dropout(0.5))
    #izlazni sloj
    #softmax genersie verovatnocu kojoj slika prippada jednoj od 10 kategorija
    model.add(Dense(10, activation='softmax'))

    #funkcija gubitka
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])


    model.fit(X_train, Y_train, 
              batch_size=32, nb_epoch=10, verbose=1)

    
    score = model.evaluate(X_test, Y_test, verbose=0)
    print(score)
 


    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

    return 1,model









