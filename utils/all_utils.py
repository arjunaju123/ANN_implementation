import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import logging

mnist = tf.keras.datasets.mnist

def model_cls():
         """Flatten the 28x28 input matrix as input layer, creating hidden layers and output layer

         Returns:
             keras.engine.sequential.Sequential: Returns an object of keras
         """

         LAYERS = [
            tf.keras.layers.Flatten(input_shape=[28,28], name="inputLayer"),
            tf.keras.layers.Dense(300, activation="relu", name="hiddenLayer1"),
            tf.keras.layers.Dense(100, activation="relu", name="hiddenLayer2"),
            tf.keras.layers.Dense(10, activation="softmax", name="outputLayer")
            ]

         model_clf = tf.keras.models.Sequential(LAYERS)

         return model_clf


def prep_data():

    logging.info("Preparing the data")
    """It is used to load the dataset, split the dataset into training ,testing and validation and predict a sample data

   Returns:
       dict: returns a dictionary of values
    """

    (X_train_full, y_train_full),(X_test, y_test) = mnist.load_data()

    X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

    X_test = X_test / 255.
    
    model_clf = model_cls()

    X_new = X_test[:3]

    y_prob = model_clf.predict(X_new)

    y_prob.round(3)

    Y_pred= np.argmax(y_prob, axis=-1)
   
    return {'X_new': X_new, 'Y_pred': Y_pred, 'y_test': y_test,'X_valid': X_valid,'X_train': X_train,'y_valid': y_valid,'y_train': y_train,'X_test': X_test}


