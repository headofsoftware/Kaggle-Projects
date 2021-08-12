


import pandas as pd 
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import  keras.optimizers 
import keras.losses
import keras.metrics
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

class Run():
    def __init__(self):
        self.read_file()
        self.seperate_matrix()
        self.cut_datas()
        # self.show_datas_shape()
        self.create_model()
        self.exe_model()
        self.visualize_result_acc()
        self.visualize_result_loss()
        
        
    def read_file(self):
        self.matrix=pd.read_csv("iris.csv")
        self.value=np.array(self.matrix)
    def seperate_matrix(self):
        self.input=self.matrix.drop(columns=["Id","Species"]).values
        self.output=self.matrix["Species"].values
        self.output = LabelBinarizer().fit_transform(self.output)
    def cut_datas(self):
        self.X_train,self.X_test,self.Y_train,self.Y_test=train_test_split(self.input,self.output,test_size=0.15)
    def create_model(self):
        self.model=Sequential()
        self.model=Sequential()
        self.model.add(Dense(4,activation="relu",input_dim=4))
        self.model.add(Dense(64,activation="relu"))
        self.model.add(Dense(256,activation="relu"))
        self.model.add(Dense(256,activation="relu"))
        self.model.add(Dense(256,activation="relu"))
        self.model.add(Dense(64,activation="relu"))
        self.model.add(Dense(3,activation="softmax"))
        
        self.model.compile(loss="categorical_crossentropy",
                            optimizer="adam",
                            metrics=['accuracy'])
    def exe_model(self):
        self.history=self.model.fit(
                    self.X_train,
                    self.Y_train,
                    epochs=38,
                    batch_size=64,
                    validation_data=(self.X_test, self.Y_test),
            )
    def visualize_result_acc(self):
        print(self.history.history.keys())
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
    def visualize_result_loss(self):
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
    def show_datas_shape(self):
        
        print("X_train:",self.X_train.shape)
        print("Y_train",self.Y_train.shape)
        print("X test :",self.X_test.shape)
        print("Y test :",self.Y_test.shape)
        

if __name__=='__main__':
    run=Run()





