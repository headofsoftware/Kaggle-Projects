


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  accuracy_score

import pickle

class Run():
    def __init__(self):
        self.file="mushrooms.csv"
        
        self.data=pd.read_csv(self.file)
        self.data_shuffle()
        self.exploring_data()
        self.split_data()
        self.build_model()
        self.model_save()
        # self.get_model()        
        
    def data_shuffle(self):
        # print(self.data.head)
        self.data=self.data.sample(frac=1).reset_index(drop=True)
        
    
    
    def exploring_data(self):
        
        
        
        
        # print(self.data.head())
        # print(self.data.shape)
        # print(self.data.isnull().sum())
        # print(self.data["habitat"].value_counts())
        
        pass
        
        
        
        
        
        
    def split_data(self):
        le=LabelEncoder()
        for col in self.data.columns:
            self.data[str(col)]=le.fit_transform(self.data[str(col)])
            
        self.inputs=self.data.drop(columns=["class"])
        self.outputs=self.data["class"]
        scale=MinMaxScaler(feature_range=(0,1))
        self.inputs=scale.fit_transform(self.inputs)
        self.X,self.x,self.Y,self.y=train_test_split(self.inputs,self.outputs,test_size=0.85)
    def build_model(self):
        print("Model running...")
        self.model=RandomForestClassifier(n_estimators=75,max_features=3,max_depth=3)#max_depth=50,random_state=100,
        self.model.fit(self.X,self.Y)
        y_pred=self.model.predict(self.x)
        Y_pred=self.model.predict(self.X)
        print("Model Accuracy Training :",accuracy_score(self.Y,Y_pred))
        print("Model Accuracy Test     :" ,accuracy_score(self.y,y_pred))
        # print("Model Score   :" , model.score(self.x,self.y))#alternative Model evaluate

    
    def model_save(self):
        pickle.dump(self.model,open("model.sav", 'wb'))

    def get_model(self):
        loaded_model=pickle.load(open("model.sav","rb"))
        print(loaded_model.score(self.x,self.y))
        
        
if  __name__ == '__main__':
    run=Run()
