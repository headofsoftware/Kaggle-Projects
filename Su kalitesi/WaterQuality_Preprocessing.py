


import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class Run():
    def __init__(self):
        
        self.file="water_potability.csv"
        self.read_file()
        self.check_data()
        self.separate()
        self.rescale()
        self.check_variance()
        
        
        
        
        
    def read_file(self):
        print(" "*30,"READ FİLE")
        self.data=pd.read_csv(self.file)
        print(self.data.head())
    def check_data(self):#Delete rows that Null values
        print("-"*50,"\n"," "*30,"CHECK DATA")
        print("\t\t\tPrevious Shape:{}".format(self.data.shape))
        self.data.fillna(self.data.mean(),inplace=True)
        # self.data.dropna(inplace=True)
        print("\t\t\tNext Shape:   ",self.data.shape)
        print("Removed rows that NaN Value")
        
        print(self.data.isnull().sum())
        
    
        
        
    
    
    
    def separate(self):
        print("-"*50,"\n"," "*30,"SEPERATE")
        self.inputs=self.data.drop(columns=["Potability"])
        self.outputs=self.data["Potability"]
            
        

        self.X,self.x,self.Y,self.y=train_test_split(self.inputs,self.outputs,test_size=0.25,random_state=42)   
        
        print("Train Input shape:",self.X.shape)
        print("Test Input shape",self.x.shape)
        print("Train Output shape",self.Y.shape)
        print("Test Output shape",self.y.shape)

    def rescale(self):
        print("-"*50,"\n"," "*30,"RESCALE ")
        stds=MinMaxScaler(feature_range=(0,1))
        self.X=pd.DataFrame(stds.fit_transform(self.X))
        self.x=pd.DataFrame(stds.fit_transform(self.x))


    def check_variance(self):#i examine variance of columns
        print("-"*50,"\n"," "*30,"CHECK VARİANCE ")
        print("\t\t\t\tBEFORE")
        for i in self.X.columns:
            print(i,"--->",np.var(self.X[i]))
            
        #Delete high variance of columns
        self.X.drop(columns=[2,5,8],inplace=True)
        self.x.drop(columns=[2,5,8],inplace=True)
        print("-"*30)
        print("\t\t\t\tAFTER")
        for i in self.X.columns:
            print(i,"--->",np.var(self.X[i]))
            
            
    
        
        
        
        
if __name__ == '__main__':
    run=Run()



