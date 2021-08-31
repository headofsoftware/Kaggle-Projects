









import WaterQuality_Preprocessing as preprocessing
from sklearn.ensemble        import RandomForestClassifier




class Run():
    def __init__(self):
        
        self.process=preprocessing.Run()
        self.model3()
    def model3(self)   :
        app=RandomForestClassifier(n_estimators=350,max_depth=15).fit(self.process.X,self.process.Y)
        acc=app.score(self.process.X,self.process.Y)
        print("-"*40)
        print("Random Forest Algorithms  Acc:",acc)
        
        
        
     
        
        
        
if __name__ == '__main__':
    run=Run()

























