import pandas as pd
import numpy as np

import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsRegressor

import dill

class Estimator(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    def __init__ (self):
        self.averageByCity ={}
    
    def fit(self, df):
        try:
            self.averageByCity=df.groupby(by=['city'])['stars'].mean()
        except:
            self.averageByCity={}
        return self
    
    def predict(self,X):
        try:
            return self.averageByCity[X['city']]
        except:
            return 0

def city_model(record):
    df=pd.read_csv ("./city.txt", sep="|",low_memory=False)  
    
    estimator = Estimator()  # initialize
    estimator.fit(df)  # fit data
    f=open("city_model","wb")
    dill.dump(estimator, f)
    f.close()
    return float(estimator.predict(record))
    
#OK, try to use the longitude and latitude


class kEstimator(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    def __init__ (self):
        self.neigh=KNeighborsRegressor(n_neighbors=5)
        
    def fit(self,X, y):
        self.neigh.fit(X, y) 
        return self
    
    def predict(self,X):
        try:
            return self.neigh.predict(X)
        except:
            return 0
            
def lat_long_model(record):
    df=pd.read_csv ("./location.txt", sep="|",low_memory=False)  
    Xsubset_np=df[['longitude','latitude']].as_matrix()
    Ysubset_np=df[['stars']].as_matrix()
    test=np.array([record['longitude'], record['latitude']])
    q2 = kEstimator()  # initialize
    q2.fit(Xsubset_np,Ysubset_np)  # fit data
    f=open("lat_long_model","wb")
    dill.dump(q2, f)
    f.close()
    return float(q2.predict(test))
  
#data=pd.read_csv ("./location.txt", sep="|",low_memory=False)
#Xsubset=data[['longitude','latitude']]
#Xsubset_np=Xsubset.as_matrix()
#Ysubset=data[['stars']]
#Ysubset_np=Ysubset.as_matrix()  
    
X={}
X['longitude']=-90
X['latitude']= 45

print lat_long_model(X)


#city_model(X)

