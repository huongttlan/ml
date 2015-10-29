import pandas as pd
import numpy as np

import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets.twenty_newsgroups import strip_newsgroup_footer
from sklearn.datasets.twenty_newsgroups import strip_newsgroup_quoting
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

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
    

X={}
X['city']="hey"
#city_model(X)

