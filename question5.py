##### First, read in the data first
import pandas as pd
import numpy as np

import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn import datasets, linear_model

import simplejson as js
import json 
import dill
import re

import gzip
with gzip.open('yelp_train_academic_dataset_business.json.gz', 'rb') as f:
    file_content = f.read()   
    
nonspace = re.compile(r'\S')
def iterparse(j):
    decoder = js.JSONDecoder()
    pos = 0
    while True:
        matched = nonspace.search(j, pos)
        if not matched:
            break
        pos = matched.start()
        decoded, pos = decoder.raw_decode(j, pos)
        yield decoded
data_lst=list(iterparse(file_content))




##### First, read in the data first
import gzip
with gzip.open('yelp_train_academic_dataset_business.json.gz', 'rb') as f:
    file_content = f.read()   
    
nonspace = re.compile(r'\S')
def iterparse(j):
    decoder = js.JSONDecoder()
    pos = 0
    while True:
        matched = nonspace.search(j, pos)
        if not matched:
            break
        pos = matched.start()
        decoded, pos = decoder.raw_decode(j, pos)
        yield decoded
data_lst=list(iterparse(file_content))
#print data_lst[0]

city=[]
stars=[]
categories=[]
att=[]

for item in data_lst:
    city.append(item.get('city'))
    categories.append(item.get('categories')) 
    att.append(item.get('attributes'))
    stars.append(item.get('stars'))

df=pd.DataFrame({'stars': stars})
 
#Just get the list in the data:
for x in data_lst:
    del x['stars']
    #del x['hours']    
    # who cares for the hours
    
#####################################
#####################################
#Question 1: City model

# my feature data is a list of dictionaries with city; long; lat; cat stuffs
# my y value is star rating 
class myCityModel(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    def __init__(self):
        self.v = DictVectorizer(sparse=False)

    def fit(self, X, y=None):
        city = [{'city':x['city']} for x in X] 
        self.v.fit_transform(city)    
        return self
    
    def transform(self,X):
        city = [{'city':x['city']} for x in X] 
        retval = self.v.transform(city)
        return retval

class myLongLatModel(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
#    def __init__(self,y):
        
        
    def fit(self, X, y=None):
        return self
    
    def transform(self,X):
        retval = [[x['longitude'],x['latitude']] for x in X] 
        return retval
    
class myCatModel(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    def __init__(self):
        self.v = DictVectorizer(sparse=False)

    def fit(self, X, y=None):
        def mylocalfun(x): 
            return {y:1 for y in x}
        categories = [x['categories'] for x in X]
        categories = map(mylocalfun, categories)    
        self.v.fit_transform(categories)
        return self
    
    def transform(self,X):
        def mylocalfun(x): 
            return {y:1 for y in x}                
        categories = [x['categories'] for x in X]
        categories = map(mylocalfun, categories)                
        retval = self.v.transform(categories)
        return retval


class myAttModel(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):    
    def __init__(self):
        self.v = DictVectorizer(sparse=False)

    def fit(self, X, y=None):
        def myFlatterLocal(u,v):
            if type(v) == list:
                v = { z:1 for z in v}
            if type(v) == dict:
                retVal = v.values()
                retKey = [re.sub(' ','', u +"_"+x) for x in v.keys()]
                retPair = {x:y for x,y in zip(retKey, retVal)}
            else:
                retPair = {u:v}
            return retPair

        def merge_two_dicts(x, y):
            z = x.copy()
            z.update(y)
            return z

        def myFlatter(myTestData):
            if(len(myTestData)==0):
                return {}
            retval = []
            for myKey in myTestData.keys():
                retval.append(myFlatterLocal(myKey, myTestData[myKey]))
            retval = reduce(merge_two_dicts,retval)
            return retval    
        tmp = [x['attributes'] for x in X]
        attributes= map(myFlatter, tmp)
        self.v.fit_transform(attributes)
        return self
    
    def transform(self,X):
        def myFlatterLocal(u,v):
            import re
            if type(v) == list:
                v = { z:1 for z in v}
            if type(v) == dict:
                retVal = v.values()
                retKey = [re.sub(' ','', u +"_"+x) for x in v.keys()]
                retPair = {x:y for x,y in zip(retKey, retVal)}
            else:
                retPair = {u:v}
            return retPair

        def merge_two_dicts(x, y):
            z = x.copy()
            z.update(y)
            return z

        def myFlatter(myTestData):
            if(len(myTestData)==0):
                return {}
            retval = []
            for myKey in myTestData.keys():
                retval.append(myFlatterLocal(myKey, myTestData[myKey]))
            retval = reduce(merge_two_dicts,retval)
            return retval    

        tmp = [x['attributes'] for x in X]
        attributes= map(myFlatter, tmp)
        retval = self.v.transform(attributes)
        return retval
    
tmpModel1 = myCityModel()   
tmpModel2 = myLongLatModel()
tmpModel3 = myCatModel()
tmpModel4 = myAttModel()

combined_features = FeatureUnion([('city',tmpModel1),('longlat',tmpModel2),\
                                    ('categories',tmpModel3) , ('attributes',tmpModel4)])

combined_features.fit(data_lst,stars)
 
f=open("combined_features_model","wb")
dill.dump(combined_features, f)
f.close()

X_features = combined_features.transform(data_lst)

from sklearn import linear_model
clf = linear_model.LinearRegression()
clf.fit(X_features,stars)

f=open("clf_model","wb")
dill.dump(clf, f)
f.close()

def full_model(record):

    f=open("combined_features_model", "r")
    a=dill.load(f)
    f.close()

    f=open("clf_model","r")
    b=dill.load(f)
    f.close()

    test_features=a.transform([record])
        
    try:
        return float(b.predict(test_features)[0])
    except:
        return 0
        
#test={u'city': u'Apache Junction', u'review_count': 3, u'name': u'Brown Bear BBQ', u'neighborhoods': [], u'open': True, u'business_id': u'ElEF5b3n27IBzbA4R-1M1g', u'full_address': u'Chevron\n3940 S Ironwood Dr\nApache Junction, AZ 85120', u'hours': {u'Tuesday': {u'close': u'19:00', u'open': u'07:00'}, u'Friday': {u'close': u'19:00', u'open': u'07:00'}, u'Monday': {u'close': u'19:00', u'open': u'07:00'}, u'Thursday': {u'close': u'19:00', u'open': u'07:00'}, u'Wednesday': {u'close': u'19:00', u'open': u'07:00'}}, u'state': u'AZ', u'longitude': -111.564202, u'latitude': 33.379277, u'attributes': {u'Take-out': True, u'Parking': {u'garage': False, u'street': False, u'validated': False, u'lot': False, u'valet': False}, u'Good For': {u'dessert': False, u'latenight': False, u'lunch': False, u'dinner': False, u'breakfast': False, u'brunch': False}, u'Attire': u'casual', u'Waiter Service': False, u'Takes Reservations': False, u'Accepts Credit Cards': True, u'Price Range': 2}, u'type': u'business', u'categories': [u'Food', u'Street Vendors', u'Barbeque', u'Restaurants']}
#hey=data_lst[0]
