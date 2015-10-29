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

#print city_model(X)

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
  
#X={}
#X['longitude']=-90
#X['latitude']= 45
#print lat_long_model(X)

#Question 3
#OK, read in data and deal with lst instead of categories

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
                
cat_lst=[]

for i in categories:
    #cat_set= cat_set.union(set(i))
    temp_dict={}
    for j in i:
        temp_dict[j]=1
    cat_lst.append(temp_dict)
#print cat_lst


#test out DictVectorizer
v=DictVectorizer(sparse=False)
X=v.fit_transform(cat_lst)
f=open("dict_category_model","wb")
dill.dump(v, f)
f.close()

#print type(X)


#{u'categories': [u'Food', u'Automotive', u'Convenience Stores', u'Gas & Service Stations']}

test={u'categories': [u'Electricians', u'Home Services']}
test_tf={x:1 for x in test['categories']}

test_tf1=v.transform(test_tf)
#print type(test_tf1)
Ysubset_np=df[['stars']].as_matrix()
 
#print type(df.ix[0,'categories'])
class lEstimator(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    def __init__ (self):
        self.linear=linear_model.LinearRegression()
        
    def fit(self,X, y):
        self.linear.fit(X, y) 
        return self
    
    def predict(self,X):
        try:
            return float(self.linear.predict(X)[0][0])
        except:
            return 0

q3 = lEstimator()  # initialize
q3.fit(X,Ysubset_np)    
f=open("category_model","wb")
dill.dump(q3, f)
f.close()
# For question 3, we need to dump more than just fit model.
#print q3.predict(test_tf1)
def category_model(record):
    test_tf={x:1 for x in record['categories']}
    f=open("dict_category_model", "r")
    a=dill.load(f)
    test_tf1=a.transform(test_tf)
    f.close()
    
    f=open("category_model","r")
    b=dill.load(f)
    f.close()
    try:
        return b.predict(test_tf1)
    except:
        return 0

#print category_model(test)
#####################################
# Question 4:
#First, need to dump myFlatter into dill file

def myFlatterLocal(u,v):
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
 
retval = map(myFlatter, att)
#print retval

u=DictVectorizer(sparse=False)
Y=u.fit_transform(retval )
f=open("dict_attribute_model","wb")
dill.dump(u, f)
f.close()
#print Y

myTestData = {'attributes':{'Take-out': True, 'Price Range': 1, 'Outdoor Seating': False, 'Caters': False, 'Noise Level': 'average', 'Parking': {'garage': False, 'street': False, 'validated': False, 'lot': True, 'valet': False}, 'Delivery': False, 'Attire': 'casual', 'Has TV': True, 'Good For': {'dessert': False, 'latenight': False, 'lunch': True, 'dinner': False, 'brunch': False, 'breakfast': False}, 'Takes Reservations': False, 'Ambience': {'romantic': False, 'intimate': False, 'touristy': False, 'hipster': False, 'divey': False, 'classy': False, 'trendy': False, 'upscale': False, 'casual': False}, 'Waiter Service': True, 'Accepts Credit Cards': True, 'Good for Kids': True, 'Good For Groups': True, 'Alcohol': 'none'}}
#myTestData1=myFlatter(myTestData)
#myTestData2=u.transform(myTestData1)
#print myTestData2

q4 = lEstimator()  # initialize
q4.fit(Y,Ysubset_np)    
f=open("attribute_model","wb")
dill.dump(q4, f)
f.close()
#print q4.predict(myTestData2)

def attribute_knn_model(record):

    test_tf=myFlatter(record['attributes'])
    f=open("dict_attribute_model", "r")
    a=dill.load(f)
    test_tf1=a.transform(test_tf)
    f.close()
    
    f=open("attribute_model","r")
    b=dill.load(f)
    f.close()
    try:
        return b.predict(test_tf1)
    except:
        return 0

print attribute_knn_model(myTestData)
