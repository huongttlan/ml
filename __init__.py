from __future__ import absolute_import
import toolz

import typecheck
import fellow
import pandas as pd
import numpy as np
from .data import test_json

import dill
import re

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


def pick(whitelist, dicts):
    return [toolz.keyfilter(lambda k: k in whitelist, d)
            for d in dicts]

def exclude(blacklist, dicts):
    return [toolz.keyfilter(lambda k: k not in blacklist, d)
            for d in dicts]

@fellow.batch(name="ml.city_model")
@typecheck.test_cases(record=pick({"city"}, test_json))
@typecheck.returns("number")
def city_model(record):
    f=open("./ml/city_model","r")
    a=dill.load(f)
    f.close()
    return float(a.predict(record))


@fellow.batch(name="ml.lat_long_model")
@typecheck.test_cases(record=pick({"longitude", "latitude"}, test_json))
@typecheck.returns("number")
def lat_long_model(record):
    test=np.array([record['longitude'], record['latitude']])
    f=open("./ml/lat_long_model","r")
    a=dill.load(f)
    f.close()
    
    return float(a.predict(test))


@fellow.batch(name="ml.category_model")
@typecheck.test_cases(record=pick({"categories"}, test_json))
@typecheck.returns("number")
def category_model(record):
    test_tf={x:1 for x in record['categories']}
    f=open("./ml/dict_category_model", "r")
    a=dill.load(f)
    test_tf1=a.transform(test_tf)
    f.close()
    
    f=open("./ml/category_model","r")
    b=dill.load(f)
    f.close()
    try:
        return b.predict(test_tf1)
    except:
        return 0

#Function to flatter out dictionary inside dictionary

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

@fellow.batch(name="ml.attribute_knn_model")
@typecheck.test_cases(record=pick({"attributes"}, test_json))
@typecheck.returns("number")
def attribute_knn_model(record):

    test_tf=myFlatter(record['attributes'])
    f=open("./ml/dict_attribute_model", "r")
    a=dill.load(f)
    test_tf1=a.transform(test_tf)
    f.close()
    
    f=open("./ml/attribute_model","r")
    b=dill.load(f)
    f.close()
    try:
        return b.predict(test_tf1)
    except:
        return 0


@fellow.batch(name="ml.full_model")
@typecheck.test_cases(record=exclude({"stars"}, test_json))
@typecheck.returns("number")
def full_model(record):

    f=open("./ml/combined_features_model", "r")
    a=dill.load(f)
    f.close()

    test_features=a.transform([record])
    
    f=open("./ml/clf_model","r")
    b=dill.load(f)
    f.close()
    try:
        return float(b.predict(test_features)[0])
    except:
        return 0
