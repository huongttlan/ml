import urllib2
import re
import simplejson as js
import json 
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

from datetime import datetime

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
print data_lst[0]

city=[]
review=[]
name=[]
hood=[]
typ=[]
biz_id=[]
addr=[]
#Mon_op=[]
#Tue_op=[]
#Wed_op=[]
#Thu_op=[]
#Fri_op=[]
#Sat_op=[]
#Sun_op=[]
#Mon_cl=[]
#Tue_cl=[]
#Wed_cl=[]
#Thu_cl=[]
#Fri_cl=[]
#Sat_cl=[]
#Sun_cl=[]
state=[]
long=[]
lat=[]
stars=[]
open=[]
att=[]
hours=[]
categories=[]
list_day=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
#list_op=[Mon_op, Tue_op, Wed_op, Thu_op, Fri_op, Sat_op, Sun_op]
#list_cl=[Mon_cl, Tue_cl, Wed_cl, Thu_cl, Fri_cl, Sat_cl, Sun_cl]

# Ok, try to get the
for item in data_lst:
    city.append(item.get('city'))
    review.append(item.get('review_count'))
    name.append(item.get('name'))
    hood.append(item.get('neighborhoods'))
    typ.append(item.get('type'))
    biz_id.append(item.get('business_id'))
    addr.append(item.get('full_address'))
    hours.append(item.get('hours'))
    categories.append(item.get('categories'))

    #for i in xrange(0,len(list_day)):
        #if list_day[i] in item.get('hours').keys():
            #list_op[i].append(item.get('hours').get(list_day[i]).get('open'))
            #list_cl[i].append(item.get('hours').get(list_day[i]).get('close'))
        #else:
            #list_op[i].append("")
            #list_cl[i].append("")
    state.append(item.get('state'))
    long.append(item.get('longitude'))      
    lat.append(item.get('latitude'))  
    stars.append(item.get('stars'))
    open.append(item.get('open'))
    att.append(item.get('attributes'))

df=pd.DataFrame({'city':city, 'review_count':review, 'name':name, 'neighborhoods': hood,\
                'type': typ, 'business_id': biz_id, 'full_address':addr,'state': state,\
                'longitude': long,'latitude': lat, 'stars': stars,'open': open})

#Ok, now we can work with hours
hours_df=pd.DataFrame(hours)
hey=[]

for i in xrange(0,len(list_day)):
    a=hours_df[list_day[i]].to_dict()
    b=pd.DataFrame(a).transpose()
    b.columns=[list_day[i]+'_open', list_day[i]+ '_close']
    df=pd.concat([df,b], axis=1)


#print [datetime.strptime(i,'%H:%M') for i in b[list_day[i]+'_open']]
# Ok, need to check the attributes to see how many they have
#att_set=set()
#for i in att:
    #temp=set(i.keys())
    #att_set= att_set.union(temp)
#att_lst=list(att_set)
# Ok, now we can work through attributes 
attr_df=pd.DataFrame(att)


#print attr_df.columns.values
#print attr_df
#print attr_df['Parking']

print att[1]

#print type(att)
#print qData[0:2]

# from sklearn.feature_extraction import DictVectorizer
# vec = DictVectorizer()
# vec.fit_transform(qData)



#v=DictVectorizer(sparse=False)
#X=v.fit_transform(att)

#Ok, for garage parking, it has to divide into dictionary again
c=pd.DataFrame(attr_df['Parking'].to_dict()).transpose()
#Ok, for ambience
d=pd.DataFrame(attr_df['Ambience'].to_dict()).transpose()
attr_df1=attr_df.drop('Parking',1)
attr_df1=attr_df1.drop('Ambience',1)
attr_df1=pd.concat([attr_df1,c,d], axis=1)

#example of attribute as dictionary

#'attributes': {'By Appointment Only': True}, 'open': True, 'categories': ['Doctors', 'Health & Medical']}


#work with categories
#cat_set=set()
cat_lst=[]

for i in categories:
    #cat_set= cat_set.union(set(i))
    temp_dict={}
    for j in i:
        temp_dict[j]=1
    cat_lst.append(temp_dict)

cat_df=pd.DataFrame(cat_lst)
cat_df=cat_df.fillna(0)


#cat_df=pd.DataFrame({'categories':cat_lst})


#Ok, now create a full table

#df=pd.concat([df,attr_df1,cat_df], axis=1)
#print df


#Ok, now print out the data
#df.to_csv('Yelpdata.txt', sep='|',encoding='utf-8')
#OK, question 1: just need to deal with city and star ratings
#hey=df[['city','stars']]
#hey.to_csv("city.txt",sep="|", cols=['city', 'stars'],encoding='utf-8')
#loc=df[['city', 'stars','longitude','latitude']]
#loc.to_csv("location.txt",sep="|", cols=['longitude', 'latitude'],encoding='utf-8')
#cat=pd.concat([df[['stars']],cat_df], axis=1)
#cat.to_csv("categories.txt",sep="|",encoding='utf-8')