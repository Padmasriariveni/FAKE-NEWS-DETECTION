# FAKE-NEWS-DETECTION
import pandas as pd

true=pd.read_csv("/content/True.csv")

fake=pd.read_csv("/content/Fake.csv")

true.head()

fake.head()

from matplotlib import pyplot as plt
plt.pie(true['subject'].value_counts(),labels=true['subject'].value_counts().keys(),autopct='%1.2f%%')

print(true.describe())

print(fake.describe())

true['label']=1

fake['label']=0

news=pd.concat([fake,true],axis=0)

news.head()

news.tail()

news.isnull().sum()

news=news.drop(['title','subject','date'],axis=1)

news.head()

news.tail()

news=news.sample(frac=1) #shuffles

news.head()

news.reset_index(inplace=True)

news.head()

news.drop(['index'],axis=1,inplace=True)

news.head()

import matplotlib.pyplot as plt
print(news.groupby(['label']).count())
news.label.value_counts().plot(kind='bar',color=['grey','skyblue'])
plt.show()

import re

def wordopt(text):
  text=text.lower() #converting to lower
  text=re.sub(r'\[.*?\]','',text) #removing square brackets
  text=re.sub(r'https?://\S+|www\.\S+','',text) #removing urls
  text=re.sub(r'<.*?>+','',text) #removing html tags
  text=re.sub(r'[^\w\s]','',text) #removing punctuations
  text=re.sub(r'\n','',text) #removing new lines
  return text

news['text']=news['text'].apply(wordopt)

news['text']

x=news['text']
y=news['label']

x

y

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

x_train.shape

x_test.shape

from sklearn.feature_extraction.text import TfidfVectorizer

vect=TfidfVectorizer()

xv_train=vect.fit_transform(x_train)
xv_train

xv_test=vect.transform(x_test)
xv_test

from sklearn.linear_model import LogisticRegression

LR=LogisticRegression()

LR.fit(xv_train,y_train)

Pred_LR=LR.predict(xv_test)

LR.score(xv_test,y_test)

from sklearn.metrics import classification_report

print(classification_report(y_test,Pred_LR))

from sklearn.tree import DecisionTreeClassifier

DTC=DecisionTreeClassifier()

DTC.fit(xv_train,y_train)

pred_dtc=DTC.predict(xv_test)

DTC.score(xv_test,y_test)

print(classification_report(y_test,pred_dtc))

from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier()

rfc.fit(xv_train,y_train)

predict_rfc=rfc.predict(xv_test)

rfc.score(xv_test,y_test)

print(classification_report(y_test,predict_rfc))

from sklearn.ensemble import GradientBoostingClassifier

gbc=GradientBoostingClassifier()

gbc.fit(xv_test,y_test)

pred_gbc=gbc.predict(xv_test)

 gbc.score(xv_test,y_test)

print(classification_report(y_test,pred_gbc))

#predicting model

def output_Label(n):
  if n==0:
    return "Fake News"
  else n==1:
    return "True News"

from pickle import NEWOBJ_EX
def manual_test(news):
  testing_news={"text":[news]}
  new_def_test=pd.DataFrame(testing_news)
  new_x_test=new_def_test["text"]
  new_xv_test=vect.transform(new_x_test)
  pred_LR=LR.predict(new_xv_test)
  pred_gbc=gbc.predict(new_xv_test)
  pred_rfc=rfc.prediction(new_xv_test)
  return '\n\nLR Prediction:{} \nGBC Prediction:{} \nRFC Prediction:{}'.format(
    output_Label(pred_LR[0]),output_Label(pred_gbc[0]),output_Label(pred_rfc[0]))
