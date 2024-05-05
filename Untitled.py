#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns
sns.set(style="whitegrid", color_codes=True)

DATA_DIR = "C:/Users/ekbin/Desktop/ch06"
os.listdir(DATA_DIR)


# In[7]:


# test.csv과 test.csv를 가져온 후, 파일 순서 바꾸고 상대 경로 리스트 생성
DATA_DIR = "C:/Users/ekbin/Desktop/ch06"
data_files = sorted([os.path.join(DATA_DIR, filename)
     for filename in os.listdir(DATA_DIR)], reverse=True)
data_files


# In[8]:


# (1) 데이터프레임을 각 파일에서 읽어온 후 df_list에 추가
df_list = []
for filename in data_files:
    df_list.append(pd.read_csv(filename)) 

# (2) 두 개의 데이터프레임을 하나로 통합
df = pd.concat(df_list, sort=False) 

# (3) 인덱스 초기화 
df = df.reset_index(drop=True) 

# (4) 결과 출력
df.head(5)


# In[9]:


# (1) train.csv 데이터의 수
number_of_train_dataset = df.Survived.notnull().sum()
# (2) test.csv 데이터의 수
number_of_test_dataset = df.Survived.isnull().sum() 
# (3) train.csv 데이터의 y 값 추출
y_true = df.pop("Survived")[:number_of_train_dataset]


# In[10]:


df.head(2).T


# In[11]:


# (1) 데이터를 소수점 두 번째 자리까지 출력
pd.options.display.float_format = '{:.2f}'.format
 
# (2) 결측치 값의 합을 데이터의 개수로 나눠 비율로 출력
df.isnull().sum() / len(df) * 100


# In[12]:


df[df["Age"].notnull()].groupby(
     ["Sex"])["Age"].mean()


# In[13]:


df[df["Age"].notnull()].groupby(
     ["Pclass"])["Age"].mean()


# In[14]:


df["Age"].fillna(
 df.groupby("Pclass")["Age"].transform("mean"), inplace=True)

df.isnull().sum() / len(df) * 100


# In[15]:


df.loc[61,"Embarked"] = "S"
df.loc[829,"Embarked"] = "S"


# In[16]:


df.info()


# In[22]:


object_columns = ["PassengerId", "Pclass", "Name", "Sex", "Ticket", "Cabin", "Embarked"]
numeric_columns = ["Age", "SibSp", "Parch", "Fare"]

for col_name in object_columns:
    df[col_name] = df[col_name].astype(object)
for col_name in numeric_columns:
    df[col_name] = df[col_name].astype(float)

df["Parch"] = df["Parch"].astype(int)
df["SibSp"] = df["SibSp"].astype(int)


# In[23]:


def merge_and_get(ldf, rdf, on, how="inner", index=None):
    if index is True:
        return pd.merge(ldf,rdf, how=how,left_index=True, right_index=True)
    else:
        return pd.merge(ldf,rdf, how=how, on=on)


# In[28]:


temp_columns = ["Sex", "Pclass", "Embarked"]
for col_name in temp_columns:
    temp_df = pd.merge(
        one_hot_df[col_name], y_true, left_index=True, right_index=True)
    sns.countplot(x="Survived", hue=col_name, data=temp_df)
    plt.show()


# In[29]:


temp_df = pd.merge(one_hot_df[temp_columns], 
                   y_true, left_index=True, 
                   right_index=True)
g = sns.catplot(x="Embarked", 
                hue="Pclass", 
                col="Survived",
                data=temp_df, 
                kind="count",
                height=4, aspect=.7);


# In[30]:


temp_df = pd.merge(
    one_hot_df[temp_columns], 
    y_true, left_index=True, 
    right_index=True)
g = sns.catplot(x="Pclass", 
                hue="Sex", col="Survived",
                data=temp_df, kind="count",
                height=4, aspect=.7)


# In[31]:


temp_df = pd.merge(
    one_hot_df[temp_columns], 
    y_true, left_index=True, 
    right_index=True)
g = sns.catplot(
    x="Embarked", hue="Sex", 
    col="Survived",
    data=temp_df, kind="count",
    height=4, aspect=.7);


# In[32]:


crosscheck_columns = [col_name for col_name in one_hot_df.columns.tolist()
    if col_name.split("_")[0] in temp_columns and "_" in col_name ] + ["Sex"]

# temp 열
temp_df = pd.merge(one_hot_df[crosscheck_columns], y_true, left_index=True, right_index=True)

corr = temp_df.corr()
sns.set()
ax = sns.heatmap(corr, annot=True, linewidths=.5, cmap="YlGnBu")


# In[ ]:




