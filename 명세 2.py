#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

train_id = train_df["PassengerId"].values
test_id = test_df["PassengerId"].values

all_df = train_df.append(test_df).set_index('PassengerId')


# In[2]:


all_df["Sex"] = all_df["Sex"].replace({"male":0,"female":1})

# 데이터 중 age 값의 빈칸의 값을 `class의 평균값으로 채운다.
all_df["Age"].fillna(
    all_df.groupby("Pclass")["Age"].transform("mean"), inplace=True)


# In[3]:


all_df["cabin_count"] = all_df["Cabin"].map(
   lambda x : len(x.split()) if type(x) == str else 0)


# In[4]:


def transform_status(x):
    if "Mrs" in x or "Ms" in x:
        return "Mrs" 
    elif "Mr" in x:
        return "Mr"
    elif "Miss" in x:
        return "Miss"
    elif "Master" in x:
        return "Master"
    elif "Dr" in x:
        return "Dr"
    elif "Rev" in x:
        return "Rev"
    elif "Col" in x:
        return "Col"
    else:
        return "0"
all_df["social_status"] = all_df["Name"].map(lambda x : transform_status(x))


# In[5]:


all_df["social_status"].value_counts()


# In[6]:


all_df[all_df["Embarked"].isnull()]


# In[9]:


import numpy as np
all_df = all_df.drop([62,830], errors='ignore') 
train_id =np.delete(train_id, [61,829])


# In[25]:


p_v_cap_s = sum((np_data[:, 0] == 1) & (np_data[:, 1] == 1)) / len(np_data)
p_v_cap_s


# In[27]:


all_df[all_df["Embarked"].isnull()]


# In[28]:


all_df.loc[all_df["Fare"].isnull(), "Fare"] = 12.415462


# In[29]:


all_df["cabin_type"] = all_df["Cabin"].map(lambda x : x[0] if type(x) == str else "99") 


# In[30]:


del all_df["Cabin"]
del all_df["Name"]
del all_df["Ticket"]


# In[31]:


y = all_df.loc[train_id, "Survived"].values
del all_df["Survived"]


# In[32]:


X_df = pd.get_dummies(all_df)
X = X_df.values


# In[33]:


from sklearn.preprocessing import MinMaxScaler
minmax_scaler = MinMaxScaler()

minmax_scaler.fit(X)
X = minmax_scaler.transform(X)


# In[34]:


X_train = X[:len(train_id)]
X_test = X[len(train_id):]


# In[35]:


from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score 

test_accuracy = []
train_accuracy = []
for idx in range(3, 20):
    df = DecisionTreeClassifier(min_samples_leaf=idx)
    acc = cross_val_score(df, X_train, y, scoring="accuracy", cv=5).mean()
    train_accuracy.append(
accuracy_score(df.fit(X_train, y).predict(X_train), y))
    test_accuracy.append(acc)

result = pd.DataFrame(train_accuracy, index=range(3,20), columns=["train"])
result["test"] = test_accuracy

result.plot()


# In[36]:


from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

algorithmes = [LogisticRegression(), DecisionTreeClassifier()]

c_params = [0.1, 5.0, 7.0, 10.0, 15.0, 20.0, 100.0]
params = []


params.append([{
    "solver" : ["saga"],
    "penalty" : ["l1"],
    "C" : c_params
    },{
    "solver" : ['liblinear'],
    "penalty" : ["l2"],
    "C" : c_params
    }
    ])
params.append({
    "criterion" : ["gini", "entropy"],
    "max_depth" : [10,8,7,6,5,4,3,2],
    "min_samples_leaf": [1,2,3,4,5,6,7,8,9]})


# In[38]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

scoring = ['accuracy']
estimator_results = []
for i, (estimator, params) in enumerate(zip(algorithmes,params)):
    gs_estimator = GridSearchCV(
        refit="accuracy", estimator=estimator, param_grid=params, scoring=scoring, cv=5, verbose=1, n_jobs=4)

    gs_estimator.fit(X_train, y)
    estimator_results.append(gs_estimator)


# In[39]:


estimator_results[0].best_score_


# In[41]:


estimator_results[1].best_score_


# In[45]:


import pandas as pd
from pandas import DataFrame
from collections import defaultdict

result_df_dict = {}
result_attributes = ["model", "accuracy", "penalty", "solver", "C", "criterion", "max_depth", "min_samples_leaf"]
result_dict = defaultdict(list)
algorithm_name = ["LogisticRegression", "DecisionTreeClassifier"]

for i, estimators in enumerate(estimator_results):
    number_of_estimators = len(estimators.cv_results_["mean_fit_time"])

    for idx_estimator in range(number_of_estimators):
        result_dict["model"].append(algorithm_name[i])
        result_dict["accuracy"].append(estimators.cv_results_["mean_test_accuracy"][idx_estimator])

    for param_value in estimators.cv_results_["params"]:
        for k, v in param_value.items():
            result_dict[k].append(v)

    for attr_name in result_attributes:
        if len(result_dict[attr_name]) < len(result_dict["accuracy"]):
            result_dict[attr_name].extend([None for i in range(number_of_estimators)])


# In[46]:


result_df = DataFrame(result_dict, columns=result_attributes)
result_df.sort_values("accuracy",ascending=False).head(n=10)


# In[50]:


estimator_results[1].best_estimator_.feature_importances_


# In[51]:


X_df.columns


# In[62]:


coef = estimator_results[1].best_estimator_.feature_importances_.argsort()[::-1]


# In[64]:


X_df.columns[coef.argsort()[::-1]][:5]


# In[65]:


get_ipython().system('pip install pydotplus')


# In[66]:


import os
os.environ["PATH"] += os.pathsep + 'C:\Program Files\Graphviz/bin/'


# In[68]:


import pydotplus
from six import StringIO
from sklearn import tree

best_tree = estimator_results[1].best_estimator_
column_names = X_df.columns

dot_data = StringIO()
tree.export_graphviz(best_tree, out_file=dot_data,feature_names=column_names) 

graph = pydotplus.pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png("titanic.png")
from IPython.core.display import Image
Image(filename='titanic.png')


# In[ ]:




