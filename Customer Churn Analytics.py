#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import Libraries

import numpy as np 
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression
import matplotlib
import matplotlib.ticker as mtick 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from mpl_toolkits.mplot3d import Axes3D 
get_ipython().system('pip install chart-studio')
import chart_studio.plotly as py
from plotly import __version__
from IPython.display import display, HTML


# In[2]:


df = pd.read_csv('Churn_Data.csv')


# In[3]:


df.head()


# In[4]:


df.dtypes


# In[5]:


df.shape


# In[6]:


for item in df.columns:
    print(item)
    print (df[item].unique())


# In[7]:


df.describe()


# In[8]:


for col in ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'DeviceProtection', 'Contract']:
    plt.figure(figsize=(8,5))
    sns.countplot(x=col, hue='Churn', data=df, palette="viridis")
    plt.show()


# In[9]:


dfobject=df.select_dtypes(['object'])
len(dfobject.columns)


# In[10]:


def labelencode(columnname):
    df[columnname] = LabelEncoder().fit_transform(df[columnname])


# In[11]:


for i in range(1,len(dfobject.columns)):
    labelencode(dfobject.columns[i])


# In[12]:


df.info()


# In[13]:


df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
def uni(columnname):
    print(columnname,"--" ,df[columnname].unique())


# In[14]:


for i in range(1,len(dfobject.columns)):
     uni(dfobject.columns[i])


# In[15]:


fig, axes = plt.subplots(nrows = 3,ncols = 5,figsize = (25,15))
sns.countplot(x = "gender", data = df,palette="viridis", ax=axes[0][0])
sns.countplot(x = "Partner", data = df,palette="viridis", ax=axes[0][1])
sns.countplot(x = "Dependents", data = df,palette="viridis", ax=axes[0][2])
sns.countplot(x = "PhoneService", data = df,palette="viridis", ax=axes[0][3])
sns.countplot(x = "MultipleLines", data = df,palette="viridis", ax=axes[0][4])
sns.countplot(x = "InternetService", data = df,palette="viridis", ax=axes[1][0])
sns.countplot(x = "OnlineSecurity", data = df,palette="viridis", ax=axes[1][1])
sns.countplot(x = "OnlineBackup", data = df,palette="viridis", ax=axes[1][2])
sns.countplot(x = "DeviceProtection", data = df,palette="viridis", ax=axes[1][3])
sns.countplot(x = "TechSupport", data = df,palette="viridis", ax=axes[1][4])
sns.countplot(x = "StreamingTV", data = df,palette="viridis", ax=axes[2][0])
sns.countplot(x = "StreamingMovies", data = df,palette="viridis", ax=axes[2][1])
sns.countplot(x = "Contract", data = df,palette="viridis", ax=axes[2][2])
sns.countplot(x = "PaperlessBilling",palette="viridis", data = df, ax=axes[2][3])
ax = sns.countplot(x = "PaymentMethod", data = df,palette="viridis", ax=axes[2][4])
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.show(fig)


# In[16]:


def show_correlations(df, show_chart = True):
    fig = plt.figure(figsize = (20,10))
    corr = df.corr()
    if show_chart == True:
        sns.heatmap(corr, 
                    xticklabels=corr.columns.values,
                    yticklabels=corr.columns.values,
                    annot=True, cmap= "Greens")
    return corr

correlation_df = show_correlations(df,show_chart=True)


# In[17]:


churn_rate = df.Churn.value_counts() / len(df.Churn)
labels = 'Non-Churn', 'Churn'

fig, ax = plt.subplots()
ax.pie(churn_rate, labels=labels,radius=1.2, colors=['seagreen','darkslateblue'], autopct='%.f%%', explode=[0,0.1])  
ax.set_title('Churn vs Non Churn', fontsize=14)


# In[18]:


# Drop Customer Id for modelling
df1 = df.drop(['customerID'], axis = 1)


# In[19]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


# In[20]:


X = df1.drop('Churn', 1)
y = df1['Churn']


# In[21]:


pip install xgboost


# In[22]:


pip install --upgrade xgboost


# In[23]:


from xgboost import XGBClassifier


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.30, random_state = 100)

classifiers = [['DecisionTree :',DecisionTreeClassifier()],
               ['RandomForest :',RandomForestClassifier()], 
               ['Naive Bayes :', GaussianNB()],
               ['KNeighbours :', KNeighborsClassifier()],
               ['SVM :', SVC()],
               ['LogisticRegression :', LogisticRegression(max_iter=500)],
               ['Neural Network :', MLPClassifier()],
               ['ExtraTreesClassifier :', ExtraTreesClassifier()],
               ['AdaBoostClassifier :', AdaBoostClassifier()],
               ['XGBoost :', XGBClassifier(use_label_encoder=False, disable_default_eval_metric=True)],
               ['GradientBoostingClassifier: ', GradientBoostingClassifier()]]

predictions_df = pd.DataFrame()
predictions_df['actual_labels'] = y_test

for name,classifier in classifiers:
    classifier = classifier
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    predictions_df[name.strip(" :")] = predictions
    print(name, accuracy_score(y_test, predictions))

