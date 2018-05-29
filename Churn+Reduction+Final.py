
# coding: utf-8

# In[1]:


#importing required libraries
import pandas as pd
import numpy as np
import re
import os
from scipy.stats import chi2_contingency
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

#libraries for visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[2]:


#loading data
train_df = pd.read_csv('C:/Users/BATMAN/Desktop/1st project working copy/Train_data.csv')
test_df = pd.read_csv('C:/Users/BATMAN/Desktop/1st project working copy/Test_data.csv')
os.chdir("C:/Users/BATMAN/Desktop/1st project working copy")
final_submission = test_df


# ## Exlopratory Data Analysis (EDA) 

# In[3]:


#Looking at data
train_df


# In[4]:


#Missing value analysis
print(train_df.isnull().sum())
print('***************')
print(test_df.isnull().sum())


# #### From the above analysis we can see that there are no missing values in the data-set

# In[5]:


# Dropping variables unessential for analysis.
train_df = train_df.drop(['state','area code','phone number'], axis =1)
test_df = test_df.drop(['state','area code','phone number'], axis =1)


# In[6]:


#Replacing yes and no with 1 and 0/ true false with 1 and 0
#train
train_df['international plan'] = train_df['international plan'].replace(' yes','1')
train_df['international plan'] = train_df['international plan'].replace(' no','0')
train_df['voice mail plan'] = train_df['voice mail plan'].replace(' yes','1')
train_df['voice mail plan'] = train_df['voice mail plan'].replace(' no','0')
train_df['Churn'] = train_df['Churn'].replace(' True.','1')
train_df['Churn'] = train_df['Churn'].replace(' False.','0')
#test
test_df['international plan'] = test_df['international plan'].replace(' yes','1')
test_df['international plan'] = test_df['international plan'].replace(' no','0')
test_df['voice mail plan'] = test_df['voice mail plan'].replace(' yes','1')
test_df['voice mail plan'] = test_df['voice mail plan'].replace(' no','0')
test_df['Churn'] = test_df['Churn'].replace(' True.','1')
test_df['Churn'] = test_df['Churn'].replace(' False.','0')
##Converting into category
test_df['international plan'] = test_df['international plan'].astype('category')
test_df['international plan'] = test_df['international plan'].astype('category')
test_df['voice mail plan'] = test_df['voice mail plan'].astype('category')
test_df['voice mail plan'] = test_df['voice mail plan'].astype('category')
test_df['Churn'] = test_df['Churn'].astype('category')
test_df['Churn'] = test_df['Churn'].astype('category')


# #### For international calls no = 0 yes = 1
# #### For voice mail plan no = 0 yes = 1
# #### For churn False = 0 True = 1

# ## Feature Selection

# In[7]:


#Correlation alnalysis using heatmap
df_corr = train_df.iloc[:,4:16]
f, ax = plt.subplots(figsize=(10,10))
plt.title('Correlation between numerical predictors',size=14,y=1.05)
corr = df_corr.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap= sns.diverging_palette(220,10, as_cmap = True), square=True,
            annot = True,ax=ax)


# ###### From the above heat-map we can infer the following:
# ######        - total day minutes & total day charge are highly +vely correlated.
# ######        - total eve minutes & total eve charge are highly +vely correlated.
# ######        - total night minutes & total night charge are highly +vely correlated.
# ######        - total intl minutes & total intl charge are highly +vely correlated.
# 
# ### Therefore we will drop the  total day charge, total eve charge,  total night charge i.e variables carrying redundant information  

# In[8]:


train_df = train_df.drop(['total day minutes','total eve minutes','total night minutes','total intl minutes'], axis =1)
test_df = test_df.drop(['total day minutes','total eve minutes','total night minutes','total intl minutes'], axis =1)


# In[9]:


cat_names = ['international plan','voice mail plan']


# In[10]:


for i in cat_names:
    print(i)
    chi2, p, dof, ex = chi2_contingency(pd.crosstab(train_df['Churn'],train_df[i]))
    print(p)


# #### As p-value of both the categorical variables is < 0.05 therefor we will reject the null hypothesis and will consider both variables for further analysis.

# ### Outlier Analysis

# In[11]:


#Fiding outliers and replacing with NA
cnames = ['total day calls', 'total day charge','total eve calls', 'total eve charge','total night calls', 'total night charge', 'total intl calls', 'total intl charge', 'number customer service calls']
for i in cnames:
    for j in range(len(train_df)):
        Q1 = train_df[i].quantile(0.25)
        Q3 = train_df[i].quantile(0.75)
        IQR = Q3 - Q1
        if (train_df[i].iloc[j] <= (Q1 - 1.5*IQR) or train_df[i].iloc[j] >= (Q3 + 1.5*IQR)):
            train_df[i].iloc[j] = np.nan
for k in cnames:
    for l in range(len(test_df)):
        Q1 = test_df[k].quantile(0.25)
        Q3 = test_df[k].quantile(0.75)
        IQR = Q3 - Q1
        if (test_df[k].iloc[l] <= (Q1 - 1.5*IQR) or test_df[k].iloc[l] >= (Q3 + 1.5*IQR)):
            test_df[k].iloc[l] = np.nan


# In[12]:


#impute Nan with mean
cnames = ['total day calls', 'total day charge','total eve calls', 'total eve charge','total night calls', 'total night charge', 'total intl calls', 'total intl charge', 'number customer service calls']
for i in cnames:
    train_df[i] = train_df[i].fillna(train_df[i].mean())
    test_df[i] = test_df[i].fillna(test_df[i].mean())


# ## Feature Scaling

# In[13]:


cnames = ['account length','number vmail messages','total day calls','total day charge','total eve calls','total eve charge','total night calls','total night charge','total intl calls','total intl charge','number customer service calls']
for i in cnames:
    print(i)
    train_df[i] = (train_df[i] - train_df[i].mean())/train_df[i].std()
    test_df[i] = (test_df[i] - test_df[i].mean())/test_df[i].std() 


# # Model implementation

# In[14]:


X_train = train_df.drop("Churn", axis=1)
Y_train = train_df["Churn"]
X_test  = test_df.drop("Churn", axis=1)
X_train.shape, Y_train.shape, X_test.shape


# In[15]:


# Gaussian Naive Bayes
NB_model = GaussianNB().fit(X_train,Y_train)
Y_pred = NB_model.predict(X_test)
NBtest_predict_prob = NB_model.predict_proba(X_test)
y1 = accuracy_score(test_df['Churn'],Y_pred)
print(y1*100)


# In[16]:


# Logistic Regression
from sklearn.metrics import accuracy_score
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
train_predict = logreg.predict(X_train)     #implement the model on train data for acuuracy
Y_pred = logreg.predict(X_test)      #implement the model on test data for acuurac
LRtest_predict_prob = logreg.predict_proba(X_test)
y2 = accuracy_score(test_df['Churn'],Y_pred)
print(y2*100)


# In[17]:


#KNN
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
y3 = accuracy_score(test_df['Churn'],Y_pred)
print(y3*100)


# In[18]:


models = pd.DataFrame({
    'Model': ['Gaussian Naive Bayes', 'Logistic Regression', 'KNN',],
    'Score': [y1,y2,y3]})
models.sort_values(by='Score', ascending=False)


# ####  - As we have to calculate the churn score; therefore we will need an output in terms of probablity.
# ####  - Gaussian Naive Bayes and Logistic Regression gives probablity as output  
# ####  - As accuracy of Logictic regression is higher as compared to Naive Bayes we will use Logistic Regression.

# In[19]:


final_submission = final_submission.drop(['state', 'account length', 'area code','international plan', 'voice mail plan', 'number vmail messages', 'total day minutes', 'total day calls', 'total day charge', 'total eve minutes', 'total eve calls', 'total eve charge', 'total night minutes', 'total night calls', 'total night charge', 'total intl minutes', 'total intl calls', 'total intl charge', 'number customer service calls','Churn'], axis =1)


# In[20]:


final_submission['Churn Score'] = pd.DataFrame(LRtest_predict_prob).iloc[:,1]


# In[21]:


final_submission.to_csv('Churn_Score_python.csv', index=False)

