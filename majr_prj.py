#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB,BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.decomposition import PCA


# In[74]:


ds1=pd.read_csv(r"C:\Users\DELL\Downloads\molecular_positive-out.csv")
ds1["Inhibitor"]=1


# In[75]:


ds2=pd.read_csv(r"C:\Users\DELL\Downloads\molecular_1-out (1).csv")
ds2["Inhibitor"]=0
ds2


# In[76]:


ds1 = ds1.drop('Title',axis=1)
ds1


# In[77]:


ds2=ds2.drop('Title',axis=1)
ds2


# In[78]:


ds3=pd.concat([ds1,ds2])
ds3


# In[79]:


from sklearn.preprocessing import LabelEncoder as ll

l = ll()

for i in ds3.columns:
    if ds3[i].dtype == 'O':
        ds3[i] = l.fit_transform(ds3[i])


# In[80]:


ds3.describe()


# In[81]:


from joblib import dump, load
from scipy import special
np.random.seed(1234)


# In[82]:


def remove_same_value_features(ds3):
    return [e for e in ds3.columns if ds3[e].nunique() == 1]


# In[109]:


drop_col = remove_same_value_features(ds3)
len(drop_col)


# In[84]:


new_df_columns = [e for e in ds3.columns if e not in drop_col]
new_df = ds3[new_df_columns]
new_df.count()


# In[90]:


new_df.to_csv("dataset_unique_columns.csv", index=False)
df2= pd.read_csv('dataset_unique_columns.csv')
df2


# In[91]:


X = df2.drop(columns=['Inhibitor'])  # Drop the target class column to get features
y= df2['Inhibitor']  # Extract the target class column


# In[92]:


X = X.loc[:, X.isna().sum() > 400]


# In[93]:


X.replace(np.nan,0,inplace=True)


# In[94]:


cor_matrix = X.corr().abs()
cor_matrix


# In[95]:


import numpy as np

upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
upper_tri


# In[96]:


to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
X.drop(X[to_drop], axis=1)


# In[97]:


X = X.drop(X[to_drop], axis=1)
X.to_csv("dataset_unrelated_features.csv", index=False)


# In[98]:


X


# In[99]:


y.value_counts()


# In[100]:


X_train, X_pseudo_test, y_train, y_pseudo_test = train_test_split( X, y, train_size=0.8, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_pseudo_test, y_pseudo_test, test_size = 0.5,                                                stratify=y_pseudo_test, random_state=42) 


# In[101]:


len(X_train),len(X_test),len(X_val),len(y_test),len(y_val)


# In[102]:


print(y_train.value_counts(), y_test.value_counts(), y_val.value_counts())


# In[103]:


X_train['labels'] = 'Training_set'
X_test['labels'] = 'Test_set'
X_val['labels'] = 'Validation_set'


# In[104]:


X = pd.concat([X_train, X_test, X_val],axis=0)


# In[105]:


X['labels'].value_counts()


# In[106]:


X_scaled = StandardScaler().fit_transform(X.iloc[:,:-1])


# In[107]:


pca = PCA (n_components= 3)
principalComponents = pca.fit_transform(X_scaled)


# In[108]:


principalComponents


# In[110]:


principal_df = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2',  'principal component 3'])


# In[111]:


principal_df.head()


# In[112]:


principal_df['labels'] = X['labels']
principal_df['labels'].value_counts()


# In[113]:


print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))


# In[114]:


fig = plt.figure()
plt.rcParams['figure.figsize'] = (35,19)
ax = fig.add_subplot(111, projection = '3d')
plt.xlabel('Principal Component - 1',fontsize=15)
plt.ylabel('Principal Component - 2',fontsize=15)
ax.set_zlabel('Principal Component - 3',fontsize=15)
targets = ['Training_set', 'Test_set', 'Validation_set',]
colors = ['y', 'r', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = X['labels'] == target
    ax.scatter(principal_df.loc[indicesToKeep, 'principal component 1']
               , principal_df.loc[indicesToKeep, 'principal component 2'],
               principal_df.loc[indicesToKeep, 'principal component 3'], c = color, s = 50)

ax.legend(targets,prop={'size': 15})
ax.view_init(30,15)


# In[115]:


clf1 = BernoulliNB()
clf1.fit(X_train.iloc[0:,:-1], y_train.values.ravel())

y_predict1 = clf1.predict(X_test.iloc[0:,:-1])
clf1.score(X_test.iloc[0:,:-1], y_test)


# In[116]:


param = {'var_smoothing' : [0.15,0.015,0.0015,0.00015,0.000015,0.0000015]}
model_rs = GridSearchCV(estimator=GaussianNB(),param_grid=param, scoring = 'accuracy',n_jobs=4,cv=10,refit=True,return_train_score=True)
model_rs.fit(X_train.iloc[0:,:-1], y_train.values.ravel())
y_pred = model_rs.predict(X_test.iloc[0:,:-1])
model_rs.score(X_test.iloc[0:,:-1], y_test)


# In[117]:


clf2 = GaussianNB(var_smoothing= 0.15)
clf2.fit(X_train.iloc[0:,:-1], y_train.values.ravel())

y_predict2 = clf2.predict(X_test.iloc[0:,:-1])
clf2.score(X_test.iloc[0:,:-1], y_test)


# In[118]:


for i in range(1,10):
    lr = LogisticRegression(C=i,multi_class='ovr', max_iter=500)
    lr.fit(X_train.iloc[0:,:-1], y_train.values.ravel())
    pred = lr.predict(X_test.iloc[0:,:-1])
    w = accuracy_score(pred, y_test)
    print(i, w)


# In[119]:


for i in range(1,5):
    lr = LogisticRegression(C=i,multi_class='multinomial', max_iter=500)
    lr.fit(X_train.iloc[0:,:-1], y_train.values.ravel())
    pred = lr.predict(X_test.iloc[0:,:-1])
    w = accuracy_score(pred, y_test)
    print(i, w)


# In[120]:


clf3 = LogisticRegression(C=3, multi_class='ovr', max_iter=500 )
clf3.fit(X_train.iloc[0:,:-1], y_train.values.ravel())
y_predict3= clf3.predict(X_test.iloc[0:,:-1])
clf3.score(X_test.iloc[0:,:-1], y_test)


# # KNN

# In[121]:


for i in range(1,15):
    knn = KNeighborsClassifier(n_neighbors=i, metric = 'euclidean')
    knn.fit(X_train.iloc[0:,:-1], y_train.values.ravel())
    pred = knn.predict(X_test.iloc[0:,:-1])
    w = accuracy_score(pred, y_test)
    print(i, w)


# In[122]:


for i in range(1,11):
    knn = KNeighborsClassifier(n_neighbors=i, metric = 'minkowski')
    knn.fit(X_train.iloc[0:,:-1], y_train.values.ravel())
    pred = knn.predict(X_test.iloc[0:,:-1])
    w = accuracy_score(pred, y_test)
    print(i, w)


# In[123]:


clf4 = KNeighborsClassifier(n_neighbors= 4, metric= 'euclidean')
clf4.fit(X_train.iloc[0:,:-1], y_train.values.ravel())

y_predict4 = clf4.predict(X_test.iloc[0:,:-1])
clf4.score(X_test.iloc[0:,:-1], y_test)


# ## 5) Random Forest Classifier# 

# In[124]:


for i in range(80,90):
    rfc = RandomForestClassifier(n_estimators= i, criterion="entropy", random_state= 42)
    rfc.fit(X_train.iloc[0:,:-1], y_train.values.ravel())
    y_pred = rfc.predict(X_test.iloc[0:,:-1])
    w = rfc.score(X_test.iloc[0:,:-1], y_test)
    print(i, w)


# In[125]:


for i in range(1,10):
    rfc = RandomForestClassifier(n_estimators= i, criterion="gini", random_state= 42)
    rfc.fit(X_train.iloc[0:,:-1], y_train.values.ravel())
    y_pred = rfc.predict(X_test.iloc[0:,:-1])
    w = rfc.score(X_test.iloc[0:,:-1], y_test)
    print(i, w)


# In[126]:


clf5= RandomForestClassifier(n_estimators= 86, criterion="entropy", random_state= 42)  
clf5.fit(X_train.iloc[0:,:-1], y_train.values.ravel())

y_predict5 = clf5.predict(X_test.iloc[0:,:-1])
clf5.score(X_test.iloc[0:,:-1], y_test)


# ## 6) Support Vector Machines# 

# In[127]:


for i in range(1,10):
    svc = SVC(C=i, kernel='poly')
    svc.fit(X_train.iloc[0:,:-1], y_train.values.ravel())
    y_pred = svc.predict(X_test.iloc[0:,:-1])
    w = svc.score(X_test.iloc[0:,:-1], y_test)
    print(i, w)


# In[128]:


for i in range(1,10):
    svc = SVC(C=i, kernel='rbf')
    svc.fit(X_train.iloc[0:,:-1], y_train.values.ravel())
    y_pred = svc.predict(X_test.iloc[0:,:-1])
    w = svc.score(X_test.iloc[0:,:-1], y_test)
    print(i, w)


# In[129]:


clf6 = SVC(C=8, kernel='rbf') 
clf6.fit(X_train.iloc[0:,:-1], y_train.values.ravel())

y_predict6= clf6.predict(X_test.iloc[0:,:-1])
clf6.score(X_test.iloc[0:,:-1], y_test)


# In[130]:


cm = confusion_matrix(y_test, y_predict6)
cm = pd.DataFrame(cm)
cm


# In[131]:


for i in range (cm.shape[0]):
    TP = cm.iloc[i,i]
    FP = cm.iloc[i,:].sum()- TP
    FN = cm.iloc[:,i].sum() - TP
    TN = cm.sum().sum() - TP-FP-FN
    Accuracy = (TP + TN)/cm.sum().sum()
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    f1 = (2*precision*recall)/(precision+recall)
    print (cm.index[i], Accuracy, precision, recall, f1)


# In[132]:


f1_score(y_test, y_predict6, average= 'weighted')


# In[133]:


dump(SVC(C=8.0, kernel='rbf') , 'Model.joblib')


# In[ ]:




