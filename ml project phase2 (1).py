#!/usr/bin/env python
# coding: utf-8

# In[1]:


# # **importing data**

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder 
   


# In[2]:


# # **importing and splitting**




dataset=pd.read_csv('Train_Data.csv')
Train_Data=pd.read_csv('Train_Data.csv')
Traindata_classlabels=pd.read_csv('Traindata_classlabels.csv')
Test_Data=pd.read_csv('testdata.csv')


# In[3]:


# Create an instance of One-hot-encoder 
enc = OneHotEncoder() 
  
# Passing encoded columns 
en=enc.fit_transform( 
    Train_Data[['blue','dual_sim','fc','four_g','n_cores','pc','three_g','touch_screen','wifi']]).toarray()
column_names = enc.get_feature_names_out(['blue','dual_sim','fc','four_g','n_cores','pc','three_g','touch_screen','wifi']) 
enc_data = pd.DataFrame(en,columns=column_names) 

# Merge with main 
Train_Data = Train_Data.join(enc_data) 
  
print(Train_Data)


# In[4]:


# Create an instance of One-hot-encoder 
enc = OneHotEncoder() 
  
# Passing encoded columns 
en1=enc.fit_transform( 
    Test_Data[['blue','dual_sim','fc','four_g','n_cores','pc','three_g','touch_screen','wifi']]).toarray()
column_names = enc.get_feature_names_out(['blue','dual_sim','fc','four_g','n_cores','pc','three_g','touch_screen','wifi']) 
enc_data = pd.DataFrame(en1,columns=column_names) 

# Merge with main 
Test_Data = Test_Data.join(enc_data) 
  
print(Test_Data)


# In[5]:


Train_Data_train, Train_Data_test, Traindata_classlabels_train, Traindata_classlabels_test = train_test_split(Train_Data, Traindata_classlabels, test_size=0.4, random_state=53)


# In[6]:


# # **Data set visualization**



dataset.head()


# In[7]:


dataset.isnull().sum()


# In[8]:


dataset.info()
dataset.describe()


# In[9]:


plt.figure(figsize=(20,20))
sns.heatmap(dataset.corr(),annot=True)
plt.show()


# In[10]:


import numpy as np

# Convert lists to NumPy arrays
Train_Data_train = np.array(Train_Data_train)
Traindata_classlabels_train = np.array(Traindata_classlabels_train)
Train_Data_test = np.array(Train_Data_test)
Traindata_classlabels_test = np.array(Traindata_classlabels_test)


# In[11]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

# Assuming Train_Data_train, Traindata_classlabels_train, Train_Data_test, Traindata_classlabels_test are defined somewhere

arr = []

for k in range(3, 100, 1):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(Train_Data_train, Traindata_classlabels_train)
    pred = knn.predict(Train_Data_test)
    f1 = f1_score(Traindata_classlabels_test, pred, average='macro')
    arr.append(f1)

    


# In[12]:


x=[]
for i in range(3,100,1):
    x.append(i)
plt.plot(x,arr)


# In[13]:


plt.plot(x[10:20],arr[10:20])


# In[14]:


knn = KNeighborsClassifier(n_neighbors=16)
knn.fit(Train_Data_train,Traindata_classlabels_train)
pred = knn.predict(Train_Data_test)
pred_acc = accuracy_score(Traindata_classlabels_test,pred)
pred_f = f1_score(Traindata_classlabels_test,pred,average='macro')
print("prediction accuracy = "+str(pred_acc))
print("prediction f measure = "+str(pred_f))
print(confusion_matrix(pred,Traindata_classlabels_test))


# In[15]:


# # **Decision Tree**

clf = DecisionTreeClassifier(random_state=40) 
clf_parameters = {
            'criterion':('gini', 'entropy'), 
            'max_features':('auto', 'sqrt', 'log2',None),
            'max_depth':(15,30,45,60),
            'ccp_alpha':(0.009,0.005,0.05)
            } 
grid_search = GridSearchCV(estimator=clf,param_grid=clf_parameters,scoring='f1_macro',cv=5)
grid_search.fit(Train_Data_train,Traindata_classlabels_train)
print(grid_search.best_estimator_)
print("Decision tree score = ")
grid_search.best_estimator_.score(Train_Data_test,Traindata_classlabels_test)


# In[16]:


pred = grid_search.best_estimator_.predict(Train_Data_test)
pred_acc = accuracy_score(Traindata_classlabels_test,pred)
pred_f = f1_score(Traindata_classlabels_test,pred,average='macro')
print(grid_search.best_estimator_)
print("prediction accuracy = "+str(pred_acc))
print("prediction f measure = "+str(pred_f))
print(confusion_matrix(pred,Traindata_classlabels_test))


# In[17]:


a=0
arr=[]
x=[]
while(a<0.01):
    clf = DecisionTreeClassifier(ccp_alpha=a, criterion='entropy', max_depth=15,random_state=40)
    clf.fit(Train_Data_train,Traindata_classlabels_train)
    pred = clf.predict(Train_Data_test)
    arr.append(f1_score(Traindata_classlabels_test,pred,average='macro'))
    x.append(a)
    a=a+0.0001


# In[18]:


plt.plot(x,arr)


# In[19]:


plt.plot(x[40:60],arr[40:60])


# In[20]:


clf = DecisionTreeClassifier(ccp_alpha=0.00425, criterion='entropy', max_depth=15,random_state=40)
clf.fit(Train_Data_train,Traindata_classlabels_train)
pred = clf.predict(Train_Data_test)
pred_acc = accuracy_score(Traindata_classlabels_test,pred)
pred_f = f1_score(Traindata_classlabels_test,pred,average='macro')
print("prediction accuracy = "+str(pred_acc))
print("prediction f measure = "+str(pred_f))
print(confusion_matrix(pred,Traindata_classlabels_test))



# In[21]:


# # **Random forest classifier**


clf = RandomForestClassifier(n_estimators=200)
clf_parameters = {
            'criterion':('entropy','gini'),       
            'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 300, num = 100)],
            'max_depth':(10,20,30,50,100,200)
            } 
grid_search = GridSearchCV(estimator=clf,param_grid=clf_parameters,scoring='f1_macro',cv=5)
grid_search.fit(Train_Data_train,Traindata_classlabels_train)
print(grid_search.best_estimator_)
print("random forest score = ")
grid_search.best_estimator_.score(Train_Data_test,Traindata_classlabels_test)


# In[22]:


pred = grid_search.best_estimator_.predict(Train_Data_test)
pred_acc = accuracy_score(Traindata_classlabels_test,pred)
pred_f = f1_score(Traindata_classlabels_test,pred,average='macro')
print(grid_search.best_estimator_)
print("prediction accuracy = "+str(pred_acc))
print("prediction f measure = "+str(pred_f))
print(confusion_matrix(pred,Traindata_classlabels_test))


# In[23]:


# # **Gaussian Naive bayes**


clf = GaussianNB()
clf_parameters = {
            'var_smoothing':np.logspace(0,-13,num=100)
            }
grid_search = GridSearchCV(estimator=clf,param_grid=clf_parameters,scoring='f1_macro',cv=5)
grid_search.fit(Train_Data_train,Traindata_classlabels_train)
print(grid_search.best_estimator_)
print("gaussian score = ")
grid_search.best_estimator_.score(Train_Data_test,Traindata_classlabels_test)


# In[ ]:





# In[24]:


from sklearn.metrics import precision_score
pred = grid_search.best_estimator_.predict(Train_Data_test)
pred_acc = accuracy_score(Traindata_classlabels_test,pred)
pred_f = f1_score(Traindata_classlabels_test,pred,average='macro')
print(grid_search.best_estimator_)
print("prediction accuracy = "+str(pred_acc))
print("prediction f measure = "+str(pred_f))
pred_prec = precision_score(Traindata_classlabels_test,pred,average='macro')
print("prediction precision = "+str(pred_prec))
print(confusion_matrix(pred,Traindata_classlabels_test))


# In[25]:


# # **Support vector machine**

clf = svm.SVC(class_weight='balanced',probability=True)
clf_parameters = {
            'C':[0.01,0.1,1,10,100],
            'gamma': [1,0.1,0.01,0.001],
            'kernel':('linear','rbf','polynomial','sigmoid')
            }
grid_search = GridSearchCV(estimator=clf,param_grid=clf_parameters,scoring='f1_macro',cv=5)
grid_search.fit(Train_Data_train,Traindata_classlabels_train)
print(grid_search.best_estimator_)
print("svm score = ")
grid_search.best_estimator_.score(Train_Data_test,Traindata_classlabels_test)



# In[26]:


pred = grid_search.best_estimator_.predict(Train_Data_test)
pred_acc = accuracy_score(Traindata_classlabels_test,pred)
pred_f = f1_score(Traindata_classlabels_test,pred,average='macro')
print(grid_search.best_estimator_)
print("prediction accuracy = "+str(pred_acc))
print("prediction f measure = "+str(pred_f))
print(confusion_matrix(pred,Traindata_classlabels_test))



# In[27]:


a=0.0005
arr=[]
x=[]
while(a<0.02):
    clf = svm.SVC(C=a, class_weight='balanced', kernel='linear', probability=True)
    clf.fit(Train_Data_train,Traindata_classlabels_train)
    pred = clf.predict(Train_Data_test)
    arr.append(f1_score(Traindata_classlabels_test,pred,average='macro'))
    x.append(a)
    a=a+0.0005


# In[28]:


plt.plot(x,arr)


# In[29]:


plt.plot(x[0:100],arr[0:100])


# In[30]:


np.max(arr)


# In[31]:


print(arr[10],x[10])



# In[32]:


clf = svm.SVC(C=0.0055, class_weight='balanced', kernel='linear', probability=True)
clf.fit(Train_Data_train,Traindata_classlabels_train)
pred = clf.predict(Train_Data_test)
pred_acc = accuracy_score(Traindata_classlabels_test,pred)
pred_f = f1_score(Traindata_classlabels_test,pred,average='macro')
precision = precision_score(Traindata_classlabels_test,pred,average='macro')
print("prediction precision = "+str(precision))
print("prediction accuracy = "+str(pred_acc))
print("prediction f measure = "+str(pred_f))
print(confusion_matrix(pred,Traindata_classlabels_test))


# In[33]:


# # **Logistic regression**


clf = LogisticRegression(multi_class="multinomial" )
clf_parameters = {
     "C":np.logspace(-6,6,num=50,base=2),
     "penalty":["l1","l2",'elasticnet'],
     'solver':['newton-cg','lbfgs','liblinear']}
grid_search = GridSearchCV(estimator=clf,param_grid=clf_parameters,scoring='f1_macro',cv=5)
grid_search.fit(Train_Data_train,Traindata_classlabels_train)
print(grid_search.best_estimator_)
print("logistic regression score = ")
grid_search.best_estimator_.score(Train_Data_test,Traindata_classlabels_test)


# In[34]:


# l1 lasso l2 ridge



print(grid_search.best_estimator_)


# In[35]:


clf = LogisticRegression(C=0.015625, multi_class='multinomial', solver='newton-cg')
clf.fit(Train_Data_train,Traindata_classlabels_train)
pred = clf.predict(Train_Data_test)
pred_acc = accuracy_score(Traindata_classlabels_test,pred)
pred_f = f1_score(Traindata_classlabels_test,pred,average='macro')
precision = precision_score(Traindata_classlabels_test,pred,average='macro')
print("prediction precision = "+str(precision))
print("prediction accuracy = "+str(pred_acc))
print("prediction f measure = "+str(pred_f))
print(confusion_matrix(pred,Traindata_classlabels_test))


# In[36]:


a=0.0005
arr=[]
x=[]
while(a<0.02):
    clf = LogisticRegression(C=a, multi_class='multinomial', solver='newton-cg')
    clf.fit(Train_Data_train,Traindata_classlabels_train)
    pred = clf.predict(Train_Data_test)
    arr.append(f1_score(Traindata_classlabels_test,pred,average='macro'))
    x.append(a)
    a=a+0.0005


# In[37]:


plt.plot(x,arr)


# In[38]:


np.max(arr)


# In[39]:


print(arr[9],x[9])


# In[40]:


clf = LogisticRegression(C=0.005, multi_class='multinomial', solver='newton-cg')
clf.fit(Train_Data_train,Traindata_classlabels_train)
pred = clf.predict(Train_Data_test)
pred_acc = accuracy_score(Traindata_classlabels_test,pred)
pred_f = f1_score(Traindata_classlabels_test,pred,average='macro')
precision = precision_score(Traindata_classlabels_test,pred,average='macro')
print("prediction precision = "+str(precision))
print("prediction accuracy = "+str(pred_acc))
print("prediction f measure = "+str(pred_f))
print(confusion_matrix(pred,Traindata_classlabels_test))


# In[41]:


# # **Prediction of data**
# 
# We have found that the support vector machine has the highest f value as compared to other models used in this project. So I am using this model to predict the target values of the test data




clf=svm.SVC(C=0.0055, class_weight='balanced', kernel='linear', probability=True)
clf.fit(Train_Data_train,Traindata_classlabels_train)
predict = clf.predict(Test_Data)
predict


# In[ ]:




