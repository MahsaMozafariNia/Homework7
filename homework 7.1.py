# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 17:11:59 2020

@author: homework 7.1

"""
 
import pandas as pd
import numpy as np 
import sklearn.preprocessing as pre
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score 
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


df = pd.read_csv('D:/Old_Data/math/Data science toseeh/Files/blocks.csv')
df.columns
head=df.head()
df.shape
df["block"].unique()
df.dtypes
y=df["block"]
X=df.iloc[:,1:]


#chon y keifi as bayad kami shavad ama dar motoghayre pasokh ham mitavan lable encoder(tartibi)
#estefade kard va ham get_dummies(gheire tartibi)

le=pre.LabelEncoder()
le.fit(y)
y=le.fit_transform(y)
np.unique(y)

#train_test_esplit
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1)
#LDA
lda=LinearDiscriminantAnalysis()
lda.fit(X_train,y_train)
lda.predict(X_test)

#QDA
qda=QuadraticDiscriminantAnalysis()
qda.fit(X_train,y_train)
qda.predict(X_test)


#Naive Bayes
nb=GaussianNB()
nb.fit(X_train,y_train)
nb.predict(X_test)

#logistic regression
lr=LogisticRegression(penalty='l2',solver='newton-cg')
lr.fit(X_train,y_train)
lr.predict(X_test)

#KNN
knn=KNeighborsClassifier(n_neighbors=1,metric='minkowski',p=2) 
knn.fit(X_train,y_train)
knn.predict(X_test)
knn.score(X_train,y_train)
#yaftane parametr n_nieghbors
#ebteda bareye k kochak va bozorg mizan deghat ra bebinim va bad hoddod tayin konim.

l=[]
for k in range(1,101):
    
    knn=KNeighborsClassifier(n_neighbors=k,metric='minkowski',p=2) 
    knn.fit(X_train,y_train)
    knn.predict(X_test)
    l.append(knn.score(X_train,y_train))
    
#bishtarin meghdare score baraye k=1 ast va nozooli ast pas faghat kafi ast masalan baraye k=1 va 2 cross ra esab konim.

knnmatrix=np.empty((26,2))
counter=-1
for k in range(1,27):
    counter+=1
    knn=KNeighborsClassifier(n_neighbors=k,metric='minkowski',p=2) 
    knnmatrix[counter,:]=np.array([k,np.mean(cross_val_score(knn,X_train,y_train,cv=10))])
    
print(np.argmax(knnmatrix[:,1]))
#mitavan range(1,100) masalan dar nazar gereft. dar kol behtarin k meghdare 1 ast.
#pas ba k=1 kar mikonim va ejra mikonim.
    
knn=KNeighborsClassifier(n_neighbors=1,metric='minkowski',p=2) 
knn.fit(X_train,y_train)
knn.predict(X_test)


#c
lda_list=[]; qda_list=[]; nb_list=[]; lr_list=[]; knn_list=[]
for i in range(1000):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1)
    lda=LinearDiscriminantAnalysis()
    qda=QuadraticDiscriminantAnalysis()
    nb=GaussianNB()
    lr=LogisticRegression(penalty='l2',solver='newton-cg')
    knn=KNeighborsClassifier(n_neighbors=1,metric='minkowski',p=2)
    lda.fit(X_train,y_train)
    qda.fit(X_train,y_train)
    nb.fit(X_train,y_train)
    lr.fit(X_train,y_train)
    knn.fit(X_train,y_train)#tavajoh k=1 farze shode.
    lda_list.append(lda.score(X_test,y_test))
    qda_list.append(qda.score(X_test,y_test))
    nb_list.append(nb.score(X_test,y_test))
    lr_list.append(lr.score(X_test,y_test))
    knn_list.append(knn.score(X_test,y_test))
    
print("\n Mean\n","\n lda \n ",np.mean(lda_list),"\n qda \n ",np.mean(qda_list),"\n nb \n ",np.mean(nb_list),
      "\n lr \n ",np.mean(lr_list),"\n knn \n ",np.mean(knn_list))
#lda ba 0.4837 darsa test accuracy az hame behtar amal karde.
#اما برای بررسی اماری باید بازه اطمینان تعیین کرد. توجه سایز نمونه 1000 است.
#چون 1000 مقدار هر بار داریم و میانگین و انحراف معیار ان را حساب میکنیم.
print("\n Sdr \n","\n lda \n ",np.std(lda_list),"\n qda \n ",np.std(qda_list),"\n nb \n ",np.std(nb_list),
      "\n lr \n ",np.std(lr_list),"\n knn \n ",np.std(knn_list))

print("\n baze etminan lda \n", np.mean(lda_list)-1.96*(np.std(lda_list)/np.sqrt(1000)),np.mean(lda_list)+1.96*(np.std(lda_list)/np.sqrt(1000)))
print("\n baze etminan qda \n", np.mean(qda_list)-1.96*(np.std(qda_list)/np.sqrt(1000)),np.mean(qda_list)+1.96*(np.std(qda_list)/np.sqrt(1000)))
print("\n baze etminan nb \n", np.mean(nb_list)-1.96*(np.std(nb_list)/np.sqrt(1000)),np.mean(nb_list)+1.96*(np.std(nb_list)/np.sqrt(1000)))
print("\n baze etminan lr \n", np.mean(lr_list)-1.96*(np.std(lr_list)/np.sqrt(1000)),np.mean(lr_list)+1.96*(np.std(lr_list)/np.sqrt(1000)))
print("\n baze etminan knn \n", np.mean(knn_list)-1.96*(np.std(knn_list)/np.sqrt(1000)),np.mean(knn_list)+1.96*(np.std(knn_list)/np.sqrt(1000)))

#d
lda=LinearDiscriminantAnalysis()
qda=QuadraticDiscriminantAnalysis()
nb=GaussianNB()
lr=LogisticRegression(penalty='l2',solver='newton-cg')
knn=KNeighborsClassifier(n_neighbors=1,metric='minkowski',p=2)
lda.fit(X_train,y_train)
qda.fit(X_train,y_train)
nb.fit(X_train,y_train)
lr.fit(X_train,y_train)
knn.fit(X_train,y_train)
lda_confusion=confusion_matrix(y_test,lda.predict(X_test))
qda_confusion=confusion_matrix(y_test,qda.predict(X_test))
nb_confusion=confusion_matrix(y_test,nb.predict(X_test))
lr_confusion=confusion_matrix(y_test,lr.predict(X_test))
knn_confusion=confusion_matrix(y_test,knn.predict(X_test))
print("\n lda confusion matrix:\n",lda_confusion)
print("\n qda confusion matrix:\n",qda_confusion)
print("\n nb confusion matrix:\n",nb_confusion)
print("\n lr confusion matrix:\n",lr_confusion)
print("\n knn confusion matrix:\n",knn_confusion)


counter=-1
matrix=np.zeros((5,5))
for method in [lda_confusion,qda_confusion,nb_confusion,lr_confusion,knn_confusion]:
    counter+=1
    matrix[counter,:]=np.array([method[0,0]/np.sum(method[0,:]),method[1,1]/np.sum(method[1,:]),method[2,2]/np.sum(method[2,:]),method[3,3]/np.sum(method[3,:]),method[4,4]/np.sum(method[4,:])])


print(matrix)   

#e
#behtarin racesh dar ghesmate c raveshe lda ast hala pca emal karde va dobare in ravesh ra fit mikonim


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1)
pca=PCA(whiten=True)
pca.fit(X_train)
plt.bar(range(1,11),pca.explained_variance_,color="blue",edgecolor="Red")#scree plot


lr_pca=LogisticRegression(penalty='l2',solver='newton-cg')
pca_parameter_matrix=np.zeros((10,2))
counter=-1
for i in range(1,11):
    counter+=1
    print(counter)
    X_train_pca=pca.transform(X_train)[:,:i]
    lr_pca=LogisticRegression(penalty='l2',solver='newton-cg')
    lr_pca.fit(X_train_pca,y_train)
    pca_parameter_matrix[counter,:]=np.array([i,np.mean(cross_val_score(lr_pca,X_train_pca,y_train,cv=10))])

print(pca_parameter_matrix)    

#behtarin meghdar baraye tedade pc score ha ya haman vizhegi haye jadid barabar ba 1 ast.
#yani dar fazaye jadid faghat vizhegi aval ra dar nazar begir.



lr_list_no_pca=[]
lr_list_pca=[]

for i in range(1000):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1)
    pca=PCA(whiten=True)#فرقی نداره باشه یا نه. فقط یه جا باید فراخوانی شده باشه. 
    #هر  بار pca فیت میشه در حقیقت خودش عملیات اسکیل را انجام میده اما داده تغییر نمیکنه و بعد جهت ها را پیدا میکنه
    #با ترنسفورم کردن pcscore ها محاسبه میشه.
    lr=LogisticRegression(penalty='l2',solver='newton-cg')
    lr_pca=LogisticRegression(penalty='l2',solver='newton-cg')
    pca.fit(X_train)
    X_train_pca=pca.transform(X_train)
    X_test_pca=pca.transform(X_test)
    lr.fit(X_train,y_train)
    lr_pca.fit(X_train_pca,y_train)
    lr_list_no_pca.append(lr.score(X_test,y_test))
    lr_list_pca.append(lr_pca.score(X_test_pca,y_test))
   
    
print("\n lr \n",np.mean(lr_list_no_pca),"\n lr_pca \n",np.mean(lr_list_pca))

#lda_pca is better

