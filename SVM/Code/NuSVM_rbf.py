# -*- coding: utf-8 -*-

"""
Created on Fri Jun  8 23:38:49 2018

@author: Fakrul-IslamTUSHAR
"""

# =============================================================================
# Import Libraries
# =============================================================================
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.svm import NuSVC
from sklearn.utils import shuffle
from sklearn.externals import joblib


# =============================================================================
# Load the data and data making
# =============================================================================
filename = 'hw3data.csv' #Provide File name
raw_data = open(filename, 'rt') #load data
data = np.loadtxt(raw_data, delimiter=",") 
print(data.shape)
x1=data[:,:-1]
x1= normalize(x1,norm='l2')

y1=data[:,-1]

#Making the dataset -1 to
for i in range(len(y1)): 
    if (y1[i]==-1):
        y1[i]=0
    else:
        y1[i]=1

#Splitting the data into 5 equal part
x_set, x_set1, y_set, y_set1 = train_test_split(x1, y1, test_size=0.20,random_state=1)
x_main1, x_main2, y_main1, y_main2 = train_test_split(x_set, y_set, test_size=0.50,random_state=1 )
x_set2, x_set3, y_set2, y_set3 = train_test_split(x_main1, y_main1, test_size=0.50,random_state=1 )
x_set4, x_set5, y_set4, y_set5 = train_test_split(x_main1, y_main1, test_size=0.50,random_state=1 )

##Making the Subset of dataset in 5 different part
"Test x_set1,y_set1"
Datset1_train_features=np.concatenate((x_set2,x_set3,x_set4,x_set5), axis=0) #x_set1 as test_data
Datset1_train_labels=np.concatenate((y_set2,y_set3,y_set4,y_set5), axis=0) # y_set1 as test_labels
"Test x_set2,y_set2"
Datset2_train_features=np.concatenate((x_set1,x_set3,x_set4,x_set5), axis=0) #x_set2 as test_data
Datset2_train_labels=np.concatenate((y_set1,y_set3,y_set4,y_set5), axis=0) # y_set2 as test_labels
"Test x_set3,y_set3"
Datset3_train_features=np.concatenate((x_set1,x_set2,x_set4,x_set5), axis=0) #x_set3 as test_data
Datset3_train_labels=np.concatenate((y_set1,y_set2,y_set4,y_set5), axis=0) # y_set3 as test_labels
"Test x_set4,y_set4"
Datset4_train_features=np.concatenate((x_set1,x_set2,x_set3,x_set5), axis=0) #x_set4 as test_data
Datset4_train_labels=np.concatenate((y_set1,y_set2,y_set3,y_set5), axis=0) # y_set4 as test_labels
"Test x_set5,y_set5"
Datset5_train_features=np.concatenate((x_set1,x_set2,x_set3,x_set4), axis=0) #x_set4 as test_data
Datset5_train_labels=np.concatenate((y_set1,y_set2,y_set3,y_set4), axis=0) # y_set4 as test_labels


"Define the dataset You want to use"
# =============================================================================
# Defining the dataset to use
# =============================================================================
train_data=Datset1_train_features
train_labels=Datset1_train_labels
test_data=x_set1 #Test Set
test_lables=y_set1 #Test set labels
# =============================================================================
# Model
# =============================================================================
#Suffle The data
X,Y=shuffle(train_data,train_labels,random_state=2)
#train test split
x_train,x_val,y_train,y_val=train_test_split(X,Y,test_size=0.50,random_state=2)

#Building the model
clf = NuSVC(nu=0.5, kernel='rbf',probability=True)#probability=True
clf.fit(x_train, y_train)


## Visualizing the AUC
predicted_probas = clf.predict_proba(test_data)
skplt.metrics.plot_roc(test_lables, predicted_probas)
plt.show()
