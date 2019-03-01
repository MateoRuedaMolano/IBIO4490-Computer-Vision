#!/usr/bin/python
"""
Created on Sun Feb 24 19:53:05 2019
@author: Sergio y Mateo
"""
#download the database
import os
import urllib.request
import tarfile

#Download the dataset CIFAR-10
cwd = os.getcwd()
if os.path.exists(cwd +'/'+'cifar-10-python.tar.gz') == False:
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    print('Downloading the database...')
    u = urllib.request.urlopen(url)
    data = u.read()
    u.close()
    with open(cwd+'/'+'cifar-10-python.tar.gz','wb') as f :
        f.write(data)
    print('Database downloaded.')
    f.close()
#Extract files
tar = tarfile.open("cifar-10-python.tar.gz")
tar.extractall()
tar.close()
#Load CIFAR-10:
def unpickle(file):
    import pickle
    import numpy as np
    with open(file, 'rb') as fo:
        _dict = pickle.load(fo, encoding='latin1')
        _dict['labels'] = np.array(_dict['labels'])
        _dict['data'] = _dict['data'].reshape(_dict['data'].shape[0], 3, 32, 32).transpose(0,2,3,1)

    return _dict

def get_data(data, sliced=1):
    from skimage import color
    import numpy as np
    data_x = data['data']
    data_x = color.rgb2gray(data_x)
    data_x = data_x[:int(data_x.shape[0]*sliced)]
    data_y = data['labels']
    data_y = data_y[:int(data_y.shape[0]*sliced)]
    return data_x, data_y

def merge_dict(dict1, dict2):
    import numpy as np
    if len(dict1.keys())==0: return dict2
    new_dict = {key: (value1, value2) for key, value1, value2 in zip(dict1.keys(), dict1.values(), dict2.values())}
    for key, value in new_dict.items():
        if key=='data':
            new_dict[key] = np.vstack((value[0], value[1]))
        if key=='labels':
            new_dict[key] = np.hstack((value[0], value[1]))
        elif key=='batch_label':
            new_dict[key] = value[1]
        else:
            new_dict[key] = value[0] + value[1]
    return new_dict


def load_cifar10_1(meta='cifar-10-batches-py', mode=1):
    assert mode in [1, 2, 3, 4, 5, 'test']
    _dict = {}
    import os
    if isinstance(mode, int):
        for i in range(mode):
            file_ = os.path.join(meta, 'data_batch_'+str(mode))
            _dict = merge_dict(_dict, unpickle(file_))
    else:
        file_ = os.path.join(meta, 'test_batch')
        _dict = unpickle(file_)
    return _dict
def load_cifar10_2(meta='cifar-10-batches-py', mode=2):
    assert mode in [1, 2, 3, 4, 5, 'test']
    _dict = {}
    import os
    if isinstance(mode, int):
        for i in range(mode):
            file_ = os.path.join(meta, 'data_batch_'+str(mode))
            _dict = merge_dict(_dict, unpickle(file_))
    else:
        file_ = os.path.join(meta, 'test_batch')
        _dict = unpickle(file_)
    return _dict
def load_cifar10_3(meta='cifar-10-batches-py', mode=3):
    assert mode in [1, 2, 3, 4, 5, 'test']
    _dict = {}
    import os
    if isinstance(mode, int):
        for i in range(mode):
            file_ = os.path.join(meta, 'data_batch_'+str(mode))
            _dict = merge_dict(_dict, unpickle(file_))
    else:
        file_ = os.path.join(meta, 'test_batch')
        _dict = unpickle(file_)
    return _dict
def load_cifar10_4(meta='cifar-10-batches-py', mode=4):
    assert mode in [1, 2, 3, 4, 5, 'test']
    _dict = {}
    import os
    if isinstance(mode, int):
        for i in range(mode):
            file_ = os.path.join(meta, 'data_batch_'+str(mode))
            _dict = merge_dict(_dict, unpickle(file_))
    else:
        file_ = os.path.join(meta, 'test_batch')
        _dict = unpickle(file_)
    return _dict
def load_cifar10_5(meta='cifar-10-batches-py', mode=5):
    assert mode in [1, 2, 3, 4, 5, 'test']
    _dict = {}
    import os
    if isinstance(mode, int):
        for i in range(mode):
            file_ = os.path.join(meta, 'data_batch_'+str(mode))
            _dict = merge_dict(_dict, unpickle(file_))
    else:
        file_ = os.path.join(meta, 'test_batch')
        _dict = unpickle(file_)
    return _dict
def load_cifar10_test(meta='cifar-10-batches-py', mode='test'):
    assert mode in [1, 2, 3, 4, 5, 'test']
    _dict = {}
    import os
    if isinstance(mode, int):
        for i in range(mode):
            file_ = os.path.join(meta, 'data_batch_'+str(mode))
            _dict = merge_dict(_dict, unpickle(file_))
    else:
        file_ = os.path.join(meta, 'test_batch')
        _dict = unpickle(file_)
    return _dict

import numpy as np
import matplotlib.pyplot as plt
data_1,labels_1 = get_data(load_cifar10_1())
data_2,labels_2 = get_data(load_cifar10_2())

BigData = np.concatenate((data_1,data_2))
BigLabels = np.concatenate((labels_1,labels_2))
#Balancing the classes,a represents the number of images per class to train
a=100
clase_0 = np.zeros((a,32,32))
clase_1 = np.zeros((a,32,32))
clase_2 = np.zeros((a,32,32))
clase_3 = np.zeros((a,32,32))
clase_4 = np.zeros((a,32,32))
clase_5 = np.zeros((a,32,32))
clase_6 = np.zeros((a,32,32))
clase_7 = np.zeros((a,32,32))
clase_8 = np.zeros((a,32,32))
clase_9 = np.zeros((a,32,32))

cont=0;
i = 0;
while(cont<a and i<20000):
   if BigLabels[i]==0:
    clase_0[cont][:][:]=BigData[i]
    cont=cont+1
   i = i+1
cont=0;
i = 0;
while(cont<a and i<20000):
   if BigLabels[i]==1:
       clase_1[cont][:][:]=BigData[i]
       cont=cont+1
   i = i+1
cont=0;
i = 0;
while(cont<a and i<20000):
   if BigLabels[i]==2:
       clase_2[cont][:][:]=BigData[i]
       cont=cont+1
   i = i+1
cont=0;
i = 0;
while(cont<a and i<20000):
   if BigLabels[i]==3:
       clase_3[cont][:][:]=BigData[i]
       cont=cont+1
   i = i+1
cont=0;
i = 0;
while(cont<a and i<20000):
   if BigLabels[i]==4:
       clase_4[cont][:][:]=BigData[i]
       cont=cont+1
   i = i+1
cont=0;
i = 0;
while(cont<a and i<20000):
   if BigLabels[i]==5:
       clase_5[cont][:][:]=BigData[i]
       cont=cont+1
   i = i+1
cont=0;
i = 0;
while(cont<a and i<20000):
   if BigLabels[i]==6:
       clase_6[cont][:][:]=BigData[i]
       cont=cont+1
   i = i+1
cont=0;
i = 0;
while(cont<a and i<20000):
   if BigLabels[i]==7:
       clase_7[cont][:][:]=BigData[i]
       cont=cont+1
   i = i+1
cont=0;
i = 0;
while(cont<a and i<20000):
   if BigLabels[i]==8:
       clase_8[cont][:][:]=BigData[i]
       cont=cont+1
   i = i+1
cont=0;
i = 0;
while(cont<a and i<20000):
   if BigLabels[i]==9:
       clase_9[cont][:][:]=BigData[i]
       cont=cont+1
   i = i+1

DataBalanced = np.concatenate((clase_0,clase_1,clase_2,clase_3,clase_4,clase_5,
                           clase_6,clase_7,clase_8,clase_9))

LabelsBalanced = np.concatenate((np.zeros((a)),np.zeros((a))+1,np.zeros((a))+2,
                             np.zeros((a))+3,np.zeros((a))+4,np.zeros((a))+5,
                             np.zeros((a))+6,np.zeros((a))+7,np.zeros((a))+8,
                             np.zeros((a))+9))
import sys
k = 16*10

sys.path.append('python')

#Create a filter bank with default params
from fbCreate import fbCreate
fb = fbCreate(support=2, startSigma=0.6) # fbCreate(**kwargs, vis=True) for visualization

from fbRun import fbRun
acum = DataBalanced[0,:,:]
for i in range(0,len(DataBalanced)-1):
    acum = np.hstack((acum,DataBalanced[i+1,:,:]))

filterResponses = fbRun(fb,acum)

from computeTextons import computeTextons
map, textons = computeTextons(filterResponses, k)

from assignTextons import assignTextons
import numpy
tmap = numpy.zeros((len(DataBalanced),32,32), dtype=float)
for i in range(0,len(DataBalanced)):
    tmap[i,:,:] = assignTextons(fbRun(fb,DataBalanced[i,:,:]),textons.transpose())

def histc(X, bins):
    import numpy as np
    map_to_bins = np.digitize(X,bins)
    r = np.zeros(bins.shape)
    for i in map_to_bins:
        r[i-1] += 1
    return np.array(r)

histogramas = numpy.zeros((len(DataBalanced),k), dtype=float)
for i in range(0,len(tmap)):
    histogramas[i] = histc(tmap[i].flatten(), np.arange(k))/tmap[i].size

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest

#Apllying Chi-Square
histogramas = SelectKBest(chi2, k=2).fit_transform(histogramas, LabelsBalanced)
#Fitting the models
modelo_KNN = KNeighborsClassifier(n_neighbors=50,weights = 'distance',p=1)
modelo_KNN.fit(histogramas, LabelsBalanced)
prediction_KNN = modelo_KNN.predict(histogramas)

modelo_TREE = RandomForestClassifier(n_estimators=100, max_depth=500, random_state=0)
modelo_TREE.fit(histogramas,LabelsBalanced)
prediction_TREE = modelo_TREE.predict(histogramas)

from sklearn.metrics import accuracy_score
ACA_KNN = accuracy_score(prediction_KNN,LabelsBalanced)
ACA_TREE = accuracy_score(prediction_TREE, LabelsBalanced)

class_names = np.arange(0,10)

import itertools
from sklearn.metrics import confusion_matrix

confusionmat_KNN = confusion_matrix(LabelsBalanced,prediction_KNN)
confusionmat_TREE = confusion_matrix(LabelsBalanced,prediction_TREE)

print('El ACA de KNN es de: '+ str(ACA_KNN))
print('El ACA de RandomForest es de: '+ str(ACA_TREE))
plt.close()
#Saving the models
from sklearn.externals import joblib
filename = 'model_KNN.pk1'
joblib.dump(modelo_KNN, filename)

from sklearn.externals import joblib
filename = 'model_RandomForest.pk1'
joblib.dump(modelo_TREE, filename)

import test_final
test_final

