#!/usr/bin/python
"""
Created on Sun Feb 24 19:53:05 2019
@author: Sergio y Mateo
"""

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

data_test,labels_test=get_data(load_cifar10_test())

import random as ar
#Choose 4 random images
rN1 = ar.randint(0,100)
rN2 = ar.randint(0,100)
rN3 = ar.randint(0,100)
rN4 = ar.randint(0,100)

demoImgs = np.zeros((4,32,32))
demoImgs[0] = data_test[rN1]
demoImgs[1] = data_test[rN2]
demoImgs[2] = data_test[rN3]
demoImgs[3] = data_test[rN4]

data_test = demoImgs

labels = demoImgs = np.zeros((4))
labels[0] = labels_test[rN1]
labels[1] = labels_test[rN2]
labels[2] = labels_test[rN3]
labels[3] = labels_test[rN4]
 
labels_test = labels

import sys
sys.path.append('python')

#Create a filter bank with default params
from fbCreate import fbCreate
fb = fbCreate(support=2, startSigma=0.6) # fbCreate(**kwargs, vis=True) for visualization

from fbRun import fbRun
acum = data_test[0,:,:]
for i in range(0,len(data_test)-1):
    acum = np.hstack((acum,data_test[i+1,:,:]))

filterResponses = fbRun(fb,acum)

k = 16*10

from computeTextons import computeTextons
map, textons = computeTextons(filterResponses, k)

from assignTextons import assignTextons
import numpy
tmap = numpy.zeros((len(data_test),32,32), dtype=float)
for i in range(0,len(data_test)):
    tmap[i,:,:] = assignTextons(fbRun(fb,data_test[i,:,:]),textons.transpose())

def histc(X, bins):
    import numpy as np
    map_to_bins = np.digitize(X,bins)
    r = np.zeros(bins.shape)
    for i in map_to_bins:
        r[i-1] += 1
    return np.array(r)

histogramas = numpy.zeros((len(data_test),k), dtype=float)
for i in range(0,len(tmap)):
    histogramas[i] = histc(tmap[i].flatten(), np.arange(k))/tmap[i].size            
   
from sklearn.externals import joblib
modelo_TREE = joblib.load("model_RandomForest.pk1")
modelo_KNN = joblib.load("model_KNN.pk1")

from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
histogramas = SelectKBest(chi2, k=2).fit_transform(histogramas, labels_test)

prediction_TREE = modelo_TREE.predict(histogramas)

prediction_KNN = modelo_KNN.predict(histogramas)

demoMaps = numpy.zeros((4,32,32))
demoMaps[0] = tmap[0]
demoMaps[1] = tmap[1]
demoMaps[2] = tmap[2]
demoMaps[3] = tmap[3]

plt.figure(figsize=(8,8))

for k in range(4):
    plt.subplot(2,4,k+1)
    #plt.title(str(i)+'.jpg')
    plt.imshow(data_test[k])
    plt.text(10,10,prediction_KNN[k],fontsize=12)
    plt.text(20,20,prediction_TREE[k],fontsize=12)
for k in range(4,8):
    plt.subplot(2,4,k+1)
    #plt.title(str(i)+'.jpg')
    plt.imshow(demoMaps[k-4])    
plt.show()

from sklearn.metrics import accuracy_score
ACA_TREE = accuracy_score(prediction_TREE,labels_test)
ACA_KNN = accuracy_score(prediction_KNN,labels_test)
print('Predictions:'+str(prediction_KNN)+str(prediction_TREE))
print('Labels:'+str(labels_test))

print('ACA KNN : '+ str(ACA_KNN))
print('ACA RandomForest : '+ str(ACA_TREE))
