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

#tests with 1000 images form the test set
data_test = data_test[0:1000,:,:]
labels_test = labels_test[0:1000]

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

from sklearn.metrics import accuracy_score
ACA_TREE = accuracy_score(prediction_TREE,labels_test)
ACA_KNN = accuracy_score(prediction_KNN,labels_test)

class_names = np.arange(0,10)

import itertools
from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    
    print(cm)
   
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
confusionmat_KNN = confusion_matrix(labels_test,prediction_KNN)
confusionmat_TREE = confusion_matrix(labels_test,prediction_TREE)



plt.figure()
plot_confusion_matrix(confusionmat_KNN, class_names, normalize =True,
                      title='Normalized confusion matrix - KNN')

plt.figure()
plot_confusion_matrix(confusionmat_TREE, class_names, normalize=True,
                      title='Normalized confusion matrix - RandomForest')  

plt.show()
print('ACA KNN : '+ str(ACA_KNN))
print('ACA RandomForest : '+ str(ACA_TREE))
plt.close()