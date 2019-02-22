#!/usr/bin/python
"""
Created on Tue Feb 19 22:11:34 2019

@author: ms.rueda10 & ss.leal10
"""
import numpy 
from scipy import misc
from scipy import ndimage
from skimage.filters import gaussian
import cv2
import os
#Downloading all the images:
cwd = os.getcwd()
if os.path.exists(cwd +'\Hybrid_Lab4.zip') == False:
    url = "https://www.dropbox.com/s/cp6vnffng7bjsfe/Hybrid_Lab4.zip?dl=1"
    import zipfile
    print('Downloading the database...')
    import urllib.request
    u = urllib.request.urlopen(url)
    data = u.read()
    u.close()
    with open(cwd+'/'+'Hybrid_Lab4.zip','wb') as f :
        f.write(data)
    print('Database downloaded.')
    f.close()
    
    #Unzip
    print('Unzipping the database...')
    zip_Archivo = zipfile.ZipFile(cwd +'\Hybrid_Lab4.zip', 'r')
    zip_Archivo.extractall(cwd)
    zip_Archivo.close()
    print('Unzipping done.') 

#Generating the hybrid image:
A = ndimage.imread("uribe1.jpg", flatten=False)
B = ndimage.imread("duque1.jpg", flatten=False)

lowPassedA = gaussian(A,sigma=10,multichannel=True)
misc.imsave("low-passed-A.png", (lowPassedA))

lowPassedB = gaussian(B,sigma=70,multichannel=True)
misc.imsave("low-passed-B.png", numpy.real(lowPassedB))

BLow = ndimage.imread("low-passed-B.png", flatten=False)
BHighPassed = cv2.subtract(B,BLow)
misc.imsave("high-passed-B.png", numpy.real(BHighPassed))

highPassed = ndimage.imread("high-passed-B.png", flatten=False)
lowPassed = ndimage.imread("low-passed-A.png", flatten=False)

hybrid = cv2.add(highPassed,lowPassed)
misc.imsave("HybridImage.png", numpy.real(hybrid))