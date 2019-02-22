#!/usr/bin/python
"""
Created on Sun Feb 17 17:07:06 2019

@author: ms.rueda10 & ss.leal10
"""
import numpy
from scipy import misc
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage.filters import gaussian as gauss
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

A = ndimage.imread("parte6.jpg", flatten=False)
B = ndimage.imread("parte8.jpg", flatten=False)

test =numpy.concatenate((A[:,0:512,:],B[:,512:,:]),axis=1)
misc.imsave("Blended_NotPyr.jpg", numpy.real(test))
plt.imshow(test)

#downsampling:
x1_A = misc.imresize(gauss(A,sigma=   0.1, multichannel=True, preserve_range = True),(512,512), interp='bicubic', mode = None)
s1_A= cv2.pyrUp(x1_A)
A_fl = A.astype(float)
s1_A_fl = s1_A.astype(float)
l1_A_fl = numpy.absolute(A_fl-s1_A_fl)
l1_A = l1_A_fl.astype(numpy.uint8)
plt.imshow(x1_A)

x2_A = misc.imresize(gauss(x1_A,sigma=0.1, multichannel=True, preserve_range = True),(256,256), interp='bicubic', mode = None)
s2_A= cv2.pyrUp(x2_A)
x1_A_fl = x1_A.astype(float)
s2_A_fl = s2_A.astype(float)
l2_A_fl = numpy.absolute(x1_A_fl-s2_A_fl)
l2_A = l2_A_fl.astype(numpy.uint8)
plt.imshow(x2_A)

x3_A = misc.imresize(gauss(x2_A,sigma=0.1, multichannel=True, preserve_range = True),(128,128), interp='bicubic', mode = None)
s3_A= cv2.pyrUp(x3_A)
x2_A_fl = x2_A.astype(float)
s3_A_fl = s3_A.astype(float)
l3_A_fl = numpy.absolute(x2_A_fl-s3_A_fl)
l3_A = l3_A_fl.astype(numpy.uint8)
plt.imshow(l3_A)


x4_A = misc.imresize(gauss(x3_A,sigma=0.1, multichannel=True, preserve_range = True),(64,64), interp='bicubic', mode = None)
s4_A= cv2.pyrUp(x4_A)
x3_A_fl = x3_A.astype(float)
s4_A_fl = s4_A.astype(float)
l4_A_fl = numpy.absolute(x3_A_fl-s4_A_fl)
l4_A = l4_A_fl.astype(numpy.uint8)
plt.imshow(x4_A)

x1_B = misc.imresize(gauss(B,sigma=0.1, multichannel=True, preserve_range = True),(512,512), interp='bicubic', mode = None)
s1_B= cv2.pyrUp(x1_B)
B_fl = B.astype(float)
s1_B_fl = s1_B.astype(float)
l1_B_fl = numpy.absolute(B_fl-s1_B_fl)
l1_B = l1_B_fl.astype(numpy.uint8)
plt.imshow(x1_B)

x2_B = misc.imresize(gauss(x1_B,sigma=0.1, multichannel=True, preserve_range = True),(256,256), interp='bicubic', mode = None)
s2_B= cv2.pyrUp(x2_B)
x1_B_fl = x1_B.astype(float)
s2_B_fl = s2_B.astype(float)
l2_B_fl = numpy.absolute(x1_B_fl-s2_B_fl)
l2_B = l2_B_fl.astype(numpy.uint8)
plt.imshow(x2_B)

x3_B = misc.imresize(gauss(x2_B,sigma=0.1, multichannel=True, preserve_range = True),(128,128), interp='bicubic', mode = None)
s3_B= cv2.pyrUp(x3_B)
x2_B_fl = x2_B.astype(float)
s3_B_fl = s3_B.astype(float)
l3_B_fl = numpy.absolute(x2_B_fl-s3_B_fl)
l3_B = l3_B_fl.astype(numpy.uint8)
plt.imshow(x3_B)

x4_B = misc.imresize(gauss(x3_B,sigma=0.1, multichannel=True, preserve_range = True),(64,64), interp='bicubic', mode = None)
s4_B= cv2.pyrUp(x4_B)
x3_B_fl = x3_B.astype(float)
s4_B_fl = s4_B.astype(float)
l4_B_fl = numpy.absolute(x3_B_fl-s4_B_fl)
l4_B = l4_B_fl.astype(numpy.uint8)
plt.imshow(x4_B)

#upsampling:

#concatenate x4
x4_halves =numpy.concatenate((x4_A[:,0:32,:],x4_B[:,32:,:]),axis=1)
plt.imshow(x4_halves)

#pyrUp(g4)
g4_halves= cv2.pyrUp(x4_halves)
plt.imshow(g4_halves)

#concatenate l4
l4_halves =numpy.concatenate((l4_A[:,0:64,:],l4_B[:,64:,:]),axis=1)
plt.imshow(l4_halves)

#sum g4+l4=x3
x3_halves = cv2.add(g4_halves,l4_halves)
plt.imshow(x3_halves)

#pyrup(g3)
g3_halves= cv2.pyrUp(x3_halves)
plt.imshow(g3_halves)

#concatenate l3
l3_halves =numpy.concatenate((l3_A[:,0:128,:],l3_B[:,128:,:]),axis=1)
plt.imshow(l3_halves)

#sum g3+l3=x2
x2_halves = cv2.add(g3_halves,l3_halves)
plt.imshow(x2_halves)

#pyrup(g2)
g2_halves= cv2.pyrUp(x2_halves)
plt.imshow(g2_halves)

#concatenate l2
l2_halves =numpy.concatenate((l2_A[:,0:256,:],l2_B[:,256:,:]),axis=1)
plt.imshow(l2_halves)

#sum g2+l2=x1
x1_halves = cv2.add(g2_halves,l2_halves)
plt.imshow(x1_halves)

#pyrup(g1)
g1_halves= cv2.pyrUp(x1_halves)
plt.imshow(g1_halves)

#concatenate l1
l1_halves =numpy.concatenate((l1_A[:,0:512,:],l1_B[:,512:,:]),axis=1)
plt.imshow(l1_halves)

#sum g1+l1=x0
x0_halves = cv2.add(g1_halves,l1_halves)
plt.imshow(x0_halves)
misc.imsave("Blended_pyr.jpg", numpy.real(x0_halves))

#Subplotting
fig = plt.figure(figsize=(10,10))

fig.add_subplot(3,5,1)
plt.imshow(A)
plt.title("Original")
fig.add_subplot(3,5,2)
plt.imshow(x1_A)
plt.title("1/2")
fig.add_subplot(3,5,3)
plt.imshow(x2_A)
plt.title("1/4") 
fig.add_subplot(3,5,4)
plt.imshow(x3_A)
plt.title("1/8")
fig.add_subplot(3,5,5)
plt.imshow(x4_A)
plt.title("1/16")

fig.add_subplot(3,5,6)
plt.imshow(B)
plt.title("Original")
fig.add_subplot(3,5,7)
plt.imshow(x1_B)
plt.title("1/2")
fig.add_subplot(3,5,8)
plt.imshow(x2_B)
plt.title("1/4")
fig.add_subplot(3,5,9)
plt.imshow(x3_B)
plt.title("1/8")
fig.add_subplot(3,5,10)
plt.imshow(x4_B)
plt.title("1/16")
           
fig.add_subplot(3,5,11)
plt.imshow(x4_halves)
plt.title("1/16")
fig.add_subplot(3,5,12)
plt.imshow(x3_halves)
plt.title("1/8")
fig.add_subplot(3,5,13)
plt.imshow(x2_halves)
plt.title("1/4")
fig.add_subplot(3,5,14)
plt.imshow(x1_halves)
plt.title("1/2")
fig.add_subplot(3,5,15)
plt.imshow(x0_halves)
plt.title("Final")
fig.savefig('Pyrs.png')
