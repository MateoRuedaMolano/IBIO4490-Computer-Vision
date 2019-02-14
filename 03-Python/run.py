#!/usr/bin/python
"""
Created on Wed Feb 13 21:21:31 2019

@author: mates
"""
#import time to set the time
import time
start= time.time()
#First we need to import os, cause we need to get the cwd.
import os
#We need the library urllib to download the dataset
import urllib.request
import urllib
#library to get the random number
import random
#Zip library to uncompress the dataset
import zipfile
#Import xlrd because labels are in excel
import xlrd
#PIL for labelling images
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
#matplotbib for plotting images
import matplotlib.pyplot as pln
#import shutil to erase the folder
import shutil
#URL to download the data
url = "https://www.dropbox.com/s/4o9lzqs5tvvoh9i/data_samples.zip?dl=1"
#Unify the data

#Check if the directory has already been downloaded
if os.path.exists("data_samples.zip") == False:
#Download data if not
      urllib.request.urlretrieve(url, "data_samples.zip")
      print('Dataset downloaded')
else:
      print('The database is in your cwd')
#Unzip the folder
if  os.path.exists("data_samples") == False:
  with zipfile.ZipFile("data_samples.zip","r") as unzip:
#Extract all files
   unzip.extractall('data_samples')
#close
   unzip.close()
   print('Dataset unzipped')
else:
  print('You already have the data-set unzipped')
  
#Get the labels
cwd = os.getcwd()
#Path of the labels
label_route = os.path.join(cwd,"data_samples","ETHZShapeClasses-V1.2","labels.xlsx")
#Retrieve the book
wb = xlrd.open_workbook(label_route)
#Retrieve the sheet
ws = wb.sheet_by_name("Hoja1")
#Retrieve the column labels
labels = ws.col_values(1)
del labels[0]


#Path of the images
images_route = os.path.join(cwd,"data_samples","ETHZShapeClasses-V1.2","data")
#names of the images
names = ws.col_values(0)
del names[0]

#Chose number
num = 9
#Random positions of images
pos_random = random.sample(range(0,len(labels)), num)
#Random names and labels of those positions
random_names = [names[i] for i in pos_random] 
random_labels = [labels[i] for i in pos_random]
random_labels = [ int(x) for x in random_labels]
#Path of the directory of re-sized images
size_route= os.path.join(cwd,"data_samples","ETHZShapeClasses-V1.2","size_img")
#If the folder already exists...
if  os.path.exists(size_route) == False:
    os.mkdir(size_route)
    print('Size_imgs folder created')
else:
    print('Size_imgs folder already exists')
## Loop inside random image names
for i in range(0, len(random_names)):
    #Path of random images
    path_rm = os.path.join(images_route, random_names[i])
    #Open random images
    img_rm = Image.open(path_rm)
    #Resize them
    siz_img_rm = img_rm.resize((256,256))
    #Label the image
    draw = ImageDraw.Draw(siz_img_rm)
    fontt = ImageFont.truetype("arial.ttf",80)
    draw.text((100, 100),str(random_labels[i]),(255), font = fontt)
    siz_img_rm.save(os.path.join(size_route,random_names[i]))
    
## Subplot
for i in range(0, len(random_names)):
    path_f= os.path.join(size_route, random_names[i])
    imagen = Image.open(path_f)
    pln.subplot(num/3,num/3,i+1)
    pln.imshow(imagen)
    pln.savefig(os.path.join(size_route,'figure.png'))
fig = Image.open(os.path.join(size_route,'figure.png'))
fig.show()
# Erase the folder
shutil.rmtree(os.path.join(size_route))
end = time.time()
print('The total time is '+ str(end-start))