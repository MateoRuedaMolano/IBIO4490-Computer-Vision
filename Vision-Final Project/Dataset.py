import os
import numpy as np 
from torch.utils.data import Dataset, DataLoader
from Get_labels_multitarget import get_labels_multitarget
import cv2
import glob
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from skimage import io
import random
class Data(Dataset):
    def __init__(self, demo=False, test = False,val = False,transform=None):
        
        #Get image paths and their labels depending the set
        if val:          
            img_path= os.path.join('Val','*.png')
            images=glob.glob(img_path)
            labels=get_labels_multitarget(images)
        elif test: 
            img_path= os.path.join('Test','*.png')
            images=glob.glob(img_path)
            labels=get_labels_multitarget(images)
        elif demo:
            img_path= os.path.join('/media/user_home2/vision/Test','test','*.png')
            images=glob.glob(img_path)
            aux= []

            for i in listaAleatorios(4):
                aux.append(images[i])
            images=aux
            labels= get_labels_multitarget(images)
        else:
            img_path= os.path.join('Train','*.png')
            images=glob.glob(img_path)
            labels=get_labels_multitarget(images)
            
        self.image_files=images
        self.labels=labels

        #Define the transforms of each set
        if test:
            self.transform= transforms.Compose([
                transforms.Resize(256),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:
            self.transform= transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        img= self.image_files[idx]
        label = self.labels[idx]      

        img= Image.open(img).convert('RGB')
                
        # Transform image to tensor
        if transforms:
            img= self.transform(img)
        
        return (img, label)

def listaAleatorios(n):
    lista = [0]  * n
    for i in range(n):
        lista[i] = random.randint(0, 20000)
    return lista
