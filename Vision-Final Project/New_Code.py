#from __future__ import print_function 
#from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
#from torchvision import datasets, models, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from torch.utils.data import DataLoader

import matplotlib as plt
import time
import numpy as np
import Dataset
import Models
import random
from sklearn.metrics import precision_score, roc_auc_score, roc_curve

#Set random seem for reproducibility
manualSeed = 777
random.seed(manualSeed)
torch.manual_seed(manualSeed)

def train (model, dataloader, criterion, optimizer, scheduler,epoch):
   
    model.train()

    for batch_idx, (data,target) in tqdm(enumerate(dataloader),desc="[TRAIN] Epoch:{}".format(epoch)):
        
        data = data.to(device)
        target= target.type(torch.Tensor).squeeze(1).to(device)
        output= model(data)

        loss= criterion(output,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def val (model, dataloader, criterion, optimizer, scheduler,epoch):
    
    model.eval()

    loss_val= []

    for batch_idx, (data,target) in tqdm(enumerate(dataloader),desc="[VAL] Epoch:{}".format(epoch)):
        
        data = data.to(device)
        target= target.type(torch.Tensor).squeeze(1).to(device)
        output= model(data)

        loss_val.append(criterion(output,target))
    
    loss_val= torch.Tensor.mean(loss_val)
    return loss_val

def AUROC (labels,predictions,num_classes):
    
    #Get the auroc score per class. The method get the precision of the class in a possible exception
    score= []
    curve= []
    labels= labels.cpu().numpy()
    predictions = predictions.cpu().detach().numpy()
    
    for i in range (num_classes):
        try:
            score.append(roc_auc_score(labels[:,i],predictions[:,i]))
            curve.append(roc_curve(labels[:,i],predictions[:,i]))
            print('A')
        except ValueError:
            score.append(precision_score(labels[:,i],np.round(predictions[:,i])))
            print('P')
            pass
    return score,curve

def test (model, dataloader,num_classes):
    
    NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
        'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

    labels = torch.FloatTensor().to(device)

    predictions = torch.FloatTensor().to(device)

    model.eval()

    for batch_idx, (data,target) in tqdm(enumerate(dataloader),desc="[TEST]"):
        
        data= data.to(device)
        target = target.type(torch.Tensor).squeeze(1).to(device)
        labels = torch.cat((labels, target), 0)
        output= model(data)
        predictions = torch.cat((predictions, output.data), 0)
    
    #Auroc individual per class and fpr,tpr to graph de roc curve
    aurocIndividual,curve = AUROC(labels,predictions, num_classes)
    aurocMean = np.array(aurocIndividual).mean()

    #Open and write a txt file with the performance of the method
    f= open("Performance_Desnet161.txt","w+")    
    print ('AUROC mean ', aurocMean)
    f.write('AUROC mean' + ' '  + str(aurocMean) + '\n')
    
    for i in range (0, len(aurocIndividual)):
        print (NAMES[i], ' ', aurocIndividual[i])
        f.write(NAMES[i] + ' ' + str(aurocIndividual[i])+ '\n')
    f.close()

    #Plot the Roc Curve
    plt.figure()
    for i in range(len(curve)):
        plt.plot(curve[i][0],curve[i][1],label=NAMES[i])

    plt.xlabel('False Positive Rate - 1-Specificity')
    plt.ylabel('True Positive Rate - Sensibility')
    plt.title('Roc Curve')
    plt.legend()
    plt.savefig('Roc_Curve.png')
    plt.show()



if __name__=='__main__':
    
    # Detect if we have a GPU available
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    #Specifications
    num_classes= 14
    batch_size= 16
    num_epochs= 5

    #Model inicialization 
    model= Models.DenseNet121(num_classes,True)
    model= model.to(device)

    #Optimizer, loss and scheduler
    optimizer= optim.Adam(model.densenet121.parameters(), lr=0.1, weight_decay=1e-8)
    scheduler= ReduceLROnPlateau(optimizer,factor=0.1,patience=3,mode='min')
    criterion = nn.BCEWithLogitsLoss(size_average=True)

    #Datasets
    train_dataset=Dataset.Data(transform=True)
    train_dataloader=DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    val_dataset=Dataset.Data(val=True,transform=True)
    val_dataloader= DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    test_dataset=Dataset.Data(test=True,transform=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    #Training 
    loss_opt= 999 
    for epoch in range(num_epochs):
        
        train(model,train_dataloader,criterion,optimizer,scheduler,epoch)
        loss_val= val(model,val_dataloader,criterion,optimizer,scheduler,epoch)
        
        print ('loss_val =' + loss_val.data[0])

        scheduler.step(loss_val.data[0])

        if loss_val.data[0] < loss_opt:
            loss_opt = loss_val.data[0]
            torch.save(model.state_dict(),'Model_Desnet121.pth')
    
    test(model,test_dataloader,num_classes)
    

    