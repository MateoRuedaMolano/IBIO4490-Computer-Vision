#%matplotlib inline
from __future__ import print_function 
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import time
import os
import copy
import Dataset
import random
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, roc_auc_score, roc_curve
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

#Set random seem for reproducibility
manualSeed = 777
random.seed(manualSeed)
torch.manual_seed(manualSeed)


# Top level data directory. Here we assume the format of the directory conforms 
#   to the ImageFolder structure
#data_dir = 'Train'#Dataset_Trai_Val'

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "densenet"

# Number of classes in the dataset
num_classes = 14

# Batch size for training (change depending on how much memory you have)
batch_size = 15

# Number of epochs to train for 
num_epochs = 20

# Flag for feature extracting. When False, we finetune the whole model, 
#   when True we only update the reshaped layer params
feature_extract = True

#Begin models train
def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, num_classes, num_epochs=25):
    since = time.time()
    val_acc_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in range(2):
            if phase == 0:
                dataset= train_dataloader
                model.train()  # Set model to training mode
            else:
                dataset= val_dataloader
                model.eval()   # Set model to evaluate mode

            running_loss = []
            #Variables to concatenated the labels and prediction as a matrix in order to obtain the auroc score
            labels_auroc= torch.FloatTensor().to(device)
            preds_auroc = torch.FloatTensor().to(device)

            # Iterate over data.
            for batch_idx, (data,target) in tqdm(enumerate(dataset), total=len(dataset)):
                data = data.to(device)
                target = target.type(torch.Tensor).squeeze(1).to(device)
                #inputs= data
                #labels = target
                labels_auroc=torch.cat((labels_auroc, target), 0)

                # track history if only in train
                with torch.set_grad_enabled(phase == 0):
                    # Get model outputs, predictions and calculate loss
                    #outputs = model(inputs)
                    outputs = model(data)
                    outputs = torch.sigmoid(outputs)
                    #loss = criterion(outputs, labels)
                    loss = criterion(outputs, target)
                    preds_auroc=torch.cat((preds_auroc,outputs),0)

                    # backward + optimize only if in training phase
                    if phase == 0:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                # statistics
                running_loss.append(loss.item())

            #Get the loss per epoch
            epoch_loss = np.array(running_loss).mean()
            #change the lr rate if the epoch_loss not change in 3 epochs (scheduler's patience)
            if phase == 1:
                scheduler.step(epoch_loss)

            epoch_acc,_=AUROC(labels_auroc,preds_auroc, num_classes)
            epoch_acc=np.array(epoch_acc).mean()

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model and save the best model depending the performance
            if phase == 1 and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 1:
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights and save the model
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(),'Model_Desnet.pth')
    return model, val_acc_history


def set_parameter_requires_grad(model, feature_extracting):
    # Parameters of the model
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    
    model_ft = None
    input_size = 0
    if model_name== "densenet":
        """ Densenet
        """
        model_ft = models.densenet161(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnext":
        """ Resnext50_32x4d
        """
        model_ft = models.resnext50_32x4d(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet50":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet101":
        """ Resnet101
        """
        model_ft = models.resnet101(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()
    
    return model_ft, input_size


def AUROC (labels,predictions,num_classes):
    #Get the auroc score per class. The method get the precision of the class in a possible exception
    out= []
    curve= []
    labels= labels.cpu().numpy()
    predictions = predictions.cpu().detach().numpy()
    
    for i in range (num_classes):
        try:
            out.append(roc_auc_score(labels[:,i],predictions[:,i]))
            curve.append(roc_curve(labels[:,i],predictions[:,i]))
            print('A')
        except ValueError:
            
            out.append(precision_score(labels[:,i],np.round(predictions[:,i])))
            print('P')
            pass
    return out,curve

def test (test_dataloader,model,epoch,num_classes):
    #Get the predictions and auroc score per class in all test set
    NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
        'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
            
    labels = torch.FloatTensor().to(device)

    predictions = torch.FloatTensor().to(device)
       
    model.eval()
        
    for batch_idx, (data, target) in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc='[TEST]'):
            
        target = target.type(torch.Tensor).squeeze(1).to(device)
        labels = torch.cat((labels, target), 0)
        varInput=data.to(device)    
        output = model(varInput)
        output = torch.sigmoid(output)
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
         
    return 

# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

# Print the model we just instantiated
print(model_ft)

print("Initializing Datasets and Dataloaders...")

#Datasets
train=Dataset.Data(transform=True)
train_dataloader=DataLoader(dataset=train, batch_size=batch_size, shuffle=True, num_workers=0)

val=Dataset.Data(val=True,transform=True)
val_dataloader= DataLoader(dataset=val, batch_size=batch_size, shuffle=False, num_workers=0)

test_dataset=Dataset.Data(test=True,transform=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)


# Detect if we have a GPU available
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#device='cpu'
print(device)

# Send the model to GPU
model_ft = model_ft.to(device)

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are 
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)


# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(params_to_update, lr=0.001, weight_decay=1e-8)
scheduler= ReduceLROnPlateau(optimizer_ft,factor=0.4,patience=3,mode='min')

# Setup the loss fxn
criterion = nn.BCEWithLogitsLoss(size_average=True)

# Train and evaluate
model_ft, hist = train_model(model_ft, train_dataloader, val_dataloader, criterion, optimizer_ft, num_classes, num_epochs=num_epochs)

test(test_dataloader,model_ft,num_epochs,num_classes)

