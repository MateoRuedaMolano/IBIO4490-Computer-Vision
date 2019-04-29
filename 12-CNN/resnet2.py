# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 03:23:27 2019

@author: mates
"""

"""
Created on Thu Apr 25 15:35:24 2019

@author: mates
"""

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
import tqdm
import torch.utils.data as utils
import os
from skimage import io
from skimage.transform import resize
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score as ac
#import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd 
from PIL import Image
from sklearn.metrics import f1_score

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print('Device:',device)
def print_network(model, name):
    num_params=0
    for p in model.parameters():
        num_params+=p.numel()
    print(name)
    print(model)
    print("The number of parameters {}".format(num_params))

class CustomDatasetFromImages(Dataset):
    def __init__(self, csv_path,stage):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.to_tensor = transforms.ToTensor()
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=None)
        if stage == 'train':
        # First column contains the image paths
            self.image_arr = np.asarray(self.data_info.iloc[1:100000, 0])
            # Second column is the labels
            self.label_arr = np.asarray(self.data_info.iloc[1:100000, 1:11]).astype(np.int32)
            self.label_arr = np.where(self.label_arr>0,1,0)
            # Calculate len
            self.data_len = 100000
        elif stage == 'val':
        # First column contains the image paths
            self.image_arr = np.asarray(self.data_info.iloc[162771:182638, 0])
            # Second column is the labels
            self.label_arr = np.asarray(self.data_info.iloc[162771:182638, 1:11]).astype(np.int32)
            self.label_arr = np.where(self.label_arr>0,1,0)
            # Calculate len
            self.data_len = 19867
            
        elif stage == 'test':
        # First column contains the image paths
            self.image_arr = np.asarray(self.data_info.iloc[182638:202600, 0])
            # Second column is the labels
            self.label_arr = np.asarray(self.data_info.iloc[182638:202600, 1:11]).astype(np.int32)
            self.label_arr = np.where(self.label_arr>0,1,0)
            # Calculate len
            self.data_len = 19962 
    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Open image
        img_as_img = Image.open('img_align_celeba'+'/'+single_image_name)

        # Transform image to tensor
        img_as_tensor = self.to_tensor(img_as_img)
        

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]
        #single_image_label = self.to_tensor(single_image_label)
        
        return (img_as_tensor, single_image_label)

    def __len__(self):
        return self.data_len
        


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, verbose=False):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x,verbose=False):
        x = self.conv1(x)
        if verbose: "Output Layer by layer"
        if verbose: print(x.size())
        
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        if verbose: print(x.size())
        x = self.layer2(x)
        if verbose: print(x.size())
        x = self.layer3(x)
        if verbose: print(x.size())
        x = self.layer4(x)
        if verbose: print(x.size())

        x = self.avgpool(x)
        if verbose: print(x.size())
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
    def training_params(self):
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.003, momentum=0.9, weight_decay=0.0001)
        self.Loss = nn.BCEWithLogitsLoss()



def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 1, 2, 1], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


def resnext50_32x4d(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], groups=32, width_per_group=4, **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnext50_32x4d']))
    return model


def resnext101_32x8d(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], groups=32, width_per_group=8, **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnext101_32x8d']))
    return model
    



def train(data_loader, model, epoch):
    model.train()
    loss_cum = []
    Acc = 0
    for batch_idx, (data,target) in tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc="[TRAIN] Epoch: {}".format(epoch)):
        data = data.to(device)
        target = target.type(torch.FloatTensor).squeeze(1).to(device)
        
        output = model(data)
        model.optimizer.zero_grad()
        loss = model.Loss(output,target)   
        loss.backward()
        model.optimizer.step()
        loss_cum.append(loss.item())
        prediction = torch.where(output.data.cpu() > 0, torch.Tensor([1]), torch.Tensor([0]))
        Acc += (torch.eq(target.data.cpu().long(),prediction.long())).sum()
        n_target = (target.data).cpu().numpy()
        n_target.clip(0)
        n_prediction = (prediction.data).cpu().numpy()
        f_measure = f1_score(n_target, n_prediction, average='macro', sample_weight=None)
        
    
    print("Loss: %0.3f | Acc: %0.2f"%(np.array(loss_cum).mean(), f_measure))
    
    
    
    
def val(data_loader, model, epoch):
   model.eval()
   loss_cum = []
   Acc = 0
   for batch_idx, (data,target) in tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc="[VAL] Epoch: {}".format(epoch)):
        data = data.to(device).requires_grad_(False)
        target = target.type(torch.FloatTensor).squeeze(1).to(device).requires_grad_(False)

        output = model(data)
        loss = model.Loss(output,target)   
        loss_cum.append(loss.item())
        #_, arg_max_out = torch.max(output.data.cpu(), 1)
        prediction = torch.where(output.data.cpu() > 0, torch.Tensor([1]), torch.Tensor([0]))      
        Acc += (torch.eq(target.data.cpu().long(),prediction.long())).sum()
        n_target = (target.data).cpu().numpy()
        n_target.clip(0)
        n_prediction = (prediction.data).cpu().numpy()
        f_measure = f1_score(n_target, n_prediction, average='macro', sample_weight=None)
    
   print("Loss: %0.3f | Acc: %0.2f"%(np.array(loss_cum).mean(), f_measure))

def test(data_loader, model, epoch):
    model.eval() 
    open("Resnet_Results.txt","w")
    file = open("Resnet_Results.txt","a")
    for batch_idx, (data,_) in tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc="[TEST] Epoch: {}".format(epoch)):
        data = data.to(device).requires_grad_(False)

        output = model(data)
        prediction = torch.where(output.data.cpu() > 0, torch.Tensor([1]), torch.Tensor([0]))
        cont = 1;
        for i in range(prediction.shape[0]):
            number_image = '{0:06}'.format(182637+cont)
            cont = cont +1
            filename=number_image+'.jpg'
            file.write(filename+",")
            for j in range(prediction.shape[1]):
                n_pred = prediction.numpy()
                n_pred.astype(int)
                res = n_pred[i][j]
                if j < prediction.shape[1]:
                    file.write(str(res)+",")   
            file.write("\n") 
    file.close()         
    
if __name__=='__main__':
    epochs=100
    batch_size=20
    TEST=True
    

    celebA_images_train = CustomDatasetFromImages('annotations.csv',stage='train')
    celebA_loader_train = DataLoader(dataset=celebA_images_train,batch_size=batch_size,shuffle=True)
    
    celebA_images_val = CustomDatasetFromImages('annotations.csv',stage='val')
    celebA_loader_val = DataLoader(dataset=celebA_images_val,batch_size=batch_size,shuffle=False)
       
    model = resnet18()
    model.to(device)
    model.training_params()
    print_network(model, 'Conv network/Resnet50() Reduced')    
    #Exploring model
    data, _ = next(iter(celebA_loader_train))
    _ = model(data.to(device).requires_grad_(False), verbose=True)
    for epoch in range(epochs): 
        train(celebA_loader_train, model, epoch)
        val(celebA_loader_val, model, epoch)

    if TEST:
        celebA_images_test = CustomDatasetFromImages('annotations.csv',stage='test')
        celebA_loader_test = DataLoader(dataset=celebA_images_test,batch_size=1,shuffle=False)
        test(celebA_loader_test, model, epoch)          
        print("TEST Results printed.")