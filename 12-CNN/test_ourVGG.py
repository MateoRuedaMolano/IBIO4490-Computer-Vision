import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
import tqdm
#import torch.utils.data as utils
#import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd 
from PIL import Image
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:',device)
def print_network(model, name):
    num_params=0
    for p in model.parameters():
        num_params+=p.numel()
    print(name)
    print(model)
    print("The number of parameters {}".format(num_params))

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=10, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        #self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 1024),
            #nn.Linear(48 * 2 * 2, 192),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 256),
            #nn.Linear(192, 96),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, num_classes),
            #nn.Linear(96, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x, verbose=False):
        if verbose: "Output Layer by layer"
        if verbose: print(x.size())
        x = self.features(x)
        if verbose: print(x.size())
        x = self.avgpool(x)
        if verbose: print(x.size())
        x = x.view(x.size(0), -1)
        if verbose: print(x.size())
        x = self.classifier(x)
        if verbose: print(x.size())
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def training_params(self):
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.003, momentum=0.9, weight_decay=0.00001)
        self.Loss = nn.BCEWithLogitsLoss()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

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
            self.image_arr = np.asarray(self.data_info.iloc[1:162771, 0])
            # Second column is the labels
            self.label_arr = np.asarray(self.data_info.iloc[1:162771, 1:11]).astype(np.int32)
            self.label_arr = np.where(self.label_arr>0,1,0)
            # Calculate len
            self.data_len = 162770
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

cfg = {
    'A': [64, 'M', 128, 'M', 192, 'M', 256, 'M', 256, 'M'],
    #'A': [6, 'M', 12, 'M', 24, 24, 'M', 48, 48, 'M', 48, 48, 'M'],
}


def vgg11(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11']))
    return model


def vgg11_bn(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']))
    return model

def test(data_loader, model):
    model.eval() 
    open("SelfArch16_Results.txt","w")
    file = open("SelfArch16_Results.txt","a")
    cont = 1;
    for batch_idx, (data,_) in tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc="[TEST]"):
        data = data.to(device).requires_grad_(False)

        output = model(data)
        prediction = torch.where(output.data.cpu() > 0, torch.Tensor([1]), torch.Tensor([0]))

        number_image = '{0:06}'.format(182637+cont)
        filename=number_image+'.jpg'
        file.write(filename+",")
        for j in range(prediction.shape[1]):
                res = (prediction[0][j]).long().item()
                file.write(str(res)+",")
        file.write(": \n") 
        cont = cont +1
    file.close() 
    
if __name__=='__main__':
    epochs=16
    batch_size=25

    model = vgg11_bn()
    model.load_state_dict(torch.load('SelfArch16.pth',map_location='cuda:0'))
    model.to(device)
    #model.eval()
    celebA_images_test = CustomDatasetFromImages('annotations.csv',stage='test')
    celebA_loader_test = DataLoader(dataset=celebA_images_test,batch_size=1,shuffle=False)
    test(celebA_loader_test, model)          
    print("TEST Results printed.")