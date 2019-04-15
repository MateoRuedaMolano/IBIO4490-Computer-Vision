
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchvision import datasets, transforms
import numpy as np
import tqdm
import torch.utils.data as utils
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:',device)
def print_network(model, name):
    num_params=0
    for p in model.parameters():
        num_params+=p.numel()
    print(name)
    print(model)
    print("The number of parameters {}".format(num_params))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #layer with 64 2d convolutional filter of size 3x3
        self.conv1 = nn.Conv2d(1, 84, kernel_size=3) #Channels input: 1, c output: 48, filter of size 3
        self.norm1 = nn.BatchNorm2d(84)
        self.conv2 = nn.Conv2d(84,72 , kernel_size=3)
        self.norm2 = nn.BatchNorm2d(72)
        self.conv3 = nn.Conv2d(72, 128, kernel_size=3)
        self.norm3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(2048, 1024)   
        self.fc2 = nn.Linear(1024, 7)   
    
    def forward(self, x, verbose=False):
        if verbose: "Output Layer by layer"
        if verbose: print(x.size())
        x = self.norm1(F.max_pool2d(F.relu(self.conv1(x)), 2)) #Perform a Maximum pooling operation over the nonlinear responses of the convolutional layer
        if verbose: print(x.size())
        x = self.norm2(F.max_pool2d(F.relu(self.conv2(x)), 2))
        #if verbose: print(x.size())
        #x = F.dropout(x, 0.50, training=self.training)#Try to control overfit on the network, by randomly excluding 25% of neurons on the last #layer during each iteration
        if verbose: print(x.size())
        x = self.norm3(F.max_pool2d(F.relu(self.conv3(x)), 2))
        if verbose: print(x.size())
        x = F.dropout(x, 0.50, training=self.training)
        if verbose: print(x.size())
        #ipdb.set_trace()
        x = x.view(-1, 2048)
        if verbose: print(x.size())
        x = F.relu(self.fc1(x))
        if verbose: print(x.size())
        x = self.fc2(x)
        if verbose: print(x.size())
        return x
    def training_params(self):
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.00001, momentum=0.9, weight_decay=0.00001)
        self.Loss = nn.CrossEntropyLoss()
        
#def get_data(batch_size):
    #transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
def get_data():
    cwd = os.getcwd()
    if os.path.isfile(cwd+'/'+'fer2013.zip') == False:
        url = "https://www.dropbox.com/s/ngq9ntcb8p3m8y3/fer2013.zip?dl=1"
        import zipfile
        print('Downloading the database...')
        import urllib.request
        u = urllib.request.urlopen(url)
        data = u.read()
        u.close()
        with open(cwd+'/'+'fer2013.zip','wb') as f :
            f.write(data)
        print('Database downloaded.')
        f.close()
        
        #Unzip
        print('Unzipping the database...')
        zip_Archivo = zipfile.ZipFile(cwd +'/'+'fer2013.zip', 'r')
        zip_Archivo.extractall(cwd)
        zip_Archivo.close()
        
        #untar
        import tarfile
        tar = tarfile.open('fer2013.tar.gz', "r:gz")
        tar.extractall()
        tar.close()
        print('Unzipping done.') 
        
    # angry, disgust, fear, happy, sad, surprise, neutral
    with open("fer2013/fer2013.csv") as f:
        content = f.readlines()

    lines = np.array(content)
    num_of_instances = lines.size
    print("number of instances: ",num_of_instances)
    print("instance length: ",len(lines[1].split(",")[1].split(" ")))

    x_train, y_train, x_test, y_test = [], [], [], []

    for i in range(1,num_of_instances):
        emotion, img, usage = lines[i].split(",")
        pixels = np.array(img.split(" "), 'float32')
        #emotion = 1 if int(emotion)==3 else 0 # Only for happiness
        if 'Training' in usage:
            y_train.append(emotion)
            x_train.append(pixels)
        elif 'PublicTest' in usage:
            y_test.append(emotion)
            x_test.append(pixels)

    #------------------------------
    #data transformation for train and test sets
    x_train = np.array(x_train, 'float64')
    y_train = np.array(y_train, 'float64')
    x_test = np.array(x_test, 'float64')
    y_test = np.array(y_test, 'float64')

    x_train /= 255 #normalize inputs between [0, 1]
    x_test /= 255

    x_train = x_train.reshape(x_train.shape[0], 48, 48)
    x_test = x_test.reshape(x_test.shape[0], 48, 48)
    y_train = y_train.reshape(y_train.shape[0], 1)
    y_test = y_test.reshape(y_test.shape[0], 1)

    #Augmentating images with reflection
    xtrain_flip= np.zeros((28709,48,48))
    for i in range(len(x_train)):
        xtrain_flip[i] = np.fliplr(x_train[i])
            
    x_train = np.concatenate((x_train, xtrain_flip), axis=0)
    y_expand1 = y_train

    y_train =  np.concatenate((y_train, y_expand1), axis=0)


    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'validation samples')
    #test will be used for validation
    return x_train, y_train, x_test, y_test

def get_test_data():
    import cv2
    import os

    cwd = os.getcwd()
    if os.path.isfile(cwd+'/'+'Emotions_test.zip') == False:
        url = "http://bcv001.uniandes.edu.co/Emotions_test.zip"
        import zipfile
        print('Downloading the test database...')
        import urllib.request
        u = urllib.request.urlopen(url)
        data = u.read()
        u.close()
        with open(cwd+'/'+'Emotions_test.zip','wb') as f :
            f.write(data)
        print('Test database downloaded.')
        f.close()
        
        #Unzip
        print('Unzipping the database...')
        zip_Archivo = zipfile.ZipFile(cwd +'/'+'Emotions_test.zip', 'r')
        zip_Archivo.extractall(cwd)
        zip_Archivo.close()
        print('Unzipping done.')
        
    images = np.zeros((1610,48,48))
    import face_recognition
    for i in tqdm.tqdm(range(1610), desc = "Detecting,cropping and resizing(48,48) test faces,wait..."):
        filename = os.listdir('Emotions_test')[i]
        print('file:',filename)
        image = face_recognition.load_image_file(os.path.join('Emotions_test',filename))
        face_locations = face_recognition.face_locations(image)
        #img = cv2.imread(os.path.join('Emotions_test',filename))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        #faces = face_cascade.detectMultiScale(img,1.1,5,0)   
        crop = gray[face_locations[0][0]:face_locations[0][2],face_locations[0][3]:face_locations[0][1]]
        img = cv2.resize(crop, dsize=(48, 48), interpolation=cv2.INTER_CUBIC)
        images[i]= img
        return images
    
def train(data_loader, model, epoch):
    model.train()
    loss_cum = []
    Acc = 0
    for batch_idx, (data,target) in tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc="[TRAIN] Epoch: {}".format(epoch)):
        data = data.to(device)
        target = target.type(torch.LongTensor).squeeze(1).to(device)
        
        output = model(data)
        model.optimizer.zero_grad()
        loss = model.Loss(output,target)   
        loss.backward()
        model.optimizer.step()
        loss_cum.append(loss.item())
        _, arg_max_out = torch.max(output.data.cpu(), 1)
        Acc += arg_max_out.long().eq(target.data.cpu().long()).sum()
    
    print("Loss: %0.3f | Acc: %0.2f"%(np.array(loss_cum).mean(), float(Acc*100)/len(data_loader.dataset)))

def val(data_loader, model, epoch):
    model.eval()
    loss_cum = []
    Acc = 0
    for batch_idx, (data,target) in tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc="[VAL] Epoch: {}".format(epoch)):
        data = data.to(device).requires_grad_(False)
        target = target.type(torch.LongTensor).squeeze(1).to(device).requires_grad_(False)

        output = model(data)
        loss = model.Loss(output,target)   
        loss_cum.append(loss.item())
        _, arg_max_out = torch.max(output.data.cpu(), 1)
        Acc += arg_max_out.long().eq(target.data.cpu().long()).sum()
    
    print("Loss Val: %0.3f | Acc Val: %0.2f"%(np.array(loss_cum).mean(), float(Acc*100)/len(data_loader.dataset)))
 
def test(data_loader, model, epoch):
    model.eval() 
    file = open("convEmotions_Results.txt","w")
    for batch_idx, (data,target) in tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc="[TEST] Epoch: {}".format(epoch)):
        data = data.to(device).requires_grad_(False)
        output = model(data)
        _,prediction =torch.max(output.data.cpu(),1)
        for i in range(len(prediction)):
           filename = os.listdir('Emotions_test_crop')[(batch_size*batch_idx)+i]
           #print('file:',filename)
           res = prediction[i].item()
           file.write(filename + "," + str(res)+":\n" )
    file.close()        

if __name__=='__main__':
    epochs=100
    batch_size=25 
    TEST=True
    x_train, y_train, x_val, y_val = get_data()
    
    x_train = x_train[:, np.newaxis]
    x_val =  x_val[:, np.newaxis]
    
    tensor_x_train = torch.stack([torch.Tensor(i) for i in x_train]) # transform to torch tensors
    tensor_y_train = torch.stack([torch.Tensor(i) for i in y_train])
    
    train_dataset = utils.TensorDataset(tensor_x_train,tensor_y_train) # create your dataset
    train_dataloader = utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # create your dataloader
    
    tensor_x_val = torch.stack([torch.Tensor(i) for i in x_val]) # transform to torch tensors
    tensor_y_val = torch.stack([torch.Tensor(i) for i in y_val])
    
    val_dataset = utils.TensorDataset(tensor_x_val,tensor_y_val) # create your dataset
    val_dataloader = utils.DataLoader(val_dataset, batch_size=batch_size,shuffle=False) # create your dataloader
    
    model = Net()
    model.to(device)
    model.training_params()
    print_network(model, 'Conv network + fc 2 layer non-linearity')    
    #Exploring model
    data, _ = next(iter(train_dataloader))
    _ = model(data.to(device).requires_grad_(False), verbose=True)
    for epoch in range(epochs): 
        train(train_dataloader, model, epoch)
        val(val_dataloader, model, epoch)

    if TEST: 
        x_test = get_test_data()
        print('xtest',x_test.shape)
        y_test = np.zeros((x_test.shape[0]))
        x_test = x_test[:,np.newaxis]
        y_test = y_test[:,np.newaxis]
        tensor_x_test = torch.stack([torch.Tensor(i) for i in x_test])
        tensor_y_test = torch.stack([torch.Tensor(i) for i in y_test])
        test_dataset = utils.TensorDataset(tensor_x_test,tensor_y_test) # create your dataset
        test_dataloader = utils.DataLoader(test_dataset, batch_size=batch_size, shuffle=False) # create your dataloader
        test(test_dataloader, model, epoch)
        print("TEST Results printed.")

