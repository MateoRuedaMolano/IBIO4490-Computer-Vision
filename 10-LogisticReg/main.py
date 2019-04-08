
# read kaggle facial expression recognition challenge dataset (fer2013.csv)
# https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge
import numpy as np
old_settings = np.seterr(all='print')
import matplotlib.pyplot as plt
import pickle    
import cv2
import os

def sigmoid(x):
    return 1/(1+np.exp(-x))

def get_data():
    # angry, disgust, fear, happy, sad, surprise, neutral
    # get the data
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
        emotion = 1 if int(emotion)==3 else 0 # Only for happiness
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

    #reshape to make images and labels
    x_train = x_train.reshape(x_train.shape[0], 48, 48)
    x_test = x_test.reshape(x_test.shape[0], 48, 48)
    y_train = y_train.reshape(y_train.shape[0], 1)
    y_test = y_test.reshape(y_test.shape[0], 1)
    
    x_val = x_train[0:8000,:,:]
    y_val = y_train[0:8000]
    x_train = x_train[8000:-1,:,:]
    y_train = y_train[8000:-1]
    
    print(x_train.shape[0], 'train samples')
    print(x_val.shape[0], 'val samples')
    print(x_test.shape[0], 'test samples')
    print(y_test)

    # plt.hist(y_train, max(y_train)+1); plt.show()

    return x_train, y_train, x_val, y_val, x_test, y_test

class Model():
    def __init__(self):
        params = 48*48 # image reshape
        out = 1 # smile label
        self.lr = 0.00001 # Change if you want
        self.W = np.random.randn(params, out) #Initialize W and b
        self.b = np.random.randn(out)

    def forward(self, image):
        image = image.reshape(image.shape[0], -1)
        out = np.dot(image, self.W) + self.b #Multiply image with W and b
        return out

    def compute_loss(self, pred, gt):
        #Get the loss 
        J = (-1/pred.shape[0]) * np.sum(np.multiply(gt, np.log(sigmoid(pred))) + np.multiply((1-gt), np.log(1 - sigmoid(pred))))
        return J

    def compute_gradient(self, image, pred, gt):
        image = image.reshape(image.shape[0], -1)
        W_grad = np.dot(image.T, pred-gt)/image.shape[0]
        self.W -= W_grad*self.lr

        b_grad = np.sum(pred-gt)/image.shape[0]
        self.b -= b_grad*self.lr

def train(model):
    pkl_filename = "pickle_model.pkl"  
    x_train, y_train, x_val, y_val, _ , _ = get_data()
    batch_size = 50 # Change if you want
    epochs = 40000 # Change if you want
    aux = []
    losses = []
    losses_val = []
    for i in range(epochs):
        loss = []
        for j in range(0,x_train.shape[0], batch_size):
            _x_train = x_train[j:j+batch_size]
            _y_train = y_train[j:j+batch_size]
            out = model.forward(_x_train)
            loss.append(model.compute_loss(out, _y_train))
            model.compute_gradient(_x_train, out, _y_train)
        #out, loss_test = test(model) 
        out = model.forward(x_val)                
        loss_val = model.compute_loss(out, y_val)
        print('Epoch {:6d}: train {:.5f} | val: {:.5f}'.format(i, np.array(loss).mean(),loss_val))
        losses.append(np.array(loss).mean())
        losses_val.append(loss_val)
        aux.append(i)  
        
    with open(pkl_filename, 'wb') as file: 
             pickle.dump(model, file)
        
    plot(losses,losses_val,aux)

def plot(loss,losses_test,epochs): # Add arguments
    # CODE HERE
    # Save a pdf figure with train and test losses
    
    #x = range(epochs)
    x = epochs
    
    plt.plot(x, loss, label='train')
    plt.plot(x, losses_test, label='val')
    plt.legend()
    plt.xlabel("iterations(Epochs)")
    plt.ylabel("loss(error)")
    plt.savefig('figure.pdf') 

def test(model):
    #_, _, x_test, y_test = get_data()
    # YOU CODE HERE
    # Show some qualitative results and the total accuracy for the whole test set
   
    _, _, _, _, x_test, y_test = get_data()
    y_score = model.forward(x_test)  
    y_score = sigmoid(y_score)
    #threshold, upper, lower = 0.5, 1, 0
    y_score[y_score >= 0.5] = 1
    y_score[y_score < 0.5] = 0
#    #y_score = np.where(y_score>threshold, upper, lower)
#    #PR curve, F1 and normalized ACA.
#    #PR
#    #from sklearn.metrics import average_precision_score
#    #average_precision = average_precision_score(y_test, y_score)  
#    print('Average precision-recall score: {0:0.2f}'.format(average_precision))                
#    #loss_test = model.compute_loss(out, y_test)         
#    from sklearn.metrics import precision_recall_curve
#    import matplotlib.pyplot as plt
#    from sklearn.utils.fixes import signature
#    
#    precision, recall, _ = precision_recall_curve(y_test, y_score)
#    step_kwargs = ({'step': 'post'}
#                   if 'step' in signature(plt.fill_between).parameters
#                   else {})
#    plt.step(recall, precision, color='b', alpha=0.2,
#             where='post')
#    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
#    
#    plt.xlabel('Recall')
#    plt.ylabel('Precision')
#    plt.ylim([0.0, 1.05])
#    plt.xlim([0.0, 1.0])
#    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
#    plt.show()
    #F1 
    from sklearn.metrics import f1_score
    f1 = f1_score(y_test, y_score, average='macro')  
    print('F1 score: {0:0.2f}'.format(f1))
   
    #normalized ACA

    from sklearn.metrics import confusion_matrix
    conf = confusion_matrix(y_test, y_score)
    print('confmat',conf)
    import seaborn as sn
    import pandas as pd
    import matplotlib.pyplot as plt
    array = conf
    df_cm = pd.DataFrame(array, index = [i for i in range(2)],
                                         columns = [i for i in range(2)])
    plt.figure(figsize = (70,70))
    sn.heatmap(df_cm, annot=True)
    plt.show()
    cont1 = 0
    cont2 = 0
    for i in range(conf.shape[1]):
        cont1 = cont1 + conf[i,i]/sum(conf[:,i])
        cont2 = cont2 +1
    ACA = cont1/cont2    
    print ('ACA: {0:0.3f}'.format(ACA))


def load_images_from_folder(folder):
    from skimage import color
    images = np.zeros((11,48,48))
    for i in range(0,10):
        filename = os.listdir('images')[i]
        img = cv2.imread(os.path.join(folder,filename))
        img = cv2.resize(img, dsize=(48, 48), interpolation=cv2.INTER_CUBIC)
        img = color.rgb2gray(img)
        images[i,:,:]= img
    return images

def demo(model):
    x_demo = load_images_from_folder('images')
    #print(x_demo.type, x_demo.shape)
    #x_demo = np.array(x_demo)
    #x_demo = x_demo.astype('float64')
    y_demo = [0,0,1,0,1,0,1,1,0,1,0]
    y_demo = np.array(y_demo, 'float64')
    y_demo = y_demo.reshape(y_demo.shape[0], 1)
    #x_demo = x_demo.reshape(x_demo.shape[0], 48, 48)
    y_score = model.forward(x_demo)  
    y_score = sigmoid(y_score)
    #threshold, upper, lower = 0.5, 1, 0
    y_score[y_score >= 0.5] = 1
    y_score[y_score < 0.5] = 0
    
#    #y_score = np.where(y_score>threshold, upper, lower)
#    #PR curve, F1 and normalized ACA.
#    #PR
#    #from sklearn.metrics import average_precision_score
#    #average_precision = average_precision_score(y_test, y_score)  
#    print('Average precision-recall score: {0:0.2f}'.format(average_precision))                
#    #loss_test = model.compute_loss(out, y_test)         
#    from sklearn.metrics import precision_recall_curve
#    import matplotlib.pyplot as plt
#    from sklearn.utils.fixes import signature
#    
#    precision, recall, _ = precision_recall_curve(y_test, y_score)
#    step_kwargs = ({'step': 'post'}
#                   if 'step' in signature(plt.fill_between).parameters
#                   else {})
#    plt.step(recall, precision, color='b', alpha=0.2,
#             where='post')
#    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
#    
#    plt.xlabel('Recall')
#    plt.ylabel('Precision')
#    plt.ylim([0.0, 1.05])
#    plt.xlim([0.0, 1.0])
#    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
#    plt.show()
    #F1 
    from sklearn.metrics import f1_score
    f1 = f1_score(y_demo, y_score, average='macro')  
    print('F1 score: {0:0.2f}'.format(f1))
   
    #normalized ACA
    from sklearn.metrics import confusion_matrix
    conf = confusion_matrix(y_demo, y_score)
    print('confmat',conf)
    import seaborn as sn
    import pandas as pd
    import matplotlib.pyplot as plt
    array = conf
    df_cm = pd.DataFrame(array, index = [i for i in range(2)],
                                         columns = [i for i in range(2)])
    plt.figure(figsize = (70,70))
    sn.heatmap(df_cm, annot=True)
    plt.show()    
    cont1 = 0
    cont2 = 0
    for i in range(conf.shape[1]):
        cont1 = cont1 + conf[i,i]/sum(conf[:,i])
        cont2 = cont2 +1
    ACA = cont1/cont2    
    print ('ACA: {0:0.3f}'.format(ACA))
       
if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()

    #parser.add_argument('--validation') # If you use more please add them to this list.
    parser.add_argument('--test', action='store_true', default= False)
    parser.add_argument('--demo', action='store_true', default= False)
    args = parser.parse_args()
    
    if args.test == True:
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
            print('Unzipping done.') 
            
            #untar
            import tarfile
            tar = tarfile.open('fer2013.tar.gz', "r:gz")
            tar.extractall()
            tar.close()

       with open("pickle_model.pkl", 'rb') as file:  
             pickle_model = pickle.load(file)
       test(pickle_model)   
       
       
    elif args.demo == True:
       with open("pickle_model.pkl", 'rb') as file:  
             pickle_model = pickle.load(file)
       demo(pickle_model)
       
       
    else:
        
        #import os     
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
            print('Unzipping done.') 
            
            #untar
            import tarfile
            tar = tarfile.open('fer2013.tar.gz', "r:gz")
            tar.extractall()
            tar.close()
            
        model = Model()
        print("Training model, this may take a while")
        train(model)
    #    #plot(losses,losses_val,aux)
    #    with open("pickle_model.pkl", 'rb') as file:  
    #             pickle_model = pickle.load(file)
        test(model)   
    #test(model,1)
    
    
