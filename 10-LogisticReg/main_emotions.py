
# read kaggle facial expression recognition challenge dataset (fer2013.csv)
# https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge
import numpy as np
old_settings = np.seterr(all='print')
import matplotlib.pyplot as plt
import pickle
import os
import cv2

#stable  version of softmax
def softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator/denominator
    return softmax

def get_data():
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
    
    x_val = x_train[0:8000,:,:]
    y_val = y_train[0:8000]
    x_train = x_train[8000:-1,:,:]
    y_train = y_train[8000:-1]
    
    print(x_val.shape[0], 'val samples')
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # plt.hist(y_train, max(y_train)+1); plt.show()
    return x_train, y_train, x_val, y_val, x_test, y_test

class Model():
    def __init__(self):
        self.lr = 0.00001 # Change if you want       
        self.W = np.random.randn(48*48, 7)
        self.b = np.random.randn(1,7)

    def forward(self, image):
        image = image.reshape(image.shape[0], -1)
        dot = (np.dot(image, self.W) + self.b)+10
        self.y_given_x = np.apply_along_axis(softmax,1,dot)
        return self.y_given_x
    
    #def cross_entropy(X,y):
    def compute_loss(self, pred, gt):
        nb_classes = 7
        #one hot coding annotations
        y_gt = gt.astype(np.int64)
        targets_gt = np.array([[y_gt]]).reshape(-1)
        one_hot_gt = np.eye(nb_classes)[targets_gt]
        #Computing loss
        log_likelihood = -np.multiply(np.log(pred),one_hot_gt)
        loss = np.sum(log_likelihood)/gt.shape[0]
        return loss

    def compute_gradient(self, image, pred, gt):
        image = image.reshape(image.shape[0], -1)
        nb_classes = 7
        #one hot coding annotations
        y_gt = gt.astype(np.int64)
        targets_gt = np.array([[y_gt]]).reshape(-1)
        one_hot_gt = np.eye(nb_classes)[targets_gt]
        image = image.reshape(image.shape[0], -1)
        W_grad = np.dot(image.T, pred-one_hot_gt)/image.shape[0]
        self.W -= W_grad*self.lr
        b_grad = np.sum(pred-gt)/image.shape[0]
        self.b -= b_grad*self.lr

def train(model):
    pkl_filename = "pickle_model_emotions.pkl"  
    x_train, y_train, x_val, y_val, _ , _ = get_data()
    batch_size = 100# Change if you want
    epochs = 10 # Change if you want
    aux = []
    losses = []
    losses_val = []
    for i in range(epochs):
        loss = []
        for j in range(0,x_train.shape[0], batch_size):
            _x_train = x_train[j:j+batch_size]
            _y_train = y_train[j:j+batch_size]
            out = model.forward(_x_train)
            out = out.astype(np.float64)
            loss.append(model.compute_loss(out, _y_train))
            model.compute_gradient(_x_train, out, _y_train)
        out = model.forward(x_val)                
        loss_val = model.compute_loss(out, y_val)
        print('Epoch {:6d}: {:.5f} | test: {:.5f}'.format(i, np.array(loss).mean(), loss_val))
        losses.append(np.array(loss).mean())
        losses_val.append(loss_val)
        aux.append(i)  
        with open(pkl_filename, 'wb') as file: 
             pickle.dump(model, file)
    #with open(pkl_filename, 'wb') as file: 
            #pickle.dump(model, file)
        
    plot(losses,losses_val,aux)

def plot(loss,losses_test,epochs): # Add arguments
    x = epochs
    
    plt.plot(x, loss, label='train')
    plt.plot(x, losses_test, label='val')
    plt.legend()
    plt.xlabel("iterations(Epochs)")
    plt.ylabel("loss(error)")
    plt.savefig('figure.pdf') 

def test(model):
 
    _, _, _, _, x_test, y_test = get_data()
    y_score = model.forward(x_test)  
    prediction =np.argmax(y_score, axis=1)
    pred = np.transpose(np.array([prediction],dtype = np.float64))
    #F1
    from sklearn.metrics import f1_score
    f1 = f1_score(y_test, pred, average='macro')  
    print('F1 score: {0:0.2f}'.format(f1))
    #normalized ACA
    from sklearn.metrics import classification_report
    target_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    print(classification_report(y_test, pred, target_names=target_names))
    from sklearn.metrics import confusion_matrix
    conf = confusion_matrix(y_test, pred)
    print('confmat',conf)
    import seaborn as sn
    import pandas as pd
    import matplotlib.pyplot as plt
    array = conf
    df_cm = pd.DataFrame(array, index = [i for i in range(7)],
                                         columns = [i for i in range(7)])
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
    y_demo = [6,0,3,2,3,1,3,3,5,3,4]
    #y_demo = np.array(y_demo, 'float64')
    #y_demo = y_demo.reshape(y_demo.shape[0], 1)
    #x_demo = x_demo.reshape(x_demo.shape[0], 48, 48)
    y_score = model.forward(x_demo)  
    prediction =np.argmax(y_score, axis=1)
    y_score = prediction.tolist()

    #y_score = np.transpose(np.array([prediction],dtype = np.float64))
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
    from sklearn.metrics import classification_report
    conf = confusion_matrix(y_demo, y_score)
    print('confmat',conf)
    import seaborn as sn
    import pandas as pd
    import matplotlib.pyplot as plt
    array = conf
    df_cm = pd.DataFrame(array, index = [i for i in range(7)],
                                         columns = [i for i in range(7)])
    plt.figure(figsize = (70,70))
    sn.heatmap(df_cm, annot=True)
    plt.show()
    target_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    print(classification_report(y_demo, y_score, target_names=target_names))
    #cont1 = 0
    #cont2 = 0
    #for i in range(conf.shape[1]):
     #   cont1 = cont1 + conf[i,i]/sum(conf[:,i])
     #   cont2 = cont2 +1
    #ACA = cont1/cont2    
    print('confmat',conf)
    #print ('ACA: {0:0.3f}'.format(ACA))
       
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
        
        
        with open("pickle_model_emotions.pkl", 'rb') as file:  
             pickle_model = pickle.load(file)

            
        test(pickle_model)  
       
       
       
       
    elif args.demo == True:
        
       with open("pickle_model_emotions.pkl", 'rb') as file:  
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
    #    with open("pickle_model_emotions.pkl", 'rb') as file:  
    #             pickle_model = pickle.load(file)
        test(model)   
    #test(model,1)
    
    

# Sources:
#http://deeplearning.net/tutorial/logreg.html   
#https://gluon.mxnet.io/chapter02_supervised-learning/softmax-regression-scratch.html 
#https://medium.com/data-science-bootcamp/understand-the-softmax-function-in-minutes-f3a59641e86d
#http://neuralnetworksanddeeplearning.com/chap3.html
#https://gombru.github.io/2018/05/23/cross_entropy_loss/
#https://medium.com/tebs-lab/how-to-classify-mnist-digits-with-different-neural-network-architectures-39c75a0f03e3
#https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a
#https://towardsdatascience.com/demystifying-cross-entropy-e80e3ad54a8
#https://rstudio-pubs-static.s3.amazonaws.com/337306_79a7966fad184532ab3ad66b322fe96e.html
#https://gombru.github.io/2018/05/23/cross_entropy_loss/