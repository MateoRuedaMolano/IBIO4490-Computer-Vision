import numpy as np 

def traduce_labels(str_label):
    #1=Atelectasis; 2=Cardiomegaly; 3=Effusion; 4=Infiltration; 5=Mass; 6=Nodule; 7=Pneumonia; 8=Pneumothorax;
    #9=Consolidation; 10=Edema; 11=Emphysema; 12=Fibrosis; 13=Pleural_Thickening; 14=Hernia
    
    labels= str_label.split('|')
    lb= np.zeros(14,dtype=int)
    #lb=np.ones(14,dtype=int)*-1

    for i in range(len(labels)):
        if labels[i]=='Atelectasis':
            lb[0]=1
        elif labels[i]=='Cardiomegaly':
            lb[1]=1
        elif labels[i]=='Effusion':
            lb[2]=1
        elif labels[i]=='Infiltration':
            lb[3]=1
        elif labels[i]=='Mass':
            lb[4]=1
        elif labels[i]=='Nodule':
            lb[5]=1
        elif labels[i]=='Pneumonia':
            lb[6]=1
        elif labels[i]=='Pneumothorax':
            lb[7]=1
        elif labels[i]=='Consolidation':
            lb[8]=1
        elif labels[i]=='Edema':
            lb[9]=1
        elif labels[i]=='Emphysema':
            lb[10]=1
        elif labels[i]=='Fibrosis':
            lb[11]=1
        elif labels[i]=='Pleural_Thickening':
            lb[12]=1
        elif labels[i]=='Hernia':
            lb[13]=1
    return lb


def get_labels_multitarget(v_paths):
    import csv
    import numpy as np
    
    labels=[]
    lab= []
    img=[]
    with open('Data_Entry_2017.csv', 'r') as csvFile:
            reader = csv.reader(csvFile)
            i=-1
            for row in reader:
                if i==-1:
                    i +=1
                else:
                    img.append(row[0])
                    lab.append(row[1])
    csvFile.close()
    
    img= np.array(img)
    j=0
    for path in v_paths:
        print (j)
        #print (path)
        ind= np.where(img==path.split('/')[1])
        #print (lab[ind[0][0]])
        labels.append(traduce_labels(lab[ind[0][0]]))
        j+=1
    return labels

