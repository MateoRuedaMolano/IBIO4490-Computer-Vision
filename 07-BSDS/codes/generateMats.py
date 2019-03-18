#!/home/afromero/anaconda3/envs/python2/bin/ipython
"""
Created on Mon Mar  4 19:34:53 2019

@author: mates
"""


def saveImages(folder):
    from main import check_dataset
    import scipy.io as sio
    import numpy

        #download dataset and unzip it
    check_dataset()
    import os
    filepath = 'BSR'+'/'+'BSDS500'+'/'+'data'+'/'+'images'+ '/'+ folder
    mylist = os.listdir(filepath)
    #for fichier in mylist[:]: # filelist[:] makes a copy of filelist.
        #if not(fichier.endswith(".jpg")):
            #mylist.remove(fichier)
    from Segment import segmentByClustering
    k = [20,15,10,5,3]
    seg = [0]*len(k)
    segs = numpy.array(seg, dtype=numpy.object)
    
    for i in mylist:
        l= 0
        for j in k:
          s = segmentByClustering(i,"lab","gmm",j)
          s.transpose()
          s=s.astype(numpy.uint16)
          segs[l]= s
          segs= segs+1
          l+=1
        name= i.split('/')
        name= name[0].split('.')
        cell_mat = name[0] + '.mat'
        #name_kmeans= folder_k + '/' + name[0] + '.mat'
        sio.savemat(cell_mat,{'segs':segs})
        #sio.savemat(name_kmeans,{'segs':Seg_kmeans})
        print (i)

saveImages('test') 
    ##jaccard_final = np.mean(jaccards)
    #      jaccards[i,j] = segmentByClustering(mylist[j],"lab",methods[i],4)
    #
    #jaccard_methods = np.zeros((len(methods)))
    #jaccard_methods[0] = np.mean(jaccards[0,:])
    #jaccard_methods[1] = np.mean(jaccards[1,:])
    #jaccard_methods[2] = np.mean(jaccards[2,:])
    #jaccard_methods[3] = np.mean(jaccards[3,:])
    #
    #x = np.arange(0,4,1)
    #plt.figure()
    #plt.plot(x,jaccard_methods, 'ro')
    #plt.show()  
    #
    #print(str(np.amax(jaccard_methods)))
    #print(str(np.argmax(jaccard_methods)))
    #      
    
