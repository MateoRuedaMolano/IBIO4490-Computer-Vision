#!/home/afromero/anaconda3/envs/python2/bin/ipython
"""
Created on Mon Mar  4 17:01:05 2019

@author: msrueda10 & ssleal10
"""

def segmentByClustering(rgbImage, colorSpace, clusteringMethod, numberOfClusters):
    #import dataset function from main
    #Import function check_dataset
    from main import check_dataset
    #download dataset and unzip it
    check_dataset()
    #Import libraries 
    import matplotlib.pyplot as plt
    #import os
    import os
    #import io, color
    from skimage import io, color
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    from sklearn.cluster import AgglomerativeClustering
    from skimage.feature import peak_local_max
    from skimage.morphology import watershed
    from scipy import ndimage
    import cv2
    from main import imshow
    from scipy import misc
    from skimage.transform import resize 

    #get the current cwd
    #cwd = os.getcwd()
    #get the image path 
    #img_file = os.path.join('BSDS_small',rgbImage)
    img_file = 'BSR'+'/'+'BSDS500'+'/'+'data'+'/'+'images'+ '/'+ 'test'+ '/'+ rgbImage
    
    #show the groundtruth segmentation 
    #groundtruth(img_file)
    #show the groundtruth edges
#    from main import groundtruth_edges
#    groundtruth_edges(img_file)
    #read the image (rgb)
    imag= io.imread(img_file)
    from skimage.filters import gaussian
    rgb = gaussian(imag,sigma=1.5,multichannel=True)

    #Check all possible color spaces
    
    #HSV color space
    if (colorSpace == 'hsv'):
           #convert rgb2hsv
           hsv = color.rgb2hsv(rgb)
           hsv = cv2.normalize(hsv, np.zeros((hsv.shape), dtype=np.uint8), alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
           #return hsv
            #plot the image in hsv
           #plt.imshow(hsv)
           #plt.show()
           #shape of the image
           (fils, cols, channels) = hsv.shape
           
           #Check all clustering Methods
           
           #Kmeans
           if (clusteringMethod == 'kmeans'):
               #reshape the image for kmeans
               X = hsv.reshape(fils*cols, 3)
               #Convert data to float 
               #X.astype(float)
               #kmeans with numberofClusters as a parameter
               kmeans = KMeans(n_clusters=numberOfClusters, random_state=0).fit(X)
               #obtaining the segmented image
               segmented_img = kmeans.cluster_centers_[kmeans.labels_]
               #reshape the image to the original size
               segmented_img = kmeans.labels_
               #reshape labels to obtain 2d image
               segmented_img = segmented_img.reshape((fils,cols))
               #show the segmentation
               #imshow(rgb,segmented_img,title = rgbImage)
               
               
           elif (clusteringMethod == 'gmm'):
               X = hsv.reshape(fils*cols, 3)
               gmm = GaussianMixture(n_components=numberOfClusters)
               gmm = gmm.fit(X)
               cluster = gmm.predict(X)
               cluster = cluster.reshape(fils,cols)
               #cluster = cluster.astype(np.uint8)
               segmented_img = cluster
               #imshow(rgb,segmented_img,title = rgbImage)
               
           elif (clusteringMethod == 'hierarchical'):
               hsv2 = resize(hsv, (int(hsv.shape[0] / 4), int(hsv.shape[1] / 4)),
                       anti_aliasing=True)
               hsv2 = cv2.normalize(hsv2, np.zeros((hsv2.shape), dtype=np.uint8), alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
               (fils, cols, channels) = hsv2.shape
               X = hsv2.reshape(fils*cols, 3)
               cluster = AgglomerativeClustering(n_clusters=numberOfClusters, affinity='manhattan', linkage='complete')
               cluster.fit_predict(X) 
               labels = cluster.labels_
               segmented_img = labels.reshape(fils,cols)
               #segmented_img = segmented_img.astype(np.uint8)
               segmented_img= misc.imresize(segmented_img,(hsv.shape[0],hsv.shape[1]),interp='bicubic')
               #imshow(rgb,segmented_img,title = rgbImage) 
               
           elif (clusteringMethod == 'watershed'):
               hsv=np.mean(hsv,axis=2)
               local_maxima = peak_local_max(-1*hsv, min_distance=15,indices=False,num_peaks=numberOfClusters)
               marks=ndimage.label(local_maxima)[0]
               segmented_img=watershed(hsv,marks)
               segmented_img =segmented_img
               #imshow(rgb,segmented_img, title= rgbImage) 
               
               
               
               
                        
               
               
        #Check lab color space
    elif (colorSpace == 'lab'):
           #convert rgb image to lab
           lab = color.rgb2lab(rgb)
           lab = cv2.normalize(lab, np.zeros((lab.shape), dtype=np.uint8), alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
           #show of the image in lab color space
           #shape of the image
           (fils, cols, channels) = lab.shape
           
           #Check all possible clustering methods
           #Kmeans
           if (clusteringMethod == 'kmeans'):
               #reshape the image for kmeans
               X = lab.reshape(fils*cols, 3)
               #Convert data to float 
               #X.astype(float)
               #kmeans with numberofClusters as a parameter
               kmeans = KMeans(n_clusters=numberOfClusters, random_state=0).fit(X)
               #obtaining the segmented image (labels)
               segmented_img = kmeans.labels_
               #reshape labels to obtain 2d image
               segmented_img = segmented_img.reshape((fils,cols))
               #convert it to uint8
               #segmented_img = segmented_img.astype(np.uint8)
               #show the segmentation
               #imshow(rgb,segmented_img,title = rgbImage)
               
               
           elif (clusteringMethod == 'gmm'):
               X = lab.reshape(fils*cols, 3)
               gmm = GaussianMixture(n_components=numberOfClusters)
               gmm = gmm.fit(X)
               cluster = gmm.predict(X)
               cluster = cluster.reshape(fils,cols)
               #cluster = cluster.astype(np.uint8)
               segmented_img = cluster
               #imshow(rgb,segmented_img,title = rgbImage)
               
           elif  (clusteringMethod == 'hierarchical'):
               lab2 = resize(lab, (int(lab.shape[0] / 4), int(lab.shape[1] / 4)),
                       anti_aliasing=True)
               lab2 = cv2.normalize(lab2, np.zeros((lab2.shape), dtype=np.uint8), alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
               (fils, cols, channels) = lab2.shape
               X = lab2.reshape(fils*cols, 3)
               cluster = AgglomerativeClustering(n_clusters=numberOfClusters, affinity='euclidean', linkage='ward')
               cluster.fit_predict(X) 
               labels = cluster.labels_
               segmented_img = labels.reshape(fils,cols)
               #segmented_img = segmented_img.astype(np.uint8)
               segmented_img= misc.imresize(segmented_img,(rgb.shape[0],rgb.shape[1]),interp='bicubic')
               #imshow(rgb,segmented_img,title = rgbImage) 
               
           elif  (clusteringMethod == 'watershed'):
               lab=np.mean(lab,axis=2)
               local_maxima = peak_local_max(-1*lab, min_distance=15,indices=False,num_peaks=numberOfClusters)
               marks=ndimage.label(local_maxima)[0]
               segmented_img=watershed(lab,marks)
               segmented_img =segmented_img
               #imshow(rgb,segmented_img, title= rgbImage) 
               
   

        
           
        #Check rgb+xy color space
    elif (colorSpace == 'rgb_xy'):
           #size of the image
           (fils, cols, channels) = rgb.shape
           pos_x = np.ones((fils,cols),dtype='uint8')
           pos_y = np.ones((fils,cols),dtype='uint8')
           for i in range(0,fils):
               for j in range(0,cols):
                   pos_x[i,:] = i
                   pos_y[:,j] = j
           pos_x= cv2.normalize(pos_x, np.zeros((pos_x.shape), dtype=np.uint8), alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
           pos_y= cv2.normalize(pos_y, np.zeros((pos_y.shape), dtype=np.uint8), alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
           rgb_xy = np.dstack((rgb,pos_x,pos_y))
           
           if (clusteringMethod == 'kmeans'):
               X = rgb_xy.reshape(fils*cols, 5)
               #X.astype(float)
               kmeans = KMeans(n_clusters=numberOfClusters, random_state=0).fit(X)
               segmented_img = kmeans.labels_
               segmented_img = segmented_img.reshape((fils,cols))
               #segmented_img = segmented_img.astype(np.uint8)
               #imshow(rgb,segmented_img,title = rgbImage)
               
           elif (clusteringMethod == 'gmm'):
               X = rgb_xy.reshape(fils*cols, 5)
               gmm = GaussianMixture(n_components=numberOfClusters)
               gmm = gmm.fit(X)
               cluster = gmm.predict(X)
               cluster = cluster.reshape(fils,cols)
               #cluster = cluster.astype(np.uint8)
               segmented_img = cluster
               #imshow(rgb,segmented_img,title = rgbImage)
               
           elif  (clusteringMethod == 'hierarchical'):
               #X = rgb_xy.reshape(fils*cols, 5)
               rgb2 = resize(rgb, (int(rgb.shape[0] / 4), int(rgb.shape[1] / 4)),
                       anti_aliasing=True)
               rgb2 = cv2.normalize(rgb2, np.zeros((rgb2.shape), dtype=np.uint8), alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
               (fils, cols, channels) = rgb2.shape
               pos_x = np.ones((fils,cols),dtype='uint8')
               pos_y = np.ones((fils,cols),dtype='uint8')
               for i in range(0,fils):
                   for j in range(0,cols):
                       pos_x[i,:] = i
                       pos_y[:,j] = j
               rgb_xy = np.dstack((rgb2,pos_x,pos_y))
               X = rgb_xy.reshape(fils*cols, 5)
               cluster = AgglomerativeClustering(n_clusters=numberOfClusters, affinity='euclidean', linkage='ward')
               cluster.fit_predict(X) 
               labels = cluster.labels_
               segmented_img = labels.reshape(fils,cols)
               #segmented_img = segmented_img.astype(np.uint8)
               segmented_img= misc.imresize(segmented_img,(rgb.shape[0],rgb.shape[1]),interp='nearest')
               #imshow(rgb,segmented_img,title = rgbImage)
               
           elif  (clusteringMethod == 'watershed'):
               rgb_xy= np.mean(rgb,axis=2)
               local_maxima = peak_local_max(-1*rgb_xy, min_distance=15,indices=False,num_peaks=numberOfClusters)
               marks=ndimage.label(local_maxima)[0]
               segmented_img=watershed(rgb_xy,marks)
               segmented_img =segmented_img
               #imshow(rgb,segmented_img, title= rgbImage)                      
               
               
               
               
           

    elif (colorSpace == 'lab_xy'):
           #size of the image
           lab = color.rgb2lab(rgb)
           lab2 = cv2.normalize(lab, np.zeros((lab.shape), dtype=np.uint8), alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
           (fils, cols, channels) = lab.shape
           pos_x = np.ones((fils,cols))#dtype='uint8')
           pos_y = np.ones((fils,cols))#dtype='uint8')
           for i in range(0,fils):
               for j in range(0,cols):
                   pos_x[i,:] = i
                   pos_y[:,j] = j
           pos_x= cv2.normalize(pos_x, np.zeros((pos_x.shape), dtype=np.uint8), alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
           pos_y= cv2.normalize(pos_y, np.zeros((pos_y.shape), dtype=np.uint8), alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
           lab_xy = np.dstack((lab,pos_x,pos_y))
           
           
           if (clusteringMethod =='kmeans'):
               X = lab_xy.reshape(fils*cols, 5)
               #X.astype(float)
               kmeans = KMeans(n_clusters=numberOfClusters).fit(X)
               segmented_img = kmeans.labels_
               segmented_img = segmented_img.reshape((fils,cols))
               #segmented_img = segmented_img.astype(np.uint8)
               #imshow(rgb,segmented_img,title = rgbImage)
               
           elif  (clusteringMethod == 'gmm'):
               X = lab_xy.reshape(fils*cols, 5)
               gmm = GaussianMixture(n_components=numberOfClusters)
               gmm = gmm.fit(X)
               cluster = gmm.predict(X)
               cluster = cluster.reshape(fils,cols)
               #cluster = cluster.astype(np.uint8)
               segmented_img = cluster
               #imshow(rgb,segmented_img,title = rgbImage)
               
           elif  (clusteringMethod == 'hierarchical'):
               lab2 = resize(lab, (int(lab.shape[0] / 4), int(lab.shape[1] / 4)),
                       anti_aliasing=True)
               lab2 = cv2.normalize(lab2, np.zeros((lab2.shape), dtype=np.uint8), alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
               (fils, cols, channels) = lab2.shape
               pos_x = np.ones((fils,cols),dtype='uint8')
               pos_y = np.ones((fils,cols),dtype='uint8')
               for i in range(0,fils):
                   for j in range(0,cols):
                       pos_x[i,:] = i
                       pos_y[:,j] = j
               lab_xy = np.dstack((lab2,pos_x,pos_y))
               X = lab_xy.reshape(fils*cols, 5)
               cluster = AgglomerativeClustering(n_clusters=numberOfClusters, affinity='euclidean', linkage='ward')
               cluster.fit_predict(X) 
               labels = cluster.labels_
               segmented_img = labels.reshape(fils,cols)
               #segmented_img = segmented_img.astype(np.uint8)
               segmented_img= misc.imresize(segmented_img,(lab.shape[0],lab.shape[1]),interp='bicubic')
               #imshow(rgb,segmented_img,title = rgbImage) 
           
           elif  (clusteringMethod == 'watershed'):
               lab_xy=np.mean(lab,axis=2)
               local_maxima = peak_local_max(-1*lab_xy, min_distance=15,indices=False,num_peaks=numberOfClusters)
               marks=ndimage.label(local_maxima)[0]
               segmented_img=watershed(lab_xy,marks)
               segmented_img =segmented_img
               #imshow(rgb,segmented_img, title= rgbImage)           
           
            
            
            
            
            
            
           #Check if the color space is hsv
    elif (colorSpace == 'hsv_xy'):
           #size of the image
           hsv = color.rgb2hsv(rgb)
           hsv = cv2.normalize(hsv, np.zeros((hsv.shape), dtype=np.uint8), alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
           (fils, cols, channels) = hsv.shape
           pos_x = np.ones((fils,cols),dtype='uint8')
           pos_y = np.ones((fils,cols),dtype='uint8')
           for i in range(0,fils):
               for j in range(0,cols):
                   pos_x[i,:] = i
                   pos_y[:,j] = j
           pos_x= cv2.normalize(pos_x, np.zeros((pos_x.shape), dtype=np.uint8), alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
           pos_y= cv2.normalize(pos_y, np.zeros((pos_y.shape), dtype=np.uint8), alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
           hsv_xy = np.dstack((hsv,pos_x,pos_y)) 
           
           
           if (clusteringMethod == 'kmeans'):
               X = hsv_xy.reshape(fils*cols, 5)
               #X.astype(float)
               kmeans = KMeans(n_clusters=numberOfClusters, random_state=0).fit(X)
               segmented_img = kmeans.labels_
               segmented_img = segmented_img.reshape((fils,cols))
               #segmented_img = segmented_img.astype(np.uint8)
               #imshow(rgb,segmented_img,title = rgbImage) 

           elif (clusteringMethod == 'gmm'):
               X = hsv_xy.reshape(fils*cols, 5)
               gmm = GaussianMixture(n_components=numberOfClusters)
               gmm = gmm.fit(X)
               cluster = gmm.predict(X)
               cluster = cluster.reshape(fils,cols)
               #cluster = cluster.astype(np.uint8)
               segmented_img = cluster
               #imshow(rgb,segmented_img,title = rgbImage) 
               
           elif (clusteringMethod == 'hierarchical'):
               hsv2 = resize(hsv, (int(hsv.shape[0] / 4), int(hsv.shape[1] / 4)),
                       anti_aliasing=True)
               hsv2 = cv2.normalize(hsv2, np.zeros((hsv2.shape), dtype=np.uint8), alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
               (fils, cols, channels) = hsv2.shape
               pos_x = np.ones((fils,cols),dtype='uint8')
               pos_y = np.ones((fils,cols),dtype='uint8')
               for i in range(0,fils):
                   for j in range(0,cols):
                       pos_x[i,:] = i
                       pos_y[:,j] = j
               hsv_xy = np.dstack((hsv2,pos_x,pos_y))
               X = hsv_xy.reshape(fils*cols, 5)
               cluster = AgglomerativeClustering(n_clusters=numberOfClusters, affinity='euclidean', linkage='ward')
               cluster.fit_predict(X) 
               labels = cluster.labels_
               segmented_img = labels.reshape(fils,cols)
               #segmented_img = segmented_img.astype(np.uint8)
               #imshow(rgb,segmented_img,title = rgbImage)    
               
           elif (clusteringMethod == 'watershed'):
               #hsv_xy=hsv_xy[:,:,2]
               hsv_xy=np.mean(hsv,axis=2)
               local_maxima = peak_local_max(-1*hsv_xy, min_distance=15,indices=False,num_peaks=numberOfClusters)
               marks=ndimage.label(local_maxima)[0]
               segmented_img=watershed(hsv_xy,marks)
               segmented_img =segmented_img
               #imshow(rgb,segmented_img, title= rgbImage)

               
               
               
        
           
         
    elif (colorSpace == 'rgb'):
        (fils, cols, channels) = rgb.shape
          
        if (clusteringMethod == 'kmeans'):
               X = rgb.reshape(fils*cols, 3)
               #X.astype(float)
               kmeans = KMeans(n_clusters=numberOfClusters, random_state=0).fit(X)
               segmented_img = kmeans.labels_
               segmented_img = segmented_img.reshape((fils,cols))
               #segmented_img = segmented_img.astype(np.uint8)
               #imshow(rgb,segmented_img,title = rgbImage)
               
        elif (clusteringMethod == 'gmm'):
               X = rgb.reshape(fils*cols, 3)
               gmm = GaussianMixture(n_components=numberOfClusters)
               gmm = gmm.fit(X)
               cluster = gmm.predict(X)
               cluster = cluster.reshape(fils,cols)
               #cluster = cluster.astype(np.uint8)
               segmented_img = cluster
               #imshow(rgb,segmented_img,title = rgbImage)
               
        elif (clusteringMethod == 'hierarchical'):
               rgb2 = resize(rgb, (int(rgb.shape[0] / 4), int(rgb.shape[1] / 4)),
                       anti_aliasing=True)
               rgb2 = cv2.normalize(rgb2, np.zeros((rgb2.shape), dtype=np.uint8), alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
               (fils, cols, channels) = rgb2.shape
               X = rgb2.reshape(fils*cols, 3)
               cluster = AgglomerativeClustering(n_clusters=numberOfClusters, affinity='euclidean', linkage='ward')
               cluster.fit_predict(X) 
               labels = cluster.labels_
               segmented_img = labels.reshape(fils,cols)
               #segmented_img = segmented_img.astype(np.uint8)
               segmented_img= misc.imresize(segmented_img,(rgb.shape[0],rgb.shape[1]),interp='bicubic')
               #imshow(rgb,segmented_img,title = rgbImage)
               
        elif (clusteringMethod == 'watershed'):
               rgb= np.mean(rgb,axis=2)
               local_maxima = peak_local_max(-1*rgb, min_distance=15,indices=False,num_peaks=numberOfClusters)
               marks=ndimage.label(local_maxima)[0]
               segmented_img=watershed(rgb,marks)
               segmented_img =segmented_img
               #imshow(rgb,segmented_img, title= rgbImage)
               
               
    return segmented_img        
#    groundtruth(img_file)
#        # Get x-gradient in "sx"
#    sx = ndimage.sobel(segmented_img,axis=0,mode='constant')
#        # Get y-gradient in "sy"
#    sy = ndimage.sobel(segmented_img,axis=1,mode='constant')
#        # Get square root of sum of squares
#    sobel=np.hypot(sx,sy)
#        # Hopefully see some edges
#    if (clusteringMethod == 'hierarchical'):
#        for i in range(0,len(sobel)):
#          sobel[i] = sobel[i].astype(float)
#        plt.imshow(sobel,cmap=plt.cm.gray)
#        plt.show()
#    plt.imshow(sobel,cmap=plt.cm.gray)
#    plt.show()
#
#        
#        #import imageio
#        #plt.imshow(imageio.imread(img))
#        #plt.show()
#        
#        # Load .mat
#    boundaries = groundtruth_edges(img_file)
#    fil, col = sobel.shape
#    for i in range(0,fil):
#        for j in range(0, col):
#            if sobel[i,j]>0:
#               sobel[i,j] = 1
#    sobel = sobel.astype(np.uint8)
#    
#    jaccard_matrix = np.zeros((fil,col),dtype='uint8')
#    for i in range(0,fil):
#        for j in range(0, col):
#            jaccard_matrix[i,j]=sobel[i,j]+ boundaries[i,j]
#    #interseccion = np.count_nonzero(jaccard_matrix == 2)
#    #union = np.count_nonzero(jaccard_matrix == 1)
#    #jaccard = interseccion/union
#    #return jaccard
#    unique, counts = np.unique(jaccard_matrix, return_counts=True)
##    intersection = np.zeros((fil,col),dtype='uint8')
##    for i in range(0,fil):
##        for j in range(0, col):
##            intersection[i,j]=sobel[i,j]* boundaries[i,j]
#    #plt.imshow(intersection)
#    #plt.show
#    import scipy.io as sio
#    
#    
#    if (len(counts) == 3):
#      rta= counts[2]/counts[1]*100
#    else:
#        rta = 0
#    return rta
# 
             
     

           


