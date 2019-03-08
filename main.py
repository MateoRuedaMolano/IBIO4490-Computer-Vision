#!/usr/bin/python3
"""
Created on Mon Mar  4 16:49:58 2019

@author: mates
"""

def imshow(img, seg, title='Image'):
    import matplotlib.pyplot as plt
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    plt.imshow(seg, cmap=plt.get_cmap('rainbow'), alpha=0.5)
    cb = plt.colorbar()
    cb.set_ticks(range(seg.max()+1))
    plt.title(title)
    plt.axis('off')
    plt.show()
    
def groundtruth(img_file):
    import scipy.io as sio
    import imageio
    img = imageio.imread(img_file)
    gt=sio.loadmat(img_file.replace('jpg', 'mat'))
    segm=gt['groundTruth'][0,3][0][0]['Segmentation']
    imshow(img, segm, title='Groundtruth')
    
def groundtruth_edges(img_file):
    import scipy.io as sio
    import matplotlib.pyplot as plt
    gt=sio.loadmat(img_file.replace('jpg', 'mat'))
    edges=gt['groundTruth'][0,3][0][0]['Boundaries']
    plt.imshow(edges)
    plt.colorbar()
    plt.show()
    return edges
        
def check_dataset():
    import os
    import urllib.request
    import zipfile
        #URL to download the data
    url = "http://157.253.196.67/BSDS_small.zip"
        #Unify the data
        #Check if the directory has already been downloaded
    if os.path.exists("BSDS_small.zip") == False:
            #Download data if not
        urllib.request.urlretrieve(url, "BSDS_small.zip")
        print('Dataset downloaded')
    else:
        print('The database is in your cwd')
        #Unzip the folder
    if  os.path.exists("BSDS_small") == False:
        with zipfile.ZipFile("BSDS_small.zip","r") as unzip:
                #Extract all files
         unzip.extractall('BSDS_small')
                #close
         unzip.close()
         print('Dataset unzipped')
    else:
        print('You already have the data-set unzipped')

if __name__ == '__main__':
    import argparse
    from Segment import segmentByClustering # Change this line if your function has a different name
    parser = argparse.ArgumentParser()

    parser.add_argument('--color', type=str, default='rgb', choices=['rgb', 'lab', 'hsv', 'rgb_xy', 'lab_xy', 'hsv_xy']) # If you use more please add them to this list.
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--method', type=str, default='watershed', choices=['kmeans', 'gmm', 'hierarchical', 'watershed'])
    parser.add_argument('--img_file', type=str, required=True)
	
    opts = parser.parse_args()
    
    seg = segmentByClustering(opts.img_file, opts.color, opts.method, opts.k)
