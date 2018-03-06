from __future__ import print_function
from PIL import Image
import os
import numpy as np
import codecs
import tensorflow as tf
rootpath,labelfile,newlabelfile,labelidfile = None,None,None,None
namepath = ['original/','truelabel.csv','newlabel.csv','labelid.csv']
flag = 2

if flag == 1:
    rootpath,labelfile,newlabelfile,labelidfile = tuple(['data/'+i for i in namepath])

if flag == 2:
    rootpath,labelfile,newlabelfile,labelidfile = tuple(['chinesedata/'+i for i in namepath])

def classnum():
    l = codecs.open(labelidfile,'r','utf8').readlines()
    return len(l)-1

def get_path_label(rootpath,newlabelfile):
    '''
    给出相应目录下所以文件的文件名到标签值的一个映射字典{}
    rootpath:需要读取的目录
    return:字典{文件1:标签array,文件2:标签array,...}
    '''
    labelfile = codecs.open(newlabelfile,'r','utf8').readlines()
    labelfile = [i.replace('\n','') for i in labelfile]
    name2label = [i.split('<+++>') for i in labelfile]
    name2label =[[rootpath+i[0],i[1].split()] for i in name2label]
    resdict = {}
    for i in name2label:
        label = i[1]
        label = np.asarray([int(k) for k in label])
        path = i[0].encode('utf8')
        resdict[path]=label
    return resdict

def get_image(image_path,height,width):  
    """
    从给定路径中读取图片，返回的是numpy.ndarray
    image_path:string, height:图像像素高度 width:图像像素宽度
    return:numpy.ndarray的图片tensor 
    """ 
    im = Image.open(image_path).convert('L')
    b = reshape(im,height,width)
    return b

def reshape(im,height,width):
    '''
    resize
    im:PIL读取图片后的Image对象
    '''
    b = im.resize((width,height),Image.BILINEAR)
    b = np.reshape(b,[b.size[1],b.size[0],1])
    return b

def normal(data,height,width):
    '''
    归一化
    '''
    data = data.astype(np.float32)
    mean = np.sum(data)/(height*width)
    std = np.max(data) - np.min(data)
    data = (data -mean)/std
    return data

def getlabellen(label,pad):
    length = len(label)
    for i in range(length):
        if label[i]==pad:
            return i
    return length


def get(image_height,image_width,image_channel):
    pathlabel = get_path_label(rootpath,newlabelfile)
    image_num = len(pathlabel)
    inx = 0
    X = np.zeros((image_num, image_height, image_width, image_channel), np.float32)
    #Y = np.zeros((image_num, num_length), np.uint8)#最后还有一位存储数字串长度
    Y = []
    labellen = np.zeros((image_num),np.uint8)
    for path in pathlabel:#对每一个label
        data = get_image(path, image_height, image_width)
        data = normal(data,image_height,image_width)
        label = pathlabel[path]
        X[inx, :, :, :] = data
        Y.append(label)
        labellen[inx] = len(label)
        inx = inx+1
    print(X.shape)
    Y = np.array(Y)
    print(Y.shape)
    print(labellen.shape)
    return X, Y,labellen
#X,Y = get(20,100,1,15)
#print(X[0],Y[0])
