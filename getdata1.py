#coding=utf8
from __future__ import print_function
from PIL import Image
import os
import numpy as np
import tensorflow as tf
import tensorlayer as tl

def getlabel(name, length=7):
    '''
    将name字符串转换为相应的标签值.这里以10代表小数点，11代表补齐的内容
    name:字符串，在这里定义成是相应的图片的名字(原始数据里每个图片的标签放在名字上)
    return:一维np array。初步定义长度是6.会进行补齐。
    eg:
    name:"90.23" return:[9,0,10,2,3]
    name:"78"    return:[7,8,11,11,11]
    name:"78_1"  return:[7,8,11,11,11]
    '''
    #文件一半命名格式为 前缀_数字串.jpeg ,这是重名文件的标签样式
    if '_' in name:
        lineindex = name.index('_')
        name = name[lineindex+1:-5]
    else:
        name = name[0:-5]
    #开始转换
    resarray = np.zeros(length)
    resarray.fill(11)
    lengthhere = len(name)
    resarray[length-1] = lengthhere-1
    for i in range(lengthhere):
        if name[i] is '.':
            resarray[i] = 10
        else:
            resarray[i] = int(name[i])
    return resarray

def test_getlabel():
    print(getlabel("4536_31.3.jpeg"))
    print(getlabel("78"))
    print(getlabel("78_"))
#test_getlabel()

def get_path_label(rootpath):
    '''
    给出相应目录下所以文件的文件名(完整路径)到标签值的一个映射字典{}
    rootpath:需要读取的目录
    return:字典{文件1:标签array,文件2:标签array,...}
    '''
    resdict = {}
    for namepath in os.listdir(rootpath):
        labelarray = getlabel(namepath)
        resdict[rootpath+namepath] = labelarray
    return resdict

def test_get_path_label():
    print(get_path_label('/home/lihang/ocr/data/genNumCode/train_long/'))
#test_get_path_label()
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
    b = np.reshape(im, [im.size[1], im.size[0], 1])
    b = tl.prepro.imresize(b, size=(height, width), interp='bilinear')
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

def get_by_path(image_height,image_width,image_channel,num_length, rootpath):
    pathlabel = get_path_label(rootpath)
    image_num = len(pathlabel)
    inx = 0
    X = np.zeros((image_num, image_height, image_width, image_channel), np.float32)
    Y = np.zeros((image_num, num_length+1), np.uint8)#最后还有一位存储数字串长度
    for path in pathlabel:#对每一个label
        data = get_image(path, image_height, image_width)
        data = normal(data,image_height,image_width)
        label = pathlabel[path]
        X[inx, :, :, :] = data
        Y[inx, :] = label
        inx = inx+1
    print(X.shape)
    print(Y.shape)
    return X, Y

#print(get_image('/home/lihang/ocr/data/num/0.10.jpeg',30,100))