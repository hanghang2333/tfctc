from __future__ import print_function
import codecs
import numpy as np
import time
import MultiNumNet
import getdata
import os
import tensorflow as tf

this_time = time.time()
# 超参数
num_classes = 12
image_height = 30
image_width = 100
image_channel = 1
num_length = 6
# 初始化
#init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
os.environ["CUDA_VISIBLE_DEVICES"]="2"
sess = tf.Session()#全局load模型
model = MultiNumNet.MultiNet(image_height, image_width, image_channel, 0.6, num_classes,num_length)
saver = tf.train.Saver()
# 读取训练好的模型参数
saver.restore(sess, 'savedmodel/100model')
print('初始化用时：%fs' % (time.time() - this_time))
print('start predicting')

def predict_result(X):
    return model.predict(sess, X)

def preprocess(X_pre):
    '''
    输入图片也需要进行与训练时相同的预处理才能放到模型里传播
    这里假设X_pre是图片的完整路径列表,可以不只一张图片
    '''
    numbers = len(X_pre)
    X = np.zeros((numbers, image_height, image_width, image_channel), np.float32)
    inx = 0
    for i in range(numbers):
        data = getdata.get_image(X_pre[i], image_height, image_width)
        data = data.astype(np.float32)
        mean = np.sum(data)/(image_height*image_width)
        std = np.max(data) - np.min(data)
        data = (data -mean)/std
        X[inx, :, :, :] = data
        inx = inx + 1
    return X

def predict(X_pre):
    a = predict_result(preprocess(X_pre))
    print(a)

path = '/home/lihang/ocr/data/val/numstr_val/'
def test():
    pathdict = getdata.get_path_label(path)
    for i in pathdict:
        predict([i])
        print(pathdict[i])

def test_y():
    test()
    X,Y = getdata.get_by_path(image_height,image_width,image_channel,num_length,path)
    model.predict_y(sess,X,Y)

test_y()
