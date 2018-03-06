import numpy as np
from sklearn.model_selection import train_test_split
import os
import tensorflow as tf
from MultiNumNet import MultiNet
import getdata
import util
import argparse
#os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
parser = argparse.ArgumentParser()
parser.add_argument('--Kth_fold',type=int,default=1,help='the Kth Fold')
parser.add_argument('--gpu_core',type=str,default='0',help='use which gpu core')
parser.add_argument('--logs',type=str,default='logs/c01221',help='logs path')
args = parser.parse_args()
np.random.seed(1)#固定下来随机化shuffle的序列
image_height = 30#数字图片应该普遍长宽比例是这样
image_width = 120
image_channel = 1
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
# 训练，测试，持久化
#with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_core
config=tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.5 # 占用GPU90%的显存 
config.gpu_options.allow_growth=True
with tf.Session(config=config) as sess:
#with tf.Session() as sess:   
    # 完成数据的读取，使用的是tensorflow的读取图片
    X, Y,labellen = getdata.get(image_height, image_width, image_channel)
    print(max(labellen))
    num_class = getdata.classnum()+1
    print('numclass:',num_class)
    # 将数据集shuffle
    X, Y = util.shuffledata(X,Y)
        # 将数据区分为测试集合和训练集合
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.9, random_state=33)
    print('Train: ',len(X_train))
    print('Test: ',len(X_test))
    model = MultiNet(image_height, image_width, image_channel, num_class)
    model.train(sess, X_train, Y_train, X_test, Y_test,split=80,num_epoch=1000,logs=args.logs)
