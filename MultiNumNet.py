from __future__ import print_function
import tensorflow as tf
import util
from util import Data
import getdata
import numpy as np
from tensorflow.contrib.framework import get_or_create_global_step
NEAR_0 = -1e1
MAXN = 1e1
def nan_to_num(n):
    return tf.clip_by_value(n, NEAR_0, MAXN)

class MultiNet(object):
    """model"""
    def __init__(self, image_height, image_width, image_channel, classNum):
        self.X = tf.placeholder(tf.float32, [None, image_height, image_width, image_channel])
        self.y = tf.sparse_placeholder(tf.int32)
        self.sequence_length = tf.placeholder(tf.int32,[None])
        self.keep_prob = tf.placeholder(tf.float32)
        self.CLASSNUM = classNum
        self.is_training = tf.placeholder(tf.bool)
        self.init = tf.truncated_normal_initializer(0.0,0.01)#参数初始化方式
        self.buildCNN()
        #self.score = tf.nn.softmax(self.score)
        # 损失函数定义
        with tf.variable_scope('loss_scope'):
            self.loss = tf.nn.ctc_loss(labels=self.y,inputs=self.score,sequence_length=self.sequence_length,time_major=False)
        self.cost = tf.reduce_mean(self.loss)
        # 优化器定义
        initial_learning_rate = 0.0001
        learning_rate_decay_factor = 0.8
        decay_steps = 100
        self.global_epoch = tf.Variable(-1, trainable=False, name='global_epoch')
        self.global_epoch_input = tf.placeholder('int32', None, name='global_epoch_input')
        self.global_epoch_assign_op = self.global_epoch.assign(self.global_epoch_input)

        self.lr = tf.train.exponential_decay(learning_rate = initial_learning_rate,global_step = self.global_epoch,
            decay_steps = decay_steps,decay_rate = learning_rate_decay_factor,staircase = True)
        #self.lr = 0.0001
        self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)        
        with tf.control_dependencies(self.extra_update_ops):
            self.train_op = tf.train.MomentumOptimizer(self.lr, 0.9).minimize(self.loss)
        # 准确度定义
        self.score_p = tf.transpose(self.score,perm=[1,0,2])
        self.decoded, self.log_prob = tf.nn.ctc_greedy_decoder(self.score_p,self.sequence_length,merge_repeated=True)
        #self.decoded, self.log_prob = tf.nn.ctc_beam_search_decoder(self.score_p,self.sequence_length,merge_repeated=True)
        self.dense_decoded = tf.sparse_tensor_to_dense(self.decoded[0],default_value=self.CLASSNUM-1)
        # 初始化
        self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    def calcacc(self,y,yp):
        if yp.shape[1]==0:
            return 0
        ypd = []
        for i in yp:
            tmp = []
            for j in i:
                if j<self.CLASSNUM-1:#(55-1)
                    tmp.append(j)
            ypd.append(tmp)
        yd = []
        for i in y:
            tmp = []
            for j in i:
                if j<self.CLASSNUM-1:
                    tmp.append(j)
            yd.append(tmp)
        allcount = len(y)
        tmpcount = 0
        for i in range(allcount):
            if ypd[i]==yd[i]:
                tmpcount += 1
        return tmpcount*1.0/allcount

        
    def buildCNN(self):
        '''
        为了简洁使用tensorflow的layers包里的卷积层直接使用
        '''
        with tf.variable_scope('hidden1'):
            conv = tf.layers.conv2d(self.X, filters=16, kernel_size=[3, 3], padding='same')
            norm = tf.layers.batch_normalization(conv,training = self.is_training)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            hidden1 = pool
        print(hidden1.shape)
        with tf.variable_scope('hidden2'):
            conv = tf.layers.conv2d(hidden1, filters=32, kernel_size=[3, 3], padding='same')
            norm = tf.layers.batch_normalization(conv,training = self.is_training)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=1, padding='same')
            hidden2 = pool
        print(hidden2.shape)
        with tf.variable_scope('hidden3'):
            conv = tf.layers.conv2d(hidden2, filters=64, kernel_size=[3, 3], padding='same')
            norm = tf.layers.batch_normalization(conv,training = self.is_training)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 1], strides=(2,1), padding='same')
            hidden3 = pool
        print(hidden3.shape)
        with tf.variable_scope('hidden4'):
            conv = tf.layers.conv2d(hidden3, filters=128, kernel_size=[3, 3], padding='same')
            norm = tf.layers.batch_normalization(conv,training = self.is_training)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=1, padding='same')
            hidden4 = pool
        print(hidden4.shape)
        with tf.variable_scope('hidden5'):
            conv = tf.layers.conv2d(hidden4, filters=128, kernel_size=[3, 3], padding='same')
            norm = tf.layers.batch_normalization(conv,training = self.is_training)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=1, padding='same')
            hidden5 = pool
        print(hidden5.shape)
        with tf.variable_scope('hidden6'):
            conv = tf.layers.conv2d(hidden5, filters=128, kernel_size=[3,3], padding='same')
            norm = tf.layers.batch_normalization(conv,training = self.is_training)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 1], strides=(2,1), padding='same')
            hidden6 = pool
        print(hidden6.shape)
        with tf.variable_scope('hidden7'):
            hidden6 = tf.transpose(hidden6,perm=[0,2,1,3])
            hidden6 = tf.reshape(hidden6,[-1,hidden6.shape[1].value,hidden6.shape[2].value*hidden6.shape[3].value])
            print('reshapeconv',hidden6.shape)
            lstm_fw_cell = tf.contrib.rnn.LSTMCell(128,forget_bias=1.0,use_peepholes=True)
            lstm_bw_cell = tf.contrib.rnn.LSTMCell(128,forget_bias=1.0,use_peepholes=True)
            lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=self.keep_prob)
            lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, output_keep_prob=self.keep_prob)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,
                                                         lstm_bw_cell,
                                                         hidden6,
                                                         dtype=tf.float32)

            lstm_outputs = tf.concat((outputs[0],outputs[1]), axis=2)
            print(lstm_outputs.shape)
            dense = tf.layers.dense(lstm_outputs, units=self.CLASSNUM)
            print(dense.shape)
            self.score = dense
    def getlabellen(self,label):
        length = len(label)
        #for i in range(length): #    if label[i]==pad:   #        return i
        return length

    def sparse_tuple_from(self,sequences,dtype=np.int32):
        """
        Create a sparse representention of x.
        Args:
            sequences: a list of lists of type dtype where each element is a sequence
        Returns:
            A tuple with (indices, values, shape)
        """
        indices = []
        values = []
        shape = []
        maxlen = 0
        for n, seq in enumerate(sequences):
            nowlen = self.getlabellen(seq)
            maxlen = max(nowlen,maxlen)
            indices.extend(zip([n] *nowlen,range(nowlen)))
            values.extend(seq[0:nowlen])
 
        indices = np.asarray(indices, dtype=np.int64)
        values = np.asarray(values, dtype=dtype)
        if self.CLASSNUM in values:
            print('WTF')
        #shape = np.asarray([len(sequences),maxlen+1],dtype=np.int32)
        shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)
        return indices, values, shape

    def train(self, sess, X_train, Y_train,X_test, Y_test, split,num_epoch,logs):
        sess.run(self.init_op)
        seq_len = self.score.shape[1].value
        print('seq_len:',seq_len)
        # 随机生成器需要fit总体样本数据
        datagen = util.get_generator()
        saver = tf.train.Saver()
        # 因为datagen的特殊需求(bacthsize需要能够整除训练集总个数，并且这里样本也少，直接全体当batchsize)
        batch_size = int(len(X_train)/split)#取split使得尽量为200左右
        if len(X_train)%split != 0:#这里仅仅是因为图片生成那里要求batchsize需要能够和输入数据个数 整除 所以如此做以确保这一点
            remove = len(X_train)%split
            X_train = X_train[:-1*remove]#如果后续数据多了大可以不必进行图片生成或者图片数据很多却依然做图片生成时则batch_size和这个可能需要再调整
            Y_train = Y_train[:-1*remove]
        print('batch_size:', batch_size)
        tacc = 0
        for e in range(num_epoch):
            indices = list(range(len(Y_train))) # indices = the number of images in the source data set
            np.random.shuffle(indices)
            X_train = X_train[indices]
            Y_train = Y_train[indices]
            yieldData = Data(X_train,Y_train)
            #更新lr
            k = sess.run([self.global_epoch_assign_op],feed_dict={self.global_epoch_input:e})
            #lrnow = sess.run(self.lr)
            #lrnow = self.lr
            print('Epoch---------',e)
            batches = 0
            if e != 0 and e %50 == 0:
                saver.save(sess,logs+'/'+str(e)+'-'+str(tacc)+'model')
            Y_test_sparse = self.sparse_tuple_from(Y_test)
            for x_batch, y_batch in datagen.flow(X_train, Y_train, batch_size=batch_size, save_to_dir=None):
                y_sparse = self.sparse_tuple_from(y_batch)
                if batches %(split-1) == 0 and batches!=0:
                    '''训练集'''
                    denseval,costval= sess.run(
                         [self.dense_decoded,self.cost], 
                         feed_dict={self.X: x_batch, self.keep_prob: 1, self.y:y_sparse,self.sequence_length:np.ones(len(y_batch))*seq_len,self.is_training:False})
                    print("Train cost:",costval)
                    print('denseval',denseval[0:10])
                    print('train_acc',self.calcacc(y_batch,denseval))
                    '''测试集'''
                    denseval,costval= sess.run(
                           [self.dense_decoded,self.cost], 
                           feed_dict={self.X: X_test, self.keep_prob: 1, self.y:Y_test_sparse,self.sequence_length:np.ones(len(Y_test))*seq_len,self.is_training:False})
                    print("Test cost:",costval)
                    tacc = self.calcacc(Y_test,denseval)
                    print('test_acc',tacc)
                if batches<=split:
                    _,costval= sess.run([self.train_op,self.cost], 
                                         feed_dict={self.X: x_batch, self.keep_prob:0.5, self.y:y_sparse,self.sequence_length:np.ones(len(y_batch))*seq_len,self.is_training:True})
                    if batches%(split-1) == 0:
                        print("cost:",costval)
                batches += 1
                if batches==split+1:
                    break
                        #使用原始数据进行迭代
            for i in range(1):
                for batch_X,batch_Y in yieldData.get_next_batch(batch_size):
                    y_sparse = self.sparse_tuple_from(batch_Y)
                    _,lossval = sess.run([self.train_op, self.loss],
                                                    feed_dict={self.X: batch_X, self.keep_prob:0.5, self.y:y_sparse,self.sequence_length:np.ones(len(batch_Y))*seq_len,self.is_training:True})

    def predict(self,sess,X):
        scoreval= sess.run([self.score], feed_dict={self.X: X, self.keep_prob: 1.0,self.is_training:False})
        #res = [0 for i in range(self.num_length)]
        #for i in range(self.num_length):
        #    rest = tf.reshape(tf.argmax(self.score[i], axis=1), [1, -1]))
        return 0     
