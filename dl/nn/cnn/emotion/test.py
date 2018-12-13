# coding:utf8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import forward
import os
import time

MODEL_SAVE_PATH = "./model"                         #模型存储路径
MODEL_NAME = "emotion_model"                        #模型名称
TIME_INTERVAL = 5
# 参数
emotion ={0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Sad',5:'Surprise',6:'Neutral'}
dropout = 0.5
class_sum = 7
REGULARIZER = 0.0001
train_num = 200
test_num = 13

data = pd.read_csv(r'./face.csv', dtype='a')        #read_csv返回二维标记数组，列可以是不同数据类型，所读对象元素之间通常用逗号隔开
label = np.array(data['emotion'])                   #用data中emotion列生成一个数组
img_data = np.array(data['pixels'])                 #用data中的pixels生成一个多维数组
N_sample = label.size                               #取得维度 N_sample = 213
Face_data = np.zeros((N_sample, 48*48))             #zeros(shape, dtype=float, order='C') (N_sample, 48*48) N_sample是行，48*48的数为列
Face_label = np.zeros((N_sample, 7), dtype=int)

def initData():
    for i in range(N_sample):
        x = img_data[i]
        x = np.fromstring(x, dtype=float, sep=' ')  #从字符串x创建矩阵
        x = x/x.max()                               #归一化处理
        Face_data[i] = x                            #赋值
        Face_label[i, int(label[i])] = 1            #label[i]返回0-6的数，代表一种表情

def test():
    with tf.Graph().as_default() as g:
        initData()
        keep_prob = tf.placeholder(tf.float32)      # dropout减轻过拟合问题
        x = tf.placeholder(tf.float32, [None, 48*48])
        y = tf.placeholder(tf.float32, [None, class_sum])
        #调用前向传播
        pred = forward.forward(x,True,REGULARIZER)    
        #取测试数据
        test_x =Face_data [train_num : train_num+test_num, :]
        test_y = Face_label [train_num : train_num+test_num, :]
        '''
        train_x=Face_data [0: train_num, :]
        train_y= Face_label [0: train_num, :]
        '''
        global_step = tf.Variable(0, trainable = False)                             #定义变量global_step，并把它的属性设置为不可训练  

        # 评估模型
        correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))                  #比较预测值和标签值，argmax返回最大的那个数值所在的下标。 
        accuracy = tf.reduce_mean(tf.cast(correct_pred, "float"))                   #上面一句返回的是bool值，cast可以转换类型，此次将bool转换成float，即是0或1,
                                                                                    #reduce_mean()可以计算一个列表中数据的均值
        saver = tf.train.Saver(max_to_keep=1) 
        #在会话中进行训练
        while True:
            with tf.Session() as sess:
                #sess.run(tf.initialize_all_variables())#初始化
                ckpt = tf.train.get_checkpoint_state("./model")                     # 从"./model"中加载训练好的模型
                if ckpt and ckpt.model_checkpoint_path: 		            # 若ckpt和保存的模型在指定路径中存在，则将保存的神经网络模型加载到当前会话中
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    coord = tf.train.Coordinator()				    #4开启线程协调器
                    threads = tf.train.start_queue_runners(sess=sess, coord=coord)  #5
                    acc= sess.run(accuracy, feed_dict={x: test_x, y: test_y})
                    print("after "+str(global_step)+" step(s) training "
                            + "accuracy is " + "{:.3f}".format(acc))
                    coord.request_stop()
                    coord.join(threads)
                else:
                    print ("No checkpoint file found")
                    return
            time.sleep(TIME_INTERVAL)
def main():
    test()

if __name__ == '__main__':
    main()
