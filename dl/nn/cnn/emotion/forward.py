# coding:utf8

import numpy as np
import tensorflow as tf

keep_prob= 0.5
class_sum = 7
def get_weight(shape, regularizer):#第1层卷积的核长
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))	#生成shape形状的参数矩阵w，其值服从平均值和标准偏差的正态分布，如果生成的值大于平均值2倍标准偏差，丢弃该值并重新选择。
    if regularizer!= None:																	#如果启用正则化
        tf.add_to_collection('losses',											#用l2正则化w，并将结果加入losses集合
                            tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w 

def get_bais(shape):
    b = tf.Variable(tf.zeros(shape))												#生成全shape形状的全0矩阵
    return b

def conv_pool_layer(data, weights_size, biases_size,regularizer):
    weights = get_weight(weights_size,regularizer)
    biases = get_bais(biases_size)
    conv2d = tf.nn.conv2d(data, weights, strides=[1,1,1,1], padding='SAME')
    relu = tf.nn.relu(conv2d + biases)
    return tf.nn.max_pool(relu, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def linear_layer(data, weights_size, biases_size,regularizer):
    weights = get_weight(weights_size,regularizer)
    biases = get_bais(biases_size)
    return tf.add(tf.matmul(data, weights), biases)

def forward(x,train,regularizer):
    x_image=tf.reshape(x, [-1,48,48,1])#48,48是每张图片的尺寸，1是通道数，灰度图是单通道，-1是自动推断矩阵的第一维的大小，可以认为是一批处理的图片数目
    h_pool1=conv_pool_layer(x_image, [5,5,1,32], [32],regularizer)
    h_pool2=conv_pool_layer(h_pool1, [5,5,32,64], [64],regularizer)
    h_pool2_flat=tf.reshape(h_pool2, [-1, 12*12*64])
    h_fc1=tf.nn.relu(linear_layer(h_pool2_flat, [12*12*64,1024], [1024],regularizer))
    if train:
        h_fc1_drop=tf.nn.dropout(h_fc1, keep_prob)#防止过拟合，对神经网络的部分进行强化，将部分神经元暂时舍弃（将部分神经元的输出结果变成0），
                                                #keep_prob是舍弃的概率（0-1之间的数）,而对非零的输出结果进行强化，强化为原来的1/keep_prob倍，
    pred = tf.nn.softmax(linear_layer(h_fc1_drop, [1024,class_sum], [class_sum],regularizer))
    return pred
