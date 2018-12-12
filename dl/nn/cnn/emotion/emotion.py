# coding:utf8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

emotion ={0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Sad',5:'Surprise',6:'Neutral'}

data = pd.read_csv(r'./face.csv', dtype='a')#read_csv返回二维标记数组，列可以是不同数据类型，所读对象元素之间通常用逗号隔开
label = np.array(data['emotion'])#用data中emotion列生成一个数组
img_data = np.array(data['pixels'])#用data中的pixels生成一个多维数组
N_sample = label.size#取得维度 N_sample = 213
Face_data = np.zeros((N_sample, 48*48))#zeros(shape, dtype=float, order='C') (N_sample, 48*48) N_sample是行，48*48的数为列
Face_label = np.zeros((N_sample, 7), dtype=int)

print(N_sample)

for i in range(N_sample):
    x = img_data[i]
    x = np.fromstring(x, dtype=float, sep=' ')#从字符串x创建矩阵
    x = x/x.max()#归一化处理
    Face_data[i] = x#赋值
    Face_label[i, int(label[i])] = 1#label[i]返回0-6的数，代表一种表情

# 参数
dropout = 0.5
class_sum = 7

# dropout减轻过拟合问题
keep_prob = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32, [None, 48*48])
y = tf.placeholder(tf.float32, [None, class_sum])

def conv_pool_layer(data, weights_size, biases_size):
    weights = tf.Variable(tf.truncated_normal(weights_size, stddev=0.1))
    biases = tf.Variable(tf.constant(0.1, shape=biases_size))
    conv2d = tf.nn.conv2d(data, weights, strides=[1,1,1,1], padding='SAME')
    relu = tf.nn.relu(conv2d + biases)
    return tf.nn.max_pool(relu, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def linear_layer(data, weights_size, biases_size):
    weights = tf.Variable(tf.truncated_normal(weights_size, stddev=0.1))
    biases = tf.Variable(tf.constant(0.1, shape=biases_size))
    return tf.add(tf.matmul(data, weights), biases)


def convolutional_neural_network(x, keep_prob):
    x_image=tf.reshape(x, [-1,48,48,1])#48,48是每张图片的尺寸，1是通道数，灰度图是单通道，-1是自动推断矩阵的第一维的大小，可以认为是一批处理的图片数目
    h_pool1=conv_pool_layer(x_image, [5,5,1,32], [32])
    h_pool2=conv_pool_layer(h_pool1, [5,5,32,64], [64])
    h_pool2_flat=tf.reshape(h_pool2, [-1, 12*12*64])
    h_fc1=tf.nn.relu(linear_layer(h_pool2_flat, [12*12*64,1024], [1024]))

    h_fc1_drop=tf.nn.dropout(h_fc1, keep_prob)#防止过拟合，对神经网络的部分进行强化，将部分神经元暂时舍弃（将部分神经元的输出结果变成0），
                                                #keep_prob是舍弃的概率（0-1之间的数）,而对非零的输出结果进行强化，强化为原来的1/keep_prob倍，
    return tf.nn.softmax(linear_layer(h_fc1_drop, [1024,class_sum], [class_sum]))

pred = convolutional_neural_network(x, keep_prob)#前向传播计算一次,得到一个预测值,这个预测值，是本轮中一批被处理的图片的预测值

# #======取前200个作为训练数据==================
train_num = 200
test_num = 13
#取训练数据
train_x = Face_data [0:train_num, :]#0:train_num是对行做切片，后面的 : 是对列切片，不写数字代表默认接收所有
train_y = Face_label [0:train_num, :]
#取测试数据
test_x =Face_data [train_num : train_num+test_num, :]
test_y = Face_label [train_num : train_num+test_num, :]

batch_size = 20
train_batch_num = train_num / batch_size#200/20
test_batch_num = test_num / batch_size#13/20

def batch_data(x, y, batch, num):
    ind = np.arange(num)
    index = ind[batch * batch_size:(batch + 1) * batch_size]
    batch_x = x[index, :]#获取x中第index行的数据，等价于x[index]
    batch_y = y[index, :]
    return batch_x,batch_y

# 训练和评估模型
cross_entropy = -tf.reduce_sum(y*tf.log(pred))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)#使用AdamOptimizer优化器，根据交叉熵进行反向传播训练
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))#比较预测值和标签值，argmax返回最大的那个数值所在的下标。 
                                                          #1是表示返回一行中最大的元素的下标,y的一行有6个0和一个1
accuracy = tf.reduce_mean(tf.cast(correct_pred, "float"))#上面一句返回的是bool值，cast可以转换类型，此次将bool转换成float，即是0或1,
                                                         #reduce_mean()可以计算一个列表中数据的均值

total_train_loss = []
total_train_acc = []
total_test_loss = []
total_test_acc = []

train_epoch = 20

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())#初始化
    for epoch in range(0, train_epoch):#训练train_epoch轮
        Total_train_loss = 0
        Total_train_acc = 0
        for train_batch in range (0, train_batch_num):#
            batch_x,batch_y = batch_data(train_x, train_y, train_batch, train_num)
            # 优化操作
            sess.run(train_step, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})#进行前向传播，与反向传播
            if train_batch % batch_size == 0:
                # 计算损失和准确率
                loss, acc = sess.run([cross_entropy, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
                print("Epoch: " + str(epoch+1) + ", Batch: "+ str(train_batch) +
                      ", Loss= " + "{:.3f}".format(loss) +
                      ", Training Accuracy= " + "{:.3f}".format(acc))
                Total_train_loss = Total_train_loss + loss
                Total_train_acc = Total_train_acc + acc
        total_train_loss.append(Total_train_loss)
        total_train_acc.append(Total_train_acc)

plt.subplot(2,1,1)
plt.ylabel('Train loss')
plt.plot(total_train_loss, 'r')
plt.subplot(2,1,2)
plt.ylabel('Train accuracy')
plt.plot(total_train_acc, 'r')
plt.savefig("loss_acc.png")

plt.show()
