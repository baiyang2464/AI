# coding:utf8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import forward
import os

MODEL_SAVE_PATH = "./model"                         #模型存储路径
MODEL_NAME = "emotion_model"                        #模型名称
# 参数
emotion ={0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Sad',5:'Surprise',6:'Neutral'}
dropout = 0.5
class_sum = 7
REGULARIZER = 0.0001
train_num = 200
test_num = 13
batch_size = 20
train_epoch = 50                                    #最大训练轮数
print_point = 10
#训练批次数
train_batch_num = train_num / batch_size            #200/20
test_batch_num = test_num / batch_size              #13/20

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

def batch_data(x, y, batch, num):
    ind = np.arange(num)
    index = ind[batch * batch_size:(batch + 1) * batch_size]
    batch_x = x[index, :]                           #获取x中第index行的数据，等价于x[index]
    batch_y = y[index, :]
    return batch_x,batch_y

total_train_loss = []
def backward():
    initData()
    keep_prob = tf.placeholder(tf.float32)          # dropout减轻过拟合问题
    x = tf.placeholder(tf.float32, [None, 48*48])
    y = tf.placeholder(tf.float32, [None, class_sum])
    #调用前向传播
    pred = forward.forward(x,True,REGULARIZER)    
    #取测试数据
    train_x=Face_data [0: train_num, :]
    train_y= Face_label[0: train_num, :]
    global_step = tf.Variable(0, trainable = False) #定义变量global_step，并把它的属性设置为不可训练  

    # 评估模型
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))          #比较预测值和标签值，argmax返回最大的那个数值所在的下标。 
                                                                        #1是表示返回一行中最大的元素的下标,y的一行有6个0和一个1
    cross_entropy = -tf.reduce_sum(y*tf.log(pred))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy,global_step = global_step)#使用AdamOptimizer优化器，根据交叉熵进行反向传播训练
    lossuracy = tf.reduce_mean(tf.cast(correct_pred, "float"))           #上面一句返回的是bool值，cast可以转换类型，此次将bool转换成float，即是0或1,
                                                                        #reduce_mean()可以计算一个列表中数据的均值
    saver = tf.train.Saver(max_to_keep=1) 
    #在会话中进行训练
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())#初始化
        ckpt = tf.train.get_checkpoint_state("./model")                 # 从"./model"中加载训练好的模型
        if ckpt and ckpt.model_checkpoint_path: 			# 若ckpt和保存的模型在指定路径中存在，则将保存的神经网络模型加载到当前会话中
            saver.restore(sess, ckpt.model_checkpoint_path)

        coord = tf.train.Coordinator()					#4开启线程协调器
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)  #5

        for epoch in range(0, train_epoch):                             #训练train_epoch轮
            Total_train_loss= 0
            for train_batch in range (0, train_batch_num):
                batch_x,batch_y = batch_data(train_x, train_y, train_batch, train_num)
                # 优化操作
                _,step=sess.run([train_step,global_step], feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})#进行前向传播，与反向传播
                if step % print_point == 0:
                    # 计算损失
                    loss = sess.run(cross_entropy, feed_dict={x: batch_x, y: batch_y})
                    print("Epoch: "+str(epoch)+
                            " after "+str(step)+" step(s) training " 
                            +" loss is " + "{:.3f}".format(loss))

                    saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME),global_step = global_step)   # 保存当前模型
                    Total_train_loss= Total_train_loss+loss 
            total_train_loss.append(Total_train_loss)
            coord.request_stop()
            coord.join(threads)
def main():
    backward()

    plt.subplot(2,1,1)
    plt.ylabel('Train loss')
    plt.plot(total_train_loss, 'r')
    plt.savefig("lossDisplay.png")

    plt.show()
if __name__ == '__main__':
    main()
