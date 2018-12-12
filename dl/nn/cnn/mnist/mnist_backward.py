#coding:utf-8
#反向传播
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import os

LEARNING_RATE_BASE =0.1
LEARNING_RATE_DECAY=0.99
BATCH_SIZE = 200

STEPS = 10000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH="./model/"
MODEL_NAME="yang_model"
REGULARIZER = 0.0001

def backward(mnist):
    x = tf.placeholder(tf.float32,[None,mnist_forward.INPUT_NODE])
    y_ =tf.placeholder(tf.float32,[None,mnist_forward.OUTPUT_NODE])
    y = mnist_forward.forward(x,REGULARIZER)

    #指数衰减学习率
    global_step =tf.Variable(0,trainable=False)
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase =True)

    #使用交叉熵，衡量两个向量之间相似的程度,调用softmax,使输出结果符合概率分布
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))#tf.argmax返回y中最大元素的索引号（数组下标）
    cem = tf.reduce_mean(ce)
    loss = cem +tf.add_n(tf.get_collection('losses'))

    #定义训练函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    #使用滑动平均,使参数更具泛化性
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    ema_op = ema.apply(tf.trainable_variables())
    #将训练函数与,滑动平均函数绑定，每train_step一次，就计算一次滑动平均
    with tf.control_dependencies([train_step,ema_op]):
        train_op = tf.no_op(name='train')

    #保存中间状态，以备不时之需
    saver = tf.train.Saver()

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        #端点续训，加载指定路径下的断点信息，继续训练
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,ckpt.model_checkpoint_path)

        #训练STEPS轮
        for i in range(STEPS):
            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            _,loss_value,step = sess.run([train_op,loss,global_step],feed_dict={x:xs,y_:ys})
            if i % 1000 ==0:
                print("After %d training step(s),loss on training batch is %g"%(step,loss_value))
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)

def main():
    mnist = input_data.read_data_sets("./data/",one_hot=True)
    backward(mnist)

if __name__ == '__main__':
    main()
