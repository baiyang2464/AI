#coding:utf-8
import tensorflow as tf
tf.set_random_seed(1)
import numpy as np
np.random.seed(1)
import os
import ner_forward
#import ner_lstm as ner_forward
import data_helper

BATCH_SIZE = 128
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARIZER = 0.0001
STEPS = 100
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH="./model/"
MODEL_NAME="ner_model"

train_file = "data/ner.train"

emb_file = "data/ner.emb"

datautil = data_helper.DataUtil()


def backward():
    x = tf.placeholder(tf.int32, [None, None]) #输入二维数据，第一维度是batch, 第二维是句长 [batch_size, seqlen]

    word_emb = tf.Variable(datautil._word_emb, dtype=tf.float32, name='word_emb')
    x_emb = tf.nn.embedding_lookup(word_emb, x) #对输入的每个字，查找向量字典，转化为字对应的100维向量，[batch_size, seqlen, emb_size]

    y_ = tf.placeholder(tf.int32, [None, None]) #标签，第一维度是batch, 第二维是句长 [batch_size, seqlen]
    y = ner_forward.forward(x_emb, is_train=True, regularizer=REGULARIZER)

    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=y_)
    cem = tf.reduce_mean(ce)

    loss = cem + tf.add_n(tf.get_collection('losses'))

    train_step = tf.train.AdamOptimizer(0.01).minimize(loss)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        for i in range(1, STEPS+1):
            batches = datautil.gen_mini_batch(BATCH_SIZE)

            total_num, total_loss = 0, 0
            for batch_id, batch_data in enumerate(batches): #遍历所有的batch
                x_batch, label_batch = batch_data

                _, loss_value = sess.run([train_step, loss], feed_dict={x: x_batch, y_: label_batch})
                total_num = total_num + len(x_batch)
                total_loss = total_loss + loss_value * len(x_batch)

            avg_loss = total_loss/total_num #计算整个训练集的平均loss
            print("After %d training step(s), loss on training batch is %g." % (i, avg_loss))
            saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=i)



def main():
    datautil.load_emb(emb_file) #加载训练好的embedding
    datautil.load_data(train_file)#加载训练集

    backward()
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    main()
