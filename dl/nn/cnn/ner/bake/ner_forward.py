#coding:utf-8
import tensorflow as tf

INPUT_NODE = 100
OUTPUT_NODE = 7
MAX_SEQ_LEN = 100
HIDDEN_SIZE = 50
BATCH_SIZE = 128

def get_weight(shape, regularizer=None):
    w = tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    if regularizer != None: tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b

def forward(x, is_train=True, regularizer=None): #[batch, seqlen, emb_size]
    seq_len = 100
    if not is_train: #如果是训练集句长为100，为了防止内存不足，测试时句长为测试集中最长句长600
        seq_len = 600

    rnn_cell = tf.nn.rnn_cell.BasicRNNCell(HIDDEN_SIZE)
    output, _ = tf.nn.dynamic_rnn(rnn_cell, x, dtype=tf.float32)

    weight_ho = get_weight(shape=[HIDDEN_SIZE, OUTPUT_NODE], regularizer=regularizer)
    bias_ho = get_bias(shape=[OUTPUT_NODE])

    y = tf.matmul(tf.reshape(output, [-1, HIDDEN_SIZE]), weight_ho) + bias_ho
    return tf.reshape(y, [-1, seq_len, OUTPUT_NODE])

