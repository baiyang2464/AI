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
    weight_xh = get_weight(shape=[INPUT_NODE,HIDDEN_SIZE])#输入层权重
    bias_xh = get_bias(shape =[HIDDEN_SIZE])#输入层偏执
    weight_hh = get_weight(shape=[HIDDEN_SIZE,HIDDEN_SIZE])#隐藏层权重
    bias_hh = get_bias(shape =[HIDDEN_SIZE])#隐藏层偏执
    weight_ho = get_weight(shape=[HIDDEN_SIZE,OUTPUT_NODE],regularizer = regularizer)#输出层权重
    bias_ho = get_bias(shape =[OUTPUT_NODE])#输入层偏执

    h = tf.zeros(shape = [tf.shape(x)[0],HIDDEN_SIZE],dtype = tf.float32)#tf.shape(x)[0]返回x的第一维的大小

    output = []#setlen个[batch_size,output_node]
    seq_lst = tf.transpose(x,[1,0,2])#将x转换为[seqlen,batch,emb_size]
    seq_lst = tf.unstack(seq_lst,num=seq_len)#将seqlen,batch,emb_size]转化为seqlen个[batch,emb_size]的list
    with tf.variable_scope("SimpleRNN"):
        for i,inp in enumerate(seq_lst):#enumerate返回下标，和下标对应的元素，此处是List
            if i>0:
                tf.get_variable_scope().reuse_variables()#每一步，rnn的权值共享
                # h = tanh(x_t*w_xh+b_xh+h_t-1*w_hh+b_hh)
            h = tf.matmul(inp,weight_xh)+bias_xh + tf.matmul(h,weight_hh)+bias_hh
            h = tf.tanh(h)
            output.append(h)
        oytput = tf.transpose(output,[1,0,2])
    y = tf.matmul(tf.reshape(output, [-1, HIDDEN_SIZE]), weight_ho) + bias_ho
    return tf.reshape(y, [-1, seq_len, OUTPUT_NODE])

