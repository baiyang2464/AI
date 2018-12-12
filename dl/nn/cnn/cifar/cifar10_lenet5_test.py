#	 coding:utf-8
#	 file    cifar10_lenet5test.py
#	 author  yang(yanghongcs@pku.edu.cn)
 
 
import tensorflow as tf
import numpy as np
import time
import cifar10_lenet5_forward
import cifar10_lenet5_backward
import cifar10_lenet5_generateds
from tensorflow.examples.tutorials.mnist import input_data
INTERVAL_TIME = 5
TEST_NUM = 1000 #1
BATCH_SIZE = 1000
#test!

def test():
    with tf.Graph().as_default() as g:                                      
        x = tf.placeholder(tf.float32, [                                    
            BATCH_SIZE,
            cifar10_lenet5_forward.IMAGE_SIZE,
            cifar10_lenet5_forward.IMAGE_SIZE,
            cifar10_lenet5_forward.NUM_CHANNELS])
        
        y_ = tf.placeholder(tf.float32,[None, cifar10_lenet5_forward.OUTPUT_NODE]) 
        y = cifar10_lenet5_forward.forward(x,False,  None)                            

        ema = tf.train.ExponentialMovingAverage(cifar10_lenet5_backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()                                      
        saver = tf.train.Saver(ema_restore) 			                             
        
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))              
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))            

        img_batch,label_batch = cifar10_lenet5_generateds.get_tfrecord(TEST_NUM, isTrain=False)  
        

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(cifar10_lenet5_backward.MODEL_SAVE_PATH)    
                if ckpt and ckpt.model_checkpoint_path:                                   
                    saver.restore(sess, ckpt.model_checkpoint_path)                       
     
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1] 

                    coord = tf.train.Coordinator()                                        
                    threads = tf.train.start_queue_runners(sess=sess, coord=coord)        
                    xs,ys = sess.run([img_batch, label_batch])                            

                    reshaped_xs = np.reshape(xs, (                              
                        BATCH_SIZE,
                        cifar10_lenet5_forward.IMAGE_SIZE,
                        cifar10_lenet5_forward.IMAGE_SIZE,
                        cifar10_lenet5_forward.NUM_CHANNELS))             

 
                    accuracy_score = sess.run(accuracy, 
                        feed_dict={x:reshaped_xs, y_:ys})

                    print ("after %s training step(s), test accuracy = %g"
                            % (global_step, accuracy_score))

                    coord.request_stop()                    
                    coord.join(threads)                    

                else:                                                   
                    print ("No checkpoint file found")
                    return
            time.sleep(INTERVAL_TIME)                                  

def main():
    test()
    
#main function,
if __name__ == '__main__':
    main()
