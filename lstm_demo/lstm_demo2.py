import tensorflow as tf
from tensorflow.contrib import rnn
import get_dummy_set as dd
import random
import get_training_data as td
from tensorflow.python.saved_model import tag_constants
import os

#import mnist dataset
#from tensorflow.examples.tutorials.mnist import input_data
tf.reset_default_graph()
#mnist=input_data.read_data_sets("/tmp/data/",one_hot=True)

max_size = 50

#ordered_dataset_x, ordered_dataset_y = dd.get_small_dataset_data()
ordered_dataset_x_cropped, ordered_dataset_y_cropped, ordered_sequence_lengths = td.get_training_set("../video_angles", max_size)

#ordered_dataset_x_cropped, ordered_dataset_y_cropped = dd.crop_videos(ordered_dataset_x, ordered_dataset_y, crop_size)

_, num_frames, num_joints = ordered_dataset_x_cropped.shape
num_videos, num_classes = ordered_dataset_y_cropped.shape

print(num_videos, num_frames, num_classes, num_joints)

c = list(zip(ordered_dataset_x_cropped, ordered_dataset_y_cropped, ordered_sequence_lengths))
random.shuffle(c)
dataset_x, dataset_y, sequence_lengths = zip(*c)

#define constants
#unrolled through 28 time steps
time_steps=max_size
#hidden LSTM units
num_units=128
#rows of 28 pixels
n_input=num_joints
#learning rate for adam
learning_rate=0.001
#mnist is meant to be classified in 10 classes(0-9).
n_classes=num_classes
#size of batch
batch_size=40

#weights and biases of appropriate shape to accomplish above task
out_weights=tf.Variable(tf.random_normal([num_units,n_classes]), name='out_weights')
out_bias=tf.Variable(tf.random_normal([n_classes]), name='out_bias')

#defining placeholders
#input image placeholder
x=tf.placeholder("float",[None,time_steps,n_input], name='x')
#input label placeholder
y=tf.placeholder("float",[None,n_classes], name='y')

sequence_len_ph = tf.placeholder(tf.int32, [None], name='sequence_len')

#processing the input tensor from [batch_size,n_steps,n_input] to "time_steps" number of [batch_size,n_input] tensors
input=tf.unstack(x ,time_steps,1)

#defining the network
lstm_layer = rnn.BasicLSTMCell(num_units,forget_bias=1)
outputs,_ = rnn.static_rnn(lstm_layer,input, dtype="float32", sequence_length = sequence_len_ph)

saver = tf.train.Saver(save_relative_paths=True)


#converting last output of dimension [batch_size,num_units] to [batch_size,n_classes] by out_weight multiplication
temp_prediction=tf.matmul(outputs[-1],out_weights, name='temp_prediction')
prediction = temp_prediction + out_bias

#loss_function
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y), name='loss')
#optimization
opt=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

#model evaluation
correct_prediction=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
sum_accuracy = 0
sum_loss = 0

num_epochs = 5
runs_per_epoch = int(len(dataset_y)/batch_size)
num_tests = 0

#initialize variables
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for j in range(num_epochs): # Epoch
        i = 0
        while i < runs_per_epoch:
            batch_x = dataset_x[i*batch_size:i*batch_size + batch_size]
            batch_y = dataset_y[i*batch_size:i*batch_size + batch_size]
            batch_sequence_lengths = sequence_lengths[i*batch_size:i*batch_size + batch_size]
            print(len(batch_sequence_lengths))
            
            sess.run(opt, feed_dict={x: batch_x, y: batch_y, sequence_len_ph:batch_sequence_lengths})
    
            if i %10==0:
                num_tests += 1
                acc=sess.run(accuracy, feed_dict={x:batch_x, y:batch_y})
                sum_accuracy += acc
                los=sess.run(loss, feed_dict={x:batch_x, y:batch_y, sequence_len_ph:batch_sequence_lengths})
                sum_loss += los
                print("For iter ",i)
                print("Accuracy ",acc)
                print("Loss ",los)
                print("__________________")
    
            i += 1

    # Get averages
    print("Average accuracy: {}, average loss: {}".format(sum_accuracy/num_tests, sum_loss/num_tests))
    
    #calculating test accuracy
#    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))

    # Saving
    cur_path = os.getcwd()
    saver.save(sess,  "./../lstm_model/model_test")