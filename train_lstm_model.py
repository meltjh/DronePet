import tensorflow as tf
from tensorflow.contrib import rnn
import get_training_data as td
from tensorflow.python.saved_model import tag_constants
import os
from sklearn.model_selection import KFold
import sys

# Perform cross validation.
def cross_validation(sess, ordered_dataset_x, ordered_dataset_y, batch_size, num_epochs, x, y, init, accuracy, opt):
    cross_val_results = []
    kf = KFold(10)
    kf_splitted = list(kf.split(ordered_dataset_x, ordered_dataset_y))
    num_cross_val = len(kf_splitted)
    i = 0
    
    for train_idx, val_idx in kf_splitted:
        # Split in train and validation set.
        train_x = ordered_dataset_x[train_idx]
        train_y = ordered_dataset_y[train_idx]
        val_x = ordered_dataset_x[val_idx]
        val_y = ordered_dataset_y[val_idx]
        
        # Train the model and get accuracy on validation set.
        train(sess, train_x, train_y, num_epochs, batch_size, x, y, init, opt)
        acc = sess.run(accuracy, feed_dict={x: val_x, y: val_y})
        
        # Average accuracy over the folds.
        cross_val_results.append(acc)
        i += 1
        average_acc = sum(cross_val_results)/i
        
        sys.stdout.write("\rPart {}/{}: accuracy of {}, average accuracy so far: {}".format(i, num_cross_val, acc, average_acc))
        sys.stdout.flush()
    print("\nFinal Average accuracy: {}".format(sum(cross_val_results)/num_cross_val))
    
# Train the model.
def train(sess, train_x, train_y, num_epochs, batch_size, x, y, init, opt):
    sess.run(init)
    num_batches = int(len(train_y) / batch_size)
    for epoch_i in range(num_epochs):
        # Loop over the batches and train.
        for i in range(num_batches):
            index_start = i*batch_size
            index_end = i*batch_size + batch_size
            batch_x = train_x[index_start:index_end]
            batch_y = train_y[index_start:index_end]
            sess.run(opt, feed_dict={x: batch_x, y: batch_y})
    
# Initialize the model that needs to be trained.
def train_lstm_model(training_set_dir, skip_frame, hidden_units, learning_rate, batch_size, num_epochs):
    tf.reset_default_graph()
    
    ordered_dataset_x, ordered_dataset_y = td.get_training_set(training_set_dir, skip_frame)
    _, num_frames, num_features = ordered_dataset_x.shape
    num_videos, num_classes = ordered_dataset_y.shape
    print("Total number of videos: {}, number of frames per video: {}, number of classes: {}, number of features: {}"
          .format(num_videos, num_frames, num_classes, num_features))
    
    time_steps = num_frames
    learning_rate = 0.001
    
    out_weights = tf.Variable(tf.random_normal([hidden_units, num_classes]), name = "out_weights")
    out_bias = tf.Variable(tf.random_normal([num_classes]), name = "out_bias")
    
    # Placeholders for the training data and labels.
    x = tf.placeholder("float", [None, time_steps, num_features], name = "x")
    y = tf.placeholder("float", [None, num_classes], name = "y")

    input = tf.unstack(x ,time_steps,1)
    
    # The LSTM model
    lstm_layer = rnn.BasicLSTMCell(hidden_units, forget_bias = 1)
    outputs,_ = rnn.static_rnn(lstm_layer, input, dtype = "float32")

    saver = tf.train.Saver(save_relative_paths = True)
    
    # Computation of the prediction.
    temp_prediction = tf.matmul(outputs[-1], out_weights, name = "temp_prediction")
    prediction = temp_prediction + out_bias
    
    # Loss function.
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction,labels = y), name = "loss")
   
    # Optimizer function.
    opt = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)
    
    # Evaluation
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        cross_validation(sess, ordered_dataset_x, ordered_dataset_y, batch_size, num_epochs, x, y, init, accuracy, opt)
        
        # Saving
        saver.save(sess,  "lstm_model/model")

if __name__ == "__main__":
    skip_frame = 11
    training_set_dir = "training_angles"
    hidden_units = 128
    learning_rate = 0.001
    batch_size = 1
    num_epochs = 10
    train_lstm_model(training_set_dir, skip_frame, hidden_units, learning_rate, batch_size, num_epochs)