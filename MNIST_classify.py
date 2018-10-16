import os
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data

current_directory   = os.path.dirname(os.path.realpath(__file__))

# download the MNIST data in <current directory>/MNIST_data/
mnist_dataset       = input_data.read_data_sets(current_directory + "/MNIST_data/", one_hot=True) #one_hot=True: one-hot-encoding

# define the size of the layers - how many neurons in each layer
layers = {
    "input"     : 784,
    "hidden1"   : 512,
    "hidden2"   : 256,
    "hidden3"   : 128,
    "hidden4"   : 64,
    "output"    : 10 
}

# define the weights, mean = 0, stddev = 0.1
weights = {
    'w1'    : tf.Variable(tf.truncated_normal([layers["input"],   layers["hidden1"]], 0, 0.1)),
    'w2'    : tf.Variable(tf.truncated_normal([layers["hidden1"], layers["hidden2"]], 0, 0.1)),
    'w3'    : tf.Variable(tf.truncated_normal([layers["hidden2"], layers["hidden3"]], 0, 0.1)),
    'w4'    : tf.Variable(tf.truncated_normal([layers["hidden3"], layers["hidden4"]], 0, 0.1)),
    'out'   : tf.Variable(tf.truncated_normal([layers["hidden4"], layers["output"]],  0, 0.1)),
}

# define the biases 
biases = {
    'b1'    : tf.Variable(tf.constant(0.1, shape=[layers["hidden1"]])),
    'b2'    : tf.Variable(tf.constant(0.1, shape=[layers["hidden2"]])),
    'b3'    : tf.Variable(tf.constant(0.1, shape=[layers["hidden3"]])),
    'b4'    : tf.Variable(tf.constant(0.1, shape=[layers["hidden4"]])),
    'out'   : tf.Variable(tf.constant(0.1, shape=[layers["output"]]))
}

# set up the neural network
X               = tf.placeholder(tf.float32, [None, layers["input"]]) 
input_layer     = tf.add(tf.matmul(X, weights['w1']), biases['b1'])
hidden_layer_1  = tf.add(tf.matmul(input_layer, weights['w2']), biases['b2'])
hidden_layer_2  = tf.add(tf.matmul(hidden_layer_1, weights['w3']), biases['b3'])
hidden_layer_3  = tf.add(tf.matmul(hidden_layer_2, weights['w4']), biases['b4'])
output_layer    = tf.matmul(hidden_layer_3, weights['out']) + biases['out']
Y               = tf.placeholder(tf.float32, [None, layers["output"]])
loss_function   = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=output_layer)) # loss function is Cross-Entropy
learning_rate   = 1e-4
optimizer       = tf.train.AdamOptimizer(learning_rate).minimize(loss_function) # optimizing the loss function

init = tf.global_variables_initializer() # initialize a session for running the graph

with tf.Session() as sess:
    sess.run(init)

    total_batch     = 500
    batch_size      = 500
    predict_correct = tf.equal(tf.argmax(output_layer, 1), tf.argmax(Y, 1)) 
    accuracy        = tf.reduce_mean(tf.cast(predict_correct, tf.float32))
    
    # train the model
    for i in range(total_batch):
        batch_xs, batch_ys   = mnist_dataset.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={X: batch_xs, Y: batch_ys})
        batch_loss, batch_accuracy = sess.run([loss_function, accuracy], feed_dict={X: batch_xs, Y: batch_ys})
        print(str(i), ":\t loss = ", str(batch_loss), "\t accuracy = ", str(batch_accuracy))

    # test the model
    test_accuracy = sess.run(accuracy, feed_dict={X: mnist_dataset.test.images, Y: mnist_dataset.test.labels})
    print("test data accuracy:", "{:.0%}".format(test_accuracy))