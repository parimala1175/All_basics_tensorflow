######## Author : Parimala Kancharla
###################################################
from __future__ import division, print_function, absolute_import
import tensorflow as tf
from IPython.core.debugger import Pdb
pdb = Pdb()
####################### Data Processing ##########
logfile = './log'
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)
images = mnist.train.images
cross_labels = mnist.test.labels
print(images.shape())
print(cross_labels.shape())
pdb.set_trace()
learning_rate = 0.001
################
num_steps = 2000
batch_size = 128
# Network Parameters
num_input = 784 
num_classes = 10 
dropout = 0.25 
############## Model ##################
def conv_net(x, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):        
        x = tf.reshape(x, shape=[-1, 28, 28, 1])
        # Convolution Layer with 32 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
        # Convolution Layer with 64 filters and a kernel size of 3
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)
        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv2)
        # Fully connected layer (in tf contrib folder for now)
        fc1 = tf.layers.dense(fc1, 1024)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)
        # Output layer, class prediction
        out = tf.layers.dense(fc1, n_classes)
    return out	
############### loss function  and Parameters ###########
loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op,global_step=tf.train.get_global_step())
# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start Training
# Start a new TF session
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
	batch_x, _ = mnist.train.next_batch(batch_size)


################## session run #########

########     model saving graph and display and tensorboad logfiles      #############
