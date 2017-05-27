#---------------------------------------------------------------------------------------------------------
#                   This file implements the shuffle and learn layer
#                   input: video with frame representations
#                   output: the mapped representation
#---------------------------------------------------------------------------------------------------------


import tensorflow as tf
import numpy as np
from utils import random_pick_3

class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """

    hidden_size1 = 864
    hidden_size2 = 720
    batch_size = 16
    max_grad_norm = 10.0 # max gradients norm for clipping
    lr = 0.001 # learning rate

    # parameters for the first layer
    filter1_size = 5
    conv1_output_channel = 32
    pool1_length = 5

    # parameters for the second layer
    filter2_size = 3
    conv2_output_channel = 8
    pool2_length = 3

    feature_size = 1024
    input_length = 1024

    input_num_once = 3

    num_shuffle_sample = 10

    def __init__(self):
        self.batch_size = 16

class shuffleLearnModel():
    def add_placeholders(self):
        self.input_placeholder  = tf.placeholder(tf.float32, [None, Config.input_length, Config.feature_size])
        self.dropout_placeholder = tf.placeholder(tf.float32)
        #self.num_frames = tf.placeholder(tf.int32, [None])

    def create_feed_dict(self, inputs_batch):
        feed_dict = {}
        feed_dict[self.input_placeholder] = inputs_batch
        #feed_dict[self.num_frames] = num_frames
        return feed_dict

    def add_extract_op(self):

        output_list = []

        for i in range(Config.input_length):
            with tf.variable_scope("conv1", reuse = None if i == 0 else True):
                # the first convolutional layer
                filter1 = tf.get_variable("f1", [Config.filter1_size, 1, Config.conv1_output_channel])
                conv1_input = tf.reshape(self.input_placeholder[:,i,:], [-1, Config.feature_size, 1 ])
                conv1 = tf.nn.conv1d(conv1_input, filter1, stride = 2, padding= "SAME")

                # pooling
                pool1 = tf.nn.pool(conv1, [Config.pool1_length], "MAX", "SAME")

                # activate
                activ1 = tf.nn.relu(pool1)

            with tf.variable_scope("conv2", reuse = None if i == 0 else True):
                # the first convolutional layer
                filter2 = tf.get_variable("f2", [Config.filter2_size, Config.conv1_output_channel, Config.conv2_output_channel])
                conv2 = tf.nn.conv1d(activ1, filter2, stride = 2, padding= "SAME")

                # pooling
                pool2 = tf.nn.pool(conv2, [Config.pool2_length], "MAX", "SAME")

                # activate
                activ2 = tf.nn.relu(pool2)

            with tf.variable_scope("fc1", reuse = None if i == 0 else True):
                activ2_shape = activ2.get_shape().as_list()
                fc1_input1 = tf.reshape(activ2, [ -1, Config.conv2_output_channel * activ2_shape[1] ])
                fc1_output1 = tf.layers.dense(inputs=fc1_input1, units=1024, activation=tf.nn.sigmoid)

            output_list.append(fc1_output1)

        # return the output of the fully connected layer
        self.output_tensor = tf.stack(output_list, axis = 1)
        self.output_tensor = tf.reshape(self.output_tensor, [-1, Config.input_length, Config.feature_size])
        return self.output_tensor


    def add_random_combination(self, input_features, num_frames):
        """
        input features: a list (length: input_length) of element with shape [batch_size, feature_size]
        Returns:

        """
        shuffle_list, label_list = random_pick_3(num_frames, Config.num_shuffle_sample) # 3-D np array [batch_size, num_samples, 3]
       # input_embedding_list = tf.unstack(input_features, axis = 0)
        input_embedding_list = input_features
        sample_list = []
        #label_list = tf.convert_to_tensor(label_list)
        video_num = num_frames.get_shape().as_list()[0]
        label_list = tf.constant(label_list, dtype = tf.bool, shape = [video_num ,60])
        for i in range(video_num):
            # loop over the batch_size
            #shuffle_index = tf.convert_to_tensor(shuffle_list[i])
	    shuffle_index = tf.Variable(shuffle_list[i], dtype = tf.int32)
            shuffle_value = tf.nn.embedding_lookup(input_embedding_list[i], shuffle_index)
            shuffle_concat_list = []
            for j in range(shuffle_value.shape[0]):
                shuffle_concat_list.append(tf.concat([shuffle_value[j][0], shuffle_value[j][1], shuffle_value[j][2]],
                                                     axis = 0))
            sample_list.append(shuffle_concat_list)
        return sample_list, label_list

    #def add_random_pick(self, input_features):

	
    def add_shuffle_loss(self, sample_list, label_list):
        #!!!!!!!!! Need Discussion!!!!!!!!
        '''

        This function computes the shuffle loss
        Now, we use a simple softmax classifier to do this job
        It needs further discussion

        '''
            # sample_list_spread = tf.reshape(sample_list, [-1, ]) # mark!!!!!! concatenate use or not
        predictions = []
        for i in range(label_list.get_shape().as_list()[0]):
            pred_one_video = []
            for j in range(label_list.get_shape().as_list()[1]):
                with tf.variable_scope("a", reuse = None if (i == 0 and j == 0) else True):
                    a = tf.get_variable("a", [Config.feature_size * 3])
                pred_one_video.append(tf.reduce_sum(tf.multiply(a, sample_list[i][j])))
                tf.stack(pred_one_video, axis = 0)
            predictions.append(pred_one_video)
        tf.stack(predictions, axis = 0)

        # to get sample_list_spread


        self.shuffle_loss = tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=label_list)
        return self.shuffle_loss

    def __init__(self, num_frames):
        self.config = Config()
        self.input_placeholder = None
        self.dropout_placeholder = None
        self.add_placeholders()
        output_feat = self.add_extract_op()
        sample_list, label_list = self.add_random_combination(output_feat, num_frames)
        self.loss = self.add_shuffle_loss(sample_list, label_list)
        # self.loss = self.add_shuffle_loss(NONE,sample_list, label_list,-1,NONE)









