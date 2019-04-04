#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: predcoding_model.py
Created on Wed Jun 28 21:22:59 2017
Last modified: Wed Dec 15, 2017

Contributers: A. Maida, N. Elsayed, M. Hosseini, Z. Kirby

This is a convolutional LSTM prototype for predictive coding.
Input maps are:
    1. Half plane moving to the right.
    2. Associated local movement map

"""

print("""\n
      _____              _ _   _      _   
     |  __ \            | | \ | |    | |  
     | |__) | __ ___  __| |  \| | ___| |_ 
     |  ___/ '__/ _ \/ _` | . ` |/ _ \ __|
     | |   | | |  __/ (_| | |\  |  __/ |_ 
     |_|   |_|  \___|\__,_|_| \_|\___|\__|
   
 by the Biological Artificial Intelligence Lab,
      University of Louisiana at Lafayette 
         inspired by Lotter et al. (2017)
                   """)

import MultiLayerPredNet as MLP
import PredNetModel
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import tensorflow as tf

print("Python version    : ", sys.version)
print("TensorFlow version: ", tf.VERSION)
print("Current directory : ", os.getcwd())
start_time = time.time()
"""
 Logging w/ TensorBoard
 The /tmp directory is periodically cleaned, such as on reboot.
 Since you probably don't want to keep these logs around forever,
 this is a practical place to put them if you are the only
 user of the computer.
"""
LOGDIR = "/tmp/convLSTM_v3/"
LENGTH_OF_VID = 30
IM_SZ_HGT = 64  # For later experiments, modify size as necessary
IM_SZ_WID = 64
BATCH_SZ = 1
NUM_UNROLLINGS = 3
INIT_LEARNING_RATE = 0.001  # initial learning rate
NUM_TRAINING_STEPS = 250
VIDEO_CHANNELS = 1

LAYER_WISE = False  # Whether or not we train each layer as independent loss values
MOVING_MNIST = True

"""
Create Input Video with 2 Channels:
  Channel 0: Video of a moving half-plane to the right.
  Channel 1:. Local movement map (LMM) to match channel 0.
  Store each video in ndarray with three dimensions.
  Videos are used to load feed dictionary.
"""
if not MOVING_MNIST:
    # moving half-plane
    in_video = np.empty([LENGTH_OF_VID, IM_SZ_HGT, IM_SZ_WID, VIDEO_CHANNELS], dtype=np.float32)
    in_video[:, :, :, 0] = 1.0  # set background of 1st channel to white

    for f in range(LENGTH_OF_VID - 1):  # make half-plane for each frame
        for col in range(f + 1):
            in_video[f, :, col, 0] = 0.0  # set to black

"""
Load Moving MNIST
    This is a MUCH harder dataset then the simple moving plane.
    We are looking for shape retention as the MNIST objects bounce around.
"""
if MOVING_MNIST:
    with np.load('movingmnistdata.npz') as data:
        a = data['arr_0']
        a = np.squeeze(a)

    # Take first video sequence
    in_video = a[:LENGTH_OF_VID, :, :]
    in_video = np.expand_dims(in_video, axis=-1)
    for x in range(LENGTH_OF_VID):
        plt.imshow(np.squeeze(in_video[x:(x + 1), :, :]), cmap=cm.gray_r, vmin=0.0, vmax=1.0)
        plt.show()

default_prediction = np.empty([IM_SZ_HGT, IM_SZ_WID, VIDEO_CHANNELS], dtype=np.float32)
default_prediction[:, :, :] = 0.5
plt.imshow(default_prediction[:, :, 0], cmap=cm.gray_r, vmin=0.0, vmax=1.0)
plt.show()

"""
 BUILD the GRAPH
     Convolutional LSTM.
     Each node in the LSTM is a stack of two feature maps.
     Internal convolutions are 5x5 w/ stride of 1.
"""
# These vars must be equal. 
# Separate names are used because they have separate purposes.
in_channels = VIDEO_CHANNELS  # stack height of input map
core_channels = VIDEO_CHANNELS  # stack height of internal maps

# BUILD MODEL
# Keyword arg overrides default value
# We construct multiple prednet layers and add them to a list
graph = tf.Graph()
with graph.as_default():
    MLP = MLP.MultiLayerPredNet(layer_count=4,
                                img_width=IM_SZ_WID,
                                img_height=IM_SZ_HGT,
                                num_unrollings=NUM_UNROLLINGS,
                                channels=VIDEO_CHANNELS)
    prednets = MLP.build_layers_()
    prednet_0 = prednets[0]
    prednet_1 = prednets[1]
    prednet_2 = prednets[2]
    prednet_3 = prednets[3]

    # Upsample weights for conv2d, i think i need these here
    # hard coded until I finally obtain sanity
    upsample_0 = tf.Variable(tf.truncated_normal([5, 5, VIDEO_CHANNELS, 32], mean=-0.1, stddev=0.1, seed=42))
    upsample_1 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], mean=-0.1, stddev=0.1, seed=42))
    upsample_2 = tf.Variable(tf.truncated_normal([5, 5, 64, 128], mean=-0.1, stddev=0.1, seed=42))
    upsample_3 = tf.Variable(tf.truncated_normal([5, 5, 128, 256], mean=-0.1, stddev=0.1, seed=42))

"""
START TRAINING
"""
with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()

    # Create graph summary
    # Use a different log directory each time you run the program.
    msumm = tf.summary.merge_all()
    dir_index = 0
    while os.path.isdir(LOGDIR + str(dir_index)):
        dir_index += 1
    writer = tf.summary.FileWriter(LOGDIR + str(dir_index))
    writer.add_graph(sess.graph)

    for step in range(NUM_TRAINING_STEPS):
        index = step % (LENGTH_OF_VID - NUM_UNROLLINGS)  # select starting frame of video segment

        data_for_this_iteration = in_video[index:(index + NUM_UNROLLINGS), :, :, :]  # video segment is NUM_UNROLLINGS long
        # FIRST LAYER------------------------------------------------------------------
        #   We always compute loss on the bottom layer, no need to check
        feed_dict_0 = {prednet_0.p_holder: data_for_this_iteration}

        _, l_0, new_lstm_output_0, err_out_0 = sess.run([prednet_0.optimizer,
                                                         prednet_0.loss,
                                                         prednet_0.lstm_output,
                                                         prednet_0.err_output],
                                                        feed_dict=feed_dict_0)
        # SECOND LAYER-----------------------------------------------------------------
        new_width = int(IM_SZ_HGT / 2)
        new_height = int(IM_SZ_WID / 2)
        new_channels = 32  # why is this like this?

        new_data_0 = tf.nn.conv2d(err_out_0, upsample_0, [1, 1, 1, 1], padding='SAME').eval()
        new_data_0 = tf.nn.relu(new_data_0).eval()
        new_data_0 = tf.nn.max_pool(new_data_0, [1, 5, 5, 1], [1, 2, 2, 1], padding='SAME').eval()
        feed_dict_1 = {prednet_1.p_holder: new_data_0}

        if LAYER_WISE:
            _, l_1, new_lstm_output_1, err_out_1 = sess.run([prednet_1.optimizer,
                                                         prednet_1.loss,
                                                         prednet_1.lstm_output,
                                                         prednet_1.err_output],
                                                        feed_dict=feed_dict_1)
        if not LAYER_WISE:
            new_lstm_output_1, err_out_1 = sess.run([prednet_1.lstm_output,
                                                 prednet_1.err_output],
                                                feed_dict=feed_dict_1)
        # THIRD LAYER------------------------------------------------------------------
        new_width = int(new_width / 2)
        new_height = int(new_height / 2)
        new_channels = int(new_channels * 2)

        new_data_1 = tf.nn.conv2d(err_out_1, upsample_1, [1, 1, 1, 1], padding='SAME').eval()
        new_data_1 = tf.nn.relu(new_data_1).eval()
        new_data_1 = tf.nn.max_pool(new_data_1, [1, 5, 5, 1], [1, 2, 2, 1], padding='SAME').eval()
        feed_dict_2 = {prednet_2.p_holder: new_data_1}

        if LAYER_WISE:
            _, l_2, new_lstm_output_2, err_out_2 = sess.run([prednet_2.optimizer,
                                                         prednet_2.loss,
                                                         prednet_2.lstm_output,
                                                         prednet_2.err_output],
                                                        feed_dict=feed_dict_2)
        if not LAYER_WISE:
            new_lstm_output_2, err_out_2 = sess.run([prednet_2.lstm_output,
                                                 prednet_2.err_output],
                                                feed_dict=feed_dict_2)
        # FOURTH LAYER------------------------------------------------------------------
        new_width = int(new_width / 2)
        new_height = int(new_height / 2)
        new_channels = int(new_channels * 2)

        new_data_2 = tf.nn.conv2d(err_out_2, upsample_2, [1, 1, 1, 1], padding='SAME').eval()
        new_data_2 = tf.nn.relu(new_data_2).eval()
        new_data_2 = tf.nn.max_pool(new_data_2, [1, 5, 5, 1], [1, 2, 2, 1], padding='SAME').eval()
        feed_dict_3 = {prednet_3.p_holder: new_data_2}
        if LAYER_WISE:
            _, l_3, new_lstm_output_3, err_out_3 = sess.run([prednet_3.optimizer,
                                                         prednet_3.loss,
                                                         prednet_3.lstm_output,
                                                         prednet_3.err_output],
                                                        feed_dict=feed_dict_3)
        if not LAYER_WISE:
            new_lstm_output_3, err_out_3 = sess.run([prednet_3.lstm_output,
                                                 prednet_3.err_output],
                                                feed_dict=feed_dict_3)
        # Begin DEBUG outputs
        prev_l = 0
        if step % 10 == 0:
            print("Step: ", step)
            if not LAYER_WISE:
                print("> Loss               : ", l_0)
                print("> NET CHANGE         : ", l_0 - prev_l)
                prev_l = l_0
            if LAYER_WISE:
                print('LAYER-WISE LOSS')
                print("> First Layer        : ", l_0)
                print("> > Second Layer     : ", l_1)
                print("> > > Third Layer    : ", l_2)
                print("> > > > Fourth Layer : ", l_3)
            print("------------------------------------")
        if step == (NUM_TRAINING_STEPS - 1):
            print("Output at end of training from 'new_lstm_output' for {} training steps.".format(NUM_TRAINING_STEPS))
            out = np.squeeze(new_lstm_output_0)

            fig, ax = plt.subplots(ncols=NUM_UNROLLINGS, nrows=2)
            inp = np.squeeze(data_for_this_iteration)

            for x in range(NUM_UNROLLINGS):
                ax[0, x].imshow(out[x, :, :], cmap=cm.Greys_r)
                ax[0, x].set_title('Predicted_' + str(x))
                ax[1, x].imshow(inp[x, :, :], cmap=cm.Greys_r)
                ax[1, x].set_title('Actual_' + str(x))
            plt.show()

    print("\n_________Finished training__________\n")

    # The code below hasn't been updated to handle our modular models, but I leave it for reference
    #   More work to be done to validate our model's learning
'''
    print("\n_________Starting testing __________")
    # run without training
    data_for_this_iteration = np.empty([NUM_UNROLLINGS, IM_SZ_HGT, IM_SZ_WID, VIDEO_CHANNELS], dtype=np.float32)
    for step in range(1,11): # 1 to 10
        # index = step % (64-8) = step % 56 = step (b/c step varies from 1 to 10)
        index = step % (LENGTH_OF_VID - NUM_UNROLLINGS)
        
        print("Testing index: ", index, "    in_video shape:", in_video.shape)
        
        # Get 8-frame video clip starting at index
        data_for_this_iteration = np.copy(in_video[index:(index+NUM_UNROLLINGS),:, :, :])
        
        # Set as gray the default image (gray) for the number of steps we want to predict
        # for i in range(NUM_UNROLLINGS-1, 3, -1):
        for i in range(4, NUM_UNROLLINGS):   # i = 4, 5, 6, 7
            data_for_this_iteration[i,:,:,:] = 0.5
        
        feed_dict = {prednet_0.p_holder : data_for_this_iteration.reshape((NUM_UNROLLINGS, IM_SZ_HGT, IM_SZ_WID, VIDEO_CHANNELS))}
        new_lstm_output = prednet_0.lstm_output.eval(feed_dict=feed_dict)
        new_lstm_output_array = prednet_0.lstm_output_array.eval(feed_dict=feed_dict)

        new_lstm_output = np.squeeze(new_lstm_output)

        f, axarr = plt.subplots(1, NUM_UNROLLINGS)
        for x in range(NUM_UNROLLINGS):
        
            axarr[x].imshow(data_for_this_iteration[x, :, :, 0], cmap = cm.gray, vmin = 0.0, vmax = 1.0)
#        axarr[1].imshow(data_for_this_iteration[1, :, :, 0], cmap = cm.gray, vmin = 0.0, vmax = 1.0)
#        axarr[2].imshow(data_for_this_iteration[2, :, :, 0], cmap = cm.gray, vmin = 0.0, vmax = 1.0)
#        axarr[3].imshow(data_for_this_iteration[3, :, :, 0], cmap = cm.gray, vmin = 0.0, vmax = 1.0)
#        axarr[4].imshow(data_for_this_iteration[4, :, :, 0], cmap = cm.gray, vmin = 0.0, vmax = 1.0)
#        axarr[5].imshow(data_for_this_iteration[5, :, :, 0], cmap = cm.gray, vmin = 0.0, vmax = 1.0)
#        axarr[6].imshow(data_for_this_iteration[6, :, :, 0], cmap = cm.gray, vmin = 0.0, vmax = 1.0)
#        axarr[7].imshow(data_for_this_iteration[7, :, :, 0], cmap = cm.gray, vmin = 0.0, vmax = 1.0)
        plt.show()

        
        # Visualize 1st frame in video data for channel 0 for each step
        input_data_view = data_for_this_iteration[0, :, :, 0]
        input_data_view = np.resize(input_data_view, (IM_SZ_HGT, IM_SZ_WID))
        ax = plt.imshow(input_data_view, cmap = cm.gray, vmin = 0.0, vmax = 1.0)
        plt.xticks([0,   2,   4,  6,   8, 10,   12, 14,   16, 18,   20, 22,   24, 26,   28, 30,   32, 34,   36, 38,   40, 42,  44,  46,   48, 50,   52, 54,   56,  58,  60 ],
                   ["0", "", "4", "", "8", "", "12", "", "16", "", "20", "", "24", "", "28", "", "32", "", "36", "", "40", "", "44", "", "48", "", "52", "", "56", "", "60"])
        plt.show()
        print("Video 1st frame of input for step number {} on channel 0.\n".format(step))
        # look at results

        if step == (11 - 1):
            print("\n_________Looking at final results__________")
            print("These are results for step {} of testing.\n".format(step))
            print("default_prediction: ", default_prediction[:,:,0])
            plt.imshow(default_prediction[:,:,0], cmap = cm.gray, vmin = 0.0, vmax = 1.0)
            plt.show()
            print("default_prediction for channel 0.\n")

            print("1st row of last_input_chan0 at index: {}\n".format(index), in_video[index,0,:,0])
            last_input_chan0 = in_video[index, :, :, 0]
            plt.imshow(last_input_chan0, cmap = cm.gray, vmin = 0.0, vmax = 1.0)
            plt.xticks([0,   2,   4,  6,   8, 10,   12, 14,   16, 18,   20, 22,   24, 26,   28, 30,   32, 34,   36, 38,   40, 42,  44,  46,   48, 50,   52, 54,   56,  58,  60 ],
                       ["0", "", "4", "", "8", "", "12", "", "16", "", "20", "", "24", "", "28", "", "32", "", "36", "", "40", "", "44", "", "48", "", "52", "", "56", "", "60"])
            plt.show()
            print("Last input for channel 0: in_video[index,:,:,0]")
            print("First frame of an 8-frame sequence.")

            #new_lstm_output.resize((IM_SZ_HGT, IM_SZ_WID, 2))
            plt.imshow(new_lstm_output[:,:,0], cmap = cm.gray, vmin = 0.0, vmax = 1.0)
            plt.xticks([0,   2,   4,  6,   8, 10,   12, 14,   16, 18,   20, 22,   24, 26,   28, 30,   32, 34,   36, 38,   40, 42,  44,  46,   48, 50,   52, 54,   56,  58,  60 ],
                       ["0", "", "4", "", "8", "", "12", "", "16", "", "20", "", "24", "", "28", "", "32", "", "36", "", "40", "", "44", "", "48", "", "52", "", "56", "", "60"])
            plt.show()
            print("Predicted next frame for channel 0: new_lstm_output[:,:,0]")
            print("This is the most recent LSTM output.")

            last_input_chan1 = in_video[index+1,:,:,1]
    #       last_input_chan1 = data_for_this_iteration[2,:,:,1]
            plt.imshow(last_input_chan1, cmap = cm.gray, vmin = 0.0, vmax = 1.0)
            plt.xticks([0,   2,   4,  6,   8, 10,   12, 14,   16, 18,   20, 22,   24, 26,   28, 30,   32, 34,   36, 38,   40, 42,  44,  46,   48, 50,   52, 54,   56,  58,  60 ],
                       ["0", "", "4", "", "8", "", "12", "", "16", "", "20", "", "24", "", "28", "", "32", "", "36", "", "40", "", "44", "", "48", "", "52", "", "56", "", "60"])
            plt.show()
            print("Last input for channel 1.")

            plt.imshow(new_lstm_output[:,:,1], cmap = cm.gray, vmin = 0.0, vmax = 1.0)
            plt.xticks([0,   2,   4,  6,   8, 10,   12, 14,   16, 18,   20, 22,   24, 26,   28, 30,   32, 34,   36, 38,   40, 42,  44,  46,   48, 50,   52, 54,   56,  58,  60 ],
                       ["0", "", "4", "", "8", "", "12", "", "16", "", "20", "", "24", "", "28", "", "32", "", "36", "", "40", "", "44", "", "48", "", "52", "", "56", "", "60"])
            plt.show()
            print("Predicted next frame for channel 1.")
           
            """
            print(""); print("")
            plt.imshow(np.subtract(last_input_chan0, new_lstm_output[:,:,0]), cmap = cm.gray, vmin = 0.0, vmax = 1.0)
            print("Frames difference of Channel 0: ")
            plt.show()
            
            
            plt.imshow(np.subtract(last_input_chan1, new_lstm_output[:,:,1]), cmap = cm.gray, vmin = 0.0, vmax = 1.0)
            print("Frames difference of Channel 1: ")
            plt.show()
            """
            print("\n_______Output tracking is below________")
            print("Looking at eight 2-channel frames.")
            # Start with 2 to skip default output
            # Look at eight 2-channel frames
            f, axarr = plt.subplots(2, NUM_UNROLLINGS)
            for x in range(NUM_UNROLLINGS):
                axarr[0, x].imshow(new_lstm_output_array[x+1, :, :, 0], cmap = cm.gray, vmin = 0.0, vmax = 1.0)
                axarr[1, x].imshow(new_lstm_output_array[x+1, :, :, 1], cmap = cm.gray, vmin = 0.0, vmax = 1.0)
#            axarr[0, 1].imshow(new_lstm_output_array[2, :, :, 0], cmap = cm.gray, vmin = 0.0, vmax = 1.0)
#            axarr[0, 2].imshow(new_lstm_output_array[3, :, :, 0], cmap = cm.gray, vmin = 0.0, vmax = 1.0)
#            axarr[0, 3].imshow(new_lstm_output_array[4, :, :, 0], cmap = cm.gray, vmin = 0.0, vmax = 1.0)
#            axarr[0, 4].imshow(new_lstm_output_array[5, :, :, 0], cmap = cm.gray, vmin = 0.0, vmax = 1.0)
#            axarr[0, 5].imshow(new_lstm_output_array[6, :, :, 0], cmap = cm.gray, vmin = 0.0, vmax = 1.0)
#            axarr[0, 6].imshow(new_lstm_output_array[7, :, :, 0], cmap = cm.gray, vmin = 0.0, vmax = 1.0)
#            axarr[0, 7].imshow(new_lstm_output_array[8, :, :, 0], cmap = cm.gray, vmin = 0.0, vmax = 1.0)
#            
#            axarr[1, 1].imshow(new_lstm_output_array[2, :, :, 1], cmap = cm.gray, vmin = 0.0, vmax = 1.0)
#            axarr[1, 2].imshow(new_lstm_output_array[3, :, :, 1], cmap = cm.gray, vmin = 0.0, vmax = 1.0)
#            axarr[1, 3].imshow(new_lstm_output_array[4, :, :, 1], cmap = cm.gray, vmin = 0.0, vmax = 1.0)
#            axarr[1, 4].imshow(new_lstm_output_array[5, :, :, 1], cmap = cm.gray, vmin = 0.0, vmax = 1.0)
#            axarr[1, 5].imshow(new_lstm_output_array[6, :, :, 1], cmap = cm.gray, vmin = 0.0, vmax = 1.0)
#            axarr[1, 6].imshow(new_lstm_output_array[7, :, :, 1], cmap = cm.gray, vmin = 0.0, vmax = 1.0)
#            axarr[1, 7].imshow(new_lstm_output_array[8, :, :, 1], cmap = cm.gray, vmin = 0.0, vmax = 1.0)
            plt.show()
            
            
            for i in range(1, NUM_UNROLLINGS+1):
                out_i_chan0 = new_lstm_output_array[i, :, :, 0]
                out_i_chan0= np.resize(out_i_chan0, (IM_SZ_HGT, IM_SZ_WID))
                plt.imshow(out_i_chan0, cmap = cm.gray, vmin = 0.0, vmax = 1.0)
                plt.xticks([0,   2,   4,  6,   8, 10,   12, 14,   16, 18,   20, 22,   24, 26,   28, 30,   32, 34,   36, 38,   40, 42,  44,  46,   48, 50,   52, 54,   56,  58,  60 ],
                           ["0", "", "4", "", "8", "", "12", "", "16", "", "20", "", "24", "", "28", "", "32", "", "36", "", "40", "", "44", "", "48", "", "52", "", "56", "", "60"])
                plt.show()
                print("Prediction channel 0 unrollment number = ", int(i))
                out_i_chan1 = new_lstm_output_array[i, :, :, 1]
                out_i_chan1= np.resize(out_i_chan1, (IM_SZ_HGT, IM_SZ_WID))
                plt.imshow(out_i_chan1, cmap = cm.gray, vmin = 0.0, vmax = 1.0)
                plt.xticks([0,   2,   4,  6,   8, 10,   12, 14,   16, 18,   20, 22,   24, 26,   28, 30,   32, 34,   36, 38,   40, 42,  44,  46,   48, 50,   52, 54,   56,  58,  60 ],
                           ["0", "", "4", "", "8", "", "12", "", "16", "", "20", "", "24", "", "28", "", "32", "", "36", "", "40", "", "44", "", "48", "", "52", "", "56", "", "60"])
                plt.show()
                print("LMM channel 1 unrollment number  = ", int(i))
 
    print("FINAL LOSS: ", l_1)
    print("--- Run Time = %s seconds ---" % ((time.time() - start_time)))
    print("--- Run Time = %s minutes ---" % ((time.time() - start_time)/60.0))
'''""""""
