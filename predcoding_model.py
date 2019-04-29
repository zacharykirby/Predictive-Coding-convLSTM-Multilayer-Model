#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: predcoding_model.py
Created on Wed Jun 28 21:22:59 2017
Last modified: Wed Dec 15, 2017

Contributers: A. Maida, N. Elsayed, M. Hosseini, Z. Kirby

This is a convolutional LSTM prototype for predictive coding.

Input types are:
    1. Half plane moving to the right.
    2. A single moving MNIST video

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

# Custom python imports
import MultiLayerPredNet as MLP
import PredNetModel

# Basic system stuff
import os
import sys
import time

# Advanced imports
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from collections import defaultdict

# Debug prints for good practice
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

# Constant variables for training, change as needed
LOGDIR = "/tmp/convLSTM/" # where to log tensorboard
LENGTH_OF_VID = 30 # length of video to create
IM_SZ_HGT = 64  # image height
IM_SZ_WID = 64 # image width
BATCH_SZ = 1 # batch size (change later?)
NUM_UNROLLINGS = 8 # how many steps in time should our LSTM see?
INIT_LEARNING_RATE = 0.1  # initial learning rate
NUM_TRAINING_STEPS = 300 # times to run computational graph
VIDEO_CHANNELS = 1 # only change if using RGB or local movement map

LAYER_COUNT = 4 # how many layers of prednet?
LAYER_WISE = False  # Whether or not we train each layer's optimizer
MOVING_MNIST = True # use moving mnist or not (defaults to moving plane)

"""
Create Input Video with 1 Channels:
  Channel 0: Video of a moving half-plane to the right.
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
    
    This file was generated elsewhere using a script called moving_mnist.py.
"""
if MOVING_MNIST:
    with np.load('movingmnistdata.npz') as data:
        a = data['arr_0']
        a = np.squeeze(a)

    # Take first video sequence
    in_video = a[:LENGTH_OF_VID, :, :]
    in_video = np.expand_dims(in_video, axis=-1)
    #for x in range(LENGTH_OF_VID):
    #    plt.imshow(np.squeeze(in_video[x:(x + 1), :, :]), cmap=cm.gray_r, vmin=0.0, vmax=1.0)
    #    plt.show()

# Generate default prediction, which is a gray square
default_prediction = np.empty([IM_SZ_HGT, IM_SZ_WID, VIDEO_CHANNELS], dtype=np.float32)
default_prediction[:, :, :] = 0.5
plt.imshow(default_prediction[:, :, 0], cmap=cm.gray_r, vmin=0.0, vmax=1.0)
plt.title('Default Prediction')
plt.show()

"""
 BUILD the GRAPH
     Convolutional LSTM.
     Each node in the LSTM is a stack of two feature maps.
     Internal convolutions are 5x5 w/ stride of 1 (usually!).
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
    
    # Define our multilayer prednet helper object
    MLP = MLP.MultiLayerPredNet(layer_count=LAYER_COUNT,
                                img_width=IM_SZ_WID,
                                img_height=IM_SZ_HGT,
                                num_unrollings=NUM_UNROLLINGS,
                                channels=VIDEO_CHANNELS)
    
    # Build layers and return a list of them
    prednets = MLP.build_layers_()

    # Upsample weights for conv2d, i think i need these here
    upsample_W = list()
    channels = VIDEO_CHANNELS
    for x in range(LAYER_COUNT + 1):
        
        if x == 0:
            temp_W = tf.Variable(tf.zeros(1)) #who cares what's in here
        
        if x == 1:
            # For our initial upsample, we go to 32 channels (don't ask why)
            channels = VIDEO_CHANNELS
            temp_W = tf.Variable(tf.truncated_normal([5, 5, channels, 32]))
            channels = 32
            
        else:
            # Gets twice as deep each iteration
            temp_W = tf.Variable(tf.truncated_normal([5, 5, channels, channels * 2]))
            channels *= 2
            
        upsample_W.append(temp_W)

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

    # Since we don't know how many layers exist, make a defaultdict of lists
    losses = defaultdict(list)
    for step in range(NUM_TRAINING_STEPS):
        
        index = step % (LENGTH_OF_VID - NUM_UNROLLINGS)  # select starting frame of video segment
        #index = 10 # test if it can learn one single sequence

        data_for_this_iteration = in_video[index:(index + NUM_UNROLLINGS), :, :, :]  # video segment is NUM_UNROLLINGS long
        
        # Modular code is good!
        for x in range(LAYER_COUNT):
            if x == 0:
                # Again, handle layer 0 specially. No upsample
                fd = {prednets[x].p_holder: data_for_this_iteration}
                _, loss, lstm_out, err_out = sess.run([prednets[x].optimizer,
                                                       prednets[x].loss,
                                                       prednets[x].lstm_output,
                                                       prednets[x].err_output],
                                                       feed_dict=fd)
                losses['layer_'+str(x)].append(loss)
            
            elif x == 1:
                new_width = int(IM_SZ_HGT / 2)
                new_height = int(IM_SZ_WID / 2)
                new_channels = 32  # why is this like this?
                
                # THe upsample weights are defined in the graph and added to upsample_W list
                new_data = tf.nn.conv2d(err_out, upsample_W[x], [1, 1, 1, 1], padding='SAME').eval()
                new_data = tf.nn.relu(new_data).eval()
                new_data = tf.nn.max_pool(new_data, [1, 5, 5, 1], [1, 2, 2, 1], padding='SAME').eval()
                fd = {prednets[x].p_holder: new_data}
                
                if LAYER_WISE:
                    _, loss, lstm_out, err_out = sess.run([prednets[x].optimizer,
                                                       prednets[x].loss,
                                                       prednets[x].lstm_output,
                                                       prednets[x].err_output],
                                                       feed_dict=fd)
                    losses['layer_'+str(x)].append(loss)
                    
                else:
                    lstm_out, err_out = sess.run([prednets[x].lstm_output,
                                                  prednets[x].err_output],
                                                  feed_dict=fd)
            
            else:
                new_width = int(IM_SZ_HGT / 2)
                new_height = int(IM_SZ_WID / 2)
                new_channels = int(new_channels * 2)
                
                new_data = tf.nn.conv2d(err_out, upsample_W[x], [1, 1, 1, 1], padding='SAME').eval()
                new_data = tf.nn.relu(new_data).eval()
                new_data = tf.nn.max_pool(new_data, [1, 5, 5, 1], [1, 2, 2, 1], padding='SAME').eval()
                fd = {prednets[x].p_holder: new_data}
                
                if LAYER_WISE:
                    _, loss, lstm_out, err_out = sess.run([prednets[x].optimizer,
                                                       prednets[x].loss,
                                                       prednets[x].lstm_output,
                                                       prednets[x].err_output],
                                                       feed_dict=fd)
                    losses['layer_'+str(x)].append(loss)
                else:
                    lstm_out, err_out = sess.run([prednets[x].lstm_output,
                                                  prednets[x].err_output],
                                                  feed_dict=fd)
        # Begin DEBUG outputs
        if step % 10 == 0:
            print("Step: ", step)
            if not LAYER_WISE:
                print("> Loss               : ", np.mean(losses['layer_0']))
            if LAYER_WISE:
                print('LAYER-WISE LOSS')
                for x in range(LAYER_COUNT):
                    print('> Layer ' + str(x) + ': ', np.mean(losses['layer_' + str(x)]))
            print("------------------------------------")
        if step == (NUM_TRAINING_STEPS - 1):
            print("Output at end of training from 'new_lstm_output' for {} training steps.".format(NUM_TRAINING_STEPS))
            out = np.squeeze(lstm_out)

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
