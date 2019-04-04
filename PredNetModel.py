"""
PredNet Tensorflow Implementation

Authored By: A. Maida, N. Elsayed, M. Hosseini, Z. Kirby

Modelled after Lotter et al (2017) paper on Predictive Coding applied to deep learning.
The Model features an convLSTM to predict t+1 images in sequence.

"""

import tensorflow as tf


class PredNetModel:
    """
    Usage: prednetObject = PredNetGraph()
           graph = prednetObject.graph

    Params:

    Returns: Invoking PredNetGraph() returns a PredNetGraph instance whose
             "graph" attribute references a graph representing the prednet layer.
    """
    num_instances = 0  # class variables
    kind = "prednet_layer"

    def __init__(self, **kwargs):
        # establish default values
        # and use keywords when invoking to change default values
        self.IN_CHANNELS = kwargs.pop("IN_CHANNELS", 1)
        self.CORE_CHANNELS = kwargs.pop("CORE_CHANNELS", 1)
        self.NUM_UNROLLINGS = kwargs.pop("NUM_UNROLLINGS", 3)
        self.IM_SZ_HGT = kwargs.pop("IM_SZ_HGT", 64)
        self.IM_SZ_WID = kwargs.pop("IM_SZ_WID", 64)
        self.INIT_LEARNING_RATE = kwargs.pop("INIT_LEARNING_RATE", 0.1)
        self.PEEPHOLE = kwargs.pop("PEEPHOLE", False)  # Nelly
        self.LAYER = kwargs.pop('LAYER', 0)  # Kirby

        if len(kwargs) != 0:
            print("WARNING: PredNet instance constructed with spurious values.")
            for key, value in kwargs.items():
                print("Key is {}. Value is {}.".format(key, value))

        self.initial_lstm_state = None
        self.initial_lstm_output = None
        self.initial_err_input = None
        self.lstm_state = None
        self.lstm_output = None
        self.err_input = None
        self.p_holder = None
        self.loss = None
        self.learning_rate = None
        self.optimizer = None
        #       self.peephole            = False                      # Added by Nelly
        self.lstm_output_array = None  # Added by Nelly
        self.err_output = None  # Added by Kirby
        self.lstm_upsample = None

        self.__set_up_LSTM_wts()
        self.__build_the_prednet_layer()
        self.__setup_loss_and_optimizer()
        self.__setup_tensorboard_visualization()
        PredNetModel.num_instances += 1  # class variable

    def __del__(self):
        PredNetModel.num_instances -= 1

    """
     WEIGHTS
        Variable (wt) definitions. Only variables can be trained.
        Naming conventions use *Deep Learning*, Goodfellow et al, 2016.
    """

    def __set_up_LSTM_wts(self):
        #        with self.graph.as_default():
        # input update wts
        self.U = tf.Variable(
            tf.truncated_normal([5, 5, self.IN_CHANNELS, self.CORE_CHANNELS], mean=-0.1, stddev=0.1, seed=1), name="U")
        self.W = tf.Variable(
            tf.truncated_normal([5, 5, self.CORE_CHANNELS, self.IN_CHANNELS], mean=-0.1, stddev=0.1, seed=2), name="W")
        self.B = tf.Variable(tf.ones([self.CORE_CHANNELS]), name="B")

        # input gate (g_gate): input, prev output, bias
        self.Ug = tf.Variable(
            tf.truncated_normal([5, 5, self.IN_CHANNELS, self.CORE_CHANNELS], mean=-0.1, stddev=0.1, seed=3), name="Ug")
        self.Wg = tf.Variable(
            tf.truncated_normal([5, 5, self.CORE_CHANNELS, self.IN_CHANNELS], mean=-0.1, stddev=0.1, seed=4), name="Wg")
        self.Bg = tf.Variable(tf.ones([self.CORE_CHANNELS]), name="Bg")

        # forget gate (f_gate): input, prev output, bias
        self.Uf = tf.Variable(
            tf.truncated_normal([5, 5, self.IN_CHANNELS, self.CORE_CHANNELS], mean=-0.1, stddev=0.1, seed=5), name="Uf")
        self.Wf = tf.Variable(
            tf.truncated_normal([5, 5, self.CORE_CHANNELS, self.IN_CHANNELS], mean=-0.1, stddev=0.1, seed=6), name="Wf")
        self.Bf = tf.Variable(tf.ones([self.CORE_CHANNELS]), name="Bf")

        # output gate (q_gate): input, prev output, bias
        self.Uo = tf.Variable(
            tf.truncated_normal([5, 5, self.IN_CHANNELS, self.CORE_CHANNELS], mean=-0.1, stddev=0.1, seed=7), name="Uo")
        self.Wo = tf.Variable(
            tf.truncated_normal([5, 5, self.CORE_CHANNELS, self.IN_CHANNELS], mean=-0.1, stddev=0.1, seed=8), name="Wo")
        self.Bo = tf.Variable(tf.ones([self.CORE_CHANNELS]), name="Bo")

        # for peephole weight initialization
        if self.PEEPHOLE == True:
            self.Wci = tf.Variable(
                tf.random_normal([1, self.IM_SZ_HGT, self.IM_SZ_WID, self.CORE_CHANNELS], mean=-0.1, stddev=0.1,
                                 seed=9), name="Wci")
            self.Wcf = tf.Variable(
                tf.random_normal([1, self.IM_SZ_HGT, self.IM_SZ_WID, self.CORE_CHANNELS], mean=-0.1, stddev=0.1,
                                 seed=10), name="Wcf")
            self.Wco = tf.Variable(
                tf.random_normal([1, self.IM_SZ_HGT, self.IM_SZ_WID, self.CORE_CHANNELS], mean=-0.1, stddev=0.1,
                                 seed=11), name="Wxc")

    """
    BUILD PREDNET LAYER
    """

    def __build_the_prednet_layer(self):
        #        with self.graph.as_default():
        """
        INITIALIZATIONS
        """
        self.initial_lstm_state = tf.zeros([self.NUM_UNROLLINGS, self.IM_SZ_HGT, self.IM_SZ_WID, self.CORE_CHANNELS])
        self.initial_lstm_output = tf.zeros([self.NUM_UNROLLINGS, self.IM_SZ_HGT, self.IM_SZ_WID, self.CORE_CHANNELS])
        self.initial_err_input = tf.zeros([self.NUM_UNROLLINGS, self.IM_SZ_HGT, self.IM_SZ_WID, self.IN_CHANNELS])

        """
        BUILD PREDICTIVE CODING LAYER
        """
        self.lstm_state = self.initial_lstm_state
        self.lstm_output = self.initial_lstm_output
        self.err_input = self.initial_err_input
        self.lstm_output_array = self.lstm_output  # [1, 64, 64, 2]

        # p_holder holds, by default, 3 consecutive frames from the video.
        # The video has two channels.
        self.p_holder = tf.placeholder(tf.float32,
                                       [self.NUM_UNROLLINGS,
                                        self.IM_SZ_HGT,
                                        self.IM_SZ_WID,
                                        self.IN_CHANNELS])
        with tf.name_scope("PredNet"):
            for f in range(self.NUM_UNROLLINGS):  # the LSTM weights are shared across unrollings
                with tf.name_scope('convLSTM'):
                    self.lstm_state, self.lstm_output = self.__make_convLSTM_cell(self.err_input, self.lstm_state,
                                                                                  self.lstm_output)
                with tf.name_scope('error'):
                    self.err_input = self.__make_error_module(self.p_holder[f, :, :, :], self.lstm_output)
                # Save history of LSTM outputs. Added by Nelly.
                self.lstm_output_array = tf.concat([self.lstm_output_array, self.lstm_output],
                                                   axis=0)  # Tony: change from 3 to 0
                # X = tf.cast(self.err_input,tf.float32)
                # self.err_output_array = tf.concat([self.err_output_array, self.err_input], axis = 0)
                # lstm_output_array: [1, 64, 64, 2 * (NUM_UNROLLINGS + 1)]
                # 1st set of frames is default output. Possible revision is to leave out.

    """
    LOSS FUNCTION AND OPTIMIZER
    """

    def __setup_loss_and_optimizer(self):
        #        with self.graph.as_default():
        # loss
        self.loss = tf.reduce_sum(tf.abs(self.err_input))  # sums the values across each component of the tensor
        global_step = tf.Variable(0)
        # optimizer
        self.learning_rate = tf.train.exponential_decay(
            self.INIT_LEARNING_RATE, global_step, 300, 0.95, staircase=True, name='LearningRate')
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        # gradients, v = zip(*self.optimizer.compute_gradients(self.loss))
        # gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
        # self.optimizer = self.optimizer.apply_gradients(zip(gradients, v), global_step=global_step)

    """
    LSTM cell building function
        Builds a cell for a convLSTM.
        The above weights are used in this definition.
    """

    def __make_convLSTM_cell(self, err_inp, prev_s, prev_h):
        """ 
            Build a convLSTM cell w/o peephole connections.
            Input args: 
                 err_inp:  current input    (tensor: [1, 64, 64, in_channels])
                 prev_h :  previous output  (tensor: [1, 64, 64, core_channels])
                 prev_s :  previous state   (tensor: [1, 64, 64, core_channels])
            Returns: 
                     s  :  current state    (tensor: [1, 64, 64, core_channels])
                     h  :  current output   (tensor: [1, 64, 64, core_channels])
            Meaning of tensor dims:
                     [BATCH_SZ, IM_SZ_HGT, IM_SZ_WID, channels (maps)]
        """
        #        with self.graph.as_default():

        if self.PEEPHOLE == False:
            self.inp = tf.sigmoid(tf.nn.conv2d(err_inp, self.U, [1, 1, 1, 1], padding='SAME')
                                  + tf.nn.conv2d(prev_h, self.W, [1, 1, 1, 1], padding='SAME')
                                  + self.B)
            self.g_gate = tf.tanh(tf.nn.conv2d(err_inp, self.Ug, [1, 1, 1, 1], padding='SAME')
                                  + tf.nn.conv2d(prev_h, self.Wg, [1, 1, 1, 1], padding='SAME')
                                  + self.Bg, name="g_gate")
            self.f_gate = tf.sigmoid(tf.nn.conv2d(err_inp, self.Uf, [1, 1, 1, 1], padding='SAME')
                                     + tf.nn.conv2d(prev_h, self.Wf, [1, 1, 1, 1], padding='SAME')
                                     + self.Bf)
            self.q_gate = tf.sigmoid(tf.nn.conv2d(err_inp, self.Uo, [1, 1, 1, 1], padding='SAME')
                                     + tf.nn.conv2d(prev_h, self.Wo, [1, 1, 1, 1], padding='SAME')
                                     + self.Bo)
            s = tf.add(tf.multiply(self.f_gate, prev_s), tf.multiply(self.g_gate, self.inp), name="state")
            if not self.lstm_upsample:
                h = tf.multiply(self.q_gate, tf.nn.tanh(s), name="output")
            else:
                h = tf.multiply(self.q_gate, tf.nn.tanh(s), name="output")
                h = tf.image.resize_images(h, [self.NUM_UNROLLINGS, self.IM_SZ_HGT, self.IM_SZ_WID, self.IN_CHANNELS])

        elif self.PEEPHOLE == True:  # with peephole
            self.inp = tf.contrib.keras.activations.hard_sigmoid(
                tf.add((tf.nn.conv2d(err_inp, self.U, [1, 1, 1, 1], padding='SAME')
                        + tf.nn.conv2d(prev_h, self.W, [1, 1, 1, 1], padding='SAME'))
                       , tf.multiply(self.Wci, prev_s)) + self.B)
            self.g_gate = tf.contrib.keras.activations.hard_sigmoid(
                tf.nn.conv2d(err_inp, self.Ug, [1, 1, 1, 1], padding='SAME')
                + tf.nn.conv2d(prev_h, self.Wg, [1, 1, 1, 1], padding='SAME') + self.Bg)  # i_gate is more common name
            self.f_gate = tf.contrib.keras.activations.hard_sigmoid(
                tf.add((tf.nn.conv2d(err_inp, self.Uf, [1, 1, 1, 1], padding='SAME')
                        + tf.nn.conv2d(prev_h, self.Wf, [1, 1, 1, 1], padding='SAME')),
                       tf.multiply(self.Wcf, prev_s)) + self.Bf)
            self.q_gate = tf.contrib.keras.activations.hard_sigmoid(
                tf.add((tf.nn.conv2d(err_inp, self.Uo, [1, 1, 1, 1], padding='SAME')
                        + tf.nn.conv2d(prev_h, self.Wo, [1, 1, 1, 1], padding='SAME')),
                       tf.multiply(self.Wco, prev_s)) + self.Bo)
            s = tf.add(tf.multiply(self.f_gate, prev_s), tf.multiply(self.g_gate, self.inp), name="state")
            h = tf.multiply(self.q_gate, tf.tanh(s), name="output")

        return s, h  # normally above is tanh

    """
    ERROR MODULE building function
        Doesn't use variables, so doesn't undergo training.
    """

    def __make_error_module(self, image, predict):
        with tf.name_scope("ErrModule"):
            image = tf.reshape(image, [1, self.IM_SZ_HGT, self.IM_SZ_WID, self.IN_CHANNELS])
            #            print("errorModule: predict shape : ", predict.get_shape())
            #            print("errorModule: image shape: ", image.get_shape())
            net_value = tf.squared_difference(image, predict)
            #            print("errorModule: net_value:", net_value.get_shape())
            err1 = tf.identity(net_value, name="E1")
            self.err_output = err1
            return err1

    """
    SETUP TENSORBOARD VISUALIZATION
    """

    def __setup_tensorboard_visualization(self):
        #        with self.graph.as_default():
        with tf.name_scope("initializations"):
            tf.summary.image("initial_lstm_state", self.initial_lstm_state, 3)
            tf.summary.image("initial_lstm_output", self.initial_lstm_output, 3)
            tf.summary.image("initial_error1",
                             tf.slice(self.initial_err_input, [0, 0, 0, 0], [1, 64, 64, 1]), 3)
            tf.summary.image("initial_error2",
                             tf.slice(self.initial_err_input, [0, 0, 0, 1], [1, 64, 64, 1]), 3)
        #        with tf.name_scope("input"):
        #            tf.summary.image("image", image, 3)
        with tf.name_scope("lstm"):
            tf.summary.image("lstm_out", self.lstm_output, 3)
            tf.summary.image("lstm_state", self.lstm_state, 3)
            with tf.name_scope("error"):
                tf.summary.image("perror_1",
                                 tf.slice(self.err_input, [0, 0, 0, 0], [1, 64, 64, 1]), 3)
                tf.summary.image("perror_2",
                                 tf.slice(self.err_input, [0, 0, 0, 1], [1, 64, 64, 1]), 3)

            with tf.name_scope('optimizer'):
                tf.summary.scalar('loss', self.loss)
                tf.summary.scalar('learning_rate', self.learning_rate)

            with tf.name_scope('weights'):
                with tf.name_scope('1_input_update'):
                    newU1 = tf.slice(self.U, [0, 0, 0, 0], [5, 5, 1, 1])
                    newU2 = tf.slice(self.U, [0, 0, 1, 0], [5, 5, 1, 1])
                    newW = tf.slice(self.W, [0, 0, 0, 0], [5, 5, 1, 1])
                    newU1 = tf.squeeze(newU1)  # now a viewable [5x5] matrix
                    newU2 = tf.squeeze(newU2)
                    newW = tf.squeeze(newW)
                    newU1 = tf.reshape(newU1, [1, 5, 5, 1])
                    newU2 = tf.reshape(newU2, [1, 5, 5, 1])
                    newW = tf.reshape(newW, [1, 5, 5, 1])
                    tf.summary.image('U1', newU1)
                    tf.summary.image('U2', newU2)
                    tf.summary.image('W', newW)
                    tf.summary.image('B', self.B)

            with tf.name_scope('2_input_gate'):
                newUg1 = tf.slice(self.Ug, [0, 0, 0, 0], [5, 5, 1, 1])
                newUg2 = tf.slice(self.Ug, [0, 0, 1, 0], [5, 5, 1, 1])
                newWg = tf.slice(self.Wg, [0, 0, 0, 0], [5, 5, 1, 1])
                newUg1 = tf.squeeze(newUg1)  # now a viewable [5x5] matrix
                newUg2 = tf.squeeze(newUg2)
                newWg = tf.squeeze(newWg)
                newUg1 = tf.reshape(newUg1, [1, 5, 5, 1])
                newUg2 = tf.reshape(newUg2, [1, 5, 5, 1])
                newWg = tf.reshape(newWg, [1, 5, 5, 1])
                tf.summary.image('Ug1', newUg1)
                tf.summary.image('Ug2', newUg2)
                tf.summary.image('Wg', newWg)
                tf.summary.image('Bg', self.Bg)

            with tf.name_scope('3_forget_gate'):
                newUf1 = tf.slice(self.Uf, [0, 0, 0, 0], [5, 5, 1, 1])
                newUf2 = tf.slice(self.Uf, [0, 0, 1, 0], [5, 5, 1, 1])
                newWf = tf.slice(self.Wf, [0, 0, 0, 0], [5, 5, 1, 1])
                newUf1 = tf.squeeze(newUf1)  # now a viewable [5x5] matrix
                newUf2 = tf.squeeze(newUf2)
                newWf = tf.squeeze(newWf)
                newUf1 = tf.reshape(newUf1, [1, 5, 5, 1])
                newUf2 = tf.reshape(newUf2, [1, 5, 5, 1])
                newWf = tf.reshape(newWf, [1, 5, 5, 1])
                tf.summary.image('Uf1', newUf1)
                tf.summary.image('Uf2', newUf2)
                tf.summary.image('Wf', newWf)
                tf.summary.image('Bf', self.Bf)

            with tf.name_scope('4_output_gate'):
                newUo1 = tf.slice(self.Uo, [0, 0, 0, 0], [5, 5, 1, 1])
                newUo2 = tf.slice(self.Uo, [0, 0, 1, 0], [5, 5, 1, 1])
                newWo = tf.slice(self.Wo, [0, 0, 0, 0], [5, 5, 1, 1])
                newUo1 = tf.squeeze(newUo1)  # now a viewable [5x5] matrix
                newUo2 = tf.squeeze(newUo2)
                newWo = tf.squeeze(newWo)
                newUo1 = tf.reshape(newUo1, [1, 5, 5, 1])
                newUo2 = tf.reshape(newUo2, [1, 5, 5, 1])
                newWo = tf.reshape(newWo, [1, 5, 5, 1])
                tf.summary.image('Uo1', newUo1)
                tf.summary.image('Uo2', newUo2)
                tf.summary.image('Wo', newWo)
                tf.summary.image('Bo', self.Bo)
