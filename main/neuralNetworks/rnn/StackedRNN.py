import tensorflow as tf
import numpy as np


class StackedRNN:
    def __init__(self, input_size, lstm_size, number_of_layers, output_size, session, learning_rate, name="rnn"):
        self.scope = name
        self.input_size = input_size
        self.lstm_size = lstm_size
        self.number_of_layers = number_of_layers
        self.output_size = output_size

        self.session = session
        self.learning_rate = tf.constant(learning_rate)

        self.lstm_last_state = np.zeros(shape=(self.number_of_layers * 2 * self.lstm_size,))

        with tf.device('/device:GPU:0'):
            with tf.variable_scope(name_or_scope=self.scope):
                with tf.variable_scope(name_or_scope="Input"):
                    self.X = tf.placeholder(dtype=tf.float32, shape=(None, None, self.input_size), name="X")
                    self.lstm_init_value = tf.placeholder(dtype=tf.float32,
                                                          shape=(None, self.number_of_layers * 2 * self.lstm_size),
                                                          name="lstm_init_value")

                with tf.name_scope(name="RNN_Core"):
                    self.lstm_cells = [tf.contrib.rnn.BasicLSTMCell(self.lstm_size, forget_bias=1.0, state_is_tuple=False)
                                       for _
                                       in range(self.number_of_layers)]
                    self.lstm = tf.contrib.rnn.MultiRNNCell(self.lstm_cells, state_is_tuple=False)
                    self.outputs, self.lstm_current_state = tf.nn.dynamic_rnn(cell=self.lstm, inputs=self.X,
                                                                              initial_state=self.lstm_init_value,
                                                                              dtype=tf.float32)

                # with tf.name_scope(name="DropOut"):
                #     self.keep_prob = tf.placeholder(tf.float32)
                #     self.lstm_current_state = tf.nn.dropout(self.lstm_current_state, self.keep_prob)

                with tf.variable_scope(name_or_scope="Output"):
                    self.OUT_W = tf.Variable(initial_value=tf.random_normal(shape=(self.lstm_size, self.output_size),
                                                                            stddev=0.01, name="output_W"))
                    self.OUT_B = tf.Variable(initial_value=tf.random_normal(shape=(self.output_size,),
                                                                            stddev=0.01, name="output_B"))
                    self.outputs_reshaped = tf.reshape(tensor=self.outputs, shape=[-1, self.lstm_size])
                    self.net_out = tf.matmul(self.outputs_reshaped, self.OUT_W) + self.OUT_B

                    self.batch_time_shape = tf.shape(self.outputs)
                    self.final_output = tf.reshape(tf.nn.softmax(logits=self.net_out), shape=(self.batch_time_shape[0],
                                                                                              self.batch_time_shape[1],
                                                                                              self.output_size))

                    self.Y = tf.placeholder(dtype=tf.float32,
                                            shape=(None, None, self.output_size))

                    self.Y_long = tf.reshape(tensor=self.Y, shape=(-1, self.output_size))

                    self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.net_out,
                                                                                       labels=self.Y_long))

                    # self.train_op = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.9).minimize(self.cost)
                    self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        self.print_number_of_parameters()

    def run_step(self, X, init_zero_state=True):
        if init_zero_state:
            init_value = np.zeros(shape=(self.number_of_layers * 2 * self.lstm_size,))
        else:
            init_value = self.lstm_last_state

        out, next_lstm_state = self.session.run([self.final_output, self.lstm_current_state],
                                                feed_dict={self.X: [X], self.lstm_init_value: [init_value]})

        self.lstm_last_state = next_lstm_state[0]

        return out[0][0]

    def train_batch(self, Xbatch, Ybatch):
        init_value = np.zeros(shape=(Xbatch.shape[0], self.number_of_layers * 2 * self.lstm_size))

        cost, _ = self.session.run([self.cost, self.train_op], feed_dict={self.X: Xbatch, self.Y: Ybatch,
                                                                          self.lstm_init_value: init_value})

        return cost

    def print_number_of_parameters(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print("total number of parameters: ", total_parameters)
