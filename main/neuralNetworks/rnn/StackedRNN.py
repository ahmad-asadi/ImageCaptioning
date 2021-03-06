import tensorflow as tf
import numpy as np


class StackedRNN:
    def __init__(self, input_size, lstm_size, number_of_layers, output_size, session, learning_rate, batch_size,
                 attn_length=0, attn_size=0, name="rnn", cnn_graph=None):
        self.scope = name
        self.cnn_graph = cnn_graph
        self.attn_length = attn_length
        self.attn_size = attn_size
        self.input_size = input_size
        self.batch_size = batch_size
        print("input size:", input_size)
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
                    # self.lstm_init_value = tf.placeholder(dtype=tf.float32,
                    #                                       shape=(None, self.number_of_layers * 2 * self.lstm_size),
                    #                                       name="lstm_init_value")

                with tf.variable_scope(name_or_scope="DropOutParams"):
                    self.keep_prob = tf.placeholder(dtype=tf.float32)

                with tf.name_scope(name="RNN_Core"):
                    # self.lstm_cells = [tf.nn.rnn_cell.LSTMCell(self.lstm_size, forget_bias=1.0,
                    #                                            state_is_tuple=True, initializer=tf.zeros_initializer)
                    #                    for _
                    #                    in range(self.number_of_layers)]
                    self.lstm_cells = [tf.nn.rnn_cell.BasicLSTMCell(self.lstm_size, forget_bias=1.0,
                                                                    state_is_tuple=True)
                                       for _
                                       in range(self.number_of_layers)]
                    # self.lstm_cells = [tf.contrib.rnn.AttentionCellWrapper(cell=lstm_cell,
                    #                                                        attn_length=self.attn_length,
                    #                                                        state_is_tuple=True)
                    #                    for lstm_cell
                    #                    in self.lstm_cells]
                    # self.lstm_cells = [tf.nn.rnn_cell.GRUCell(num_units=self.lstm_size)
                    #                    for _
                    #                    in range(self.number_of_layers)]
                    #
                    # self.lstm_init_states = [rnn_cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
                    #                          for rnn_cell in self.lstm_cells]

                    self.lstm_cells = [tf.nn.rnn_cell.DropoutWrapper(lstm_cell,
                                                                     output_keep_prob=self.keep_prob)
                                       for lstm_cell in self.lstm_cells]
                    self.lstm = tf.nn.rnn_cell.MultiRNNCell(self.lstm_cells, state_is_tuple=True)
                    self.lstm = tf.nn.rnn_cell.DropoutWrapper(self.lstm, output_keep_prob=self.keep_prob)
                    self.lstm_init_states = self.lstm.zero_state(batch_size=tf.shape(self.X)[1], dtype=tf.float32)
                    self.outputs, self.lstm_current_state = tf.nn.dynamic_rnn(cell=self.lstm, inputs=self.X,
                                                                              dtype=tf.float32, time_major=True,
                                                                              initial_state=self.lstm_init_states)

                with tf.variable_scope(name_or_scope="Output"):
                    self.OUT_W = tf.Variable(initial_value=tf.random_normal(shape=(self.lstm_size, self.output_size),
                                                                            stddev=0.01, name="output_W"))
                    self.OUT_B = tf.Variable(initial_value=tf.random_normal(shape=(self.output_size,),
                                                                            stddev=0.01, name="output_B"))

                    self.outputs_reshaped = tf.reshape(tensor=self.outputs, shape=[-1, self.lstm_size])
                    self.net_out = tf.nn.batch_normalization(x=tf.matmul(self.outputs_reshaped, self.OUT_W) + self.OUT_B
                                                             , mean=0, variance=1, offset=0, scale=1,
                                                             variance_epsilon=1e-10)

                    self.batch_time_shape = tf.shape(self.outputs)
                    # self.softmax_net_out = tf.nn.softmax(logits=self.net_out)
                    self.softmax_net_out = self.net_out
                    self.final_output = tf.reshape(self.softmax_net_out, shape=(self.batch_time_shape[0],
                                                                                self.batch_time_shape[1],
                                                                                self.output_size))

                    self.Y = tf.placeholder(dtype=tf.float32,
                                            shape=(None, None, self.output_size))

                    self.Y_long = tf.reshape(tensor=self.Y, shape=(-1, self.output_size))

                    # self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.net_out,
                    #                                                                    labels=self.Y_long))
                    # self.cost = tf.reduce_sum(tf.losses.mean_squared_error(predictions=self.softmax_net_out,
                    #                                                        labels=self.Y_long))
                    self.cost = tf.reduce_sum(tf.losses.absolute_difference(predictions=self.softmax_net_out,
                                                                           labels=self.Y_long))

                    # self.train_op = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.9).minimize(self.cost)
                    self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
                    # self.train_op = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
                    # self.train_op = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(
                    #     self.cost)
        self.print_number_of_parameters()

    def run_step(self, X, init_zero_state=True, keep_prob=1):
        if init_zero_state:
            init_value = np.zeros(shape=(self.number_of_layers * 2 * self.lstm_size,))
        else:
            init_value = self.lstm_last_state

        out, next_lstm_state = self.session.run([self.final_output, self.lstm_current_state],
                                                feed_dict={self.X: X,
                                                           # self.lstm_init_value: [init_value],
                                                           self.keep_prob: keep_prob})

        self.lstm_last_state = next_lstm_state[len(next_lstm_state) - 1]

        return out

    def train_batch(self, Xbatch, Ybatch, keep_prob=1):
        init_value = np.zeros(shape=(Xbatch.shape[0], self.number_of_layers * 2 * self.lstm_size))

        cost, _ = self.session.run([self.cost, self.train_op], feed_dict={self.X: Xbatch, self.Y: Ybatch,
                                                                          # self.lstm_init_value: init_value,
                                                                          self.keep_prob: keep_prob})

        return cost

    @staticmethod
    def print_number_of_parameters():
        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print("total number of parameters: ", total_parameters)
