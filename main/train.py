import tensorflow as tf
import numpy as np
import random
import os
import time

from data_utils import data_helper
from main.neuralNetworks.cnn import CNN
from main.neuralNetworks.rnn import StackedRNN as RNN
from main.neuralNetworks.rnn import RNNUtils


def start():
    data_dir = "../../Dataset/MS-COCO/"

    print("loading dataset...")
    cocoHelper = data_helper.COCOHelper(data_dir + "annotations/captions_train2014.json")

    print("creating network structure")
    cnn = CNN.CNN(os.path.join(data_dir, "train2014"))
    img = cocoHelper.imgs[9]
    image_features = cnn.classify(os.path.join(data_dir, "train2014/" + img['file_name']))

    rnnUtils = RNNUtils.RNNUtils()
    rawData, vocab = rnnUtils.load_data()
    data = rnnUtils.embed_to_vocab(data_=rawData, vocab=vocab)

    RNNOptions = RNNUtils.RNNOptions(vocab=vocab, rnnUtils=rnnUtils, image_feature_size=len(image_features))

    rnn = RNN.StackedRNN(input_size=RNNOptions.input_size + RNNOptions.image_feature_size, lstm_size=RNNOptions.lstm_size,
                         number_of_layers=RNNOptions.num_layers, output_size=RNNOptions.out_size,
                         session=RNNOptions.session, learning_rate=RNNOptions.learning_rate, name=RNNOptions.name)

    print_graph(RNNOptions)

    RNNOptions.session.run(tf.global_variables_initializer())

    RNNOptions.saver = tf.train.Saver(tf.global_variables())

    maximum_iteration_count = 200
    start_time = time.time()
    batch_input = np.zeros(shape=(RNNOptions.batch_size, RNNOptions.time_step,
                                  RNNOptions.input_size + RNNOptions.image_feature_size))
    batch_labels = np.zeros(shape=(RNNOptions.batch_size, RNNOptions.time_step, RNNOptions.input_size))

    possible_batch_ids = range(data.shape[0] - RNNOptions.time_step - 1)
    costs = np.zeros(shape=maximum_iteration_count)
    for i in range(maximum_iteration_count):
        print(i)
        batch_id = random.sample(possible_batch_ids, RNNOptions.batch_size)
        image_features_reshaped = np.reshape(np.repeat(image_features, RNNOptions.batch_size),
                                             newshape=(RNNOptions.batch_size, RNNOptions.image_feature_size))
        for j in range(RNNOptions.time_step):
            input_ind = [k + j for k in batch_id]
            label_ind = [k + j + 1 for k in batch_id]
            tmp = np.concatenate([data[input_ind, :], image_features_reshaped], axis=1)
            batch_input[:, j, :] = tmp
            batch_labels[:, j, :] = data[label_ind, :]

        costs[i] = rnn.train_batch(Xbatch=batch_input, Ybatch=batch_labels)

        if (i % 100) == 0:
            end_time = time.time()
            duration = (end_time - start_time)/100
            print("batch ", i, ", train time per batch: ", duration)
            start_time = end_time

        if (i % 500) == 0:
            print("saving current model...")
            RNNOptions.saver.save(RNNOptions.session, RNNOptions.saved_model_path)
            # print("testing current model with prefix string: " + RNNOptions.test_prefix_string)
            # test_model(RNNOptions, rnn)


def print_graph(RNNOptions):
    print("writing graphs to /tmp/graph in order to show it in tensor board")
    writer = tf.summary.FileWriter("/tmp/graph")
    writer.add_graph(RNNOptions.session.graph)


def test_model(RNNOptions, rnn):

    if RNNOptions.saved_model_path != "":
        RNNOptions.saver.restore(RNNOptions.session, RNNOptions.saved_model_path)

    TEST_PREFIX = RNNOptions.test_prefix_string.lower()
    for i in range(len(TEST_PREFIX)):
        out = rnn.run_step(RNNOptions.rnn_utils.embed_to_vocab(TEST_PREFIX[i], RNNOptions.vocab), i == 0)

    print("SENTENCE:")
    gen_str = TEST_PREFIX
    for i in range(RNNOptions.test_text_length):
        # noinspection PyUnboundLocalVariable
        element = np.random.choice(range(len(RNNOptions.vocab)),
                                   p=out)  # Sample character from the network according to the generated output
        # probabilities

        gen_str += RNNOptions.vocab[element]

        out = rnn.run_step(RNNOptions.rnn_utils.embed_to_vocab(RNNOptions.vocab[element], RNNOptions.vocab), False)
    print(gen_str)

start()
