import tensorflow as tf
import numpy as np
import random
import time
import sys

from main.neuralNetworks.rnn import RNN


# Embed string to character-arrays -- it generates an array len(data) x len(vocab)
# Vocab is a list of elements


# noinspection PyShadowingNames
def embed_to_vocab(data_, vocab):
    data = np.zeros((len(data_), len(vocab)))

    cnt = 0
    for s in data_:
        v = [0.0] * len(vocab)
        v[vocab.index(s)] = 1.0
        data[cnt, :] = v
        cnt += 1

    return data


# noinspection PyShadowingNames
def decode_embed(array, vocab):
    return vocab[array.index(1)]


def test_model(TEST_PREFIX):
    # 2) GENERATE LEN_TEST_TEXT CHARACTERS USING THE TRAINED NETWORK

    if ckpt_file != "":
        saver.restore(sess, ckpt_file)

    TEST_PREFIX = TEST_PREFIX.lower()
    for i in range(len(TEST_PREFIX)):
        out = net.run_step(embed_to_vocab(TEST_PREFIX[i], vocab), i == 0)

    print("SENTENCE:")
    gen_str = TEST_PREFIX
    for i in range(LEN_TEST_TEXT):
        # noinspection PyUnboundLocalVariable
        element = np.random.choice(range(len(vocab)),
                                   p=out)  # Sample character from the network according to the generated output
        # probabilities

        gen_str += vocab[element]

        out = net.run_step(embed_to_vocab(vocab[element], vocab), False)
    print(gen_str)


ckpt_file = ""
TEST_PREFIX = "The "  # Prefix to prompt the network in test mode

print("Usage:")
print('\t\t ', sys.argv[0], ' [ckpt model to load] [prefix, e.g., "The "]')
if len(sys.argv) >= 2:
    ckpt_file = sys.argv[1]
if len(sys.argv) == 3:
    TEST_PREFIX = sys.argv[2]

# Load the data
data_ = ""
with open('/tmp/data/shakespeare.txt', 'r') as f:
    data_ += f.read()
data_ = data_.lower()

# Convert to 1-hot coding
vocab = sorted(list(set(data_)))

data = embed_to_vocab(data_, vocab)

in_size = out_size = len(vocab)
lstm_size = 256  # 128
num_layers = 4
batch_size = 64  # 128
time_steps = 100  # 50

NUM_TRAIN_BATCHES = 20000

LEN_TEST_TEXT = 500  # Number of test characters of text to generate after training the network

# Initialize the network
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)

net = RNN.RNN(in_size=in_size,
              lstm_size=lstm_size,
              num_layers=num_layers,
              out_size=out_size,
              session=sess,
              learning_rate=0.003,
              name="char_rnn_network")

sess.run(tf.global_variables_initializer())

saver = tf.train.Saver(tf.global_variables())

# 1) TRAIN THE NETWORK
if ckpt_file == "":
    last_time = time.time()

    batch = np.zeros((batch_size, time_steps, in_size))
    batch_y = np.zeros((batch_size, time_steps, in_size))

    possible_batch_ids = range(data.shape[0] - time_steps - 1)
    for i in range(NUM_TRAIN_BATCHES):
        # Sample time_steps consecutive samples from the dataset text file
        batch_id = random.sample(possible_batch_ids, batch_size)

        for j in range(time_steps):
            ind1 = [k + j for k in batch_id]
            ind2 = [k + j + 1 for k in batch_id]
            d1 = data[ind1, :]
            batch[:, j, :] = data[ind1, :]
            batch_y[:, j, :] = data[ind2, :]

        cst = net.train_batch(batch, batch_y)

        if (i % 100) == 0:
            new_time = time.time()
            diff = new_time - last_time
            last_time = new_time

            print("batch: ", i, "   loss: ", cst, "   speed: ", (100.0 / diff), " batches / s")

        if (i % 500) == 0:
            print("saving current model, please wait...")
            saver.save(sess, "saved/model.ckpt")
            test_model(TEST_PREFIX)
            print("current model has been saved")


