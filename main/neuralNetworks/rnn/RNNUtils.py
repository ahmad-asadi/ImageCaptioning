import tensorflow as tf
import numpy as np
import re

from gensim.models.word2vec import Word2Vec


class RNNUtils:
    @staticmethod
    def embed_to_vocab(data_, vocab):
        print("creating word embedding, expected embedding size: " + repr(len(data_)) + " * " + repr(len(vocab)))
        data = np.zeros((len(data_), len(vocab)))
        cnt = 0
        for s in data_:
            v = [0.0] * len(vocab)
            v[vocab.index(s)] = 1.0
            data[cnt, :] = v
            cnt += 1
        return data

    @staticmethod
    def embed_inst_to_vocab(dataInst, cocoHelper):
        current_caption_words = re.split("[\W .,?!\"\'/\\\]+", dataInst['caption'])
        data = np.zeros(shape=(len(current_caption_words), cocoHelper.word2vec.layer1_size))
        cnt = 0
        for s in current_caption_words:
            data[cnt, :] = cocoHelper.word2vec[s.lower()]
            cnt += 1
        return data

    @staticmethod
    def embed_inst_label_to_vocab(dataInst, vocab, word2ind):
        current_caption_words = re.split("[\W .,?!\"\'/\\\]+", dataInst['caption'])
        data = np.zeros(shape=(len(current_caption_words), len(vocab)))
        cnt = 0
        for s in current_caption_words:
            v = [0.0] * len(vocab)
            v[word2ind[s.lower()]] = 1.0
            data[cnt, :] = v
            cnt += 1
        return data

    @staticmethod
    def embed_to_vocab(data_, vocab, sentenceSize, word2Ind):
        print("creating word embedding, expected embedding size: " + repr(len(data_)) + " * " + repr(len(vocab)))
        data = np.zeros(shape=(len(data_), sentenceSize, len(vocab)))
        rawCaptionId = 0
        for rawCaption in data_:
            captionWordList = re.split("[\W .,?!\"\'/\\\]+", rawCaption['caption'])
            for wordInd in range(min(len(captionWordList), sentenceSize)):
                wordEmbedding = np.zeros(shape=(len(vocab)))

                wordEmbedding[word2Ind[captionWordList[wordInd].lower()]] = 1
                data[rawCaptionId, wordInd, :] = wordEmbedding

            rawCaptionId += 1
        return data

    @staticmethod
    def decode_embed(array, vocab):
        return vocab[array.index(1)]

    @staticmethod
    def load_data():
        data_ = ""
        with open('/tmp/data/shakespeare.txt', 'r') as f:
            data_ += f.read()
        data_ = data_.lower()

        # Convert to 1-hot coding
        vocab = sorted(list(set(data_)))
        return data_, vocab


class RNNOptions:
    def __init__(self, vocab, rnnUtils, image_feature_size, word_embedding_size):
        self.image_feature_size = image_feature_size
        self.num_layers = 5
        self.word_embedding_size = word_embedding_size
        self.input_size = self.word_embedding_size + self.image_feature_size
        self.out_size = len(vocab)
        self.lstm_size = 512
        self.batch_size = 512
        self.time_step = 25
        # self.learning_rate = 0.003
        self.learning_rate = 0.0001
        self.saved_model_path = "./image_caption_encoder_saved/model.ckpt"
        self.name = "Decoder"
        self.test_prefix_string = "Romeo: "
        self.vocab = vocab
        config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.session = tf.InteractiveSession(config=config)
        self.rnn_utils = rnnUtils
        self.test_text_length = 500

        self.saver = None
