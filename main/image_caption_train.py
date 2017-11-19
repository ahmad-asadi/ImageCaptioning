import tensorflow as tf
import numpy as np

import os
import time
import resource
import math
import copy
import re

from data_utils import data_helper
from main.neuralNetworks.cnn import CNN as Encoder
from main.neuralNetworks.rnn import StackedRNN as Decoder
from main.neuralNetworks.rnn import RNNUtils as DecoderUtils

from matplotlib import pyplot as plt
from multiprocessing import Process


def getNextBatch(batchId, rawCaptions, cocoHelper, rnnOptions, vocab, word2ind):
    # training.batch_sequences_with_states()
    batchInstanceCount = min(rnnOptions.batch_size, len(rawCaptions))
    batchData = np.zeros(shape=(rnnOptions.time_step, cocoHelper.word2vec.layer1_size, rnnOptions.batch_size))
    batchLabel = np.zeros(shape=(rnnOptions.time_step, len(vocab), rnnOptions.batch_size))
    batchImgs = []
    for i in range(batchInstanceCount - 1):
        embeddedCaptionInst, instImgFileName, embeddedLabel = getNextInstance(i + batchId, rawCaptions, cocoHelper,
                                                                              rnnOptions.rnn_utils, vocab,
                                                                              word2ind=word2ind)
        batchData[1:embeddedCaptionInst.shape[0], :, i] = embeddedCaptionInst[0:min(rnnOptions.time_step,
                                                                                    embeddedCaptionInst.shape[0]) - 1]
        batchLabel[0:embeddedCaptionInst.shape[0], :, i] = embeddedLabel[0:rnnOptions.time_step]
        batchImgs.append(instImgFileName)
    return batchData, batchImgs, batchLabel


def getNextInstance(iteration, data, cocoHelper, rnnUtils, vocab, word2ind):
    # training.batch_sequences_with_states()
    global testLabel

    inst_image = [cocoHelper.imgs[i] for i in cocoHelper.imgs][iteration % len(data)]
    inst_image_filename = inst_image["file_name"]
    dataInsts = [cocoHelper.anns[i] for i in cocoHelper.anns if cocoHelper.anns[i]["image_id"] == inst_image["id"]]
    dataInst = dataInsts[0]

    if testLabel is None:
        testLabel = dataInst
    embeddedCaption = rnnUtils.embed_inst_to_vocab(dataInst=dataInst, cocoHelper=cocoHelper)
    embeddedLabel = rnnUtils.embed_inst_label_to_vocab(dataInst=dataInst, vocab=vocab, word2ind=word2ind)

    return embeddedCaption, inst_image_filename, embeddedLabel


def start():
    global testLabel
    testLabel = None
    rsrc = resource.RLIMIT_DATA
    soft, hard = resource.getrlimit(rsrc)
    print('Soft limit starts as  :', soft)

    resource.setrlimit(rsrc, (32 * 1024 * 1024 * 1024, hard))  # limit to one kilobyte

    soft, hard = resource.getrlimit(rsrc)
    print('Soft limit changed to :', soft)

    plt.interactive(True)

    print("defining constants")
    data_dir, imageFeaturesSize, rnnUtils, maxIterCount = initializeParameters()

    print("loading dataset")
    # cocoHelper, captionsDict, embeddedCaptions = loadDataset(data_dir, rnnUtils)
    cocoHelper, captionsDict, rawCaptions, capWord2Ind, capInd2Word = loadDataset(data_dir, rnnUtils)

    print("creating encoder's structure")
    cnn = createEncoder(data_dir)

    print("creating decoder's structure and initialization")
    rnn, rnnOptions = createAndInitializeDecoder(captionsDict, imageFeaturesSize, rnnUtils,
                                                 word_embedding_size=cocoHelper.word2vec.layer1_size)

    printGraph(rnnOptions)

    extractNextBatch(0, capWord2Ind, captionsDict, cocoHelper, rawCaptions, rnnOptions)
    batchData = copy.copy(batch_data_buffer)
    batchImgFileName = list(batch_img_filename_buffer)
    batchLabelRaw = copy.copy(batch_label_buffer)
    batchInput = np.zeros(shape=(batchData.shape[2], batchData.shape[0],
                                 batchData.shape[1] + imageFeaturesSize))
    batchLabel = np.zeros(shape=(batchLabelRaw.shape[2], batchLabelRaw.shape[0], batchLabelRaw.shape[1]))
    for batchCnt in range(len(batchImgFileName)):
        imageFeatures = cnn.classify(os.path.join(data_dir, "train2014/" + batchImgFileName[batchCnt]))

        imageFeaturesMat = np.reshape(np.repeat(a=imageFeatures, repeats=rnnOptions.time_step).transpose(),
                                      (rnnOptions.image_feature_size, rnnOptions.time_step))
        batchInput[batchCnt, :, :] = (np.concatenate((batchData[:, :, batchCnt].transpose(), imageFeaturesMat),
                                                     axis=0)).transpose()
        batchLabel[batchCnt, 0:batchLabel.shape[1] - 1, :] = [batchLabelRaw[i + 1, :, batchCnt] for i in
                                                              range(batchLabelRaw.shape[0] - 1)]
    testInput = batchInput[0, :, :]
    print("test image name: ", batchImgFileName[0])

    print("starting to train the structure")
    global costs
    startTime = time.time()
    costs = np.zeros(maxIterCount)
    checkPoint = 1
    statShowPeriod = 1

    batchId = 0
    for i in range(maxIterCount):
        for internal_loop_counter in range(math.ceil(len(rawCaptions) / rnnOptions.batch_size)):
            print("iteration:" + repr(i) + ", internal loop: processing ",
                  100 * internal_loop_counter / math.ceil(len(rawCaptions) / rnnOptions.batch_size),
                  "%")

            batchData = copy.copy(batch_data_buffer)
            batchImgFileName = list(batch_img_filename_buffer)
            batchLabelRaw = copy.copy(batch_label_buffer)

            load_data_process = Process(target=extractNextBatch, args=(batchId, capWord2Ind, captionsDict,
                                                                       cocoHelper, rawCaptions, rnnOptions))
            load_data_process.start()

            prepare_data_and_train_structure(batchData=batchData, i=i, batchImgFileName=batchImgFileName, cnn=cnn,
                                             data_dir=data_dir, imageFeaturesSize=imageFeaturesSize, rnn=rnn,
                                             rnnOptions=rnnOptions, batchLabelRaw=batchLabelRaw)
            load_data_process.join()
            batchId += 1

        if i % statShowPeriod == 0 and i > 0:
            endTime = time.time()
            duration = (endTime - startTime) / statShowPeriod
            print("batch ", i, ", train time per batch: ", duration, " current gained cost: ", costs[i])
            startTime = endTime
            print("drawing error curve...")
            plt.plot([i - 1, i], costs[i - 1:i + 1])
            plt.show()
            plt.pause(1)

        if i % checkPoint == 0 and i > 0:
            print("saving current model...")
            rnnOptions.saver.save(rnnOptions.session, rnnOptions.saved_model_path)
            print("testing current model:")
            testModel(rnn, rnnOptions, testInput, cocoHelper, capInd2Word)


def prepare_data_and_train_structure(batchData, i, batchImgFileName, cnn, data_dir, imageFeaturesSize, rnn,
                                     rnnOptions, batchLabelRaw):
    global costs
    batchInput = np.zeros(shape=(batchData.shape[2], batchData.shape[0],
                                 batchData.shape[1] + imageFeaturesSize))
    batchLabel = np.zeros(shape=(batchLabelRaw.shape[2], batchLabelRaw.shape[0], batchLabelRaw.shape[1]))
    for batchCnt in range(len(batchImgFileName)):
        imageFeatures = cnn.classify(os.path.join(data_dir, "train2014/" + batchImgFileName[batchCnt]))

        imageFeaturesMat = np.reshape(np.repeat(a=imageFeatures, repeats=rnnOptions.time_step).transpose(),
                                      (rnnOptions.image_feature_size, rnnOptions.time_step))
        batchInput[batchCnt, :, :] = (np.concatenate((batchData[:, :, batchCnt].transpose(), imageFeaturesMat),
                                                     axis=0)).transpose()
        batchLabel[batchCnt, 0:batchLabel.shape[1] - 1, :] = [batchLabelRaw[i + 1, :, batchCnt] for i in
                                                              range(batchLabelRaw.shape[0] - 1)]

    costs[i] = rnn.train_batch(Xbatch=batchInput, Ybatch=batchLabel, keep_prob=0.9)


def train_structure(batchInput, batchLabel, rnn):
    global cost


def extractNextBatch(batchId, capWord2Ind, captionsDict, cocoHelper, rawCaptions, rnnOptions):
    global batch_data_buffer, batch_img_filename_buffer, batch_label_buffer
    batch_data_buffer, batch_img_filename_buffer, batch_label_buffer = getNextBatch(batchId=batchId,
                                                                                    rawCaptions=rawCaptions,
                                                                                    cocoHelper=cocoHelper,
                                                                                    rnnOptions=rnnOptions,
                                                                                    vocab=captionsDict,
                                                                                    word2ind=capWord2Ind)


def testModel(rnn, rnnOptions, testInput, cocoHelper, ind2word):
    global testLabel
    gen_str = ""
    out = rnn.run_step(X=testInput, init_zero_state=True)[0]
    for testInd in range(rnnOptions.time_step - 1):
        # noinspection PyUnboundLocalVariable
        element = np.random.choice(range(len(out)),
                                   p=out)  # Sample character from the network according to the generated output
        # probabilities

        new_word = ind2word[element]
        gen_str += " " + new_word
        testInput[testInd + 1, 0:cocoHelper.word2vec.layer1_size] = cocoHelper.word2vec[re.split("[\W]+", new_word)[0]]

        out = rnn.run_step(X=testInput, init_zero_state=False)[0]
    print(gen_str)
    print("Human label: ", testLabel)


def printGraph(rnnOptions):
    print("writing graphs to /tmp/graph in order to show it in tensor board")
    writer = tf.summary.FileWriter("/tmp/graph")
    writer.add_graph(rnnOptions.session.graph)


def createAndInitializeDecoder(captionsDict, imageFeaturesSize, rnnUtils, word_embedding_size):
    rnnOptions = DecoderUtils.RNNOptions(vocab=captionsDict, rnnUtils=rnnUtils, image_feature_size=imageFeaturesSize,
                                         word_embedding_size=word_embedding_size)
    rnn = Decoder.StackedRNN(input_size=rnnOptions.input_size, lstm_size=rnnOptions.lstm_size,
                             number_of_layers=rnnOptions.num_layers, output_size=rnnOptions.out_size,
                             session=rnnOptions.session, learning_rate=rnnOptions.learning_rate,
                             batch_size=rnnOptions.batch_size, name=rnnOptions.name)
    rnnOptions.session.run(tf.global_variables_initializer())
    rnnOptions.saver = tf.train.Saver(tf.global_variables())
    return rnn, rnnOptions


def createEncoder(data_dir):
    cnn = Encoder.CNN(os.path.join(data_dir, "train2014"))
    return cnn


def loadDataset(data_dir, rnnUtils):
    cocoHelper = data_helper.COCOHelper(data_dir + "annotations/captions_train2014.json")
    rawCaptions, captionsDict, capIndToWord, capWordToInd, capIndToWord = cocoHelper.extract_captions()
    # embeddedCaptions = rnnUtils.embed_to_vocab(data_=rawCaptions, vocab=captionsDict, sentenceSize=100,
    #                                            word2Ind=capWordToInd)
    return cocoHelper, captionsDict, rawCaptions, capWordToInd, capIndToWord  # , embeddedCaptions


def initializeParameters():
    rnnUtils = DecoderUtils.RNNUtils()
    data_dir = "../../Dataset/MS-COCO/"
    imageFeaturesSize = 1008
    maxIterCount = 20000
    return data_dir, imageFeaturesSize, rnnUtils, maxIterCount


start()
