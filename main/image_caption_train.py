import tensorflow as tf
import numpy as np
import nltk
import os
import time
import resource
import math
import copy
import re

import logging

from data_utils import data_helper
from main.neuralNetworks.cnn import CNN as Encoder
from main.neuralNetworks.rnn import StackedRNN as Decoder
from main.neuralNetworks.rnn import RNNUtils as DecoderUtils

from matplotlib import pyplot as plt
from multiprocessing import Process


def getNextBatch(batchId, rawCaptions, cocoHelper, rnnOptions, vocab, word2ind):
    # training.batch_sequences_with_states()
    batchInstanceCount = min(rnnOptions.batch_insts, len(rawCaptions))
    batchData = np.zeros(shape=(rnnOptions.time_step, cocoHelper.word2vec.layer1_size, rnnOptions.batch_size))
    batchLabel = np.zeros(shape=(rnnOptions.time_step, rnnOptions.word_embedding_size, rnnOptions.batch_size))
    batchImgs = []
    for i in range(batchInstanceCount - 1):
        embeddedCaptionInsts, instImgFileNames, embeddedLabels = getNextInstance(i + batchId, rawCaptions, cocoHelper,
                                                                                 rnnOptions.rnn_utils, vocab,
                                                                                 word2ind=word2ind)

        cnt = 0
        offset = i * 5
        for embeddedCaptionInst in embeddedCaptionInsts:
            batchData[1:embeddedCaptionInst.shape[0], :, offset + cnt] = embeddedCaptionInst[0:min(rnnOptions.time_step,
                                                                                                   embeddedCaptionInst.shape[
                                                                                                       0]) - 1]
            cnt += 1

        cnt = 0
        for embeddedLabel in embeddedLabels:
            batchLabel[0:embeddedLabel.shape[0], :, offset + cnt] = embeddedLabel[0:rnnOptions.time_step]
            cnt += 1
        for instImgFileName in instImgFileNames:
            batchImgs.append(instImgFileName)
    return batchData, batchImgs, batchLabel


def getNextInstance(iteration, data, cocoHelper, rnnUtils, vocab, word2ind):
    # training.batch_sequences_with_states()
    global testLabel

    inst_image = [cocoHelper.imgs[i] for i in cocoHelper.imgs][iteration % len(data)]
    dataInsts = [cocoHelper.anns[i] for i in cocoHelper.anns if cocoHelper.anns[i]["image_id"] == inst_image["id"]]
    embeddedCaptions = [rnnUtils.embed_inst_to_vocab(dataInst=dataInsts[i], cocoHelper=cocoHelper) for i in
                        range(len(dataInsts))]
    # embeddedLabels = [rnnUtils.embed_inst_label_to_vocab(dataInst=dataInsts[i], vocab=vocab, word2ind=word2ind) for i in
    #                   range(len(dataInsts))]
    embeddedLabels = [rnnUtils.embed_inst_to_vocab(dataInst=dataInsts[i], cocoHelper=cocoHelper) for i in
                      range(len(dataInsts))]
    inst_image_filenames = [inst_image["file_name"] for _ in range(len(dataInsts))]

    testLabel.append(dataInsts)

    return embeddedCaptions, inst_image_filenames, embeddedLabels


def start():
    global testLabel
    testLabel = []
    rsrc = resource.RLIMIT_DATA
    soft, hard = resource.getrlimit(rsrc)
    print('Soft limit starts as  :', soft)

    resource.setrlimit(rsrc, (32 * 1024 * 1024 * 1024, hard))  # limit to one kilobyte
    logging.getLogger().addHandler(logging.StreamHandler())  # set logger to print logs in console

    soft, hard = resource.getrlimit(rsrc)
    print('Soft limit changed to :', soft)

    plt.interactive(True)

    print("defining constants")
    data_dir, imageFeaturesSize, rnnUtils, maxIterCount = initializeParameters()

    print("loading dataset")
    # cocoHelper, captionsDict, embeddedCaptions = loadDataset(data_dir, rnnUtils)
    cocoHelper, captionsDict, rawCaptions, capWord2Ind, capInd2Word = loadDataset(data_dir, rnnUtils)
    test_cocoHelper, test_captionsDict, test_rawCaptions, test_capWord2Ind, test_capInd2Word = loadTestDataset(
        data_dir, rnnUtils)

    print("creating encoder's structure")
    cnn = createEncoder(data_dir)

    print("creating decoder's structure and initialization")
    rnn, rnnOptions = createAndInitializeDecoder(captionsDict, imageFeaturesSize, rnnUtils,
                                                 word_embedding_size=cocoHelper.word2vec.layer1_size, cnn=cnn)

    printGraph(rnnOptions)

    extractNextBatch(0, capWord2Ind, captionsDict, cocoHelper, rawCaptions, rnnOptions)
    batchData = copy.copy(batch_data_buffer)
    batchImgFileName = list(batch_img_filename_buffer)
    batchLabelRaw = copy.copy(batch_label_buffer)
    batchInput = np.zeros(shape=(batchData.shape[0], batchData.shape[2],
                                 batchData.shape[1] + imageFeaturesSize))
    batchLabel = np.zeros(shape=(batchLabelRaw.shape[0], batchLabelRaw.shape[2], batchLabelRaw.shape[1]))
    for batchCnt in range(len(batchImgFileName)):
        # imageFeatures = cnn.classify(os.path.join(data_dir, "train2014/" + batchImgFileName[batchCnt]))
        imageFeatures = cnn.extract_features(os.path.join(data_dir, "train2014/" + batchImgFileName[batchCnt]))

        imageFeaturesMat = np.reshape(np.repeat(a=imageFeatures, repeats=rnnOptions.time_step).transpose(),
                                      (rnnOptions.image_feature_size, rnnOptions.time_step))
        batchInput[:, batchCnt, :] = (np.concatenate((batchData[:, :, batchCnt].transpose(), imageFeaturesMat),
                                                     axis=0)).transpose()
        batchLabel[0:batchLabel.shape[0] - 1, batchCnt, :] = [batchLabelRaw[i + 1, :, batchCnt] for i in
                                                              range(batchLabelRaw.shape[0] - 1)]
    print("test image name: ", batchImgFileName[0])

    print("starting to train the structure")
    global costs
    startTime = time.time()
    costs = np.zeros(maxIterCount)
    checkPoint = 1
    statShowPeriod = 1

    batchId = 0
    for i in range(maxIterCount):
        loop_counter = math.ceil((len(rawCaptions) / 5) / rnnOptions.batch_size)
        loop_counter = min(loop_counter, 10)
        for internal_loop_counter in range(loop_counter):
            print("iteration:" + repr(i) + ", internal loop: processing ",
                  100 * internal_loop_counter / loop_counter,
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

        endTime = time.time()
        duration = (endTime - startTime) / statShowPeriod
        print("batch ", i, ", train time per batch: ", duration, " current gained cost: ", costs[i])
        startTime = endTime
        print("drawing error curve...")

        if i % statShowPeriod == 0 and i > 0:
            plt.plot([i - 1, i], costs[i - 1:i + 1])
            plt.show()
            plt.pause(1)

        if i % checkPoint == 0:
            print("saving current model...")
            rnnOptions.saver.save(rnnOptions.session, rnnOptions.saved_model_path)
            print("testing current model:")
            testModel(rnn, rnnOptions, test_cocoHelper, test_capInd2Word, test_capWord2Ind, test_captionsDict,
                      test_rawCaptions, cnn, data_dir)


def prepare_data_and_train_structure(batchData, i, batchImgFileName, cnn, data_dir, imageFeaturesSize, rnn,
                                     rnnOptions, batchLabelRaw):
    global costs
    batchInput = np.zeros(shape=(batchData.shape[0], batchData.shape[2],
                                 batchData.shape[1] + imageFeaturesSize))
    batchLabel = np.zeros(shape=(batchLabelRaw.shape[0], batchLabelRaw.shape[2], batchLabelRaw.shape[1]))
    for batchCnt in range(len(batchImgFileName)):
        imageFeatures = cnn.extract_features(os.path.join(data_dir, "train2014/" + batchImgFileName[batchCnt]))

        imageFeaturesMat = np.reshape(np.repeat(a=imageFeatures, repeats=rnnOptions.time_step).transpose(),
                                      (rnnOptions.image_feature_size, rnnOptions.time_step))
        batchInput[:, batchCnt, :] = (np.concatenate((batchData[:, :, batchCnt].transpose(), imageFeaturesMat),
                                                     axis=0)).transpose()
        # batchInput[:, batchCnt, rnnOptions.word_embedding_size:batchInput.shape[2]] = np.zeros(
        #     shape=(batchInput.shape[0], rnnOptions.image_feature_size))
        batchLabel[0:batchLabel.shape[0] - 1, batchCnt, :] = [batchLabelRaw[i + 1, :, batchCnt] for i in
                                                              range(batchLabelRaw.shape[0] - 1)]

    costs[i] = rnn.train_batch(Xbatch=batchInput, Ybatch=batchLabel, keep_prob=0.5)


def train_structure(batchInput, batchLabel, rnn):
    global cost


def extractNextBatch(batchId, capWord2Ind, captionsDict, cocoHelper, rawCaptions, rnnOptions, validation_mode=False):
    global batch_data_buffer, batch_img_filename_buffer, batch_label_buffer
    global test_batch_data_buffer, test_batch_img_filename_buffer, test_batch_label_buffer
    batch_data, batch_img_filename, batch_label = getNextBatch(batchId=batchId,
                                                               rawCaptions=rawCaptions,
                                                               cocoHelper=cocoHelper,
                                                               rnnOptions=rnnOptions,
                                                               vocab=captionsDict,
                                                               word2ind=capWord2Ind)
    if not validation_mode:
        batch_data_buffer = batch_data
        batch_img_filename_buffer = batch_img_filename
        batch_label_buffer = batch_label
    else:
        test_batch_data_buffer = batch_data
        test_batch_img_filename_buffer = batch_img_filename
        test_batch_label_buffer = batch_label


def testModel(rnn, rnnOptions, cocoHelper, ind2word, word2ind, vocab, raw_captions, cnn, data_dir):
    global testLabel
    testLabel = []
    extractNextBatch(0, capWord2Ind=word2ind, captionsDict=vocab, cocoHelper=cocoHelper, rawCaptions=raw_captions,
                     rnnOptions=rnnOptions, validation_mode=True)

    batchData = copy.copy(test_batch_data_buffer)
    batchImgFileName = list(test_batch_img_filename_buffer)
    batchLabelRaw = copy.copy(test_batch_label_buffer)

    batchInput = np.zeros(shape=(batchData.shape[0], batchData.shape[2],
                                 batchData.shape[1] + rnnOptions.image_feature_size))
    batchLabel = np.zeros(shape=(batchLabelRaw.shape[0], batchLabelRaw.shape[2], batchLabelRaw.shape[1]))
    for batchCnt in range(len(batchImgFileName)):
        imageFeatures = cnn.extract_features(os.path.join(data_dir, "val2014/" + batchImgFileName[batchCnt]))

        imageFeaturesMat = np.reshape(np.repeat(a=imageFeatures, repeats=rnnOptions.time_step).transpose(),
                                      (rnnOptions.image_feature_size, rnnOptions.time_step))
        batchInput[:, batchCnt, :] = (np.concatenate((batchData[:, :, batchCnt].transpose(), imageFeaturesMat),
                                                     axis=0)).transpose()
        # batchInput[:, batchCnt, rnnOptions.word_embedding_size:batchInput.shape[2]] = np.zeros(
        #     shape=(batchInput.shape[0], rnnOptions.image_feature_size))
        batchLabel[0:batchLabel.shape[0] - 1, batchCnt, :] = [batchLabelRaw[i + 1, :, batchCnt] for i in
                                                              range(batchLabelRaw.shape[0] - 1)]

    predicted_captions = ["" for _ in range(rnnOptions.batch_insts)]
    BLEUScores = np.zeros(shape=rnnOptions.batch_insts)
    test_input = np.zeros(shape=(batchInput.shape[0], 1, batchInput.shape[2]))
    for batch_inst in range(rnnOptions.batch_insts-1):
        i = batch_inst * 5
        test_input[:, 0, :] = batchInput[:, i, :]
        out = rnn.run_step(X=test_input, init_zero_state=True)[0][0]
        for testInd in range(rnnOptions.time_step - 1):
            # # noinspection PyUnboundLocalVariable
            # element = np.random.choice(range(len(out[i])),
            #                           p=out[i])  # Sample character from the network according to the generated output
            # # probabilities
            # if element == 0:  # caption finished
            #     break
            #
            # new_word = ind2word[element]
            new_word = cocoHelper.word2vec.most_similar(positive=[out], topn=1)[0][0]
            predicted_captions[batch_inst] += " " + new_word
            batchInput[testInd + 1, i, 0:cocoHelper.word2vec.layer1_size] = cocoHelper.word2vec[new_word]
            test_input[:, 0, :] = batchInput[:, i, :]
            out = rnn.run_step(X=test_input, init_zero_state=False)[testInd + 1]
            out = out[0]
        BLEUScores[batch_inst] = nltk.translate.bleu_score.sentence_bleu(
            [ref["caption"] for ref in testLabel[batch_inst]],
            predicted_captions[batch_inst], emulate_multibleu=True)
        print("prediction: ", i, ":", predicted_captions[batch_inst])
        print("human label: ", testLabel[batch_inst])
        print("BLEU SCORE: ", BLEUScores[batch_inst])


def printGraph(rnnOptions):
    print("writing graphs to /tmp/graph in order to show it in tensor board")
    writer = tf.summary.FileWriter("/tmp/graph")
    writer.add_graph(rnnOptions.session.graph)


def createAndInitializeDecoder(captionsDict, imageFeaturesSize, rnnUtils, word_embedding_size, cnn):
    rnnOptions = DecoderUtils.RNNOptions(vocab=captionsDict, rnnUtils=rnnUtils, image_feature_size=imageFeaturesSize,
                                         word_embedding_size=word_embedding_size)
    rnn = Decoder.StackedRNN(input_size=rnnOptions.input_size, lstm_size=rnnOptions.lstm_size,
                             number_of_layers=rnnOptions.num_layers, output_size=rnnOptions.out_size,
                             session=rnnOptions.session, learning_rate=rnnOptions.learning_rate,
                             batch_size=rnnOptions.batch_size, name=rnnOptions.name, cnn_graph=cnn.net.session.graph)
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


def loadTestDataset(data_dir, rnnUtils):
    cocoHelper = data_helper.COCOHelper(data_dir + "annotations/captions_val2014.json")
    rawCaptions, captionsDict, capIndToWord, capWordToInd, capIndToWord = cocoHelper.extract_captions()
    # embeddedCaptions = rnnUtils.embed_to_vocab(data_=rawCaptions, vocab=captionsDict, sentenceSize=100,
    #                                            word2Ind=capWordToInd)
    return cocoHelper, captionsDict, rawCaptions, capWordToInd, capIndToWord  # , embeddedCaptions


def initializeParameters():
    rnnUtils = DecoderUtils.RNNUtils()
    data_dir = "../../Dataset/MS-COCO/"
    imageFeaturesSize = 2048
    maxIterCount = 20000
    return data_dir, imageFeaturesSize, rnnUtils, maxIterCount


start()
