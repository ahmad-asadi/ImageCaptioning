# Interface for accessing the Microsoft COCO dataset.

# Microsoft COCO is a large image dataset designed for object detection,
# segmentation, and caption generation. pycocotools is a Python API that
# assists in loading, parsing and visualizing the annotations in COCO.
# Please visit http://mscoco.org/ for more information on COCO, including
# for the data, paper, and tutorials. The exact format of the annotations
# is also described on the COCO website. For example usage of the pycocotools
# please see pycocotools_demo.ipynb. In addition to this API, please download both
# the COCO images and annotations in order to run the demo.

# An alternative to using the API is to load the annotations directly
# into Python dictionary
# Using the API provides additional utility functions. Note that this API
# supports both *instance* and *caption* annotations. In the case of
# captions not all functions are defined (e.g. categories are undefined).

# The following API functions are defined:
#  COCO       - COCO api class that loads COCO annotation file and prepare data structures.
#  decodeMask - Decode binary mask M encoded via run-length encoding.
#  encodeMask - Encode binary mask M using run-length encoding.
#  getAnnIds  - Get ann ids that satisfy given filter conditions.
#  getCatIds  - Get cat ids that satisfy given filter conditions.
#  getImgIds  - Get img ids that satisfy given filter conditions.
#  loadAnns   - Load anns with the specified ids.
#  loadCats   - Load cats with the specified ids.
#  loadImgs   - Load imgs with the specified ids.
#  segToMask  - Convert polygon segmentation to binary mask.
#  showAnns   - Display the specified annotations.
#  loadRes    - Load result file and create result api object.
# Throughout the API "ann"=annotation, "cat"=category, and "img"=image.
# Help on each functions can be accessed by: "help COCO>function".

# See also COCO>decodeMask,
# COCO>encodeMask, COCO>getAnnIds, COCO>getCatIds,
# COCO>getImgIds, COCO>loadAnns, COCO>loadCats,
# COCO>loadImgs, COCO>segToMask, COCO>showAnns

# Microsoft COCO Toolbox.      Version 1.0
# Data, paper, and tutorials available at:  http://mscoco.org/
# Code written by Piotr Dollar and Tsung-Yi Lin, 2014.
# Licensed under the Simplified BSD License [see bsd.txt]

import json
import datetime
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import numpy as np
from skimage.draw import polygon
import copy
import re
from gensim.models.word2vec import Word2Vec


class COCOHelper:
    def __init__(self, annotation_file=None):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.dataset = {}
        self.anns = []
        self.imgToAnns = {}
        self.catToImgs = {}
        self.imgs = []
        self.cats = []
        self.word2vec = None
        self.word2vec_vocab = None
        if annotation_file is not None:
            print('loading annotations into memory...')
            time_t = datetime.datetime.utcnow()
            dataset = json.load(open(annotation_file, 'r'))
            print(datetime.datetime.utcnow() - time_t)
            self.dataset = dataset
            self.createIndex()

        self.create_word2vec()

    def createIndex(self):
        # create index
        print('creating index...')
        imgToAnns = {ann['image_id']: [] for ann in self.dataset['annotations']}
        anns = {ann['id']: [] for ann in self.dataset['annotations']}
        for ann in self.dataset['annotations']:
            imgToAnns[ann['image_id']] += [ann]
            anns[ann['id']] = ann

        imgs = {im['id']: {} for im in self.dataset['images']}
        for img in self.dataset['images']:
            imgs[img['id']] = img

        cats = []
        catToImgs = []
        if 'type' in self.dataset and self.dataset['type'] == 'instances':
            cats = {cat['id']: [] for cat in self.dataset['categories']}
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat
            catToImgs = {cat['id']: [] for cat in self.dataset['categories']}
            for ann in self.dataset['annotations']:
                catToImgs[ann['category_id']] += [ann['image_id']]

        print('index created!')

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats

    def info(self):
        """
        Print information about the annotation file.
        :return:
        """
        for key, value in self.dataset['info'].items():
            print('%s: %s' % (key, value))

    def getAnnIds(self, imgIds=None, catIds=None, areaRng=None, iscrowd=None):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param iscrowd:
        :param areaRng:
        :param catIds:
        :param imgIds:
        :return: ids (int array)       : integer array of ann ids
        """
        if areaRng is None:
            areaRng = []
        if catIds is None:
            catIds = []
        if imgIds is None:
            imgIds = []
        imgIds = imgIds if type(imgIds) == list else [imgIds]
        catIds = catIds if type(catIds) == list else [catIds]

        if len(imgIds) == len(catIds) == len(areaRng) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(imgIds) == 0:
                anns = sum([self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns], [])
            else:
                anns = self.dataset['annotations']
            anns = anns if len(catIds) == 0 else [ann for ann in anns if ann['category_id'] in catIds]
            anns = anns if len(areaRng) == 0 else [ann for ann in anns if
                                                   areaRng[0] < ann['area'] < areaRng[1]]
        if self.dataset['type'] == 'instances':
            if iscrowd is not None:
                ids = [ann['id'] for ann in anns if ann['iscrowd'] == iscrowd]
            else:
                ids = [ann['id'] for ann in anns]
        else:
            ids = [ann['id'] for ann in anns]
        return ids

    # noinspection PyIncorrectDocstring
    def getCatIds(self, catNms=None, supNms=None, catIds=None):
        """
        filtering parameters. default skips that filter.
        :param catNms (str array)  : get cats for given cat names
        :param supNms (str array)  : get cats for given supercategory names
        :param catIds (int array)  : get cats for given cat ids
        :return: ids (int array)   : integer array of cat ids
        """
        if catIds is None:
            catIds = []
        if supNms is None:
            supNms = []
        if catNms is None:
            catNms = []
        catNms = catNms if type(catNms) == list else [catNms]
        supNms = supNms if type(supNms) == list else [supNms]
        catIds = catIds if type(catIds) == list else [catIds]

        if len(catNms) == len(supNms) == len(catIds) == 0:
            cats = self.dataset['categories']
        else:
            cats = self.dataset['categories']
            cats = cats if len(catNms) == 0 else [cat for cat in cats if cat['name'] in catNms]
            cats = cats if len(supNms) == 0 else [cat for cat in cats if cat['supercategory'] in supNms]
            cats = cats if len(catIds) == 0 else [cat for cat in cats if cat['id'] in catIds]
        ids = [cat['id'] for cat in cats]
        return ids

    # noinspection PyIncorrectDocstring
    def getImgIds(self, imgIds=None, catIds=None):
        """
        Get img ids that satisfy given filter conditions.
        :param imgIds (int array) : get imgs for given ids
        :param catIds (int array) : get imgs with all given cats
        :return: ids (int array)  : integer array of img ids
        """
        if imgIds is None:
            imgIds = []
        if catIds is None:
            catIds = []
        imgIds = imgIds if type(imgIds) == list else [imgIds]
        catIds = catIds if type(catIds) == list else [catIds]

        if len(imgIds) == len(catIds) == 0:
            ids = self.imgs.keys()
        else:
            ids = set(imgIds)
            for catId in catIds:
                if len(ids) == 0:
                    ids = set(self.catToImgs[catId])
                else:
                    ids &= set(self.catToImgs[catId])
        return list(ids)

    def loadAnns(self, ids=None):
        """
        Load anns with the specified ids.
        :param ids:
        :return: anns (object array) : loaded ann objects
        """
        if ids is None:
            ids = []
        if type(ids) == list:
            # noinspection PyShadowingBuiltins
            return [self.anns[id] for id in ids]
        elif type(ids) == int:
            return [self.anns[ids]]

    def loadCats(self, ids=None):
        """
        Load cats with the specified ids.
        :param ids:
        :return: cats (object array) : loaded cat objects
        """
        if ids is None:
            ids = []
        if type(ids) == list:
            # noinspection PyShadowingBuiltins
            return [self.cats[id] for id in ids]
        elif type(ids) == int:
            return [self.cats[ids]]

    def loadImgs(self, ids=None):
        """
        Load anns with the specified ids.
        :param ids:
        :return: imgs (object array) : loaded img objects
        """
        if ids is None:
            ids = []
        if type(ids) == list:
            # noinspection PyShadowingBuiltins
            return [self.imgs[id] for id in ids]
        elif type(ids) == int:
            return [self.imgs[ids]]

    def showAnns(self, anns):
        """
        Display the specified annotations.
        :param anns:
        :return: None
        """
        if len(anns) == 0:
            return 0
        if self.dataset['type'] == 'instances':
            ax = plt.gca()
            polygons = []
            color = []
            for ann in anns:
                c = np.random.random((1, 3)).tolist()[0]
                if type(ann['segmentation']) == list:
                    # polygon
                    for seg in ann['segmentation']:
                        poly = np.array(seg).reshape((len(seg) / 2, 2))
                        polygons.append(Polygon(poly, True, alpha=0.4))
                        color.append(c)
                else:
                    # mask
                    mask = COCOHelper.decodeMask(ann['segmentation'])
                    img = np.ones((mask.shape[0], mask.shape[1], 3))
                    if ann['iscrowd'] == 1:
                        color_mask = np.array([2.0, 166.0, 101.0]) / 255
                    if ann['iscrowd'] == 0:
                        color_mask = np.random.random((1, 3)).tolist()[0]
                    for i in range(3):
                        # noinspection PyUnboundLocalVariable
                        img[:, :, i] = color_mask[i]
                    ax.imshow(np.dstack((img, mask * 0.5)))
            p = PatchCollection(polygons, facecolors=color, edgecolors=(0, 0, 0, 1), linewidths=3, alpha=0.4)
            ax.add_collection(p)
        if self.dataset['type'] == 'captions':
            for ann in anns:
                print(ann['caption'])

    def loadRes(self, resFile):
        """
        Load result file and return a result api object.
        :param resFile:
        :return: res (obj)         : result api object
        """
        res = COCOHelper()
        res.dataset['images'] = [img for img in self.dataset['images']]
        res.dataset['info'] = copy.deepcopy(self.dataset['info'])
        res.dataset['type'] = copy.deepcopy(self.dataset['type'])
        res.dataset['licenses'] = copy.deepcopy(self.dataset['licenses'])

        print('Loading and preparing results...     ')
        time_t = datetime.datetime.utcnow()
        anns = json.load(open(resFile))
        assert type(anns) == list, 'results in not an array of objects'
        annsImgIds = [ann['image_id'] for ann in anns]
        assert set(annsImgIds) == (set(annsImgIds) & set(self.getImgIds())), \
            'Results do not correspond to current coco set'
        if 'caption' in anns[0]:
            imgIds = set([img['id'] for img in res.dataset['images']]) & set([ann['image_id'] for ann in anns])
            res.dataset['images'] = [img for img in res.dataset['images'] if img['id'] in imgIds]
            # noinspection PyShadowingBuiltins
            for id, ann in enumerate(anns):
                ann['id'] = id
        elif 'bbox' in anns[0] and not anns[0]['bbox'] == []:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            # noinspection PyShadowingBuiltins
            for id, ann in enumerate(anns):
                bb = ann['bbox']
                x1, x2, y1, y2 = [bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]]
                ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                ann['area'] = bb[2] * bb[3]
                ann['id'] = id
                ann['iscrowd'] = 0
        elif 'segmentation' in anns[0]:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            # noinspection PyShadowingBuiltins
            for id, ann in enumerate(anns):
                ann['area'] = sum(ann['segmentation']['counts'][2:-1:2])
                ann['bbox'] = []
                ann['id'] = id
                ann['iscrowd'] = 0
        print('DONE (t=%0.2fs)' % ((datetime.datetime.utcnow() - time_t).total_seconds()))

        res.dataset['annotations'] = anns
        res.createIndex()
        return res

    @staticmethod
    def decodeMask(R):
        """
        Decode binary mask M encoded via run-length encoding.
        :param R:
        :return: M (bool 2D array) : decoded binary mask
        """
        N = len(R['counts'])
        M = np.zeros((R['size'][0] * R['size'][1],))
        n = 0
        val = 1
        for pos in range(N):
            val = not val
            for c in range(R['counts'][pos]):
                # noinspection PyStatementEffect
                R['counts'][pos]
                M[n] = val
                n += 1
        return M.reshape((R['size']), order='F')

    @staticmethod
    def encodeMask(M):
        """
        Encode binary mask M using run-length encoding.
        :param M:
        :return: R (object RLE)     : run-length encoding of binary mask
        """
        [h, w] = M.shape
        M = M.flatten(order='F')
        N = len(M)
        counts_list = []
        pos = 0
        # counts
        counts_list.append(1)
        diffs = np.logical_xor(M[0:N - 1], M[1:N])
        # noinspection PyTypeChecker
        for diff in diffs:
            if diff:
                pos += 1
                counts_list.append(1)
            else:
                counts_list[pos] += 1
        # if array starts from 1. start with 0 counts for 0
        if M[0] == 1:
            counts_list = [0] + counts_list
        return {'size': [h, w],
                'counts': counts_list,
                }

    # noinspection PyIncorrectDocstring
    @staticmethod
    def segToMask(S, h, w):
        """
         Convert polygon segmentation to binary mask.
         :param   S (float array)   : polygon segmentation mask
         :param   h (int)           : target mask height
         :param   w (int)           : target mask width
         :return: M (bool 2D array) : binary mask
         """
        M = np.zeros((h, w), dtype=np.bool)
        for s in S:
            N = len(s)
            rr, cc = polygon(np.array(s[1:N:2]), np.array(s[0:N:2]))  # (y, x)
            M[rr, cc] = 1
        return M

    def extract_captions(self):
        print("creating captions word dictionary...")
        rawCaptions = [self.anns[record] for record in self.anns if record < 2000]
        # rawCaptions = [self.anns[record] for record in self.anns]
        captionsDict = set()
        # captionsDict.add(".")
        # captionsDict.add(",")
        # captionsDict.add("?")
        # captionsDict.add("!")
        # captionsDict.add("\"")
        # captionsDict.add("\'")
        # captionsDict.add("\\")
        # captionsDict.add("/")
        cnt = 0

        for rawCaption in rawCaptions:
            if cnt % 10000 == 0:
                print("analysis of caption " + repr(cnt) + " out of " + repr(rawCaptions.__len__()))
            cnt += 1
            # noinspection PyTypeChecker
            splittedWords = re.split("[\W]+", rawCaption['caption'])
            for word in splittedWords:
                captionsDict.add(word.lower())
        print("captions word dictionary is created.")
        captionsDictList = sorted(captionsDict)
        captionsDictWordToInd = {captionsDictList[ind]: ind for ind in range(len(captionsDictList))}
        captionsDictIndToWord = {ind: captionsDictList[ind] for ind in range(len(captionsDictList))}
        return rawCaptions, captionsDictList, captionsDictIndToWord, captionsDictWordToInd, captionsDictIndToWord

    def create_word2vec(self):
        print("creating word embeddings structure")
        sentences = [re.split("[\W]+", (self.anns[i]["caption"]).lower()) for i in self.anns]
        iteration_count = 10
        self.word2vec = Word2Vec(iter=iteration_count, min_count=0, size=1000, workers=8)
        self.word2vec_vocab = self.word2vec.build_vocab(sentences=sentences)
        print("training word embeddings")
        self.word2vec.train(sentences=sentences, total_examples=len(sentences), epochs=iteration_count)
