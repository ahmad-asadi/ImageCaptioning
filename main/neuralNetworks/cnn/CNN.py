import tensorflow as tf

from main.neuralNetworks.cnn.inception import inception


class CNN:
    def __init__(self, data_dir):
        inception.data_dir = data_dir
        inception.maybe_download()
        with tf.name_scope("Encoder"):
            self.net = inception.Inception()

    def classify(self, image_path):
        pred = self.net.classify(image_path=image_path)
        # self.net.print_scores(pred=pred, k=1008, only_first_name=True)
        return pred

    def extract_features(self, image_path):
        return self.net.transfer_values(image_path=image_path)
