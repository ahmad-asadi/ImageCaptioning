from main.neuralNetworks.cnn.inception import inception


class CNN:
    def __init__(self, data_dir):
        inception.data_dir = data_dir
        inception.maybe_download()
        self.net = inception.Inception()

    def classify(self, image_path):
        pred = self.net.classify(image_path=image_path)
        # self.net.print_scores(pred=pred, k=1008, only_first_name=True)
        return pred

    def fine_tune(self, error_vector):
        self.net.__reduce__()
