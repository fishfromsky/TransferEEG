import math


class Config:
    def __init__(self):
        self.dataset = 'SEED'

        if self.dataset == 'SEED':
            self.channel = 62
            self.band = 5
            self.source_number = 15
            self.total_epoch = 100
            self.batch_size = 64
            self.num_classes = 3
            self.threshold = 0.9
            self.data_length = 3394
            self.epochs = math.ceil((self.data_length * self.total_epoch) / self.batch_size)
            self.pre_train = 20
            self.eps = 1e-4
            self.kmeans_iter = 5
            self.alpha = 1
            self.status = 'test'
            self.learning_rate = 0.001
            self.weight_decay = 0.0005
            self.file_path = '../ExtractedFeatures'
            self.result_path = 'result/SEED/cross_subject/batch_size_64/with_cluster_th_0.9/'

