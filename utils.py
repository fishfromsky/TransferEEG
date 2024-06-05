import scipy.io as sio
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import copy
import torch.nn.functional as F


def read_dataset(file_path):
    sess_data = {}
    sess_label = {}
    label_data = sio.loadmat(os.path.join(file_path, 'label.mat'))['label'][0]
    label_data += 1
    for sess in os.listdir(file_path):
        print('Processing Session ', sess)
        sess_pth = os.path.join(file_path, sess)
        if os.path.isdir(sess_pth):
            subject_data = []
            for file in os.listdir(sess_pth):
                mat_data = sio.loadmat(os.path.join(sess_pth, file))
                mat_de_data = {key: value for key, value in mat_data.items() if key.startswith('de_LDS')}
                subject_data.append(mat_de_data)
            sess_data[sess] = subject_data
            sess_label[sess] = label_data
    return sess_data, sess_label


def read_dataset_SEED_IV(file_path):
    sess_data = {}
    sess_label = {}
    session_label = [[1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
                      [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
                      [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]
                      ]
    for sess in os.listdir(file_path):
        print('Processing Session ', sess)
        sess_idx = int(sess) - 1
        sess_path = os.path.join(file_path, sess)
        subject_data = []
        for file in os.listdir(sess_path):
            mat_data = sio.loadmat(os.path.join(sess_path, file))
            mat_de_data = {key: value for key, value in mat_data.items() if key.startswith('de_LDS')}
            subject_data.append(mat_de_data)
        sess_data[sess] = subject_data
        sess_label[sess] = session_label[sess_idx]

    return sess_data, sess_label


def data_factory(file_path, config):
    if config.dataset == 'SEED':
        data, label = read_dataset(file_path)
    else:
        data, label = read_dataset_SEED_IV(file_path)
    data_package = {}
    label_package = {}
    for sess in data.keys():
        subject_data = data[sess]
        subject_label = label[sess]
        all_mats = []
        all_labels = []
        for i in range(config.source_number):
            trial_data = subject_data[i]
            data_comb = None
            label_comb = []
            for index, key in enumerate(trial_data.keys()):
                idx = int(key[6:])
                data_per_trial = trial_data[key].transpose((1, 0, 2)).reshape([-1, config.channel*config.band])
                if index == 0:
                    data_comb = data_per_trial
                else:
                    data_comb = np.vstack([data_comb, data_per_trial])
                label_comb.extend([subject_label[idx-1] for _ in range(data_per_trial.shape[0])])
            label_comb = np.array(label_comb)

            all_mats.append(data_comb)
            all_labels.append(label_comb)

        data_package[sess] = all_mats
        label_package[sess] = all_labels

    return data_package, label_package


def norminy(data):
    dataT = data.T
    for i in range(dataT.shape[0]):
        dataT[i] = normalization(dataT[i])
    return dataT.T


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


class myDataset(Dataset):
    def __init__(self, data, label):
        super(myDataset, self).__init__()
        self.data = data
        self.label = label
        self.weight = np.array([1. for _ in range(len(label))])

    def __len__(self):
        return len(self.label)

    def set_weight_label(self, weight, index):
        weight = weight.cpu().numpy()
        self.weight[index] = weight

    def __getitem__(self, item):
        return self.data[item], self.label[item], self.weight[item], item


def generate_data_loader(data, label, config):
    data_loader = DataLoader(myDataset(data, label), batch_size=config.batch_size, shuffle=True,
                             drop_last=False)
    return data_loader


def shuffle(data_x, label_x):
    indexes = np.array([i for i in range(len(label_x))])
    np.random.shuffle(indexes)
    data_x = data_x[indexes, :]
    label_x = label_x[indexes]
    return data_x, label_x


class EarlyStopping:
    def __init__(self):
        self.best_score = None
        self.save_model = None

    def __call__(self, val_acc, model):
        score = val_acc

        if self.best_score is None:
            self.best_score = score
            self.save_model = copy.deepcopy(model)

        elif score <= self.best_score:
            pass
        else:
            self.best_score = score
            self.save_model = copy.deepcopy(model)


if __name__ == '__main__':
    from config import Config
    config = Config()
    data_factory('../MS_MDA_pytorch/eeg_feature_smooth', config)
