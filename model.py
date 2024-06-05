import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cdist
import numpy as np
from torch.autograd import Variable, Function
from loss import grad_reverse, FocalLoss


class Encoder(nn.Module):
    """
    MLP encoder model for our model
    """

    def __init__(self, num_fts):
        super(Encoder, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(num_fts, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.features(x)
        return x


class PrivateEncoder(nn.Module):
    def __init__(self):
        super(PrivateEncoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layers(x)


class ClassClassifier(nn.Module):
    def __init__(self, num_cls):
        super(ClassClassifier, self).__init__()
        self.classifier = nn.Linear(32, num_cls)

    def forward(self, x):
        x = self.classifier(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        return self.layers(x)


class MS_MDA(nn.Module):
    def __init__(self, num_fts, num_cls, source_number, config, device):
        super(MS_MDA, self).__init__()

        self.config = config
        self.device = device
        self.source_number = source_number
        self.encoder = Encoder(num_fts)
        self.num_classes = num_cls
        self.private_encoder = nn.ModuleList([PrivateEncoder() for _ in range(source_number)])
        self.private_classifier = nn.ModuleList([ClassClassifier(num_cls) for _ in range(source_number)])

        self.discriminator = nn.ModuleList([Discriminator() for _ in range(source_number)])

        self.pseudo_encoder = PrivateEncoder()
        self.pseudo_classifier = ClassClassifier(num_cls)

        self.focal_loss = FocalLoss(class_num=2, gamma=5)

    def forward(self, source_data_batch, source_label_batch, source_weight_batch, target_data_batch, mark, learn_cen):
        source_feature_shared = self.encoder(source_data_batch)
        target_feature_shared = self.encoder(target_data_batch)

        source_feature_x = self.private_encoder[mark](source_feature_shared)
        target_x_list = []
        for i in range(self.source_number):
            target_x_list.append(self.private_encoder[i](target_feature_shared))

        source_feature_y = self.private_classifier[mark](source_feature_x)
        target_feature_y = self.private_classifier[mark](target_x_list[mark])

        target_feature_x = target_x_list[mark]
        prob_t = (1 + (target_feature_x.unsqueeze(1) - learn_cen.unsqueeze(0)).pow(2).sum(2) / self.config.alpha) \
            .pow(-(self.config.alpha + 1) / 2)
        t_loss_cluster = self.calculate_pq_loss(prob_t, False)

        # domain_source = self.discriminator[mark](grad_reverse(source_feature_x))
        # domain_target = self.discriminator[mark](grad_reverse(target_x_list[mark]))
        #
        # domain_source_label = Variable(torch.zeros(domain_source.size(0)).long().to(self.device))
        # domain_target_label = Variable(torch.ones(domain_target.size(0)).long().to(self.device))

        # d_loss_s = self.focal_loss(domain_source, domain_source_label)
        # d_loss_t = self.focal_loss(domain_target, domain_target_label)
        # d_loss = d_loss_s + d_loss_t

        t_loss = self.calculate_pq_loss(target_feature_y)

        prob_c = (1 + (source_feature_x.unsqueeze(1) - learn_cen.unsqueeze(0)).pow(2).sum(2) / self.config.alpha) \
            .pow(-(self.config.alpha + 1) / 2)
        c_loss_cluster = self.calculate_classify_loss(prob_c, source_label_batch, source_weight_batch, False)
        c_loss = self.calculate_classify_loss(source_feature_y, source_label_batch, source_weight_batch)

        return c_loss, t_loss, c_loss_cluster, t_loss_cluster

    def calculate_pq_loss(self, output, softmax=True):
        if softmax:
            prob_p = F.softmax(output, dim=-1)
        else:
            prob_p = output / output.sum(1, keepdim=True)

        prob_q2 = prob_p / prob_p.sum(0, keepdim=True).pow(0.5)
        prob_q2 /= prob_q2.sum(1, keepdim=True)
        prob_q = prob_q2

        if softmax:
            loss = - (prob_q * F.log_softmax(output, dim=1)).sum(1).mean()
        else:
            loss = - (prob_q * prob_p.log()).sum(1).mean()

        return loss

    def calculate_classify_loss(self, output, target, weight, softmax=True):
        if softmax:
            prob_p = F.softmax(output, dim=-1)
        else:
            prob_p = output / output.sum(1, keepdim=True)

        prob_q = Variable(torch.FloatTensor(prob_p.size()).fill_(0)).to(self.device)
        prob_q.scatter_(1, target.unsqueeze(1), torch.ones(prob_p.size(0), 1).to(self.device))

        if softmax:
            loss = - (prob_q * F.log_softmax(output, dim=1)).sum(1).mean()
        else:
            loss = - (prob_q * prob_p.log()).sum(1).mean()

        return loss

    def return_hidden_layer(self, data_batch):
        shared_feature = self.encoder(data_batch)
        hidden_feature = []
        for i in range(len(self.private_encoder)):
            feature_x = self.private_encoder[i](shared_feature)
            hidden_feature.append(feature_x)
        return hidden_feature

    def predict_class(self, data_batch):
        shared_feature = self.encoder(data_batch)
        pred = []
        for i in range(len(self.private_encoder)):
            feature_x = self.private_encoder[i](shared_feature)
            result = self.private_classifier[i](feature_x)
            pred.append(result)
        return pred

    def calculate_centroid(self, source_loader, target_loader, mark):
        self.encoder.eval()
        self.private_encoder[mark].eval()
        self.private_classifier[mark].eval()

        all_source_feature = None
        all_source_label = None
        all_target_feature = None
        all_target_label = None
        all_target_confidence = None
        flag = True

        with torch.no_grad():
            for data, label, _, _ in source_loader:
                data = data.to(self.device).float()
                label = label.to(self.device).long()

                c_feature_source = self.encoder(data)
                h_feature_source = self.private_encoder[mark](c_feature_source)
                if flag:
                    all_source_feature = h_feature_source
                    all_source_label = label
                    flag = False
                else:
                    all_source_feature = torch.cat((all_source_feature, h_feature_source), dim=0)
                    all_source_label = torch.hstack((all_source_label, label))

            flag = True
            for data, _, _, _ in target_loader:
                data = data.to(self.device).float()

                c_feature_target = self.encoder(data)
                h_feature_target = self.private_encoder[mark](c_feature_target)
                output_target = self.private_classifier[mark](h_feature_target)
                output_confidence = torch.softmax(output_target, dim=-1)
                label_target = torch.argmax(output_target, dim=-1)
                confidence_target = output_confidence.gather(1, label_target.view(-1, 1)).squeeze()
                if flag:
                    all_target_feature = h_feature_target
                    all_target_label = label_target
                    all_target_confidence = confidence_target
                    flag = False
                else:
                    all_target_feature = torch.cat((all_target_feature, h_feature_target), dim=0)
                    all_target_label = torch.hstack((all_target_label, label_target))
                    all_target_confidence = torch.hstack((all_target_confidence, confidence_target))

        mask = all_target_confidence > self.config.threshold
        all_target_feature = all_target_feature[mask, :]
        all_target_label = all_target_label[mask]

        source_id = F.one_hot(all_source_label, self.config.num_classes).float()
        source_cen = torch.matmul(source_id.transpose(0, 1), all_source_feature)
        source_centroid = (source_cen.t() / (source_id.sum(dim=0)+self.config.eps)).t()

        target_id = F.one_hot(all_target_label, self.config.num_classes).float()
        target_cen = torch.matmul(target_id.transpose(0, 1), all_target_feature)
        target_centroid = (target_cen.t() / (target_id.sum(dim=0)+self.config.eps)).t()

        return source_centroid, target_centroid








