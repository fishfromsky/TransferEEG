import torch
import torchmetrics
from torch.autograd import Variable
from config import Config
from utils import generate_data_loader, data_factory, shuffle, norminy
from model import MS_MDA
import numpy as np
from utils import EarlyStopping
import math
import torch.nn.functional as F
import pandas as pd
import os


config = Config()

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

np.random.seed(100)

metrics = torchmetrics.Accuracy().to(device)


def train_model(source_comb, source_label_comb, target_comb, target_label_comb, target, sess):
    target_loader = generate_data_loader(target_comb, target_label_comb, config)
    source_loader = []
    for source in range(len(source_comb)):
        source_loader.append(generate_data_loader(source_comb[source], source_label_comb[source], config))

    model = MS_MDA(target_comb.shape[-1], config.num_classes, len(source_loader), config, device)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    target_iter = iter(target_loader)
    source_iter_list = []
    for i in range(len(source_loader)):
        source_iter_list.append(iter(source_loader[i]))

    real_epoch = 1

    early_stopping = EarlyStopping()

    final_loss_list = []
    final_loss_s_list = []
    final_loss_t_list = []

    for i in range(len(source_loader)):
        final_loss_list.append([])
        final_loss_s_list.append([])
        final_loss_t_list.append([])

    csv_acc_list = []
    csv_loss_list = []

    total_loss_list = []
    total_train_loss_s_list = []
    total_train_loss_t_list = []

    for i in range(len(source_loader)):
        total_train_loss_s_list.append([])
        total_train_loss_t_list.append([])
        total_loss_list.append([])

    cen_list = []
    for mark in range(len(source_loader)):
        cen_temp = Variable(torch.FloatTensor(config.num_classes, 64)).fill_(0).to(device)
        cen_temp.requires_grad_(True)
        cen_list.append(cen_temp)

    for mark in range(len(source_loader)):
        source_centroid, target_centroid = model.calculate_centroid(source_loader[mark], target_loader, mark)
        cen_list[mark].data = source_centroid.data.clone()

    for epoch in range(config.epochs):
        model.train()
        
        for mark in range(len(source_loader)):

            try:
                source_data_batch, source_label_batch, source_weight_batch, _ = next(source_iter_list[mark])
            except Exception as err:
                source_iter_list[mark] = iter(source_loader[mark])
                source_data_batch, source_label_batch, source_weight_batch, _ = next(source_iter_list[mark])

            try:
                target_data_batch, target_label_batch, _, _ = next(target_iter)
            except Exception as err:
                target_iter = iter(target_loader)
                target_data_batch, target_label_batch, _, _ = next(target_iter)

            source_data_batch = source_data_batch.to(device).float()
            source_label_batch = source_label_batch.to(device).long()

            target_data_batch = target_data_batch.to(device).float()

            c_loss, t_loss, c_loss_cluster, t_loss_cluster = model(source_data_batch, source_label_batch, source_weight_batch,
                                           target_data_batch, mark, cen_list[mark])

            gamma = 2 / (1 + math.exp(-10 * epoch / config.epochs)) - 1

            loss = c_loss + gamma*t_loss + gamma*c_loss_cluster + gamma * t_loss_cluster

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss_list[mark].append(loss.item())
            total_train_loss_s_list[mark].append(c_loss.item())
            total_train_loss_t_list[mark].append(t_loss.item())

        if math.floor(((epoch + 1) * config.batch_size) / config.data_length) == real_epoch:

            model.eval()

            for m in range(len(source_loader)):
                source_centroid, target_centroid = model.calculate_centroid(source_loader[m], target_loader, m)
                centroid = Variable(torch.FloatTensor(config.num_classes, source_centroid.shape[-1])).fill_(0).to(device)
                for c in range(centroid.shape[0]):
                    non_zero = torch.nonzero(target_centroid[c, :])
                    if min(non_zero.shape) == 0:
                        print('Centroid Number is not full')
                        centroid[c, :] = source_centroid[c, :].data.clone()
                    else:
                        centroid[c, :] = ((source_centroid[c, :] + target_centroid[c, :])/2).data.clone()
                cen_list[m].data = centroid.data.clone()

            for mark in range(len(source_loader)):
                print('Train source ' + str(
                    mark + 1) + ' Total loss: {:.4f} - source loss: {:.4f} - target_loss: {:.4f}'.
                      format(np.mean(total_loss_list[mark]), np.mean(total_train_loss_s_list[mark]),
                             np.mean(total_train_loss_t_list[mark])))

                final_loss_list[mark].append(np.mean(total_loss_list[mark]))
                final_loss_s_list[mark].append(np.mean(total_train_loss_s_list[mark]))
                final_loss_t_list[mark].append(np.mean(total_train_loss_t_list[mark]))

                total_loss_list[mark] = []
                total_train_loss_s_list[mark] = []
                total_train_loss_t_list[mark] = []

            real_epoch += 1
            acc_list = []
            loss_list = []
            for t_data, t_label, _, _ in target_loader:
                t_data = t_data.to(device).float()
                t_label = t_label.to(device).long()
                t_pred = model.predict_class(t_data)
                for i in range(len(t_pred)):
                    t_pred[i] = F.softmax(t_pred[i], dim=-1)
                pred = sum(t_pred) / len(t_pred)
                cls_loss = F.nll_loss(F.log_softmax(pred, dim=1), t_label)
                acc = metrics(torch.argmax(pred, dim=-1), t_label)
                acc_list.append(acc.item())
                loss_list.append(cls_loss.item())
            print('Epoch {:.0f}: Testing Acc: {:.4f} - Testing Loss: {:.4f}'.format(real_epoch - 1, np.mean(acc_list),
                                                                                    np.mean(loss_list)))
            csv_acc_list.append(np.mean(acc_list))
            csv_loss_list.append(np.mean(loss_list))

            early_stopping(np.mean(acc_list), model)

    print('Best score recorded: ', early_stopping.best_score)
    csv_dir = os.path.join(config.result_path, str(sess), str(target), 'csv')
    model_dir = os.path.join(config.result_path, str(sess), str(target), 'model')
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(early_stopping.save_model.state_dict(), os.path.join(model_dir, 'model.pth'))

    writer = pd.ExcelWriter(os.path.join(csv_dir, 'result.xls'), engine='openpyxl')
    csv_dict = {
        'Test Accuracy': csv_acc_list,
        'Test Loss': csv_loss_list
    }
    df_test = pd.DataFrame(csv_dict)
    df_test.to_excel(writer, sheet_name='test result')
    for mark in range(len(source_loader)):
        data_dict = {
            'Train Loss Total': final_loss_list[mark],
            'Train Loss Source': final_loss_s_list[mark],
            'Train Loss Target': final_loss_t_list[mark],
        }
        df = pd.DataFrame(data_dict)
        df.to_excel(writer, sheet_name='Source' + str(mark))
    writer.save()


def train_cross_subject(sess):
    data_package, label_package = data_factory(config.file_path, config)
    data_sess, label_sess = data_package[str(sess)], label_package[str(sess)]
    for target in range(12, config.source_number):
        target_comb, target_label_comb = data_sess[target], label_sess[target]
        target_comb = norminy(target_comb)
        target_comb, target_label_comb = shuffle(target_comb, target_label_comb)

        source_comb, source_label_comb = [], []
        for source in range(config.source_number):
            if source != target:
                source_comb_temp, source_label_comb_temp = data_sess[source], label_sess[source]
                source_comb_temp = norminy(source_comb_temp)
                source_comb_temp, source_label_comb_temp = shuffle(source_comb_temp, source_label_comb_temp)
                source_comb.append(source_comb_temp)
                source_label_comb.append(source_label_comb_temp)

        train_model(source_comb, source_label_comb, target_comb, target_label_comb, target, sess)


def train_cross_session(sub):
    data_package, label_package = data_factory(config.file_path, config)

    for target in range(1, 4):
        target_comb, target_label_comb = data_package[str(target)][sub], label_package[str(target)][sub]
        target_comb = norminy(target_comb)
        target_comb, target_label_comb = shuffle(target_comb, target_label_comb)

        source_comb, source_label_comb = [], []
        for source in range(1, 4):
            if source != target:
                source_comb_temp, source_label_comb_temp = data_package[str(source)][sub], label_package[str(source)][sub]
                source_comb_temp = norminy(source_comb_temp)
                source_comb_temp, source_label_comb_temp = shuffle(source_comb_temp, source_label_comb_temp)
                source_comb.append(source_comb_temp)
                source_label_comb.append(source_label_comb_temp)

        train_model(source_comb, source_label_comb, target_comb, target_label_comb, target, sub)


if __name__ == '__main__':
    for sess in range(1, 4):
        train_cross_subject(sess)
    # for sub in range(15):
    #     train_cross_session(sub)

