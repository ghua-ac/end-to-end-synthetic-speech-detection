import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
from data import PrepASV19Dataset, PrepASV15Dataset
import models
from test import asv_cal_accuracies, cal_roc_eer
import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # TODO: Define dataset scope and data type
    # specify the data type and root path
    dataset = 19   # {'ASVspoof2019': 19, 'ASVspoof2015': 15}
    data_type = 'time_frame'  # {'time_frame', 'CQT'}

    if not os.path.exists('./trained_models/'):
        os.makedirs('./trained_models/')

    if data_type == 'time_frame':
        if dataset == 15:
            root_path = 'F:/ASVspoof2015/'
            train_protocol_file_path = root_path + 'CM_protocol/cm_train.trn.txt'
            dev_protocol_file_path = root_path + 'CM_protocol/cm_develop.ndx.txt'
            eval_protocol_file_path = root_path + 'CM_protocol/cm_evaluation.ndx.txt'
            train_data_path = root_path + 'data/train_6/'
            dev_data_path = root_path + 'data/dev_6/'
            eval_data_path = root_path + 'data/eval_6/'
        else:
            root_path = 'F:/ASVspoof2019/LA/'
            train_protocol_file_path = root_path + 'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'
            dev_protocol_file_path = root_path + 'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'
            eval_protocol_file_path = root_path + 'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt'
            train_data_path = root_path + 'data/train_6/'
            dev_data_path   = root_path + 'data/dev_6/'
            eval_data_path  = root_path + 'data/eval_6/'

    elif data_type == 'CQT':
        if dataset == 15:
            root_path = 'F:/ASVspoof2015/'
            train_protocol_file_path = root_path + 'CM_protocol/cm_train.trn.txt'
            dev_protocol_file_path = root_path + 'CM_protocol/cm_develop.ndx.txt'
            eval_protocol_file_path = root_path + 'CM_protocol/cm_evaluation.ndx.txt'
            train_data_path = root_path + 'data/train_6.4_cqt/'
            dev_data_path = root_path + 'data/dev_6.4_cqt/'
            eval_data_path = root_path + 'data/eval_6.4_cqt/'
        else:
            root_path = 'F:/ASVspoof2019/LA/'
            train_protocol_file_path = root_path + 'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'
            dev_protocol_file_path   = root_path + 'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'
            eval_protocol_file_path  = root_path + 'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt'
            train_data_path = root_path + 'data/train_6.4_cqt/'
            dev_data_path   = root_path + 'data/dev_6.4_cqt/'
            eval_data_path  = root_path + 'data/eval_6.4_cqt/'
    else:
        print("Program only supports 'time_frame' and 'CQT' data types.")
        sys.exit()

    # TODO: Prepare data and set training parameters
    if dataset == 15:
        train_set = PrepASV15Dataset(train_protocol_file_path, train_data_path, data_type=data_type)
    else:
        train_set = PrepASV19Dataset(train_protocol_file_path, train_data_path, data_type=data_type)
    weights = train_set.get_weights().to(device)  # weight used for WCE
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)

    if data_type == 'CQT':
        Net = models.SSDNet2D()  # 2D-Res-TSSDNet
    else:
        Net = models.SSDNet1D()   # Res-TSSDNet
        # Net = models.DilatedNet()  # Inc-TSSDNet
    Net = Net.to(device)

    num_total_learnable_params = sum(i.numel() for i in Net.parameters() if i.requires_grad)
    print('Number of learnable params: {}.'.format(num_total_learnable_params))

    optimizer = optim.Adam(Net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    loss_type = 'WCE'  # {'WCE', 'mixup'}

    # TODO: Training
    print('Training data: {}, Date type: {}. Training started...'.format(train_data_path, data_type))

    num_epoch = 100
    loss_per_epoch = torch.zeros(num_epoch,)
    best_d_eer = [.09, 0]

    if not os.path.exists('./trained_models/train_log/'):
        os.makedirs('./trained_models/train_log/')

    log_path = './trained_models/train_log/'
    time_name = time.ctime()
    time_name = time_name.replace(' ', '_')
    time_name = time_name.replace(':', '_')
    f = open(log_path + time_name + '.csv', 'w+')

    # for epoch in range(check_point['epoch']+1, num_epoch):
    for epoch in range(num_epoch):
        Net.train()
        t = time.time()
        total_loss = 0
        counter = 0
        for batch in train_loader:
            counter += 1
            # forward
            samples, labels, _ = batch
            samples = samples.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            if loss_type == 'mixup':
                # mixup
                alpha = 0.1
                lam = np.random.beta(alpha, alpha)
                lam = torch.tensor(lam, requires_grad=False)
                index = torch.randperm(len(labels))
                samples = lam*samples + (1-lam)*samples[index, :]
                preds = Net(samples)
                labels_b = labels[index]
                loss = lam * F.cross_entropy(preds, labels) + (1 - lam) * F.cross_entropy(preds, labels_b)
            else:
                preds = Net(samples)
                loss = F.cross_entropy(preds, labels, weight=weights)
                # loss = F.cross_entropy(preds, labels)

            # backward
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        loss_per_epoch[epoch] = total_loss/counter

        dev_accuracy, d_probs = asv_cal_accuracies(dev_protocol_file_path, dev_data_path, Net, device, data_type=data_type, dataset=dataset)
        d_eer = cal_roc_eer(d_probs, show_plot=False)
        if d_eer <= best_d_eer[0]:
            best_d_eer[0] = d_eer
            best_d_eer[1] = int(epoch)

            eval_accuracy, e_probs = asv_cal_accuracies(eval_protocol_file_path, eval_data_path, Net, device, data_type=data_type, dataset=dataset)
            e_eer = cal_roc_eer(e_probs, show_plot=False)
        else:
            e_eer = .99
            eval_accuracy = 0.00

        net_str = data_type + '_' + str(epoch) + '_' + 'ASVspoof20' + str(dataset) + '_LA_Loss_' + str(round(total_loss / counter, 4)) + '_dEER_' \
                            + str(round(d_eer * 100, 2)) + '%_eEER_' + str(round(e_eer * 100, 2)) + '%.pth'
        torch.save({'epoch': epoch, 'model_state_dict': Net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss_per_epoch}, ('./trained_models/' + net_str))

        elapsed = time.time() - t

        print_str = 'Epoch: {}, Elapsed: {:.2f} mins, lr: {:.3f}e-3, Loss: {:.4f}, d_acc: {:.2f}%, e_acc: {:.2f}%, ' \
                    'dEER: {:.2f}%, eEER: {:.2f}%, best_dEER: {:.2f}% from epoch {}.'.\
                    format(epoch, elapsed/60, optimizer.param_groups[0]['lr']*1000, total_loss / counter, dev_accuracy * 100,
                           eval_accuracy * 100, d_eer * 100, e_eer * 100, best_d_eer[0] * 100, int(best_d_eer[1]))
        print(print_str)
        df = pd.DataFrame([print_str])
        df.to_csv(log_path + time_name + '.csv', sep=' ', mode='a', header=False, index=False)

        scheduler.step()

    f.close()
    plt.plot(torch.log10(loss_per_epoch))
    plt.show()

    print('End of Program.')
