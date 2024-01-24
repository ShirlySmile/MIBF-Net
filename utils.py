import os, torch, math
from sklearn .metrics import confusion_matrix
from torch.utils.data import DataLoader
from dataset import MyData


from models import MIBfusion, IBloss_R
import numpy as np


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img1):
        for t in self.transforms:
            img1 = t(img1)
        return img1

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass



def getnet(model, in_channels1, in_channels2, cls_num):
    loss = None
    if model is 'MIBfusion':
        net = MIBfusion(in_channels1, in_channels2, cls_num)
        loss = IBloss_R()
    return net, loss



def get_training_dataloader(img1, img2, index_train_data, patch_size, ratio, label_train, BATCH_SIZE):

    transform_train = None

    train_data = MyData(img1, img2, index_train_data, patch_size, ratio, label_train, transform=transform_train)
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader

def get_test_dataloader(img1, img2, index_test_data, patch_size, ratio, label_test, BATCH_SIZE):
    transform_test = None
    test_data = MyData(img1, img2, index_test_data, patch_size, ratio, label_test, transform=transform_test)
    test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return test_loader


def get_all_dataloader(img1, img2, index_all_data, patch_size, ratio, BATCH_SIZE):
    transform_all = None
    all_data = MyData(img1, img2, index_all_data, patch_size, ratio, transform=transform_all)
    all_data_loader = DataLoader(dataset=all_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return all_data_loader



def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.BATCH_SIZE / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * args.LR


def eval_model(net, test_loader, label_test, mode):

    net.cuda()
    l = 0
    y_pred = []
    net.eval()


    for step, (ms, pan, label, gt_xy) in enumerate(test_loader):
        l = l + 1
        with torch.no_grad():
            output = net(ms.cuda(), pan.cuda())

        if isinstance(output, tuple):
            output = output[0]
        try:
            pred_y = torch.max(output, 1)[1].cuda().data.squeeze()
        except:
            output = output.unsqueeze(0)
            pred_y = torch.max(output, 1)[1].cuda().data.squeeze()
        if l == 1:
            y_pred = pred_y.cpu().numpy()
        else:
            try:
                y_pred = np.concatenate((y_pred, pred_y.cpu().numpy()), axis=0)
            except:
                pred_y = pred_y.unsqueeze(0)
                y_pred = np.concatenate((y_pred, pred_y.cpu().numpy()), axis=0)


    showlabel = label_test.cpu().numpy()
    ss = np.concatenate((showlabel[:, np.newaxis], y_pred[:, np.newaxis]), axis=1)
    con_mat = confusion_matrix(y_true=label_test, y_pred=y_pred)
    if mode == 'test':
        print('con_mat', con_mat)


    all_acr = 0
    p = 0
    column = np.sum(con_mat, axis=0)
    line = np.sum(con_mat, axis=1)
    for i, clas in enumerate(con_mat):
        precise = clas[i]
        all_acr = precise + all_acr

        acr = precise / column[i]
        recall = precise / line[i]

        f1 = 2 * acr * recall / (acr + recall)
        temp = column[i] * line[i]
        p = p + temp
        if mode is 'test':
            print("class %d: || 准确率: %.7f || 召回率: %.7f || F1: %.7f " % (i, acr, recall, f1))
    OA = np.trace(con_mat) / np.sum(con_mat)
    AA = np.mean(con_mat.diagonal() / np.sum(con_mat, axis=1))
    Pc = np.sum(np.sum(con_mat, axis=0) * np.sum(con_mat, axis=1)) / (np.sum(con_mat)) ** 2
    Kappa = (OA - Pc) / (1 - Pc)

    return OA, AA, Kappa




