import torch.nn as nn
from utils import *
import random, argparse
from dataset import getdata, splitdata



def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='MIBfusion')
    parser.add_argument('--dataname', type=str, default='MUUFL', help='dataset used for training: MUUFL/2018/2013')

    parser.add_argument('--partition', type=str, default='Cls_Rate_Same', help='the data partitioning strategy: Cls_Num_Same/Cls_Rate_Same')
    parser.add_argument('--Traindata_Num', type=float, default=20, help='the split dataset per-class number')
    parser.add_argument('--Traindata_Rate', type=float, default=0.02, help='the split dataset rate')
    parser.add_argument('--patch_size', type=int, default=7, help='the split image size')

    parser.add_argument('--BATCH_SIZE', type=int, default=64, help='input batch size for training (default: 64)')


    parser.add_argument('--LR', type=float, default=0.001, help='learning rate (default: 0.1) Berlin:0.0005/45')
    parser.add_argument('--epochs', type=int, default=300, help='number of  epochs')

    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--logdir', type=str, default='./log/', help='save message')
    args = parser.parse_args()
    return args




if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    args = get_args()
    device = torch.device(args.device)

    seed = 2333


    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)


    img1, img2, labels = getdata(args.dataname, 0.005, args.patch_size)
    index_all_data, index_train_data, index_test_data, label_train, label_test, cls_num = splitdata(labels, args.Traindata_Rate, args.Traindata_Num, args.partition, seed)


    ratio = img2.shape[1] / img1.shape[1]

    net, criterion_extra = getnet(args.model, img1.shape[0], img2.shape[0], cls_num-1)
    criterion = nn.CrossEntropyLoss().cuda()

    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)

    net.cuda()

    optimizer = torch.optim.Adam(net.parameters(),lr=args.LR,weight_decay=0.0005)
    train_loader = get_training_dataloader(img1, img2, index_train_data, args.patch_size, ratio, label_train, args.BATCH_SIZE)

    train_allloader = get_training_dataloader(img1, img2, index_train_data, args.patch_size, ratio, label_train,
                                           len(index_train_data))
    test_allloader = get_test_dataloader(img1, img2, index_test_data, args.patch_size, ratio, label_test, len(index_test_data))

    test_loader = get_test_dataloader(img1, img2, index_test_data, args.patch_size, ratio, label_test, args.BATCH_SIZE)
    all_loader = get_all_dataloader(img1, img2, index_all_data, args.patch_size, ratio, args.BATCH_SIZE)
    Aall_loader = get_all_dataloader(img1, img2, index_all_data, args.patch_size, ratio, len(index_all_data))


    for epoch in range(args.epochs):
        valid_batch = iter(test_loader)
        for step, (img1_patch, img2_patch, label, _) in enumerate(train_loader, start=epoch * len(train_loader)):

            output = net(img1_patch, img2_patch)
            if criterion_extra:
                loss = criterion_extra(output, label)
            else:
                loss = criterion(output, label)

            if isinstance(output, tuple):
                output = output[0]

            train_pred_y = torch.max(output, 1)[1].cuda().data.squeeze()
            train_acc = (train_pred_y == label).sum().item() / float(label.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                net.eval()
                img1_test, img2_test, label_t, _ = next(valid_batch)
                with torch.no_grad():
                    test_output = net(img1_test, img2_test)

                if isinstance(test_output, tuple):
                    test_output = test_output[0]
                pred_y = torch.max(test_output, 1)[1].cuda().data.squeeze()
                accuracy = (pred_y == label_t).sum().item() / float(label_t.size(0))
                print('[Epoch:%3d] || step: %3d || LR: %.6f  || train loss: %.4f || train acc: %.4f || test acc: %.4f' % (epoch, step, optimizer.param_groups[0]['lr'], loss.item(), train_acc, accuracy))
                net.train()
            adjust_learning_rate(args, optimizer, train_loader, step)

    savedir = args.logdir
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    save_model_path = savedir + 'checkpoint.pth.tar'
    torch.save({'state_dict': net.state_dict()}, save_model_path)

    net2, _ = getnet(args.model, img1.shape[0], img2.shape[0], cls_num - 1)
    checkpoint = torch.load(save_model_path)
    net2.load_state_dict(checkpoint['state_dict'])
    OA, AA, Kappa = eval_model(net2, test_loader, label_test, 'test')
    print('OA:', OA)
    print('AA:', AA)
    print('Kappa:', Kappa)










