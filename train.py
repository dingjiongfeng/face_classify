import logging
import csv
from numpy.lib.function_base import append
import pandas as pd
import os
import numpy as np
import torch
import torchvision.models as models
import argparse
import timm
import torch.nn as nn
import torch.optim as optim
from util import *
from torch.backends.cudnn import benchmark
from IPython import embed
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
from loss import *
warnings.filterwarnings('ignore')


logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler("log.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

benchmark = True
parser = argparse.ArgumentParser(
    description='train CNN to classify the fake faces')
parser.add_argument('--name', type=str, default='xxx',
                    help='the train id')
parser.add_argument('--model', type=str, default='resnet50',
                    help='the model to train')
parser.add_argument('--figsize', type=int, default='224',
                    help='the figsize of image')
parser.add_argument('--max-epoch', type=int, default='51',
                    help='the epoch that need to train the model')
parser.add_argument('--test-freq', type=int,
                    default='1', help='test frequency')
parser.add_argument('--resume', action='store_true', help='resume option')
parser.add_argument('--checkpoint-path', type=str,
                    default='resnet50_224_loss_epoch1_f1_score0.8613834444136302.pth.tar', help='checkpoint path')

args = parser.parse_args()
# dpn92 legacy_seresnext50_32x4d seresnet101 seresnext50 resnet101 /


def create_data_lists(data_dir, split=False):
    '''
    only in training environment
    read train image label from csv file and split them into
    X_train, X_val, y_train, y_val
    '''
    df = pd.read_csv(os.path.join(data_dir, train_csv_name))

    df = pd.DataFrame(df['fnames\tlabel'].str.split(
        '\t').tolist(), columns=['fname', 'label'])

    fname, label = df['fname'].values, df['label'].values  # numpy array
    X_train, y_train = fname, label

    if split:
        ss = StratifiedShuffleSplit(n_splits=1, test_size=0.1)
        for train_index, val_index in ss.split(fname, label):
            X_train, X_val = fname[train_index], fname[val_index]
            y_train, y_val = label[train_index], label[val_index]

        val_transform = T.Compose([T.ToTensor(), T.Resize((args.figsize, args.figsize)), T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        val_dataset = FaceDataset(os.path.join(data_dir, image_dir, 'train'), X_val, y_val,
                                  transform=val_transform, sift_transform=None)
        val_loader = DataLoader(
            dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_gpu)

    train_transform = T.Compose([T.ToTensor(), T.RandomHorizontalFlip(), T.RandomResizedCrop(
        (args.figsize, args.figsize)), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_dataset = FaceDataset(os.path.join(data_dir, image_dir, 'train'), X_train, y_train,
                                transform=train_transform, sift_transform=None)  # cityscapeDatasetTransform
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_gpu)
    if split:
        return train_loader, val_loader
    else:
        return train_loader


def get_resnet18():
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, 2)
    return model


def get_resnet34():
    model = models.resnet34(pretrained=True)
    model.fc = nn.Linear(512, 2)
    return model


def get_resnet50():
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(2048, 2)
    return model


def get_resnet101():
    model = models.resnet101(pretrained=False)
    model.fc = nn.Linear(2048, 2)
    return model


def get_model():
    if args.model == 'resnet18':
        model = get_resnet18()
    elif args.model == 'resnet34':
        model = get_resnet34()
    elif args.model == 'resnet50':
        model = get_resnet50()
    elif args.model == 'resnet101':
        model = get_resnet101()
    else:
        model = timm.create_model(
            args.model, pretrained=False, num_classes=2, in_chans=3)  # 输入通道数，微调的类别数
    return model


def main():

    best_f1, best_epoch = 0.92, 0
    if not os.path.exists('submission'):
        os.mkdir('submission')
    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')
    print('Loading data...')
    train_loader, val_loader = create_data_lists(data_dir, split=True)
    print('Loaded data!')

    print(f'Loading Model {args.model}...')
    from model import doublemodel
    model = doublemodel(get_model(), get_model())

    # criterion = FocalLoss(alpha=0.01, size_average=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00003)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=len(train_loader) / 10, gamma=0.95)

    # 在gpu条件下保存的模型参数 要先放到gpu上，不然会报错two device：cuda0 and cpu

    model = model.to(device)
    if args.resume:
        load_checkpoint(os.path.join(
            'checkpoints', args.checkpoint_path), model, optimizer)

    # model = nn.DataParallel(model)
    print(f'Loaded Model!')

    for epoch in range(args.max_epoch):
        f1_score = train(epoch, train_loader, model, optimizer, criterion)
        # help save checkpoint
        if epoch >= 1 and (epoch % args.test_freq == 0 or f1_score > best_f1):
            val(epoch, val_loader, model, criterion, optimizer)
            best_f1 = f1_score
            best_epoch = epoch
        if epoch - best_epoch > 5:  # early stopping
            print(f'Epoch {epoch}: train early stopping')
            break
        scheduler.step()


def train(epoch, train_loader, model, optimizer, criterion):
    model.train()
    total_loss = AverageMeter()
    acc = AverageMeter()
    precision = AverageMeter()
    recall = AverageMeter()
    f1 = AverageMeter()
    for i, (data, label) in enumerate(train_loader, 1):
        img1, imgfft, label = data[0].to(device).float(), data[1].to(
            device).float(), label.long().to(device)
        # embed()
        # print(data)
        # data = data / 1000
        output = model(img1, imgfft)
        # print(output, label)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        print(loss)
        total_loss.update(loss.item())
        acc.update(torch.sum(output.argmax(dim=1) == label), img1.size(0))
        # y_true，y_pred tensor to numpy array
        # can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
        # Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.
        label, output = label.detach().cpu().numpy(
        ), output.detach().cpu().numpy()  # output shape 32, 2
        predict = output.argmax(axis=1)
        # Classification metrics can't handle a mix of binary and continuous-multioutput targets
        precision.update(precision_score(label, predict))
        recall.update(recall_score(label, predict))
        f1.update(f1_score(label, predict))

        if i % 1 == 0:
            print('Epoch [{0:02d}/{1:d}]\tStep [{2:02d}/{3:d}] \tTrain Loss:{4:.4f} Train acc:{5:.4f} precision:{6:.4f} recall:{7:.4f} f1_score:{8:.4f}'
                  .format(epoch, args.max_epoch, i+1, len(train_loader), total_loss.avg, acc.avg, precision.avg, recall.avg, f1.avg))
            logger.info('Epoch [{0:02d}/{1:d}]\tStep [{2:02d}/{3:d}] \tTrain Loss:{4:.4f} Train acc:{5:.4f} precision:{6:.4f} recall:{7:.4f} f1_score:{8:.4f}'
                        .format(epoch, args.max_epoch, i+1, len(train_loader), total_loss.avg, acc.avg, precision.avg, recall.avg, f1.avg))
    return f1.avg


def val(epoch, val_loader, model, criterion, optimizer):
    model.eval()
    total_loss = AverageMeter()
    acc = AverageMeter()
    precision = AverageMeter()
    recall = AverageMeter()
    f1 = AverageMeter()
    with torch.no_grad():
        for i, (data, label) in enumerate(val_loader, 1):
            img1, imgfft, label = data[0].to(device).float(), data[1].to(
                device).float(), label.long().to(device)
            output = model(img1, imgfft)

            loss = criterion(output, label)

            total_loss.update(loss.item())
            acc.update(torch.sum(output.argmax(dim=1) == label), img1.size(0))

            label, output = label.detach().cpu().numpy(), output.detach().cpu().numpy()
            predict = output.argmax(axis=1)
            precision.update(precision_score(label, predict))
            recall.update(recall_score(label, predict))
            f1.update(f1_score(label, predict))

            if i % 30 == 0:
                print('Epoch [{0:02d}/{1:d}]\tStep [{2:02d}/{3:d}] \tVal Loss:{4:.4f} Val acc:{5:.4f} precision:{6:.4f} recall:{7:.4f} f1_score:{8:.4f}'
                      .format(epoch, args.max_epoch, i+1, len(val_loader),
                              total_loss.avg, acc.avg, precision.avg, recall.avg, f1.avg))
                logger.info('Epoch [{0:02d}/{1:d}]\tStep [{2:02d}/{3:d}] \tVal Loss:{4:.4f} Val acc:{5:.4f} precision:{6:.4f} recall:{7:.4f} f1_score:{8:.4f}'
                            .format(epoch, args.max_epoch, i+1, len(val_loader),
                                    total_loss.avg, acc.avg, precision.avg, recall.avg, f1.avg))
        save_checkpoint('checkpoints/{0}_{1}_loss_epoch{2}_f1_score{3}.pth.tar'.format(
            args.model, args.figsize,  epoch, f1.avg),
            model.module if type(model) == nn.DataParallel else model,
            optimizer)


def check_file_name(file_name):
    append_suffix = 0
    while(os.path.isfile(file_name)):
        arr = file_name.split(".")
        suffix = arr[-1]
        name = '.'.join(arr[0:-1])
        file_name = name + f'({append_suffix}).{suffix}'
        append_suffix += 1
    return file_name


def write_csv_data(csv_file_name, header, data):
    csv_file_name = check_file_name(csv_file_name)
    with open(csv_file_name, 'w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(header)
        f_csv.writerows(data)


def final_train():
    '''
    使用整个训练集进行训练
    '''
    best_f1, best_epoch = 0.95, 0
    print('Loading data...')
    train_loader = create_data_lists(data_dir)
    print('Loaded data!')

    print(f'Loading Model {args.model}...')
    model = get_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00003)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=len(train_loader) / 10, gamma=0.95)

    # 在gpu条件下保存的模型参数 要先放到gpu上，不然会报错two device：cuda0 and cpu
    model = model.to(device)
    if args.resume:
        load_checkpoint(os.path.join(
            'checkpoints', args.checkpoint_path), model, optimizer)

    model = nn.DataParallel(model)
    print(f'Loaded Model!')

    for epoch in range(args.max_epoch):
        f1_score = train(epoch, train_loader, model, optimizer, criterion)
        # help save checkpoint
        if f1_score > best_f1:
            save_checkpoint(f'checkpoints/{args.model}_{args.figsize}_f1score{f1_score}.pth.tar',
                            model.module if type(model) == nn.DataParallel else model, optimizer)
        if epoch - best_epoch > 5:  # early stopping
            print(f'Epoch {epoch}: train early stopping')
            break
        scheduler.step()


def predict(best_checkpoint_path):
    '''
    模型加载checkpoint的参数，输出预测
    使用测试集val进行测试，并返回submission.csv
    '''
    # model = get_model().to(device)
    from model import doublemodel
    model = doublemodel(get_model(), get_model())
    load_checkpoint(best_checkpoint_path, model)
    model = nn.DataParallel(model)
    test_image_path = os.path.join(data_dir, image_dir, 'test')
    fnames = os.listdir(test_image_path)
    test_transform = T.Compose([T.ToTensor(), T.Resize((args.figsize, args.figsize)), T.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    test_dataset = FaceDataset(
        test_image_path, fnames, labels=None, transform=test_transform)
    test_loader = DataLoader(
        test_dataset, batch_size=16, shuffle=False, pin_memory=use_gpu)

    model.eval()
    print('start prediction!')
    header = ['fnames\tlabel']
    res = []
    for _, (data, fname) in enumerate(test_loader):
        # img1, imgfft = data[0].to(device).float(), data[1].to(
        #     device).float()
        # output = model(img1, imgfft)
        data = data.to(device)
        output = model(data)

        predict = output.argmax(dim=1)
        # embed()
        predict = predict.detach().cpu().numpy()
        fname = np.array(fname)
        # 生成两列dataframe
        for i in range(len(fname)):
            cur = []
            cur.append(f"{fname[i]}\t{predict[i]}")
            res.append(cur)
        # arr, value, axis
        # break
    # 一列 fnames predict_label

    res.sort(key=lambda x: int(x[0].split('.')[0].split('_')[1]))

    print('finish prediction!')
    write_csv_data(submission_name, header, res)


def emsemble():
    '''
    利用几个预测f1_score高的csv文件生成最终的csv
    '''
    sub_files = ['submission/submission3.csv',
                 'submission/submission1.csv',
                 'submission/submission2.csv']
    n = len(sub_files)
    dfs = None
    for file in sub_files:
        df = pd.read_csv(file)
        if dfs is None:
            dfs = pd.DataFrame(df['fnames\tlabel'].str.split(
                '\t').tolist(), columns=['fnames', 'label'])  # 1.jpg 0
            # 分成两列
            dfs['label'] = dfs['label'].astype(int)
        else:
            item = pd.DataFrame(df['fnames\tlabel'].str.split(
                '\t').tolist(), columns=['fnames', 'label'])
            dfs['label'] += item['label'].astype(int)

    true_index = dfs['label'] > n/2
    dfs['label'][true_index] = 1  # 将true_index的项变成1

    result = dfs['fnames']+'\t'+dfs['label'].map(str)
    # print(result)
    return pd.DataFrame(result, columns=['fnames\tlabel'])


if __name__ == '__main__':
    # main()
    # 需要再次训练
    # final_train()
    result = predict(
        'checkpoints/resnet50_224_loss_epoch30_f1_score0.9623711857457989.pth.tar')
    
    
    # result.to_csv(submission_name, index=None)
    # result = emsemble()
    # result.to_csv('submission/submission.csv', index=None)
