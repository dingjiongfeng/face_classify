import torch
from torch.utils.data import DataLoader, Dataset
import cv2
import os.path as osp
import numpy as np
import torchvision.transforms.functional as FT
# train.py 文件要加. 表示同级目录下的utils模块 但是想运行datasets.py 不要.

'''
数据集分为训练集和测试集，训练集包括标签文件和图像文件，
其中标签文件中包含图像文件名和真伪标签信息；测试集包含一批图像文件，参赛选手需要对图像文件进行真伪鉴别。
'''

data_dir = '../data/face_data'


class FaceDataset(Dataset):
    def __init__(self, data_folder, fnames, labels=None, transform=None, sift_transform=None) -> None:
        super(FaceDataset, self).__init__()
        self.filenames = fnames
        self.fname = [osp.join(data_folder, item) for item in fnames]
        self.labels = labels
        self.transform = transform
        self.sift_transform = sift_transform

    def __getitem__(self, index):
        path = self.fname[index]
        image = cv2.imread(path)  # 1024*1024*3
        # image = Image.open(path)  # PIL对象 3*1024*1024
        # image.convert('RGB')
        # 获取特征组成新的tensor
        # print(image)
        # print("#" * 10)

        # f = np.fft.fft2(image)
        # fshift = np.fft.fftshift(f)
        # magnitude_spectrum = np.abs(fshift)
        # image = magnitude_spectrum
        # print(image)

        if self.transform:
            # 添加图形增强 TypeError: pic should be PIL Image or ndarray. Got <class 'NoneType'>
            # print("#" * 10)
            image = self.transform(image)
            # magnitude_spectrum = self.transform(magnitude_spectrum)
            # print(image)

        if self.labels is not None:
            label = self.labels[index]
            label = torch.from_numpy(
                np.array(label, dtype=np.uint8))  # 一维的 是train，val
        else:  # test dataset
            label = self.filenames[index]
        return image, label

    def __len__(self):
        return len(self.fname)

    # def collate_fn(self, batch):
    #     images = list()
    #     labels = list()
    #     for b in batch:
    #         images.append(b[0])
    #         labels.append(b[2])

    #     images = torch.stack(images, dim=0)
    #     return images, labels
        #tensor(b, h, w), list(b, n_objects, 4), list(b, n_objects), list(b, n_objects)


# dataset = FaceDataset(data_dir)
# dataloader = DataLoader(dataset=dataset, batch_size=32)
# print(next(iter(dataloader)))
# print(len(dataloader))
