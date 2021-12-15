from PIL import Image
import cv2
import os.path as osp
import glob
import os
from scipy.sparse.construct import rand
import torch
import random
from torch.utils import data
import torchvision.transforms.functional as FT
import torchvision.transforms as T
import os
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from datasets import FaceDataset
from IPython import embed
from sklearn.metrics import precision_score
from sklearn .decomposition import PCA

'''
train.labels.csv
fnames	图像文件名
label	图像真伪标签（0表示真实人脸图像，1表示伪造人脸图像）
'''
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
data_dir = '../data/face_data'
train_csv_name = 'train.labels.csv'
image_dir = 'image'
batch_size = 32
num_workers = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_gpu = torch.cuda.is_available()
submission_name = 'submission/submission.csv'


'''
    参赛者需提交文件格式
    ImageId	测试集图像Id
    IsFake	图像真伪标签
'''


class ColorTransform(object):
    def __init__(self):
        super(ColorTransform, self).__init__()

    def __call__(self, img):
        distortions = [FT.adjust_brightness,
                       FT.adjust_contrast,
                       FT.adjust_saturation,
                       FT.adjust_hue]

        random.shuffle(distortions)
        for d in distortions:
            if random.random() < 0.2:
                if d.__name__ == 'adjust_hue':
                    factor = random.uniform(-18./255, 18./255)
                else:
                    factor = random.uniform(0.5, 1.5)
                img = d(img, factor)

        return img


def save_checkpoint(path, model, optimizer):
    '''
    保存两个个键值对，整个model，optimizer（lr）
    '''
    state_dict = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()}
    torch.save(state_dict, path)


def load_checkpoint(path, model, optimizer=None):
    '''
    将path路径下的文件加载到model和optimizer， epoch赋给start_epoch
    return start_epoch 开始的epoch
    '''
    if not os.path.exists(path):
        print('Sorry, don\'t have checkpoint.pth file, continue training!')
        return
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])
    print("#" * 20)
    print(model)
    print("#" * 20)
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


def embed_sift_features(image):
    '''
    param image: numpy array of size (1024, 1024, 3)
    return tensor with embedding sift features size (1024, 1024, 4)
    '''
    sift = cv2.xfeatures2d.SIFT_create()
    _, des = sift.detectAndCompute(image, None)
    des = np.transpose(des, axes=[1, 0])
    pca = PCA(n_components=128)
    des2 = pca.fit_transform(np.array(des))
    features = np.zeros((1024, 1024))
    features[0:128, 0:128] = des2
    return features

######################
# 对数据进行增强处理
######################


def convert_color_factory(src, dst):

    code = getattr(cv2, f'COLOR_{src.upper()}2{dst.upper()}')

    def convert_color(img):
        out_img = cv2.cvtColor(img, code)
        return out_img

    convert_color.__doc__ = f"""Convert a {src.upper()} image to {dst.upper()}
        image.

    Args:
        img (ndarray or str): The input image.

    Returns:
        ndarray: The converted {dst.upper()} image.
    """

    return convert_color


class PhotoMetricDistortion(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def convert(self, img, alpha=1, beta=0):
        """Multiple with alpha and add beat with clip."""
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img):
        """Brightness distortion."""
        if np.random.randint(2):
            return self.convert(
                img,
                beta=np.random.uniform(-self.brightness_delta,
                                       self.brightness_delta))
        return img

    def contrast(self, img):
        """Contrast distortion."""
        if np.random.randint(2):
            return self.convert(
                img,
                alpha=np.random.uniform(self.contrast_lower, self.contrast_upper))
        return img

    def saturation(self, img):
        """Saturation distortion."""
        if np.random.randint(2):
            img = convert_color_factory('bgr', 'hsv')(img)
            img[:, :, 1] = self.convert(
                img[:, :, 1],
                alpha=np.random.uniform(self.saturation_lower,
                                        self.saturation_upper))
            img = convert_color_factory('hsv', 'bgr')(img)
        return img

    def hue(self, img):
        """Hue distortion."""
        if np.random.randint(2):
            img = convert_color_factory('bgr', 'hsv')(img)
            img[:, :, 0] = (img[:, :, 0].astype(int) +
                            np.random.randint(-self.hue_delta, self.hue_delta)) % 180
            img = convert_color_factory('hsv', 'bgr')(img)
        return img

    def cal(self, img):
        """Call function to perform photometric distortion on images."""

        # random brightness
        img = self.brightness(img)

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = np.random.randint(2)
        if mode == 1:
            img = self.contrast(img)

        # random saturation
        img = self.saturation(img)

        # random hue
        img = self.hue(img)

        # random contrast
        if mode == 0:
            img = self.contrast(img)

        return img


def get_crop_bbox(img, crop_size=(768, 768)):
    """Randomly get a crop bounding box."""
    margin_h = max(img.shape[0] - crop_size[0], 0)
    margin_w = max(img.shape[1] - crop_size[1], 0)
    offset_h = np.random.randint(0, margin_h + 1)
    offset_w = np.random.randint(0, margin_w + 1)
    crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
    crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]
    return crop_y1, crop_y2, crop_x1, crop_x2


def crop(img, crop_bbox):
    """Crop from ``img``"""
    crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
    img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
    return img


def cityscapeDatasetTransform(image):
    # 1024 * 1024
    ########################################################################
    # randomly scale the img and the label:
    ########################################################################
    max_ratio = 2.0
    min_ratio = 0.5
    img_scale = (1025, 1025)
    ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
    scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)

    h, w = image.shape[:2]  # 1024, 1024
    max_long_edge = max(scale)
    max_short_edge = min(scale)
    scale_factor = min(max_long_edge / max(h, w),
                       max_short_edge / min(h, w))
    new_size = int(w * float(scale_factor) +
                   0.5), int(h * float(scale_factor) + 0.5)
    # cv2.INTER_LINEAR
    # cv2.INTER_NEAREST
    image = cv2.resize(
        image, new_size, interpolation=cv2.INTER_LINEAR)

    ########################################################################
    # select a 768x768 random crop from the img and label:
    ########################################################################

    crop_bbox = get_crop_bbox(image)
    image = crop(image, crop_bbox)  # (shape: (768, 768, 3))
    ########################################################################
    # flip
    ########################################################################

    flip = np.random.randint(low=0, high=2)
    if flip == 1:
        image = cv2.flip(image, 1)  # 水平翻转
    ########################################################################
    # PhotoMetricDistortion
    ########################################################################
    a = PhotoMetricDistortion()
    image = a.cal(image)
    ########################################################################
    # normalize the img (with the mean and std for the pretrained ResNet):
    # mean=[123.675, 116.28, 103.53]
    # std=[58.395, 57.12, 57.375]
    ########################################################################
    mean = [123.675, 116.28, 103.53]
    std = [58.395, 57.12, 57.375]
    to_rgb = True
    image = image.copy().astype(np.float32)
    assert image.dtype != np.uint8
    mean = np.array(mean, dtype=np.float64).reshape(1, -1)
    stdinv = 1 / np.array(std, dtype=np.float64).reshape(1, -1)
    if to_rgb:
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB, image)  # inplace
    cv2.subtract(image, mean, image)  # inplace
    cv2.multiply(image, stdinv, image)  # inplace

    ########################################################################
    # pad
    ########################################################################
    # print(image.shape)
    img_h, img_w = image.shape[:2]
    crop_size = (768, 768)
    crop_h, crop_w = crop_size
    pad_h = max(crop_h - img_h, 0)
    pad_w = max(crop_w - img_w, 0)
    if pad_h > 0 or pad_w > 0:
        image = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                   pad_w, cv2.BORDER_CONSTANT,
                                   value=0)
    ########################################################################
    # DefaultFormatBundle
    ########################################################################
    image = np.ascontiguousarray(np.transpose(
        image, (2, 0, 1)))  # (shape: (3, 1025, 2049))
    assert not np.any(np.isnan(image))
    # convert numpy -> torch:
    image = torch.from_numpy(image)  # (shape: (3, 1025, 2049))
    return image


if __name__ == '__main__':
    # train_loader, val_loader = create_data_lists(data_dir, split=True)
    # print(len(train_loader), len(val_loader))  # 2045, 288
    # print(next(iter(train_loader)))

    image = Image.open('svm_train/train_12.jpg')
    print(image.size)
    image = cv2.imread('svm_train/train_12.jpg')
    print(image.shape)
    cityscapeDatasetTransform(image)

    # y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
    #                   0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    # y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    # print(precision_score(y_true, y_pred))
