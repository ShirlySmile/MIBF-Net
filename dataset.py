import os, cv2, torch
from skimage import io
import numpy as np
import random
from scipy.io import loadmat
from torch.utils.data import Dataset



class MyData(Dataset):
    def __init__(self, data1, data2, xy, cut_size, ratio, labels=None, transform=None):
        self.train_data1 = data1
        self.train_data2 = data2

        self.train_labels = labels
        self.gt_xy = xy
        self.size1 = cut_size
        self.size2 = cut_size * ratio
        self.ratio = ratio
        self.transform = transform
        self.augmentation = True

    def __getitem__(self, index):
        x_1, y_1 = self.gt_xy[index]
        x_2, y_2 = int(self.ratio * x_1), int(self.ratio * y_1)

        if self.size1 == 1:
            image_1 = self.train_data1[:, x_1:int(x_1 + self.size1),
                      y_1:int(y_1 + self.size1)]
        else:
            image_1 = self.train_data1[:, x_1:int(x_1 + self.size1),
                       y_1:int(y_1 + self.size1)]

        if self.size2 == 1:
            image_2 = self.train_data2[:, x_2:int(x_2 + self.size2),
                        y_2:int(y_2 +self.size2)]
        else:
            image_2 = self.train_data2[:, x_2:int(x_2 + self.size2),
                      y_2:int(y_2 + self.size2)]

        locate_xy = self.gt_xy[index]

        image_1 = torch.from_numpy(np.copy(image_1)).type(torch.FloatTensor)
        image_2 = torch.from_numpy(np.copy(image_2)).type(torch.FloatTensor)

        if self.train_labels is None:
            return image_1.cuda(), image_2.cuda(), locate_xy
        else:
            target = self.train_labels[index]
            return image_1.cuda(), image_2.cuda(), target.cuda(), locate_xy

    def __len__(self):
        return len(self.gt_xy)

def cumulativehistogram(array_data, counts, percent):
    gray_level,gray_num = np.unique(array_data,return_counts=True)
    count_percent1 = counts * percent
    count_percent2 = counts * (1 - percent)
    cutmax = 0
    cutmin = 0

    for i in range(1, len(gray_level)):
        gray_num[i] += gray_num[i - 1]
        if (gray_num[i] >= count_percent1 and gray_num[i - 1] <= count_percent1):
            cutmin = gray_level[i]
        if (gray_num[i] >= count_percent2 and gray_num[i - 1] <= count_percent2):
            cutmax = gray_level[i]
    return cutmin, cutmax


def preprocess(image, percent):
    h, w, c = image.shape[0], image.shape[1], image.shape[2]
    array_data = image[:, :, :]

    compress_data = np.zeros((h, w, c))
    for i in range(c):
        cutmin, cutmax = cumulativehistogram(array_data[:, :, i], h*w, percent)
        compress_scale = cutmax - cutmin
        if compress_scale == 0:
            print('error')

        temp = np.array(array_data[:, :, i])
        temp[temp > cutmax] = cutmax
        temp[temp < cutmin] = cutmin
        compress_data[:, :, i] = (temp - cutmin) / (cutmax - cutmin)
    return compress_data


def padimg(image, size):
    """Image edge padding"""
    Interpolation = cv2.BORDER_REFLECT_101
    top_size, bottom_size, left_size, right_size = (size, size,
                                                    size, size)
    image = cv2.copyMakeBorder(image, top_size, bottom_size, left_size, right_size, Interpolation)
    return image



def getdata(data_name, percent, size):
    print('-------------------Data loading-------------------')
    data_dir = './data/'+ data_name + '/'
    if data_name == 'MUUFL':
        data1 = loadmat(os.path.join(data_dir, 'data_HS_LR.mat'))['hsi_data']
        data2 = loadmat(os.path.join(data_dir, 'data_SAR_HR.mat'))['lidar_data']
        labels = loadmat(os.path.join(data_dir, 'gt.mat'))['labels']
        where_0 = np.where(labels == -1)
        labels[where_0] = 0
    elif data_name == '2018':
        data1 = io.imread(os.path.join(data_dir, 'HSI.tif'))
        data2 = io.imread(os.path.join(data_dir, 'LiDAR.tif'))
        labels = io.imread(os.path.join(data_dir, 'GT.tif'))
    elif data_name == '2013':
        data1 = loadmat(os.path.join(data_dir, 'HSI.mat'))['HSI']
        data2 = loadmat(os.path.join(data_dir, 'LiDAR.mat'))['LiDAR']
        labels = loadmat(os.path.join(data_dir, 'gt.mat'))['gt']



    if data1.dtype=='float32' or data1.dtype=='float64':
        data1 = data1 * 10000
    if data2.dtype=='float32' or data2.dtype=='float64':
        data2 = data2 * 100

    data1 = data1.astype(int)
    data2 = data2.astype(int)
    labels = labels.astype(np.uint8)

    print('Before Padding：', (np.shape(data1),np.shape(data2)))
    ratio = data2.shape[0] / data1.shape[0]
    data1 = padimg(data1, int(size/2))
    data2 = padimg(data2, int(size*ratio/2))
    print('After Padding：', (np.shape(data1),np.shape(data2)))

    if len(np.shape(data1)) < 3:
        data1 = np.expand_dims(data1, axis=2)
    if len(np.shape(data2)) < 3:
        data2 = np.expand_dims(data2, axis=2)


    data1 = preprocess(data1, percent)
    data2 = preprocess(data2, percent)

    data1 = np.array(data1).transpose((2, 0, 1))
    data2 = np.array(data2).transpose((2, 0, 1))

    return data1, data2, labels


def to_tensor(image):
    max_i = np.max(image)
    min_i = np.min(image)
    image = (image - min_i) / (max_i - min_i)
    return image



def splitdata(labels, Traindata_Rate,Traindata_Num, Split_MODE, SEED):
    label_list, labels_counts = np.unique(labels, return_counts=True)
    Cls_Number = len(label_list) - 1
    ground_xy = np.array([[]] * Cls_Number).tolist()

    if Split_MODE == 'Cls_Rate_Same':
        split_num_list = np.ceil(labels_counts * Traindata_Rate).astype(np.int)
    if Split_MODE == 'Cls_Num_Same':
        split_num_list = [Traindata_Num] * (Cls_Number + 1)

    print(split_num_list)
    index_train_data = []
    index_test_data = []
    label_train = []
    label_test = []
    for id in range(Cls_Number):
        label = id + 1
        ground_xy[id] = np.argwhere(labels == label)

        np.random.seed(SEED)
        np.random.shuffle(ground_xy[id])
        categories_number = labels_counts[label]
        split_num = split_num_list[label]

        index_train_data.extend(ground_xy[id][:split_num])
        index_test_data.extend(ground_xy[id][split_num:])
        label_train = label_train + [id for x in range(split_num)]
        label_test = label_test + [id for x in range(categories_number - split_num)]

    index_all_data = np.argwhere(labels != 255)
    np.random.seed(SEED)
    np.random.shuffle(index_all_data)

    label_train = np.array(label_train)
    label_test = np.array(label_test)
    index_train_data = np.array(index_train_data)
    index_test_data = np.array(index_test_data)

    shuffle_array = np.arange(0, len(label_test), 1)
    np.random.seed(SEED)
    np.random.shuffle(shuffle_array)
    label_test = label_test[shuffle_array]
    index_test_data = index_test_data[shuffle_array]

    shuffle_array = np.arange(0, len(label_train), 1)
    np.random.seed(SEED)
    np.random.shuffle(shuffle_array)
    label_train = label_train[shuffle_array]
    index_train_data = index_train_data[shuffle_array]

    label_train = torch.from_numpy(label_train).type(torch.LongTensor)
    label_test = torch.from_numpy(label_test).type(torch.LongTensor)
    index_train_data = torch.from_numpy(index_train_data).type(torch.LongTensor)
    index_test_data = torch.from_numpy(index_test_data).type(torch.LongTensor)
    index_all_data = torch.from_numpy(index_all_data).type(torch.LongTensor)

    return index_all_data, index_train_data, index_test_data, label_train, label_test, Cls_Number + 1