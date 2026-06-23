import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import math


######################## norm ###########################
def std_norm(image):  # input tensor image size with CxHxW
    mean = image.mean([1, 2], keepdim=True)
    std = image.std([1, 2], keepdim=True)
    std_result = (image - mean) / (std + 1e-5)  # 添加一个小的值避免除以0
    # (x - mean(x))/std(x) normalize each channel to mean: 0, std: 1
    return std_result

def band_norm(img): # img为numpy整张图象each band 归一化到[0, 1]
    bands = img.shape[2]
    for i in range(bands):
        img[:, :, i] = (img[:, :, i] - np.min(img[:, :, i])) / (np.max(img[:, :, i]) - np.min(img[:, :, i]))
    return img

def one_zero_norm(image):  # input tensor image size with CxHxW
    channel, height, width = image.shape
    data = image.reshape(channel, height * width)
    data_max = data.max(dim=1)[0]
    data_min = data.min(dim=1)[0]

    data = (data - data_min.unsqueeze(1))/(data_max.unsqueeze(1) - data_min.unsqueeze(1))
    # (x - min(x))/(max(x) - min(x))  normalize to (0, 1) for each channel

    return data.view(channel, height, width)

def pos_neg_norm(image):  # input tensor image size with CxHxW
    channel, height, width = image.shape
    data = image.reshape(channel, height*width)
    data_max = data.max(dim=1)[0]
    data_min = data.min(dim=1)[0]

    data = -1 + 2 * (data - data_min.unsqueeze(1))/(data_max.unsqueeze(1) - data_min.unsqueeze(1))
    # -1 + 2 * (x - min(x))/(max(x) - min(x))  normalize to (-1, 1) for each channel

    return data.view(channel, height, width)

######################## get patch ###########################
def construct_sample(img1, img2, abu, window_size=5):  # 默认patch_size=5
    _, height, width = img1.shape  # input float tensor image size with CxHxW
    half_window = int(window_size//2)

    # padding
    pad = nn.ReplicationPad2d(half_window)
    pad_img1 = pad(img1.unsqueeze(0)).squeeze(0)  # 每个波段每边补4个0像素
    pad_img2 = pad(img2.unsqueeze(0)).squeeze(0)
    pad_abu = pad(abu.unsqueeze(0)).squeeze(0)

    # get coordinates，记录每个像素的左上(h,w)和右下(h+s,w+s)坐标
    patch_coordinates = torch.zeros((height*width, 4), dtype=torch.long)
    t = 0
    for h in range(height):
        for w in range(width):
            patch_coordinates[t, :] = torch.tensor([h, h + window_size, w, w + window_size])
            t += 1

    return pad_img1, pad_img2, pad_abu, patch_coordinates


def select_sample(gt, train_ratio, val_ratio):  # input tensor data with NxCxHxW, tensor gt with HxW，default train_ratio=0.2
    gt_vector = gt.reshape(-1, 1).squeeze(1)  # 获得gt一维向量
    label = torch.unique(gt)  # label = tensor([0, 1,2,3,...,n])

    first_time = True

    for each in range(len(label)):  # each表示label中的每个类别
        indices_vector = torch.where(gt_vector == label[each])  # 一维向量中等于label[each]的索引
        indices = torch.where(gt == label[each])  # 二维矩阵中等于label[each]的索引

        indices_vector = indices_vector[0]
        indices_row = indices[0]
        indices_column = indices[1]

        class_num = torch.tensor(len(indices_vector))

        # get select_train_num
        if train_ratio < 1:
            train_num = math.ceil(train_ratio * class_num)  # 确保 train_num 至少为 1，train[each]的数量
            val_num = math.ceil(val_ratio*class_num)  # val

        else:
            train_num = train_ratio
            val_num = val_ratio

        # 训练样本大于一半时，取一半
        if train_num > class_num//2:
            select_train_num = class_num//2
            select_val_num = class_num//4

        else:
            select_train_num = train_num
            select_val_num = val_num
        
        print('class', each, ', class_num', class_num, 'train_num', select_train_num, 'val_num', select_val_num)

        select_train_num = torch.tensor(select_train_num)
        select_val_num = torch.tensor(select_val_num)

        # disorganize
        rand_indices0 = torch.randperm(class_num)  # 随机排列gt
        rand_indices = indices_vector[rand_indices0]  # 随机排列后的索引

        # Divide train and test，除了train[each]，添加了val[each],其余都是test[each]
        tr_ind0 = rand_indices0[0:select_train_num]  # gt
        vl_ind0 = rand_indices0[select_train_num:select_train_num+select_val_num]
        te_ind0 = rand_indices0[select_train_num+select_val_num:]
        tr_ind = rand_indices[0:select_train_num]  # 索引
        vl_ind = rand_indices[select_train_num:select_train_num+select_val_num]
        te_ind = rand_indices[select_train_num+select_val_num:]
        # index+Sample center coordinate
        select_tr_ind = torch.cat([tr_ind.unsqueeze(1),
                                indices_row[tr_ind0].unsqueeze(1),
                                indices_column[tr_ind0].unsqueeze(1)],
                                dim=1
                                )
        select_vl_ind = torch.cat([vl_ind.unsqueeze(1),
                                indices_row[vl_ind0].unsqueeze(1),
                                indices_column[vl_ind0].unsqueeze(1)],
                                dim=1
                                ) 
        select_te_ind = torch.cat([te_ind.unsqueeze(1),
                                indices_row[te_ind0].unsqueeze(1),
                                indices_column[te_ind0].unsqueeze(1)],
                                dim=1
                                )

        if first_time:
            first_time = False

            train_sample_center = select_tr_ind
            train_sample_num = select_train_num.unsqueeze(0)

            val_sample_center = select_vl_ind
            val_sample_num = select_val_num.unsqueeze(0)

            test_sample_center = select_te_ind
            test_sample_num = (class_num - select_train_num - select_val_num).unsqueeze(0)

        else:
            train_sample_center = torch.cat([train_sample_center, select_tr_ind], dim=0)
            train_sample_num = torch.cat([train_sample_num, select_train_num.unsqueeze(0)])

            val_sample_center = torch.cat([val_sample_center, select_vl_ind], dim=0)
            val_sample_num = torch.cat([val_sample_num, select_val_num.unsqueeze(0)])

            test_sample_center = torch.cat([test_sample_center, select_te_ind], dim=0)
            test_sample_num = torch.cat([test_sample_num, (class_num - select_train_num - select_val_num).unsqueeze(0)])


    rand_tr_ind = torch.randperm(train_sample_num.sum())    # 随机排列train_sample_num.sum()个数的索引
    train_sample_center = train_sample_center[rand_tr_ind, ]   # torch.Size([22316, 3])   22316 = 不变20377+ 变化1939
    rand_vl_ind = torch.randperm(val_sample_num.sum())
    val_sample_center = val_sample_center[rand_vl_ind, ]   # torch.Size([11158, 3])
    rand_te_ind = torch.randperm(test_sample_num.sum())
    test_sample_center = test_sample_center[rand_te_ind, ]   # torch.Size([89267, 3])

    data_sample = {'train_sample_center': train_sample_center, 'train_sample_num': train_sample_num,
                   'val_sample_center': val_sample_center, 'val_sample_num': val_sample_num,
                   'test_sample_center': test_sample_center, 'test_sample_num': test_sample_num,
                   }

    return data_sample