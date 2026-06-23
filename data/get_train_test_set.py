import torch
import data.data_preprocess as data_preprocess
from data.get_dataset import get_dataset as getdata
import configs.configs as allcfg

# cfg.data
def get_train_test_set(cfg):
    # load
    current_dataset = cfg['current_dataset']
    train_ratio = cfg['train_set_num']
    val_ratio = cfg['val_ratio']
    patch_size = cfg['patch_size']

    img1, img2, gt, abu = getdata(current_dataset)

    img1 = torch.from_numpy(img1)  # numpy to tensor
    img2 = torch.from_numpy(img2)
    gt = torch.from_numpy(gt)
    abu = torch.from_numpy(abu)

    img1 = img1.permute(2, 0, 1)   # channel放到第一维CxHxW
    img2 = img2.permute(2, 0, 1)
    abu = abu.permute(2, 0, 1)
    # label transform
    img_gt = gt

    # std_norm
    if allcfg.std:
        img1 = data_preprocess.std_norm(img1)
        img2 = data_preprocess.std_norm(img2)

    # construct patch samples
    img1_pad, img2_pad, abu_pad, patch_coordinates = data_preprocess.construct_sample(img1, img2, abu, patch_size)

    # Divide samples
    data_sample = data_preprocess.select_sample(img_gt, train_ratio, val_ratio)

    data_sample['img1_pad'] = img1_pad
    data_sample['img2_pad'] = img2_pad
    data_sample['abu_pad'] = abu_pad

    data_sample['patch_coordinates'] = patch_coordinates
    data_sample['img_gt'] = img_gt  #
    data_sample['ori_gt'] = gt

    return data_sample

# data_sample = {
#     'train_sample_center': train_sample_center,
#     'val_sample_center': val_sample_center,
#     'test_sample_center': test_sample_center,
#     'train_sample_num': train_sample_num,
#     'val_sample_num': val_sample_num,
#     'test_sample_num': test_sample_num,
#     'img1_pad': img1_pad,
#     'img2_pad': img2_pad,
#     'abu_pad': abu_pad,
#     'patch_coordinates': patch_coordinates,
#     'img_gt': img_gt,
#     'ori_gt': gt}
