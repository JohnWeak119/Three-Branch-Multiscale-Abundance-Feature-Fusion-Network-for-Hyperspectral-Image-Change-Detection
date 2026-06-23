'''
 Dataset Source:
    Farmland: http://crabwq.github.io/
    River: https://share.weiyun.com/5ugrczK
    Hermiston: https://citius.usc.es/investigacion/datasets/hyperspectral-change-detection-dataset
'''

from scipy.io import loadmat
import numpy as np
np.float = float
from tools.construct_JM import construct_JM
import configs.configs as cfg
from tools.slic_unmixing import get_slic_abu
import os
from sklearn.decomposition import PCA
if cfg.combine_abu:
    from tools.vca_unmixing import get_vca_combabu as get_vca_abu
else:
    from tools.vca_unmixing import get_vca_abu as get_vca_abu


def band_norm(img): # each band 归一化到[0, 1]
    bands = img.shape[2]
    for i in range(bands):
        img[:, :, i] = (img[:, :, i] - np.min(img[:, :, i])) / (np.max(img[:, :, i]) - np.min(img[:, :, i]))
    return img

def apply_PCA(data, num_components=10):
    new_data = np.reshape(data, (-1, data.shape[2]))
    pca = PCA(n_components=num_components, whiten=True, svd_solver='full')
    new_data = pca.fit_transform(new_data)
    new_data = np.reshape(
        new_data, (data.shape[0], data.shape[1], -1))
    return new_data

base_path = os.path.dirname(__file__)  # 获取当前文件的目录

def get_Hermiston_dataset():
    img1 = loadmat(os.path.join(base_path, '../datasets/Hermiston/hermiston1.mat'))['hermiston1'].astype('float32')
    img2 = loadmat(os.path.join(base_path, '../datasets/Hermiston/hermiston2.mat'))['hermiston2'].astype('float32')
    gt = loadmat(os.path.join(base_path, '../datasets/Hermiston/label.mat'))['label'].astype('float32')
    if cfg.superpixel_scale == 0:
        abu = get_vca_abu(img1, img2)
    else:
        abu = get_slic_abu(img1, img2)
    return img1, img2, gt, abu

def get_MultiFarmland_dataset():
    img1 = loadmat(os.path.join(base_path, '../datasets/MultiFarmland/MultiFarmland1.mat'))['MultiFarmland1'].astype('float32')
    img2 = loadmat(os.path.join(base_path, '../datasets/MultiFarmland/MultiFarmland2.mat'))['MultiFarmland2'].astype('float32')
    gt = loadmat(os.path.join(base_path, '../datasets/MultiFarmland/label.mat'))['label'].astype('float32')
    if cfg.superpixel_scale == 0:
        abu = get_vca_abu(img1, img2)
    else:
        abu = get_slic_abu(img1, img2)
    return img1, img2, gt, abu

def get_MultiHermiston_dataset():
    img1 = loadmat(os.path.join(base_path, '../datasets/MultiHermiston/MultiHermiston1.mat'))['MultiHermiston1'].astype('float32')
    img2 = loadmat(os.path.join(base_path, '../datasets/MultiHermiston/MultiHermiston2.mat'))['MultiHermiston2'].astype('float32')
    gt = loadmat(os.path.join(base_path, '../datasets/MultiHermiston/label.mat'))['label'].astype('float32')
    if cfg.superpixel_scale == 0:
        abu = get_vca_abu(img1, img2)
    else:
        abu = get_slic_abu(img1, img2)
    return img1, img2, gt, abu



def get_dataset(current_dataset):
    get_dataset_dict = {
        'Hermiston': get_Hermiston_dataset,  # Hermiston(390, 200, 159), gt[0 5]
        'MultiFarmland': get_MultiFarmland_dataset,  # MultiFarmland(430, 220, 132), gt[0 5]
        'MultiHermiston': get_MultiHermiston_dataset,  # MultiHermiston(225, 180, 159), gt[0 6]
    }
    if current_dataset in get_dataset_dict:
        return get_dataset_dict[current_dataset]()
    else:
        raise ValueError(f"Unknown dataset: {current_dataset}")