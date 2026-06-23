import numpy as np
import matplotlib.pyplot as plt
from data.data_preprocess import band_norm
from tools import hyperVca, sunsal
import pysptools as pysptools
from pysptools import abundance_maps
import configs.configs as cfg
import os
import scipy.io as sio

current_dataset = cfg.current_dataset
base_path = os.path.dirname(__file__)  # 获取当前文件的目录
save_mat_folder = os.path.join(base_path, '../datasets/', current_dataset)


####################### VCA，分别解混  #######################
def get_vca_abu(HSI1, HSI2):
    if os.path.exists(save_mat_folder + '/' + 'cat_vca_abu.mat'):
        cat_vca_abu = sio.loadmat(save_mat_folder + '/' + 'cat_vca_abu.mat')['cat_vca_abu']
        print('Exist cat_vca_abu.mat')

    elif os.path.exists(save_mat_folder + '/' + 'cat_vca_abu.mat') == False:
        print('Start Unmixing......')
        HSI1 = band_norm(HSI1)
        HSI2 = band_norm(HSI2)
        hsi_height, hsi_width, hsi_bands = HSI1.shape
        # 按左右并列拼接
        HSI_combined = np.concatenate((HSI1, HSI2), axis=1)
        HSI_reshape1 = HSI1.reshape(hsi_height * hsi_width, hsi_bands)
        HSI_reshape2 = HSI2.reshape(hsi_height * hsi_width, hsi_bands)

        ########## vca_fcls################
        q = cfg.endmember_num
        U1,_,_ = hyperVca.hyperVca(HSI1.reshape(-1, hsi_bands).T, q)
        U2,_,_ = hyperVca.hyperVca(HSI2.reshape(-1, hsi_bands).T, q)
        if hsi_height * hsi_width > 300*300 or q >5:
        # sunsal
            abundance1,res_p1,res_d1,i1 = sunsal.sunsal(U1, HSI_reshape1.T, positivity = True, addone = True)
            abundance2,res_p2,res_d2,i2 = sunsal.sunsal(U2, HSI_reshape2.T, positivity = True, addone = True)
            abundance1 = abundance1.reshape(q, hsi_height, hsi_width).transpose(1, 2, 0)
            abundance2 = abundance2.reshape(q, hsi_height, hsi_width).transpose(1, 2, 0)
        else:
            # pysptools
            abundance1 = pysptools.abundance_maps.amaps.FCLS(HSI_reshape1, U1.T).reshape(hsi_height, hsi_width, q)
            abundance2 = pysptools.abundance_maps.amaps.FCLS(HSI_reshape2, U2.T).reshape(hsi_height, hsi_width, q)

        fig, axs = plt.subplots(1, q, figsize=(2*q, 5))  # 创建一个1行q列的子图
        fig.suptitle('Abundance1')
        for i in range(q):
            axs[i].imshow(abundance1[:, :, i], cmap='gray')
            axs[i].set_title('VCA_Band %d' % i)
            axs[i].axis('off')  # 关闭坐标轴
        plt.savefig(save_mat_folder + '/' + 'VCA_ABU1.png')
        plt.close()

        fig, axs = plt.subplots(1, q, figsize=(2*q, 5))  # 创建一个1行q列的子图
        fig.suptitle('Abundance2')
        for i in range(q):
            axs[i].imshow(abundance2[:, :, i], cmap='gray')
            axs[i].set_title('VCA_Band %d' % i)
            axs[i].axis('off')  # 关闭坐标轴
        plt.savefig(save_mat_folder + '/' + 'VCA_ABU2.png')
        plt.close()

        # 按照端元U和丰度abundance重构影像
        reconstructed1 = np.dot(U1, abundance1.reshape(-1,q).T).T.reshape(hsi_height, hsi_width, hsi_bands)
        reconstructed2 = np.dot(U2, abundance2.reshape(-1,q).T).T.reshape(hsi_height, hsi_width, hsi_bands)
        difference1 = HSI1 - reconstructed1
        difference2 = HSI2 - reconstructed2
        # 计算每个像素的均方根误差
        rmse1 = np.sqrt(np.mean(np.square(difference1)))
        rmse2 = np.sqrt(np.mean(np.square(difference2)))
        print('RMSE1:{:.4f}, RMSE2:{:.4f}'.format(rmse1, rmse2))

        # cat丰度图
        cat_vca_abu = np.concatenate((abundance1, abundance2), axis=2)
        sio.savemat(save_mat_folder + '/' + 'cat_vca_abu.mat', {'cat_vca_abu': cat_vca_abu})

    return cat_vca_abu

####################### VCA，联合解混  #######################
def get_vca_combabu(HSI1, HSI2):
    if os.path.exists(save_mat_folder + '/' + 'cat_vca_combabu.mat'):
        cat_vca_combabu = sio.loadmat(save_mat_folder + '/' + 'cat_vca_combabu.mat')['cat_vca_combabu']
        print('Exist cat_vca_combabu.mat')

    elif os.path.exists(save_mat_folder + '/' + 'cat_vca_combabu.mat') == False:
        print('Start Unmixing......')
        HSI1 = band_norm(HSI1)
        HSI2 = band_norm(HSI2)
        hsi_height, hsi_width, hsi_bands = HSI1.shape
        # 按左右并列拼接
        HSI_combined = np.concatenate((HSI1, HSI2), axis=1)
        HSI_combined_reshape = HSI_combined.reshape(-1, hsi_bands)

        ########## vca_fcls################
        q = cfg.endmember_num
        U,_,_ = hyperVca.hyperVca(HSI_combined_reshape.T, q)
        if hsi_height * hsi_width > 300*300 or q >5:
        # sunsal
            abundance,res_p1,res_d1,i1 = sunsal.sunsal(U, HSI_combined_reshape.T, positivity = True, addone = True)
            abundance = abundance.reshape(q, hsi_height, 2* hsi_width).transpose(1, 2, 0)
        else:
        # pysptools
            abundance = pysptools.abundance_maps.amaps.FCLS(HSI_combined_reshape, U.T).reshape(hsi_height, 2* hsi_width, q)

        fig, axs = plt.subplots(1, q, figsize=(2*q, 5))  # 创建一个1行q列的子图
        fig.suptitle('Combined Abundance')
        for i in range(q):
            axs[i].imshow(abundance[:, :, i], cmap='gray')
            axs[i].set_title('Combined Band %d' % i)
            axs[i].axis('off')  # 关闭坐标轴
        plt.savefig(save_mat_folder + '/' + 'Comb_ABU.png')
        plt.close()

        for i in range(q):
            plt.figure(figsize=(5, 5))
            plt.imshow(abundance[:, :, i], cmap='gray')
            plt.title('Combined Band %d' % i)
            plt.axis('off')  # 关闭坐标轴
            plt.savefig(save_mat_folder + '/' + 'Combined_Band%d.png' % i)
            plt.close()
        plt.close()

        # 按照端元U和丰度abundance重构影像
        reconstructed = np.dot(U, abundance.reshape(-1,q).T).T.reshape(hsi_height, 2* hsi_width, hsi_bands)
        difference = HSI_combined - reconstructed
        # 计算每个像素的均方根误差
        comb_rmse = np.sqrt(np.mean(np.square(difference)))
        print('Comb RMSE:{:.4f}'.format(comb_rmse))

        # cat丰度图
        abundance1 = abundance[:, :hsi_width, :]
        abundance2 = abundance[:, hsi_width:, :]
        cat_vca_combabu = np.concatenate((abundance1, abundance2), axis=2)
        sio.savemat(save_mat_folder + '/' + 'cat_vca_combabu.mat', {'cat_vca_combabu': cat_vca_combabu})

    return cat_vca_combabu