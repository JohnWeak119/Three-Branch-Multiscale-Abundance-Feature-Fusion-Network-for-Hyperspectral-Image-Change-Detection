import numpy as np
import matplotlib.pyplot as plt
from data.data_preprocess import band_norm
from tools import hyperVca, SLIC, sunsal
import configs.configs as cfg
import os
import scipy.io as sio

current_dataset = cfg.current_dataset
current_model = cfg.current_model
superpixel_scale = cfg.superpixel_scale
endmember_num = cfg.endmember_num
base_path = os.path.dirname(__file__)  # 获取当前文件的目录
save_mat_folder = os.path.join(base_path, '../datasets/', current_dataset)

def get_slic_abu(HSI1, HSI2):
    if os.path.exists(save_mat_folder + '/' + 'cat_slic_abu.mat'):
        cat_slic_abu = sio.loadmat(save_mat_folder + '/' + 'cat_slic_abu.mat')['cat_slic_abu']
        print('Exist cat_slic_abu.mat')

    elif os.path.exists(save_mat_folder + '/' + 'cat_slic_abu.mat') == False:
        print('Start Unmixing......')
        HSI1 = band_norm(HSI1)
        HSI2 = band_norm(HSI2)
        hsi_height, hsi_width, hsi_bands = HSI1.shape
        # 按左右并列拼接
        HSI_combined = np.concatenate((HSI1, HSI2), axis=1)
        hsi_height1, hsi_width1, hsi_bands1 = HSI_combined.shape

        ########## avg_slic_vca_fcls################
        avg_SLIC1= SLIC.apply_SLIC(HSI1, superpixel_scale)
        avg_SLIC2= SLIC.apply_SLIC(HSI2, superpixel_scale)
        avg_HSI_combined = np.concatenate((avg_SLIC1, avg_SLIC2), axis=1)
        q = endmember_num   # 端元数
        HSI_reshape = HSI_combined.reshape(-1, hsi_bands)
        avg_HSI_reshape = avg_HSI_combined.reshape(-1, hsi_bands1)
        _, idx = np.unique(avg_HSI_reshape, axis=0, return_index=True)
        avg_HSI_reshape1 = avg_HSI_reshape[np.sort(idx)]

        # VCA
        U,_,_ = hyperVca.hyperVca(avg_HSI_reshape1.T, q)

        # unmix
        HSI_reshape = HSI_combined.reshape(-1, hsi_bands).T
        abundance,res_p,res_d,i = sunsal.sunsal(U, HSI_reshape, positivity = True, addone = True)
        abundance = abundance.reshape(q, hsi_height1, hsi_width1).transpose(1, 2, 0)
        # 按照端元U和丰度abundance重构影像
        reconstructed = np.dot(U, abundance.reshape(-1,q).T)
        reconstructed = reconstructed.T.reshape(hsi_height1, hsi_width1, hsi_bands1)
        difference = HSI_combined - reconstructed
        # 计算每个波段的均方根误差
        rmse = np.mean(np.sqrt(np.mean(np.square(difference), axis=(0, 1))))
        print('Scale: ', superpixel_scale, ' , avg_slic_RMSE: {:.4f}'.format(rmse))

        fig, axs = plt.subplots(1, q, figsize=(3*q, 5))  # 创建一个1行q列的子图
        fig.suptitle('AvgSLIC RMSE: %.4f' % rmse)
        for i in range(q):
            axs[i].imshow(abundance[:, :, i], cmap='gray')
            axs[i].set_title('AVGSLIC_Band %d' % i)
            axs[i].axis('off')  # 关闭坐标轴
        plt.savefig(save_mat_folder + '/' + 'AVGSLIC_ABU.png')
        plt.close()

        for i in range(q):
            plt.figure(figsize=(5, 5))
            plt.imshow(abundance[:, :, i], cmap='gray')
            plt.title('SLIC_Band %d' % i)
            plt.axis('off')  # 关闭坐标轴
            plt.savefig(save_mat_folder + '/' + 'SLIC_Band%d.png' % i)
            plt.close()
        plt.close()

        # 拆分cat丰度图
        abu1 = abundance[:, :hsi_width, :]
        abu2 = abundance[:, hsi_width:, :]
        cat_slic_abu = np.concatenate((abu1, abu2), axis=2)
        sio.savemat(save_mat_folder + '/' + 'cat_slic_abu.mat', {'cat_slic_abu': cat_slic_abu})

    return cat_slic_abu