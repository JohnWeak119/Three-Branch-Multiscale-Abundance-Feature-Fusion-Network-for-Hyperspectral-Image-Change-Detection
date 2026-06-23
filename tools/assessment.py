import numpy as np
import torch
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
from skimage.metrics import structural_similarity as ssim
import configs.configs as cfg


def accuracy_assessment(img_gt, changed_map):
    '''
        assess accuracy of changed map based on ground truth
    '''
    esp = 1e-6

    org_img_gt = img_gt.cpu().numpy().astype(np.float32)
    org_changed_map = changed_map.cpu().numpy().astype(np.float32)
    height, width = changed_map.shape
    changed_map_ = np.reshape(changed_map, (-1,))
    img_gt_ = np.reshape(img_gt, (-1,))

    cm = np.ones((height * width,))
    cm[changed_map_ == 1] = 2
    cm[changed_map_ == 0] = 1
    gt = np.zeros((height * width,))
    gt[img_gt_ == 1] = 2
    gt[img_gt_ == 0] = 1

    # scikit-learn 混淆矩阵函数 sklearn.metrics.confusion_matrix API 接口
    conf_mat = confusion_matrix(y_true=gt, y_pred=cm, labels=[1, 2])
    kappa_co = cohen_kappa_score(y1=gt, y2=cm, labels=[1, 2])

    # TN, FP, FN, TP
    TN, FP, FN, TP = conf_mat.ravel()
    P = TP / (TP + FP + esp)
    R = TP / (TP + FN + esp)
    F1 = 2 * P * R / (P + R + esp)
    acc = (TP + TN) / (TP + TN + FP + FN + esp)

    CH_num = np.count_nonzero(img_gt == 1)
    UN_num = np.count_nonzero(img_gt == 0)
    oa = np.sum(conf_mat.diagonal()) / (np.sum(conf_mat) + esp)
    oa_CH = TP/(CH_num + esp)
    oa_UN = TN/(UN_num + esp)
    loU = TP/(TP + FP + FN + esp)
    ssim_value = ssim(org_changed_map, org_img_gt, data_range=1.0)
    return conf_mat, oa, oa_CH, oa_UN, kappa_co, P, R, F1, acc, loU, ssim_value

def multi_class_accuracy_assessment(img_gt, changed_map):
    '''
        assess accuracy of changed map based on ground truth
    '''
    esp = 1e-6
    changed_map_reshape = np.reshape(changed_map, (-1,))
    img_gt_reshape = np.reshape(img_gt, (-1,))
    classes_ev = {}
    f1_scores = []

    # 掩膜数据处理
    unique_classes = np.unique(img_gt_reshape)
    if cfg.current_dataset in cfg.mask0data:
        unique_classes = unique_classes[unique_classes != 0]
        mask = (img_gt_reshape != 0)
        img_gt_reshape = img_gt_reshape[mask]
        changed_map_reshape = changed_map_reshape[mask]
        
    for each in unique_classes:
        each = int(each)
        temp_matrix = np.ones(len(img_gt_reshape))
        temp_matrix[changed_map_reshape == each] = 2
        temp_matrix[changed_map_reshape != each] = 1

        gt = np.ones(len(img_gt_reshape))
        gt[img_gt_reshape == each] = 2
        gt[img_gt_reshape != each] = 1

        conf_mat = confusion_matrix(y_true=gt, y_pred=temp_matrix, labels=[1, 2])
        kappa = cohen_kappa_score(y1=gt, y2=temp_matrix, labels=[1, 2])

        TN, FP, FN, TP = conf_mat.ravel()
        P = TP / (TP + FP + esp)
        R = TP / (TP + FN + esp)
        F1 = 2 * P * R / (P + R + esp)
        f1_scores.append(F1)
        acc = (TP + TN) / (TP + TN + FP + FN + esp)

        CH_num = np.count_nonzero(img_gt_reshape == each)
        UN_num = np.count_nonzero(img_gt_reshape != each)
        oa = np.sum(conf_mat.diagonal()) / (np.sum(conf_mat) + esp)
        oa_CH = TP / (CH_num + esp)
        oa_UN = TN / (UN_num + esp)
        loU = TP / (TP + FP + FN + esp)
        cl_ssim = ssim(temp_matrix, gt, data_range=1.0)

        classes_ev[each] = "Class:{:}, OA: {:.2f}%, OA_CH: {:.2f}%, OA_UN: {:.2f}%, Kappa: {:.4f}, Pre: {:.4f}, F1: {:.4f}, loU: {:.4f}, ssim:{:.4f}".format(
                    each, oa * 100, oa_CH * 100, oa_UN * 100, kappa, P, F1, loU, cl_ssim)

    overall_conf_mat = confusion_matrix(y_true=img_gt_reshape, y_pred=changed_map_reshape, labels=unique_classes)
    overall_kappa = cohen_kappa_score(y1=img_gt_reshape, y2=changed_map_reshape, labels=unique_classes)
    overall_oa = np.sum(overall_conf_mat.diagonal()) / (np.sum(overall_conf_mat) + esp)
    overall_F1 = np.mean(f1_scores)
    # 确保 changed_map_reshape 和 img_gt_reshape 是 numpy 数组
    if isinstance(changed_map_reshape, torch.Tensor):
        changed_map_reshape = changed_map_reshape.cpu().numpy()
    if isinstance(img_gt_reshape, torch.Tensor):
        img_gt_reshape = img_gt_reshape.cpu().numpy()
    overall_ssim = ssim(changed_map_reshape, img_gt_reshape, data_range=1.0)
    return classes_ev, overall_oa * 100, overall_kappa, overall_F1, overall_ssim