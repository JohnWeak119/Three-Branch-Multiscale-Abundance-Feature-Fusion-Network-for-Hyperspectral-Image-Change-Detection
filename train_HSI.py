import os
import torch.nn as nn
import configs.configs as cfg
import torch.optim as optim

from data.HSICD_data import HSICD_data
from data.get_train_test_set import get_train_test_set as get_set
from tools.train import train as fun_train
from tools.test import test as fun_test
from datetime import datetime
from tools.save_color_predict_img import save_color_img

##########################  ATTENTION  #############################
# model is initialized with img_channel & input_abu(data)_channel  #

# tools
from tools.show import *
from tools.assessment import *
# 根据cfg中的model改变导入的model
if cfg.current_model == 'TMAFN':
    from model.TMAFN import TMAFN as fun_model


def main():

    current_dataset = cfg.current_dataset 
    current_model = cfg.current_model
    model_name = current_dataset + '_' + current_model
    print('model {}'.format(model_name))
    # 加载config创建的字典
    cfg_data = cfg.data
    cfg_model = cfg.model
    cfg_train = cfg.train['train_model']
    cfg_optim = cfg.train['optimizer']
    cfg_test = cfg.test
    device = cfg.device

    #Data import and data set partition
    data_sets = get_set(cfg_data)   # get_set函数返回的是一个字典(data_sets)
    img_gt = data_sets['img_gt']
    train_data = HSICD_data(data_sets, cfg_data['train_data'])  # 获得训练的img1, img2, abu(patch)，label, index
    val_data = HSICD_data(data_sets, cfg_data['val_data'])    # 获得验证的patch，label, index
    test_data = HSICD_data(data_sets, cfg_data['test_data'])    # 获得测试的patch，label, index

    # Load model & loss function & optimizer
    model = fun_model(cfg_model['in_fea_num'], cfg_model['in_abu_ch']).to(device)

    loss_fun = nn.CrossEntropyLoss()    # 多分类问题
    
    scheduler = None    # 初始化scheduler
    if cfg_optim['typename'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg_optim['lr'], weight_decay=cfg_optim['weight_decay'])      # Adam
    elif cfg_optim['typename'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=cfg_optim['lr'], momentum=cfg_optim['momentum'], weight_decay=cfg_optim['weight_decay'])    # SGD
    else:
        raise ValueError('Undefined Optimizer')
    fun_train(train_data, val_data, model, loss_fun, optimizer, scheduler, device, cfg_train)
    # val & test
    pred_train_label, pred_train_acc = fun_test(train_data, data_sets['img_gt'], model, device, cfg_test)
    pred_val_label, pred_val_acc = fun_test(val_data, data_sets['img_gt'], model, device, cfg_test)
    pred_test_label, pred_test_acc = fun_test(test_data, data_sets['img_gt'], model, device, cfg_test)

    # Post processing
    predict_label = torch.cat([pred_train_label, pred_val_label, pred_test_label], dim=0)
    print('pred_train_acc {:.2f}%, val_acc {:.2f}%, pred_test_acc {:.2f}%'.format(pred_train_acc, pred_val_acc, pred_test_acc))
    predict_img = Predict_Label2Img(predict_label, img_gt)
    classes_ev, overall_oa, overall_kappa, overall_F1, overall_ssim = multi_class_accuracy_assessment(img_gt, predict_img)

    # Store
    save_folder = cfg_test['save_folder']
    save_name = cfg_test['save_name']

    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)
    # get assessment_result str
    config_str = 'model_name: {}, epoch: {}, lr: {}, decay: {}, \nbatch_size: {}, patch_size: {}, train_ratio: {}, val_ratio: {}'.format(
        model_name, cfg.epoch_number, cfg.lr, cfg_optim['weight_decay'], cfg.bs_number, cfg.patch_size, cfg.train_set_num, cfg.val_ratio)
    overall_assessment = "OA: {:.2f}%, Kappa: {:.4f}, F1: {:.4f}, SSIM: {:.4f}".format(overall_oa, overall_kappa, overall_F1, overall_ssim)

    # 获取当前时间并格式化为字符串
    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # 将评估结果写入文件
    write_mode = 'a'    # ‘w’覆盖写入，‘a’追加写入
    with open(save_folder + '/' + save_name + '_result.txt', write_mode) as f:
        f.write('\n\n' + current_time_str + '\n' + config_str + '\n' + overall_assessment)
        for i in range(cfg.classes):
            if cfg.current_dataset in cfg.mask0data and i == 0:  # 忽略键为0的值
                continue
            f.write('\n' + classes_ev[i])

    # 写入.mat文件
    # io.savemat(save_folder + '/' + save_name + ".mat",
    #            {"predict_img": np.array(predict_img.cpu()), "oa": overall_oa, "kappa": overall_kappa})

    save_color_img(predict_img, save_folder, save_name)

if __name__ == '__main__':
    main()