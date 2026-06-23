import torch
#########################  NOTICE!  #########################
# 1. When setting undefined datasets, please check data.get_dataset_only.py & data.get_dataset.py
#    & tools.slic_unmixing.py & tools.vca_unmixing.py
#    & datasets class & settings in this file.
# 2. When setting undefined models, please check model & train_HSI.py

current_dataset = 'MultiFarmland'   # RMSE1:0.0267, RMSE2:0.0201  Comb RMSE:0.0271  6end
# current_dataset = 'Hermiston'     # RMSE1:0.0316, RMSE2:0.0531  Comb RMSE:0.0306
# current_dataset = 'MultiHermiston'  # Comb RMSE:0.0507

# datasets class
classes6data = ['MultiFarmland', 'Hermiston']
classes7data = ['MultiHermiston']

# current_model = 'CBANet'  # 论文1
# current_model = 'MFCEN'     # 论文2
# current_model = 'MSDFFN'    # 论文3
# current_model = 'ML_EDAN'   # 论文4
# current_model = 'CSANet'    # 论文5
# current_model = 'D2AGCN'    # 论文6
# current_model = 'MSUJMF'    
# current_model = 'MSUJMC'    
current_model = 'TMAFN'  # 实验组, TMAFN
my_model = ['My_Net_ResMHCA', 'TMAFN']

# 1. data
phase = ['train', 'val', 'test', 'no_gt']
train_set_num = 0.05
val_ratio = 0.1

# 数据集参数设置
# 不采用超像素时设置为0
settings = {
    'Hermiston':        {'superpixel_scale': 50, 'endmember_num': 5, 'pca_ch':10, 'classes': 6, 'in_fea_num': 159},
    'MultiFarmland':    {'superpixel_scale': 5, 'endmember_num': 6, 'pca_ch':10, 'classes': 6, 'in_fea_num': 132},
    'MultiHermiston':   {'superpixel_scale': 100, 'endmember_num': 6, 'pca_ch':10, 'classes': 7, 'in_fea_num': 159},
}
current_setting = settings[current_dataset]
endmember_num = current_setting['endmember_num']
pca_ch = current_setting['pca_ch']
superpixel_scale = current_setting['superpixel_scale']
classes = current_setting['classes']
abu_ch = endmember_num * 2
model = {
    'in_fea_num': current_setting['in_fea_num'],
    'in_abu_ch': abu_ch, # JM时该参数为JM矩阵的通道数
}

# 0.Initial parameter
combine_abu = True     # 重新解混删除dataset中的abu.mat
lr_adjust = None
lr_step = None
lr_gamma = None
momentum = None
weight_decay = 0
std = None      # 控制两影像是否std——norm
avg_valloss = None  # 控制是否使用平均的val_loss
w_avg_valloss = None    # 控制是否使用加权平均的val_loss
reuse_model = False

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:1' if use_cuda else 'cpu')

current = current_dataset + '_' + current_model + train_set_num.__str__()
reuse_file='./weights/' + current_model + '/'  + current_dataset + '/' + current + '_Final.pth'
model_weights='./weights/' + current_model + '/'  + current_dataset + '/' + current + '_Final.pth'

# 2. model deteil

if current_model in my_model:
    patch_size = 7
    bs_number = 128
    epoch_number = 150
    weight_decay = 1e-3
    avg_valloss == False
    w_avg_valloss == False

    optim_typename='Adam'
    lr = 1e-5
    lr_gamma = 0.1
    lr_adjust = True
    lr_step=[100, 200]

    reuse_model = False
    reuse_file='./weights/' + current_model + '/'  + current_dataset + '/' + current + '_best.pth'
    model_weights='./weights/' + current_model + '/'  + current_dataset + '/' + current + '_best.pth'

# 3. data
data = dict(
    current_dataset=current_dataset,
    train_set_num=train_set_num,
    val_ratio=val_ratio,
    patch_size=patch_size,

    train_data=dict(
        phase=phase[0]
    ),
    val_data=dict(
        phase=phase[1]
    ),
    test_data=dict(
        phase=phase[2]
    ),
)

# 4. train
train = dict(
    optimizer=dict(
        typename = optim_typename,
        lr=lr,
        momentum = momentum,
        weight_decay = weight_decay
    ),
    train_model=dict(
        gpu_train=True,
        gpu_num=1,
        workers_num=12,
        epoch = epoch_number,
        batch_size = bs_number,
        lr = lr,
        lr_adjust = lr_adjust,
        lr_gamma = lr_gamma,
        lr_step = lr_step,
        train_ratio = train_set_num,

        save_folder='./weights/' + current_model + '/'  + current_dataset + '/',
        save_name = current,
        current_dataset = current_dataset,

        # 是否复用已有模型
        reuse_model = reuse_model,
        reuse_file= reuse_file,
    )
)

# 5. test
test = dict(
    batch_size=1000,
    gpu_train=True,
    gpu_num=1,
    workers_num=8,
    model_weights = model_weights,
    save_name=current,
    save_folder='./result' + '/' + current_model + '/' + current_dataset,
)