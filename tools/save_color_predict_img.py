import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import configs.configs as cfg
import numpy as np

def save_color_img (predict_img, save_folder, save_name):

    # 定义颜色映射，每个整数分配一个颜色
    if cfg.current_dataset in cfg.classes2data:
        cmap = ListedColormap(['black', 'white'])
        num_classes = 2
    elif cfg.current_dataset in cfg.classes3data:
        cmap = ListedColormap(['black', 'red', 'green'])
        num_classes = 3
    elif cfg.current_dataset in cfg.classes4data:
        cmap = ListedColormap(['black', 'red', 'green', 'blue'])
        num_classes = 4
    elif cfg.current_dataset in cfg.classes6data:
        cmap = ListedColormap(['black', 'red', 'green', 'blue', 'yellow', 'purple'])
        num_classes = 6
    elif cfg.current_dataset in cfg.classes7data:
        cmap = ListedColormap(['black', 'red', 'green', 'blue', 'yellow', 'purple', 'cyan'])
        num_classes = 7
    else:
        raise ValueError(f"Unknown dataset: {cfg.current_dataset}")

    norm = BoundaryNorm(np.arange(num_classes + 1) - 0.5, num_classes)  # 设置边界，使颜色正确映射到整数标签

    # 使用 matplotlib 保存带颜色的图片
    plt.imshow(predict_img, cmap=cmap, norm=norm, interpolation='none')  # 确保像素之间没有插值
    plt.axis('off')  # 关闭坐标轴
    plt.savefig(save_folder + '/' + save_name + '_predict.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    print('save predict_img successful!')
    return None