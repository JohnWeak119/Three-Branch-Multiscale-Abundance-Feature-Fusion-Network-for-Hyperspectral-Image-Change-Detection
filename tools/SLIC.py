import numpy as np
from skimage.segmentation import slic
from sklearn import preprocessing

class SLIC(object):
    def __init__(self, HSI, n_segments=1000, compactness=20,
                 max_num_iter=20, sigma=0, min_size_factor=0.3, max_size_factor=2):
        #n_segments 超像素数；compactness 紧密度；max_iter 迭代；sigma ；size 最大最小尺度

        self.n_segments = n_segments   # 分割数
        self.compactness = compactness
        self.max_num_iter = max_num_iter
        self.min_size_factor = min_size_factor
        self.max_size_factor = max_size_factor
        self.sigma = sigma
        self.data = HSI

    def SegmentsLabelProcess(segments):
        segments = np.array(segments, np.int64)
        H, W = segments.shape
        ls = list(set(np.reshape(segments, [-1]).tolist()))

        dic = {}
        for i in range(len(ls)):
            dic[ls[i]] = i

        new_segments = segments
        for i in range(H):
            for j in range(W):
                new_segments[i, j] = dic[new_segments[i, j]]
        return new_segments

    def get_Q_and_S_and_Segments(self):
        img = self.data
        (h, w, d) = img.shape
        segments = slic(img, n_segments=self.n_segments, compactness=self.compactness, max_iter=self.max_num_iter,
                        convert2lab=False, sigma=self.sigma, enforce_connectivity=True,
                        min_size_factor=self.min_size_factor, max_size_factor=self.max_size_factor, slic_zero=False, start_label=0)

        if segments.max()+1 != len(list(set(np.reshape(segments, [-1]).tolist()))):
            segments = SLIC.SegmentsLabelProcess(segments)
        self.segments = segments
        superpixel_count = segments.max() + 1
        self.superpixel_count = superpixel_count

        ###################################### 显示超像素图片 ######################################
        # out = mark_boundaries(img[:,:,[0,1,2]], segments)
        # plt.figure()
        # plt.imshow(out)
        # plt.show()
        ###################################### 显示超像素图片 ######################################

        segments = np.reshape(segments, [-1])
        S = np.zeros([superpixel_count, d], dtype=np.float32)
        Q = np.zeros([w * h, superpixel_count], dtype=np.float32)
        x = np.reshape(img, [-1, d])
        for i in range(superpixel_count):
            idx = np.where(segments == i)[0]
            count = len(idx)
            pixels = x[idx]
            superpixel = np.sum(pixels, 0) / count
            S[i] = superpixel
            Q[idx, i] = 1  #当前超像素索引的点为记为1

        self.S = S
        self.Q = Q
        return Q, S, self.segments #Q 列展开超像素标签；S 超像素均值；segments 原影像超像素标签

    def get_A(self, sigma: float):
        A = np.zeros(
            [self.superpixel_count, self.superpixel_count], dtype=np.float32)
        (h, w) = self.segments.shape  #原影像大小
        for i in range(h - 2):
            for j in range(w - 2):
                sub = self.segments[i:i + 2, j:j + 2]  #每2*2个像素一个sub，记录标签
                sub_max = np.max(sub).astype(np.int32)
                sub_min = np.min(sub).astype(np.int32)
                if sub_max != sub_min:
                    idx1 = sub_max
                    idx2 = sub_min
                    if A[idx1, idx2] != 0:
                        continue

                    pix1 = self.S[idx1]
                    pix2 = self.S[idx2]
                    diss = np.exp(-np.sum(np.square(pix1 - pix2)) / sigma ** 2)
                    A[idx1, idx2] = A[idx2, idx1] = diss  #A为对角矩阵，记录每2*2像素对应的超像素相似度
        return A

def apply_SLIC(HSI, scale=25):
    height, width, bands = HSI.shape
    scale = scale

    # 不采用超像素时设置为0
    if scale == 0:
        avg_SLIC = HSI
    else:
        n_segments_init = height * width / scale
        myslic = SLIC(HSI, n_segments=n_segments_init, compactness=1, sigma=1, min_size_factor=0.1, max_size_factor=2)
        Q, S, Segments = myslic.get_Q_and_S_and_Segments()
        # A = myslic.get_A(sigma=10)
        avg_SLIC = np.zeros((height, width, bands))
        for i in range(height):
            for j in range(width):
                for k in range(bands):
                    # 获取当前像素的超像素标签
                    superpixel_label = Segments[i, j]
                    # 从S中获取对应的超像素均值
                    avg_SLIC[i, j, k] = S[superpixel_label, k]
    
    return avg_SLIC