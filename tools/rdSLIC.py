import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from skimage.segmentation import slic
from sklearn import preprocessing


def SegmentsLabelProcess(labels):
    labels = np.array(labels, np.int64)
    H, W = labels.shape
    ls = list(set(np.reshape(labels, [-1]).tolist()))

    dic = {}
    for i in range(len(ls)):
        dic[ls[i]] = i

    new_labels = labels
    for i in range(H):
        for j in range(W):
            new_labels[i, j] = dic[new_labels[i, j]]
    return new_labels


class SLIC(object):
    def __init__(self, HSI, labels, n_segments=1000, compactness=20,
                 max_iter=20, sigma=0, min_size_factor=0.3, max_size_factor=2):
        #n_segments 超像素数；compactness 紧密度；max_iter 迭代；sigma ；size 最大最小尺度

        self.n_segments = n_segments   # 分割数
        self.compactness = compactness
        self.max_iter = max_iter
        self.min_size_factor = min_size_factor
        self.max_size_factor = max_size_factor
        self.sigma = sigma
        height, width, bands = HSI.shape
        data = np.reshape(HSI, [height * width, bands])
        minMax = preprocessing.StandardScaler() #将数据转换为均值为0，标准差为1的分布，这有助于加速模型的收敛并提高模型的性能
        data = minMax.fit_transform(data)
        self.data = np.reshape(data, [height, width, bands])
        self.labels = labels

    def get_Q_and_S_and_Segments(self):
        img = self.data
        (h, w, d) = img.shape
        segments = slic(img, n_segments=self.n_segments, compactness=self.compactness, max_iter=self.max_iter,
                        convert2lab=False, sigma=self.sigma, enforce_connectivity=True,
                        min_size_factor=self.min_size_factor, max_size_factor=self.max_size_factor, slic_zero=False, start_label=0)

        if segments.max()+1 != len(list(set(np.reshape(segments, [-1]).tolist()))):
            segments = SegmentsLabelProcess(segments)
        self.segments = segments
        superpixel_count = segments.max() + 1
        self.superpixel_count = superpixel_count
        print("superpixel_count", superpixel_count)

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


class LDA_SLIC(object):
    def __init__(self, data, labels, n_component):
        self.data = data
        self.init_labels = labels
        self.curr_data = data
        self.n_component = n_component
        self.height, self.width, self.bands = data.shape
        self.x_flatt = np.reshape(data, [self.width*self.height, self.bands]) #data3d转2d
        self.y_flatt = np.reshape(labels, [self.height*self.width]) #label转2d
        self.labels = labels

    def LDA_Process(self, curr_labels):
        curr_labels = np.reshape(curr_labels, [-1])  #转1d
        idx = np.where(curr_labels != 0)[0]  #1维curr_labels中不为0的idx
        x = self.x_flatt[idx]
        y = curr_labels[idx]
        lda = LinearDiscriminantAnalysis()  # n_components = self.n_component
        lda.fit(x, y-1)  #非0的x与对应y的标签进行训练，标签需从0开始
        X_new = lda.transform(self.x_flatt)  #用训练得到的LDA模型对所有self.x_flatt进行降维
        return np.reshape(X_new, [self.height, self.width, -1]) #X降到class_num-1维

    def SLIC_Process(self, img, scale=25):  #预设超参数
        n_segments_init = self.height*self.width/scale  #超像素大小
        # print("n_segments_init",n_segments_init)                      #  210.25
        myslic = SLIC(img, n_segments=n_segments_init, labels=self.labels,
                      compactness=1, sigma=1, min_size_factor=0.1, max_size_factor=2)
        Q, S, Segments = myslic.get_Q_and_S_and_Segments()
        A = myslic.get_A(sigma=10)
        return Q, S, A, Segments

    def simple_superpixel(self, scale):
        curr_labels = self.init_labels
        X = self.LDA_Process(curr_labels)  #降维
        Q, S, A, Seg = self.SLIC_Process(X, scale=scale)  #分割
        # Q, S, A, Seg = self.SLIC_Process(self.data, scale=scale)
        return Q, S, A, Seg

    def simple_superpixel_no_LDA(self, scale):
        Q, S, A, Seg = self.SLIC_Process(self.data, scale=scale)
        return Q, S, A, Seg
