"""
Author: Jolin
Usage : feeder
Date  : 2024-12-05
"""
import numpy as np
import yaml
import torch
from torch.utils.data import Dataset
from feeders import tools
from .bone_pairs import ntu_pairs
import random

# import tools
# from bone_pairs import ntu_pairs


class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', 
                 interpolate = 0, Contrastive_Learning = False,
                 random_choose=False, random_shift=False,random_scale_shift=False,
                 random_move=False, random_rot=False, 
                 window_size=-1, normalization=False, debug=False, use_mmap=False,
                 bone=False, vel=False):
        """ 
        :param data_path:
        :param label_path:
        :param split: training set or test set
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param random_rot: rotate skeleton around xyz axis
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        :param bone: use bone modality or not
        :param vel: use motion modality or not
        :param only_label: only load label for ensemble score compute
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.random_scale_shift = random_scale_shift
        if random_scale_shift:
            self.ScaleAndTranslate = SkeletonScaleAndTranslate()
        self.bone = bone
        self.vel = vel
        self.interpolate = interpolate
        self.Contrastive_Learning = Contrastive_Learning
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M
        npz_data = np.load(self.data_path)
        if self.split == 'train':
            print("--------begin load data(5min)--------")
            self.data = npz_data['x_train'] # (40091, 300, 150)
            print("--------npz_data['x_train'] done--------")
            self.label = np.where(npz_data['y_train'] > 0)[1]
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.split == 'test':
            self.data = npz_data['x_test']
            self.label = np.where(npz_data['y_test'] > 0)[1]
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]#一大串的表头名称
        else:
            raise NotImplementedError('data split only supports train/test')
        N, T, _ = self.data.shape
        #self.data = self.data.reshape((N, T, 2, 25, 5)).transpose(0, 4, 1, 3, 2)
        self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)    # (40091, 3, 300, 25, 2)
        
    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]   
        label = self.label[index]  
        data_numpy = np.array(data_numpy)  
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)    
        # reshape Tx(MVC) to CTVM
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)   
        if self.random_rot:    
            random_number = random.random()
            if random_number<= self.random_rot:
                data_numpy = tools.random_rot(data_numpy).numpy()
        if self.random_scale_shift:
            random_number = random.random()
            if random_number<= self.random_scale_shift:
                data_numpy = self.ScaleAndTranslate(data_numpy)                
        if self.bone: 
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in ntu_pairs:
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            data_numpy = bone_data_numpy
        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0
        if self.Contrastive_Learning:
            data_numpy_aug = tools.random_rot(data_numpy)
            data_numpy_reverse = data_numpy[:,::-1,:,:].copy()
            return data_numpy, data_numpy_aug, data_numpy_reverse, label
        else:            
            return data_numpy, label

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)

class SkeletonScaleAndTranslate(object):
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2., translate_range=0.2):
        """
        初始化缩放和平移参数。

        :param scale_low: 缩放因子的下界。
        :param scale_high: 缩放因子的上界。
        :param translate_range: 平移范围，在 [-translate_range, translate_range] 之间。
        """
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.translate_range = translate_range

    def __call__(self, skeleton):
        """
        对输入的骨架数据进行缩放和平移。

        :param skeleton: 形状为 [C, T, V, M] 的骨架数据。
        :return: 经过缩放和平移后的骨架数据。
        """
        C, T, V, M = skeleton.shape

        scale_factors = np.random.uniform(low=self.scale_low, high=self.scale_high, size=(C, 1, 1,1))
        translations = np.random.uniform(low=-self.translate_range, high=self.translate_range, size=(C, 1, 1,1))

        skeleton = skeleton * scale_factors + translations

        return skeleton

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


if __name__ == '__main__':
    import tools
    config_path = r"./config/NTU120_CS_default.yaml"
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    feeder = Feeder(**config['train_feeder_args'])
    print(feeder[0])