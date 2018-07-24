import torch
from torch.utils import data
import os
import re
import numpy as np
from netCDF4 import Dataset
import transforms as joint_transforms
import time

class WData(data.Dataset):
    def __init__(self, root, transforms, data_list = None):
        self.root = root
        self.data_list = data_list

        for _, _, files in os.walk(self.root):
            continue
        self.files = files
        # reorder
        self.pattern = re.compile("(\d+)")
        self.files.sort(self.my_cmp)
        for i in range(len(self.files)):
            if '2010' in self.files[0] or '2011' in self.files[0] or '2012' in self.files[0]:
                self.files.pop(0)
        print(files)

        self.length_files = []
        for i in self.files:
            weather_factor = Dataset(self.root + i)
            self.length_files.append(weather_factor['t'].shape[0])

        self.transforms = transforms

    def __len__(self):
        return sum(self.length_files) - 4

    def __getitem__(self, idx):
        # load data_file
        for i in range(1, len(self.length_files) + 1):
            if idx < sum(self.length_files[:i]) and idx >= sum(self.length_files[:(i-1)]):
                idx_ = idx - sum(self.length_files[:(i-1)])
                weather_factor = Dataset(self.root + self.files[i-1])
                if (idx + 1) >= sum(self.length_files[:i]):
                    weather_factor_next = Dataset(self.root + self.files[i])
                    idx_next = 0
                else:
                    weather_factor_next = weather_factor
                    idx_next = idx_ + 1
                break

        tem_data = weather_factor['t'][idx_].astype(np.float32)
        hum_data = weather_factor['r'][idx_].astype(np.float32)
        xwin_data = weather_factor['u'][idx_].astype(np.float32)
        ywin_data = weather_factor['v'][idx_].astype(np.float32)
        # load next data

        tem_data_next = weather_factor_next['t'][idx_next].astype(np.float32)
        hum_data_next = weather_factor_next['r'][idx_next].astype(np.float32)
        xwin_data_next = weather_factor_next['u'][idx_next].astype(np.float32)
        ywin_data_next = weather_factor_next['v'][idx_next].astype(np.float32)

        # load label file
        rainfall = Dataset("/data/caoyong/EC_TP_2010-2016.nc")
        # find time id
        time_id = np.where(weather_factor['time'][idx_] == rainfall['time'][:])[0][0]
        # rainfall_label_past = rainfall['tp'][time_id + 1].astype(np.float32)
        # rainfall_label = rainfall['tp'][time_id + 2].astype(np.float32)
        # rainfall_label -= rainfall_label_past

        if idx % 2 != 0:
            rainfall_future = rainfall['tp'][time_id + 4].astype(np.float32)
            # rainfall_past = rainfall['tp'][time_id + 2].astype(np.float32) -\
            #                 rainfall['tp'][time_id].astype(np.float32)
        else:
            rainfall_future = (rainfall['tp'][time_id + 4].astype(np.float32) -
                               rainfall['tp'][time_id + 2].astype(np.float32))
            # rainfall_past = rainfall['tp'][time_id + 2].astype(np.float32)

        data = self.transforms(*[tem_data, hum_data, xwin_data, ywin_data,
                                 tem_data_next, hum_data_next, xwin_data_next, ywin_data_next,
                                 rainfall_future])

        data_tensor = torch.cat([data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]], dim=0)
        # question about 4 dim
        data_tensor.resize_(8, 37, 256, 256)

        return data_tensor, self.tolabel1(data[-1])

    def my_cmp(self, v1, v2):
        d1 = [int(i) for i in self.pattern.findall(v1)][0]
        d2 = [int(i) for i in self.pattern.findall(v2)][0]
        return cmp(d1, d2)

    def tolabel1(self, image):
        image = image.numpy()
        y = np.zeros(image.shape, dtype=np.int32)
        y[image <= 00001] = 0
        y[image > 0.00001] = 1
        return torch.from_numpy(y)

if __name__ == '__main__':
    transform_train = joint_transforms.Compose([
        joint_transforms.RandomCrop(416),
        joint_transforms.ToTensor(),
    ])
    dataset = WData('/data/caoyong/train_data/', transform_train)
    print(1)
    for i in range(len(dataset)):
        data_tensor, label = dataset[i]
        print('{} : {}'.format(i, sum(sum(label == 0))/(416.*416.)))
