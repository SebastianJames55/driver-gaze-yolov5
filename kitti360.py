import os
import numpy as np
import math

import torch
from torch.utils.data import Dataset
import torchvision
from PIL import Image


class Kitti360(Dataset):
    """
    Kitti360 feature class.
    """
    def __init__(self, subset, file, feature_path, threshold, lstm, seqlen):
        """
        Args:

        """
        self.subset = subset
        self.file = file
        self.feature_path = feature_path
        self.threshold = threshold
        self.mean = torch.zeros(1024)
        self.std = torch.ones(1024)
        self.lstm = lstm
        self.seqlen = seqlen
        self._parse_list()
        self.transform = torchvision.transforms.Compose(
                [torchvision.transforms.Resize([36,64]),
                torchvision.transforms.ToTensor()])



    def _parse_list(self):

        self.img_list = []

        all_entries = os.listdir(self.file)
        img_list = [f for f in all_entries if os.path.isfile(os.path.join(self.file, f))]

        if self.lstm:
            self.img_dict = {}

            clips = list(set([x.split('_')[0] for x in open(self.file)]))

            for clip in clips:
                self.img_dict[clip] = []

            for item in img_list:
                img_name = item.split('.')[0]
                feature_name = img_name + ".pt"
                clip = item.split('.')[0].split('_')[0]
                img_nr = item.split('.')[0].split('_')[1]

                feature_path = os.path.join(self.feature_path,feature_name)

                if os.path.exists(feature_path):
                    self.img_list.append(item)
                    self.img_dict[clip].append(img_nr)
                else:
                    print('error loading feature:', feature_path)

            for key in self.img_dict:
                self.img_dict[key].sort()
            print('video number in %s: %d'%(self.subset,(len(self.img_list))))
        else:
            for item in img_list:
                img_name = item.split('.')[0]
                feature_name = img_name + ".pt"

                feature_path = os.path.join(self.feature_path,feature_name)
                if os.path.exists(feature_path):
                    self.img_list.append(item)
                else:
                    print('error loading feature:', feature_path)


        print('video number in %s: %d'%(self.subset,(len(self.img_list))))


    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        """
        """

        if self.lstm:
            record = self.img_list[index]
            img_name = record.split('.')[0]
            feature_name = img_name + ".pt"

            clip = record.split('.')[0].split('_')[0]
            img_nr = record.split('.')[0].split('_')[1]
            dict_idx = self.img_dict[clip].index(img_nr)

            feature_path = os.path.join(self.feature_path,feature_name)
            feature = torch.load(feature_path)

            # create list with previous features, last one is original
            feature_list = []
            first = dict_idx-(self.seqlen-1)
            duplicate = 0
            if first < 0:
                duplicate = abs(first) # if there are not enough previous features, we duplicate original to get seqlen
                first = 0
            for idx in range(first, dict_idx+1):
                feature_name2 = clip+'_'+self.img_dict[clip][idx]+ ".pt"
                feature_path2 = os.path.join(self.feature_path,feature_name2)
                feature2 = torch.load(feature_path2)
                feature_list.append(feature2)
            if duplicate:
                for i in range(duplicate):
                    feature_list.append(feature)
            feature = torch.stack(feature_list)
        else:
            record = self.img_list[index]
            img_name = record.split('.')[0]
            feature_name = img_name + ".pt"
            feature_path = os.path.join(self.feature_path,feature_name)
            feature = torch.load(feature_path)

            return feature, img_name
