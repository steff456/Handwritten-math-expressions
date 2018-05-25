# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
from PIL import Image
import os.path as osp
import torch.utils.data as data


class MathException(Exception):
    pass


class Modes:
    CLASSIFICATION = "classification"
    RECOGNITION = "recognition"


class MathDataset(data.Dataset):
    """Mathematical symbols loader"""
    FILE_NAME = '{0}_index.txt'
    # FILE_REGEX = re.compile(r'\d+_+.+_\d+')
    FOLDER_ROOT = {'generatedSymbols': ['iso', 'png_generatedSymbols'],
                   'junkSymbols': ['junk', 'png_junkSymbols'],
                   'trainingSymbols': ['iso', 'png_trainingSymbols'],
                   'valSymbols': ['BOTH', 'png_valSymbols'],
                   'testSymbols': ['iso', 'png_testSymbols']}
    SPLITS = {'train': ['generatedSymbols', 'trainingSymbols', 'junkSymbols'],
              'test': ['testSymbols'],
              'val': ['valSymbols']}

    def __init__(self, data_root, split, split_folder="data",
                 mode=None, transform=None, annotation_transform=None,
                 num_images={'generatedSymbols': None,
                             'trainingSymbols': None, 'junkSymbols': None}):
        super(MathDataset, self).__init__()
        self.data_root = data_root
        self.split_folder = split_folder
        self.split = split
        self.mode = mode
        self.dataset = []
        self.labels = {}
        self.transform = transform
        self.annotation_transform = annotation_transform
        self.split_file = osp.join(self.split_folder, self.split) + '.pth'
        self.labels_file = osp.join(self.split_folder, 'labels.pth')

        if split not in self.SPLITS:
            raise MathException(
                "Split {0} not in {1}".format(split, self.SPLITS))

        if (len(set(num_images.keys()) & set(self.SPLITS[self.split])) !=
                len(num_images)):
            raise MathException(
                "Part bounds are incomplete/overcomplete "
                "{0} expected {1}".format(
                    num_images.keys(), self.SPLITS[self.split]))

        if not osp.exists(self.split_folder):
            self.process_dataset()

        self.dataset_dict = torch.load(self.split_file)
        for cat in self.dataset_dict:
            parts = self.dataset_dict[cat]
            for part in parts:
                imgs = parts[part][:num_images[part]]
                self.dataset += list(zip(imgs, [cat] * len(imgs)))
        self.labels = torch.load(self.labels_file)

    def _load_txt(self, filename, folder, ext):
        '''Loads the routes of the png files'''
        placeholder = '{0}{1}{2}'
        with open(filename) as f:
            a = f.readlines()
        d = {}
        for line in a:
            line = line.strip()
            parts = line.split(',')
            data = parts[0].split('_')
            name = data[-1].strip()
            cat = parts[1].strip()
            file_name = osp.join(folder, placeholder.format(ext, name, '.png'))
            if cat in d:
                d[cat].append(file_name)
            else:
                d[cat] = [file_name]
        return d

    def process_dataset(self):
        os.makedirs(self.split_folder)
        print("Generating splits from {0}".format(self.data_root))
        labels = []
        for split in self.SPLITS:
            data_split = {}
            for part in self.SPLITS[split]:
                prefix, folder_name = self.FOLDER_ROOT[part]
                folder_name = folder_name.format(part)
                filename = osp.join(self.data_root, self.FILE_NAME.format(part))
                maps = self._load_txt(filename, folder_name, prefix)
                for cat in maps:
                    if cat not in data_split:
                        data_split[cat] = {}
                    partial = data_split[cat]
                    if part not in partial:
                        partial[part] = []
                    partial_list = partial[part]
                    partial_list += maps[cat]
                    labels.append(cat)
            split_file = osp.join(self.split_folder, '{0}.pth'.format(split))
            torch.save(data_split, split_file)
        labels = set(labels)
        labels = dict(zip(list(labels), range(len(labels))))
        torch.save(labels, osp.join(self.split_folder, 'labels.pth'))

    def __len__(self):
        return len(self.dataset)

    def pull_item(self, idx):
        image_file, label = self.dataset[idx]
        image_file = osp.join(self.data_root, image_file)
        image = Image.open(image_file)
        return image, label

    def __getitem__(self, idx):
        image, label = self.pull_item(idx)
        target = self.labels[label]
        image = np.array(image)
        if image.ndim == 2:
            image = np.stack([image] * 3, dim=-1)
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        if self.annotation_transform is not None:
            target = self.annotation_transform(target)
        return image, target
