# -*- coding: utf-8 -*-

import os
import numpy as np
from PIL import Image
import os.path as osp
import torch.data as data

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
                   'junkSymbols' : ['junk', 'png_junkSymbols'],
                   'trainingSymbols' : ['iso', 'png_trainingSymbols'],
                   'valSymbols' : ['BOTH', 'png_valSymbols'],
                   'testSymbols' : ['iso', 'png_testSymbols']
                   }
    SPLITS = {'train': [
                'generatedSymbols', 'trainingSymbols', 'junkSymbols'],
              'test' : ['testSymbols'],
              'val' : ['valSymbols']}

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
        self.transform = transform
        self.annotation_transform = annotation_transform
        self.split_file = osp.join(self.split_folder, self.split) + '.pth'

        if split not in self.SPLITS:
            raise MathException(
                "Split {0} not in {1}".format(split, self.SPLITS))

        if ((set(num_images.keys()) & set(self.SPLITS[self.split])) !=
                len(num_images)):
            raise MathException("Part bounds are incomplete/overcomplete")

        if not osp.exists(self.split_folder):
            self.process_dataset()

        self.dataset_dict = torch.load(self.split_file)
        for cat in self.dataset_dict:
            parts = self.dataset_dict[cat]
            for part in parts:
                imgs = parts[part][:num_images[part]]
                self.dataset += list(zip(imgs, [cat] * len(imgs)))

    def _load_txt(filename, folder, ext):
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
            file_name = osp.join(folder, placeholder.format(ext, name, '.png'))
            if parts[1] in d:
                d[parts[1]].append(file_name)
            else:
                d[parts[1]] = [file_name]
        return d

    def process_dataset(self):
        os.makedirs(self.split_folder)
        print("Generating splits from {0}".format(self.data_root))
        for split in self.SPLITS:
            data_split = {}
            for part in self.SPLITS[split]:
                prefix, folder_name = self.FOLDER_ROOT[part]
                folder_name = folder_name.format(part)
                filename = osp.join(self.data_root, self.FILE_NAME.format(part))
                maps = self._load_txt(filename, folder_name, prefix)
                for cat in maps:
                    partial = data_split.get(cat, {})
                    partial_list = partial[part].get(partial, [])
                    partial_list.append(maps[cat])
                    partial[part] = partial_list
                    data_split[cat] = partial
            split_file = osp.join(self.split_folder, '{0}.pth'.format(split))
            torch.save(data_split, split_file)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_file, label = self.dataset[idx]
        image_file = osp.join(self.data_root, image_file)
        image = Image.open(image_file)

        if self.transform is not None:
            image = self.transform(image)
        if self.annotation_transform is not None:
            target = self.annotation_transform(target)
        return image, label

