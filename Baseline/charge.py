#!/usr/bin/env python
import os
import numpy as np


# Charges the handwritten symbols to the workspace
def get_images():
    # Directory of simple symbols
    root_dir = './Handwritten-symbols'

    # Stores the directory for train images and test images
    # Dictionary, key: category, val: list of dir images
    print('------ Loading information ------')
    train_images = {}
    test_images = {}
    for root, dirs, files in os.walk(root_dir, topdown=False):
        tam_dir = len(files)
        half = int(tam_dir/2)
        count = 0
        data = root.split('/')
        category = data[-1]
        for f in files:
            act_f = root + '/' + f
            if category in list(train_images.keys()) and count < half:
                list_im = train_images[category]
                list_im.append(act_f)
            elif category not in list(train_images.keys()) and count < half:
                train_images[category] = [act_f]
            if category in list(test_images.keys()) and count >= half:
                list_im = test_images[category]
                list_im.append(act_f)
            elif category not in list(test_images.keys()) and count >= half:
                test_images[category] = [act_f]
            count = count + 1

    return train_images, test_images


def save_var(name_file, variable):
    np.save(name_file, variable)
