#!/usr/bin/env python
import os
import numpy as np


def get_images(numberImagesTrainPerCategory,
               root_dir='/datadisks/disk1/galindojimenez/Data_Symbols/'):
    # Dictionaries for train and test
    train_images = {}
    test_images = {}
    # original lists from the .txt files
    train_list_org = open(root_dir+'trainingSymbols_index.txt').readlines()
    train_list_gen = open(root_dir+'generatedSymbols_index.txt').readlines()
    train_list_junk = open(root_dir+'junkSymbols_index.txt').readlines()
    # super list containing all information
    train_list_total = train_list_org+train_list_gen
    # Test list
    test_list = open(root_dir+'testSymbols_index.txt').readlines()
    # fill the train_images dictionary
    count = 0
    print('Inicié train')
    for f in range(len(train_list_total)):
        # extract the line of the list to get category and name of the image
        f_act = train_list_total[f]
        category = f_act.split(',')[-1].split()[0]
        if category == '/':
            category = 'slash'
        num = f_act.split(',')[0].split('_')[-1]
        # Generate the complete route according to the list the file comes from
        if f < len(train_list_org)-1:
            root_train = root_dir+'/png_trainingSymbols/'
            name = root_train+'iso'+num+'.png'
        else:
            root_train = root_dir+'/png_generatedSymbols/'
            name = root_train+'iso'+num+'.png'
        # fill the dictionaries
        # fill with 101 categories
        if category in list(train_images.keys()) and len(train_images[category]) < numberImagesTrainPerCategory:
            list_category = train_images[category]
            list_category.append(name)
        elif category not in list(train_images.keys()):
            train_images[category] = [name]
        count = count+1
        # print(count)
    # fill with the junk Category
    print('Pase a junk')
    count = 0
    for j in range(len(train_list_junk)):
        # extract the line of the junk list to get the
        # category and name of the image
        f_act = train_list_junk[j]
        category = 'junk'
        num = f_act.split(',')[0].split('_')[-1]
        # give the correct name
        root_train = root_dir+'/png_junkSymbols/'
        name = root_train+'junk'+num+'.png'
        # fill the junk category
        if category in list(train_images.keys()) and len(train_images[category]) < numberImagesTrainPerCategory:
            list_category = train_images[category]
            list_category.append(name)
        elif category not in list(train_images.keys()):
            train_images[category] = [name]
        count = count+1
        # print(count)
    # fill the test_images dictionary
    print('Pase a Test')
    count = 0
    for n in range(len(test_list)):
        # extract the line of the list to get category and name of the image
        f_act = test_list[n]
        category = f_act.split(',')[-1].split()[0]
        if category == '/':
            category = 'slash'
        num = f_act.split(',')[0].split('_')[-1]
        # Generate the complete route according to the list the file comes from
        # give the correct name
        root_train = root_dir+'/png_testSymbols/'
        name = root_train+'iso'+num+'.png'
        # fill the dictionary with 102 categories
        if category in list(test_images.keys()):
            list_category = test_images[category]
            list_category.append(name)
        elif category not in list(test_images.keys()):
            test_images[category] = [name]
        count = count+1
    return train_images, test_images
