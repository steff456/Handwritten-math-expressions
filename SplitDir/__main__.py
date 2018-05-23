# -*- coding: utf-8 -*-
# !/usr/bin/env python3

"""Main for splitting the directories"""
import os
import os.path as osp
import argparse
import logging
import coloredlogs

from AugmData.new_inkml import generate

LOG_FORMAT = ('%(levelname) -10s %(asctime)s %(name) -30s %(funcName) '
              '-35s %(lineno) -5d: %(message)s')
LOGGER = logging.getLogger(__name__)
coloredlogs.install(level='info')

parser = argparse.ArgumentParser(
    description='Divide per directory')

parser.add_argument('--train-dir', type=str, default='./png_trainingSymbols',
                    help='directory of the training png files')
parser.add_argument('--gen-dir', type=str, default='./png_generatedSymbols',
                    help='directory of the generated png files')
parser.add_argument('--junk-dir', type=str, default='./png_junkSymbols',
                    help='directory of the junk train png files')
parser.add_argument('--val-dir', type=str, default='./png_valSymbols',
                    help='directory of the validation png files')
parser.add_argument('--test-dir', type=str, default='./png_testSymbols',
                    help='directory of the test png files')
parser.add_argument('--train-txt', type=str, default='./training_index.txt',
                    help='file with annotations of train')
parser.add_argument('--gen-txt', type=str, default='./generated_index.txt',
                    help='file with annotations of generated')
parser.add_argument('--val-txt', type=str, default='./val_index.txt',
                    help='file with annotations of val')
parser.add_argument('--test-txt', type=str, default='./test_index.txt',
                    help='file with annotations of test')
parser.add_argument('--out-dir', type=str, default='./orgDataJunk',
                    help='directory for the new dataset')

args = parser.parse_args()


def load_txt(file, directory, ext):
    '''Loads the routes of the png files'''
    with open(file) as f:
        a = f.readlines()
    d = {}
    for line in a:
        line = line.strip()
        parts = line.split(',')
        data = parts[0].split('_')
        name = data[-1].strip()
        if parts[1] in d:
            d[parts[1]].append(directory + '/' + ext + name + '.png')
        else:
            d[parts[1]] = [directory + '/' + ext + name + '.png']
    return d


def organize_junk(junk, out_dir):
    # generate train directory
    act = out_dir + '/train'
    os.mkdir(act)
    # generate junk in train
    out = act + '/junk'
    os.mkdir(out)
    for key in junk:
        listact = junk[key]
        for i in listact:
            print(i)
            os.system('cp ' + i + ' ' + out)


def organize_trainO(org, out_dir):
    # generate the original in train
    for key in org:
        print('-------------')
        print(key)
        if key == '/':
            name = 'slash'
        else:
            name = key
        out = act + '/' + name
        osp.mkdir(out)
        listorg = org[key]
        for i in listorg:
            os.system('cp ' + i + ' ' + out)


def organize_trainG(gen, out_dir):
    # generate the gen in train
    for key in gen:
        print('-------------')
        print(key)
        if key == '/':
            name = 'slash'
        else:
            name = key
        out = act + '/' + name
        osp.mkdir(out)
        listgen = gen[key]
        for i in listgen:
            os.system('cp ' + i + ' ' + out)


def main():
    trainOr = load_txt(args.train_txt, args.train_dir, 'iso')
    trainGen = load_txt(args.gen_txt, args.gen_dir, 'iso')
    val = load_txt(args.val_txt, args.val_dir, 'BOTH')
    test = load_txt(args.test_txt, args.test_dir, 'iso')
    trainJunk = load_txt(args.junk_txt, args.junk_dir, 'junk')
    if osp.exists(args.out_dir):
        os.system('rm -rf ' + args.out_dir)
    os.mkdir(args.out_dir)
    organize_junk(trainJunk, args.out_dir)
    organize_trainO(trainOr, args.out_dir)
    organize_trainG(trainGen, args.out_dir)
    organize_val(val)
    organize_test(test)
    # generate(original, args.num_im, args.out_dir)
# trainOr = load_txt('./Data_Symbols/training_index.txt', './Data_Symbols/png_trainingSymbols', 'iso')
# trainGen = load_txt('./Data_Symbols/generated_index.txt', './Data_Symbols/png_generatedSymbols', 'iso')
# val = load_txt('./Data_Symbols/val_index.txt', './Data_Symbols/png_valSymbols', 'BOTH')
# test = load_txt('./Data_Symbols/test_index.txt', './Data_Symbols/png_testSymbols', 'iso')
# trainJunk = load_txt('./Data_Symbols/junk_index.txt', './Data_Symbols/png_junkSymbols', 'junk')
# out_dir = './prueba'
# if osp.exists(out_dir):
#     os.system('rm -rf ' + out_dir)
# d
# os.mkdir(out_dir)


if osp.exists(out_dir):
    os.system('rm -rf ' + out_dir)

os.mkdir(out_dir)
organize_train(trainOr, trainGen, trainJunk, out_dir)
organize_val(val)
organize_test(test)

if __name__ == '__main__':
    main()
