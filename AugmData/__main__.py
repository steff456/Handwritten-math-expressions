# -*- coding: utf-8 -*-
# !/usr/bin/env python3

"""Main for the generation of new inkml"""
import argparse
import logging
import coloredlogs

from AugmData.new_inkml import generate

LOG_FORMAT = ('%(levelname) -10s %(asctime)s %(name) -30s %(funcName) '
              '-35s %(lineno) -5d: %(message)s')
LOGGER = logging.getLogger(__name__)
coloredlogs.install(level='info')

parser = argparse.ArgumentParser(
    description='Increase number of images')

parser.add_argument('--directory', type=str, default='./trainingSymbols',
                    help='directory of the original inkml files')
parser.add_argument('--index-txt', type=str, default='./training_index.txt',
                    help='file with annotations')
parser.add_argument('--num-im', type=int, default=5,
                    help='number of new images')
parser.add_argument('--out-dir', type=str, default='./generatedSymbols',
                    help='directory of the generated inkml files')

args = parser.parse_args()

def load_txt(file, directory):
    '''Loads the routes of the existing inkml files'''
    with open(file) as f:
        a = f.readlines()
    d = {}
    for line in a:
        line = line.strip()
        parts = line.split(',')
        data = parts[0].split('_')
        name = data[-1]
        if parts[1] in d:
            d[parts[1]].append(directory + '/' + 'iso' + name + '.inkml')
        else:
            d[parts[1]] = [directory + '/' + 'iso' + name + '.inkml']

    return d


def main():
    print('*** Loading original txt ***')
    original = load_txt(args.index_txt, args.directory)
    print('*** Generating ***')
    generate(original, args.num_im, args.out_dir)


if __name__ == '__main__':
    main()
