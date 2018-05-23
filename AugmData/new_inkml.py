# -*- coding: utf-8 -*-
import math
import random
import xml.etree.ElementTree as ET
# import matplotlib.pyplot as plt
# import numpy as np


def new_coordinates(route):
    ''' Calculate new coordinates from inkml of the route '''
    NS = {'ns': 'http://www.w3.org/2003/InkML',
          'xml': 'http://www.w3.org/XML/1998/namespace'}
    # charge image
    tree = ET.parse(route)
    root = tree.getroot()
    # find id and annotation
    a = root.findall('ns:annotation', namespaces=NS)
    a
    # find traces
    b = root.findall('ns:trace', namespaces=NS)
    strokes = []
    coord_strokes = {}
    # check all strokes
    # rand_x = random.random()*3
    # rand_y = random.random()*3
    for i in range(0, len(b)):
        strokes.append(b[i].text.strip())
        coord = strokes[-1].split(',')
        new_coord = ''
        first = True
        # generate new pairs
        for pair in coord:
            pair = pair.strip()
            pair = pair.split(' ')
            x = float(pair[0])
            y = float(pair[1])
            rand_x = random.random()/600.0
            # rand_x = 0.01
            nx = x + rand_x
            rand_y = random.random()/600.0
            # rand_y = 0.01
            ny = y + rand_y
            # print(str(x) + ';' + str(nx))
            # print(str(y) + ';' + str(ny))
            # print('---')
            strx = str(nx)
            stry = str(ny)
            if first:
                new_coord = strx + ' ' + stry
                first = False
            else:
                new_coord = new_coord + ', ' + strx + ' ' + stry
        coord_strokes[i] = new_coord

    return coord_strokes


def writeInkML(org_inkml, n_coord, out_dir, count):
    '''Write the generated inkml from the original inkml'''
    name = out_dir + '/iso' + str(count) + '.inkml'
    outputfile = open(name, 'w')

    with open(org_inkml) as f:
        a = f.readlines()

    change = False
    count = 0
    for line in a:
        if line.startswith('<trace id='):
            change = True
            # print(line)
            outputfile.write(line)
            outputfile.write(n_coord[count] + '\n')
            # print(n_coord[count])
            count += 1
        elif change:
            change = False
        else:
            # print(line)
            outputfile.write(line)

    outputfile.close()
    return name


def writeLabels(route, label, name, count):
    '''Write in a txt the label of the generated inkml'''
    with open('./test.txt', "a") as f:
        f.write(name + "_" + str(count) + ", " + label + '\n')


def generate(or_dict, num_im, out_directory):
    '''Function that generates the desired number of images per class'''
    n_dict = or_dict
    count = 0
    for key in or_dict.keys():
        print(key)
        in_count = 0
        while(len(n_dict[key]) < num_im):
            i = in_count % len(or_dict[key])
            inkml_act = or_dict[key][i]
            # print(inkml_act)
            n_coord = new_coordinates(inkml_act)
            name = writeInkML(inkml_act, n_coord, out_directory, count)
            writeLabels(out_directory, key, name, count)
            n_dict[key].append(name)
            in_count += 1
            count += 1
