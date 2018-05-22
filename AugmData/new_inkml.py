import xml.etree.ElementTree as ET
# import matplotlib.pyplot as plt
# import numpy as np


def new_inkml(route='./trainingSymbols/iso4.inkml'):
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
    for i in range(0, len(b)):
        strokes.append(b[i].text.strip())
        coord = strokes[-1].split(',')
        xcoord = []
        ycoord = []
        for pair in coord:
            pair = pair.strip()
            pair = pair.split(' ')
            x = pair[0]
            y = pair[1]
            xcoord.append(int(x))
            ycoord.append(int(y))
            print(x)
            print(y)
            print('---')
        coord_strokes[i] = [xcoord, ycoord]

    # plt.imshow(f_arr, cmap=plt.get_cmap('gray'))
    # plt.show()


def generate(or_dict, num_im=7940):
    '''Function that generates the desired number of images per class'''
    n_dict = or_dict
    for key in n_dict.keys():
        count = 0
        while(len(n_dict[key]) < num_im):
            i = count % len(or_dict[key])
            inkml_act = or_dict[key][i]
            n = new_inkml(inkml_act)
            n_dict[key].append(n)
            count += 1
