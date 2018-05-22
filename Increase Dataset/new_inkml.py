import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np

NS = {'ns': 'http://www.w3.org/2003/InkML',
      'xml': 'http://www.w3.org/XML/1998/namespace'}

# charge image
tree = ET.parse('./trainingSymbols/iso4.inkml')
root = tree.getroot()
# find id and annotation
a = root.findall('ns:annotation', namespaces=NS)

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

# find size of final image
min_x = 100000
min_y = 100000
max_x = 0
max_y = 0
for i in coord_strokes:
    # plt.scatter(xcoord, ycoord)
    xcoord = coord_strokes[i][0]
    ycoord = coord_strokes[i][1]
    if np.min(xcoord) < min_x:
        min_x = np.min(xcoord)
    if np.min(ycoord) < min_y:
        min_y = np.min(ycoord)
    if np.max(xcoord) > max_x:
        max_x = np.max(xcoord)
    if np.max(ycoord) > max_y:
        max_y = np.max(ycoord)
    # plt.plot(xcoord, ycoord, '-o')

diff_x = max_x - min_x
diff_y = max_y - min_y
f_arr = np.ones((diff_y+1, diff_x+1))

# generate image from coordinates
for i in coord_strokes:
    xcoord = coord_strokes[i][0]
    ycoord = coord_strokes[i][1]
    xant = 0
    yant = 0
    bol = False
    for x, y in zip(xcoord, ycoord):
        xp = x-min_x
        yp = y-min_y
        f_arr[yp][xp] = 0
        if not bol:
            if xant <= xp and yant <= yp:
                f_arr[yant:yp+1][xant:xp+1] = 0
            if xant <= xp and yant >= yp:
                f_arr[yp:yant+1][xant:xp+1] = 0
            if xant >= xp and yant <= yp:
                f_arr[yant:yp+1][xp:xant+1] = 0
            if xant >= xp and yant >= yp:
                f_arr[yp:yant+1][xp:xant+1] = 0
            print("ant: " + str(xant) + ";" + str(yant))
            print("act: " + str(xp) + ";" + str(yp))
        xant = xp
        yant = yp
        bol = False

plt.imshow(f_arr, cmap=plt.get_cmap('gray'))
plt.show()

file = 'trainingSymbols/iso_GT.txt'
with open(file) as f:
    a = f.readlines()

d = {}
for line in a:
    line = line.strip()
    parts = line.split(',')
    if parts[1] in d:
        d[parts[1]].append(parts[0])
    else:
        d[parts[1]] = [parts[0]]

total = 0
for key in d.keys():
    total += len(d[key])
    print(key + ';' + str(len(d[key])))
