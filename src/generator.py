import os

print(os.getcwd())
with open('data/data.txt', 'w') as fileout:
    for i in os.listdir('data/label'):
        name = 'data/image/' + i[:-3] + 'jpg'
        with open('data/label/' + i, 'r') as filein:
            x1, y1, x2, y2 = map(int, filein.readline().split(','))
            filein.close()
        mid_x = int((x1 + x2) / 2)
        mid_y = int((y1 + y2) / 2)
        len_x = x2 - x1
        len_y = y2 - y1
        ave = int((len_x + len_y) / 2)
        r = mid_x - ave // 2
        c = mid_y - ave // 2
        h = ave
        w = ave
        string = "{} {} {} {}".format(r, c, h, w)
        fileout.write(name + '\n')
        fileout.write(str(1) + '\n')
        fileout.write(string + '\n')
