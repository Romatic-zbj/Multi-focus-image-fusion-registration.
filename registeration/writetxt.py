import os
import glob
from pathlib import Path
import random

def getPath():
    '''

    :param path1:
    :return: list of path with
    '''
    imgList = [str(x)[14:] for x in sorted(glob.glob("./Data/MFIF-V/V-1/*.jpg",recursive=True),)]
    matchList = []

    for i in range(10):
        for j in range(len(imgList)):
            diff = random.randint(1, 20)
            temp = imgList[j]+" "+imgList[(j+diff) % len(imgList)]
            matchList.append(temp)
    # print(matchList)
    return matchList
def write_strings_to_file(strings, filename):

    with open(filename, 'a') as f:
        for s in strings:
            match1,match2 = s.split(" ")
            pre1, name1 = match1.split("\\")
            pre2,name2 = match2.split("\\")
            s = pre1 + r"\\" + name1 + " " + pre2 + r"\\" + name2
            f.write(s +'\n')
def write_test(filename):
    imgList = [str(x)[14:] for x in sorted(glob.glob("./Data/MFIF-V/V-6/*.jpg", recursive=True), )]
    matchList = []
    index = random.sample(range(0, len(imgList)),20)#在图片列表中随机选取10张图片的下标用于测试

    for j in index:
        diff = random.randint(1, 5)
        temp = imgList[j] + " " + imgList[(j + diff) % len(imgList)]
        matchList.append(temp)
    strings = matchList
    with open(filename, 'a') as f:
        for s in strings:
            match1,match2 = s.split(" ")
            pre1, name1 = match1.split("\\")
            pre2,name2 = match2.split("\\")
            s = pre1 + r"\\" + name1 + " " + pre2 + r"\\" + name2
            f.write(s +'\n')

if __name__ == '__main__':
    path = "./Data/MFIF-V/"
    # path = Path(path)
    #
    strings = getPath()
    #
    filename = 'trainMicro_06.txt'
    write_strings_to_file(strings, filename)
    # write_test(filename)

