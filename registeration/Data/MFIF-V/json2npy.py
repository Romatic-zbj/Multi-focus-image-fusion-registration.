import json
import os
import numpy as np
import re
import glob

file_list = os.listdir("./estimate-V2")
file_list = sorted(file_list,key=lambda x:(int(re.sub('\D', '', x)), x))
num=0
print(len(file_list))
for file in file_list:
    img_list = [x for x in sorted(glob.glob("./estimate-V2/"+file+"/*.jpg",recursive=True))]
    # print(img_list)
    json_path = glob.glob("./estimate-V2/"+file+"/*.json",recursive=True)
    # print(json_path)
    npy_path = "../Coordinate-V4/"
    for img_path in img_list:
        # print(img_path)
        if img_path[-5] == 't':
            name1 = img_path
            name1 = img_path.split('\\')[-1]
            name1, _ = os.path.splitext(name1)
            name1 = name1.split("_")[0]
            name1 = name1+".jpg"
        else:
            name2 = img_path.split('\\')[-1]
    if num<3:
        name1 = "V-1_" + name1
        name2 = "V-1_" + name2
    if 3 <= num < 5:
        name1 = "V-2_" + name1
        name2 = "V-2_" + name2
    if 5 <= num < 7:
        name1 = "V-4_" + name1
        name2 = "V-4_" + name2
    if 7 <= num < 10:
        name1 = "V-5_" + name1
        name2 = "V-5_" + name2
    if 80 <= num < 100:
        name1 = "V-5_" + name1
        name2 = "V-5_" + name2
    if 100 <= num < 120:
        name1 = "V-6_" + name1
        name2 = "V-6_" + name2
    print(file)
    with open(json_path[0], 'r') as f:
        json_data = json.load(f)
    source_pic_name = (json_data['Template']['Path'].split('/')[-1]).split('\\')[-1]
    aligned_pic_name = (json_data['Sample']['Path'].split('/')[-1]).split('\\')[-1]
    match_pts = []
    for i in range(6):
        match_pts.append([tuple(json_data['Template']['Points'][i]), tuple(json_data['Sample']['Points'][i])])
    new_data = {'path1': name1, 'path2': name2, 'matche_pts': match_pts}
    np.save("../Coordinate-V4/" + name1 + '_' + name2 + '.npy', np.array(new_data))
    num = num + 1