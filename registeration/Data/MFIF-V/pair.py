import os
import cv2

pair_list = list(open("./testMicro_06.txt"))
base_path = os.getcwd()
i=0
for pair in pair_list:
    img_path1, img_path2 = pair.split(' ')
    #将其resize，移动

    img1 = cv2.imread(img_path1)
    img1 = cv2.resize(img1, (640,360))
    img2 = cv2.imread(img_path2[:-1])
    img2 = cv2.resize(img2,(640,360))
    pair_path = "./Pair/pair{}".format(i)
    if not os.path.exists(pair_path):
        os.makedirs(pair_path)
    img_name1 = img_path1[5:]
    img_name2 = img_path2[5:-1]

    img_name1, _ = os.path.splitext(img_name1)
    img_name1 = img_name1 + '_t.jpg'
    cv2.imwrite(os.path.join(pair_path,img_name1),img1)
    cv2.imwrite(os.path.join(pair_path,img_name2),img2)
    i=i+1

