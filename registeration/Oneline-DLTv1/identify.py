import numpy as np
import cv2 as cv
import os

def geometricDistance(correspondence, h):
    """
    Correspondence err
    :param correspondence: Coordinate，包含x,y,z信息，可能没有z
    :param h: Homography
    :return: L2 distance
    """

    p1 = np.transpose(np.matrix([correspondence[0][0], correspondence[0][1], 1]))# 将由坐标形成的矩阵转置
    estimatep2 = np.dot(h, p1)
    estimatep2 = (1/estimatep2.item(2))*estimatep2

    p2 = np.transpose(np.matrix([correspondence[1][0], correspondence[1][1], 1]))
    error = p2 - estimatep2
    return np.linalg.norm(error)

RE = ['V-1']
LT = ['V-2']
LL = ['V-3']
SF = ['V-4']
LF = ['V-5']
DF = ['V-6']

MSE_RE = []
MSE_LT = []
MSE_LL = []
MSE_SF = []
MSE_LF = []
MSE_DF = []

result_files = "./tranditional_res"
result_txt = "result_ours_exp_identify.txt"
res_txt = os.path.join(result_files, result_txt)
f = open(res_txt, "w")
npy_path = "../Data/Coordinate-V3/"
pair_list = list(open("../Data/testMicro_06.txt"))
i=0
for pair in pair_list:
    img_path1, img_path2 = pair.split(' ')
    print(img_path1[5:])

    video_name = img_path1.split('\\')[0]
    npy_name = video_name + "_" + img_path1[5:] + '_' + video_name + "_" + img_path2[5:-1] + '.npy'
    npy_id = npy_path + npy_name

    img1 = cv.imread("../Data/Test/"+img_path1)
    img1 = cv.resize(img1, (640, 360))
    img2 = cv.imread("../Data/Test/"+img_path2[:-1])
    img2 = cv.resize(img2, (640, 360))

    M = np.array([[1,0,0],[0,1,0],[0, 0, 1]])
    point_dic = np.load(npy_id, allow_pickle=True)
    data = point_dic.item()
    err_img = 0.0
    for j in range(6):
        points_LR = data['matche_pts'][j]
        points_RL = [points_LR[1], points_LR[0]]

        err_LR = geometricDistance(points_LR,
                                   M)  # because of the order of the Coordinate of img_A and img_B is inconsistent
        err_RL = geometricDistance(points_RL, M)  # the data annotator has no fixed left or right when labelling

        err = min(err_LR, err_RL)
        err_img += err

    err_avg = err_img / 6
    name = "0" * (8 - len(str(i))) + str(i)
    line = name + ":" + str(err_avg) + "\n"

    f.write(line)
    print("{}:{}".format(i, err_avg))
    if video_name in RE:
        MSE_RE.append(err_avg)
    elif video_name in LT:
        MSE_LT.append(err_avg)
    elif video_name in LL:
        MSE_LL.append(err_avg)
    elif video_name in SF:
        MSE_SF.append(err_avg)
    elif video_name in LF:
        MSE_LF.append(err_avg)
    elif video_name in DF:
        MSE_DF.append(err_avg)
    i = i + 1
MSE_RE_avg = np.mean(MSE_RE)
MSE_LT_avg = np.mean(MSE_LT)
MSE_LL_avg = np.mean(MSE_LL)
MSE_SF_avg = np.mean(MSE_SF)
MSE_LF_avg = np.mean(MSE_LF)
MSE_DF_avg = np.mean(MSE_DF)
res = {'RE': MSE_RE_avg, 'LT': MSE_LT_avg, 'LL': MSE_LL_avg, 'SF': MSE_SF_avg, 'LF': MSE_LF_avg, 'DF': MSE_DF_avg}
print(res)
f.write(str(res))