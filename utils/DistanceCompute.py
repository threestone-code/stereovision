import math
import os.path

import cv2
import numpy as np
from PIL import Image

"""
功能：输入左图及对应的视差图，相机内参及基线距离，目标检测框坐标，输出距离标注图
obj_3d = [[x1, y1, x2, y2, z], [x1, y1, x2, y2, z], [x1, y1, x2, y2, z],...] 相机坐标
boxes = [[x1, y1, x2, y2], [x1, y1, x2, y2], [x1, y1, x2, y2],...] 像素坐标
len(obj_3d) == len(boxes) == 检测到的目标数
"""
# 相机内参
fx = 721.5377  # x轴方向焦距/像素
fy = 721.5377  # y轴方向焦距/像素
u0 = 609.5593  # 光轴与像素平面的焦点(u0, v0)
v0 = 172.8540
baseline = 0.54 * 1000
# 目标检测
boxes = [[6, 173, 119, 254],
         [754, 150, 915, 213],
         [1075, 163, 1228, 250]]


def compute_distance(point1, point2):
    # [x1, y1, x2, y2, z] x1 < x2, y1 < y2
    # 比较x方向是否有重叠
    if point1[2] < point2[0]:
        x1, x2 = point1[2], point2[0]
    elif point1[0] > point2[2]:
        x1, x2 = point1[0], point2[2]
    else:
        x1 = x2 = 0
    # 比较y方向是否有重叠
    if point1[3] < point2[1]:
        y1, y2 = point1[3], point2[1]
    elif point1[1] > point2[3]:
        y1, y2 = point1[1], point2[3]
    else:
        y1 = y2 = 0
    z1, z2 = point1[4], point2[4]
    distance = math.sqrt(math.pow(x1-x2, 2) + math.pow(y1-y2, 2) + math.pow(z1-z2, 2))
    # print('point1: ', x1, y1, z1)
    # print('point2: ', x2, y2, z2)
    # print('distance: ', distance)

    return distance


def coordinate_p2c(disp_map, boxes):
    """
    boxes:[[x1,y1,x2,y2]，[x1,y1,x2,y2]，[x1,y1,x2,y2]...]
    """
    obj_3d = []
    for box in boxes:
        # 1、识别框内视差值的中位数
        box_d = float(np.median(disp_map[box[1]:box[3], box[0]:box[2]])) / 256.0
        # print('disp is \t', box_d)
        # print('depth is \t', fx * baseline / box_d)
        z = fx * baseline / box_d  # 视差值——》深度值，单位mm
        # 像素坐标变相机坐标，单位mm
        x1 = (box[0] - u0) * z / fx
        y1 = (box[1] - v0) * z / fx
        x2 = (box[2] - u0) * z / fx
        y2 = (box[3] - v0) * z / fx
        obj_3d.append([x1, y1, x2, y2, z])
        # print([x1, y1, x2, y2, z])

    # for i in range(len(obj_3d)):
        # print("object‘s 3D coordinate is:\t", obj_3d[i])
    return obj_3d


def imshow_distance(img, boxes, obj_3d, target=None):
    """target：指定参照物"""
    assert len(boxes) == len(obj_3d)
    font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体

    for i in range(len(boxes)):
        # 画目标框
        cv2.rectangle(img, (boxes[i][0], boxes[i][1]), (boxes[i][2], boxes[i][3]), color=(255, 0, 255), thickness=2)
        # 写识别的物体的名称/标签和序号
        # cv2.putText(img, 'car ' + str(i+1), (boxes[i][0], boxes[i][1]), font, 0.5, (0, 255, 0), thickness=1)

    # 计算并显示距离
    if target and isinstance(target, int):  # 目标序号就是目标检测识别出的顺序
        for i in range(len(boxes)):
            if target == (i+1):
                continue
            print('obj index: ', target-1, i)

            # 文字坐标
            text_coor = (abs(boxes[target-1][0] - boxes[i][0]) // 2 + min(boxes[target-1][0], boxes[i][0]),
                         abs(boxes[target-1][1] - boxes[i][1]) // 2 + min(boxes[target-1][1], boxes[i][1]))
            # print('text coordinate:\t', text_coor)
            # 画线
            cv2.line(img, (boxes[target-1][0], boxes[target-1][1]), (boxes[i][0], boxes[i][1]), color=(0, 255, 0))
            # print('object distance:\t', compute_distance(obj_3dcoor[target-1], obj_3dcoor[i]), '\n')

            distance = compute_distance(obj_3d[target - 1], obj_3d[i])
            # 显示距离，变换单位米
            cv2.putText(img, f'dis{target, i+1}= ' + str(np.around(distance/1000, 2)) + 'm',
                        text_coor, font, 0.75, (0, 0, 255), 2)
    else:
        for i in range(len(obj_3d)-1):
            for j in range(len(obj_3d)-i-1):
                print('obj index: ', i, j+i+1)
                # 文字坐标
                text_coor = (abs(boxes[i][0] - boxes[j+i+1][0]) // 2 + min(boxes[i][0], boxes[j+i+1][0]),
                             abs(boxes[i][1] - boxes[j+i+1][1]) // 2 + min(boxes[i][1], boxes[j+i+1][1]))

                cv2.line(img, (boxes[i][0], boxes[i][1]), (boxes[j+i+1][0], boxes[j+i+1][1]), color=(0, 255, 0))

                distance = compute_distance(obj_3d[i], obj_3d[j+i+1])
                cv2.putText(img, f'dis{i+1, j+i+2}=' + str(np.around(distance/1000, 2)) + 'm',
                            text_coor, font, 0.75, (0, 0, 255), 2)

    cv2.imwrite('resultTrueDepth4.png', img)


def convertPNG(pngfile, outdir):
    # READ THE DEPTH
    im_depth = cv2.imread(pngfile, cv2.IMREAD_UNCHANGED)
    # apply colormap on depth image(image must be converted to 8-bit per pixel first)
    im_color = cv2.applyColorMap(cv2.convertScaleAbs(im_depth, alpha=15), cv2.COLORMAP_JET)
    # convert to mat png
    im = Image.fromarray(im_color)
    # save image
    im.save(os.path.join(outdir, os.path.basename(pngfile)))


if __name__ == '__main__':
    disp_map = cv2.imread("./pre_depth_map.png", flags=cv2.IMREAD_UNCHANGED)
    # print('depth map‘s type:', disp_map.dtype, disp_map.shape, disp_map.max())

    left_img = cv2.imread("./000022_10l.png", flags=cv2.IMREAD_UNCHANGED)
    # cv2.imshow('left_img', left_img)
    # cv2.waitKey(0)

    # 计算目标的相机坐标
    obj_3d = coordinate_p2c(disp_map, boxes)

    # 计算目标距离并绘制图像
    # imshow_distance(left_img, boxes, obj_3d, target=3)
    imshow_distance(left_img, boxes, obj_3d)
