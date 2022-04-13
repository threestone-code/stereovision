import os
import re

import cv2
import numpy as np
import sys


def generate_image_list(data_dir):
    """Generate absolute path list of images for 'FlyingThings3D -Sceneflow Dataset'"""
    left_data_dir = os.path.join(data_dir, 'image_2')
    right_data_dir = os.path.join(data_dir, 'image_3')
    label_data_dir = os.path.join(data_dir, 'disp_noc_0')

    left_data_files = [os.path.join(left_data_dir, img) for img in os.listdir(left_data_dir) if img.find('_10') > -1]
    right_data_files = [os.path.join(right_data_dir, img) for img in os.listdir(right_data_dir) if img.find('_10') > -1]
    label_files = [os.path.join(label_data_dir, img) for img in os.listdir(label_data_dir) if img.find('_10') > -1]

    return left_data_files, right_data_files, label_files


def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale


if __name__ == '__main__':
    # generate_image_list(r'D:\Users\yzl\Documents\pycharm-workplace\KITTIStereo2015_data_scene_flow\training')

    # fmt = '{:06}_10.png'
    # for i in range(100):
    #     print(fmt.format(i))

    # print('abcd.10_png'.find('10_'))

    # with open(r'D:\Users\yzl\Documents\pycharm-workplace\github_code\Sampler\FlyingThings3D\disparity\0006.pfm', 'rb') as pfmFile:
    #     header = pfmFile.readline().decode().rstrip()
    #     channels = 3 if header == 'PF' else 1
    #     h, w = pfmFile.readline().decode().rstrip().split(' ')
    #     h = int(h)
    #     w = int(w)
    #     shape = (h, w, 3) if channels == 3 else (h, w)
    #     scale = float(pfmFile.readline().decode().rstrip())
    #     if scale < 0:
    #         endian = '<'
    #         scale = -scale
    #     else:
    #         endian = '>'
    #     # print('channels:\t', channels)
    #     # print('shape:\t', shape)
    #     # print('scale:\t', scale)
    #
    #     data = np.fromfile(pfmFile, endian + 'f')
    #     data = np.reshape(data, shape)
    #     data = np.flipud(data)
    #     # print(data, scale)

    data, scale = readPFM(r'D:\Users\yzl\Documents\pycharm-workplace\github_code\Sampler\FlyingThings3D\disparity\0006.pfm')
    img = np.flipud(data).astype('uint8')
    cv2.imshow('a', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


    print('End.')

