
import torch.utils.data as data
from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def dataloader(filepath):
    r"""
    数据集的目录格式：
        monkaa_frames_cleanpass_webp
            |——class1
                |——left
                |——right
            |——class2
                ...
            。。。
        monkaa_disparity
            |——class1
                    |——left
                    |——right
        driving_frames_cleanpass_webp
            ...
        driving_disparity
            ...
        flyingthings3d_frames_cleanpass_webp
            ...
        flyingthings3d_disparity
            ...
    :param filepath: 数据集所在目录
    :return: 数据路径列表
    """
    # 返回参数
    all_left_img   = []
    all_right_img  = []
    all_left_disp  = []
    test_left_img  = []
    test_right_img = []
    test_left_disp = []

    classes = [d for d in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, d))]  # 3类数据集
    image   = [img for img in classes if img.find('frames_cleanpass') > -1]
    disp    = [dsp for dsp in classes if dsp.find('disparity') > -1]

    # monkaa数据集
    monkaa_path = filepath + [x for x in image if 'monkaa' in x][0]
    monkaa_disp = filepath + [x for x in disp if 'monkaa' in x][0]
    monkaa_dir  = os.listdir(monkaa_path)  # monkaa数据集的子类别

    i = 0
    j = 0
    for dd in monkaa_dir:
      for im in os.listdir(monkaa_path+'/'+dd+'/left/'):
        if is_image_file(monkaa_path+'/'+dd+'/left/'+im):
          if i % 2 == 0:  # 新增拆分为训练集和测试集
              all_left_img.append(monkaa_path+'/'+dd+'/left/'+im)
              all_left_disp.append(monkaa_disp+'/'+dd+'/left/'+im.split('.')[0]+'.pfm')
          else:
              test_left_img.append(monkaa_path+'/'+dd+'/left/'+im)
              test_left_disp.append(monkaa_disp + '/' + dd + '/left/' + im.split('.')[0] + '.pfm')
          i += 1
      for im in os.listdir(monkaa_path+'/'+dd+'/right/'):
        if is_image_file(monkaa_path+'/'+dd+'/right/'+im):
            if j % 2 == 0:
                all_right_img.append(monkaa_path+'/'+dd+'/right/'+im)
            else:
                test_right_img.append(monkaa_path+'/'+dd+'/right/'+im)
            j += 1

    # # flyingthings3d数据集
    # flying_path = filepath + [x for x in image if x == 'frames_cleanpass'][0]
    # flying_disp = filepath + [x for x in disp if x == 'frames_disparity'][0]
    # flying_dir = flying_path+'/TRAIN/'
    # subdir = ['A','B','C']
    #
    # for ss in subdir:
    #   flying = os.listdir(flying_dir+ss)
    #
    #   for ff in flying:
    #     imm_l = os.listdir(flying_dir+ss+'/'+ff+'/left/')
    #     for im in imm_l:
    #       if is_image_file(flying_dir+ss+'/'+ff+'/left/'+im):
    #         all_left_img.append(flying_dir+ss+'/'+ff+'/left/'+im)
    #
    #       all_left_disp.append(flying_disp+'/TRAIN/'+ss+'/'+ff+'/left/'+im.split(".")[0]+'.pfm')
    #
    #       if is_image_file(flying_dir+ss+'/'+ff+'/right/'+im):
    #         all_right_img.append(flying_dir+ss+'/'+ff+'/right/'+im)
    #
    # flying_dir = flying_path+'/TEST/'
    #
    # subdir = ['A','B','C']
    #
    # for ss in subdir:
    #   flying = os.listdir(flying_dir+ss)
    #
    #   for ff in flying:
    #     imm_l = os.listdir(flying_dir+ss+'/'+ff+'/left/')
    #     for im in imm_l:
    #       if is_image_file(flying_dir+ss+'/'+ff+'/left/'+im):
    #         test_left_img.append(flying_dir+ss+'/'+ff+'/left/'+im)
    #
    #       test_left_disp.append(flying_disp+'/TEST/'+ss+'/'+ff+'/left/'+im.split(".")[0]+'.pfm')
    #
    #       if is_image_file(flying_dir+ss+'/'+ff+'/right/'+im):
    #         test_right_img.append(flying_dir+ss+'/'+ff+'/right/'+im)
    #
    # # driving数据集
    # driving_dir = filepath + [x for x in image if 'driving' in x][0] + '/'
    # driving_disp = filepath + [x for x in disp if 'driving' in x][0]
    #
    # subdir1 = ['35mm_focallength', '15mm_focallength']
    # subdir2 = ['scene_backwards', 'scene_forwards']
    # subdir3 = ['fast', 'slow']
    #
    # for i in subdir1:
    #   for j in subdir2:
    #     for k in subdir3:
    #         imm_l = os.listdir(driving_dir+i+'/'+j+'/'+k+'/left/')
    #         for im in imm_l:
    #           if is_image_file(driving_dir+i+'/'+j+'/'+k+'/left/'+im):
    #             all_left_img.append(driving_dir+i+'/'+j+'/'+k+'/left/'+im)
    #
    #           all_left_disp.append(driving_disp+'/'+i+'/'+j+'/'+k+'/left/'+im.split(".")[0]+'.pfm')
    #
    #           if is_image_file(driving_dir+i+'/'+j+'/'+k+'/right/'+im):
    #             all_right_img.append(driving_dir+i+'/'+j+'/'+k+'/right/'+im)

    return all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp


