from __future__ import print_function
import argparse
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import time
from PIL import Image
from models.dispnet import dispnets, dispnetc
from models.dispnet import DispNet

# 2012 data KITTIStereo2015_data_scene_flow/testing

device = torch.device('cpu')

parser = argparse.ArgumentParser(description='DispNet')
parser.add_argument('--KITTI', default='2015',
                    help='KITTI version')
parser.add_argument('--datapath', default='KITTIStereo2015_data_scene_flow/testing',
                    help='select model')
parser.add_argument('--loadmodel', default='./pretrained/checkpoint_0.tar',
                    help='loading model')
parser.add_argument('--leftimg', default='./testImg/kitti2015/000000_10l.png',
                    help='load model')
parser.add_argument('--rightimg', default='./testImg/kitti2015/000000_10r.png',
                    help='load model')
parser.add_argument('--model', default='dispnets',
                    help='select model')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.model == 'dispnets':
    model = DispNet.DispNetS()
elif args.model == 'dispnetc':
    model = DispNet.DispNetC(args.maxdisp)
else:
    print('no model')

# model = nn.DataParallel(model, device_ids=[0])
# model.cuda()
model.to(device)

if args.loadmodel is not None:
    print('load DispNet')
    state_dict = torch.load(args.loadmodel, map_location=device)
    model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


def test(imgL, imgR):
    model.eval()

    if args.cuda:
        imgL = imgL.to(device)
        imgR = imgR.to(device)

    with torch.no_grad():
        disp = model(imgL, imgR)

    disp = torch.squeeze(disp)
    pred_disp = disp.data.cpu().numpy()

    return pred_disp


def main():
    normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]}
    infer_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(**normal_mean_var)])

    imgL_o = Image.open(args.leftimg).convert('RGB')
    imgR_o = Image.open(args.rightimg).convert('RGB')

    imgL = infer_transform(imgL_o)
    imgR = infer_transform(imgR_o)

    # pad to width and hight to 64 times
    if imgL.shape[1] % 64 != 0:
        times = imgL.shape[1] // 64
        top_pad = (times + 1) * 64 - imgL.shape[1]
    else:
        top_pad = 0

    if imgL.shape[2] % 64 != 0:
        times = imgL.shape[2] // 64
        right_pad = (times + 1) * 64 - imgL.shape[2]
    else:
        right_pad = 0

    imgL = F.pad(imgL, (0, right_pad, top_pad, 0)).unsqueeze(0)
    imgR = F.pad(imgR, (0, right_pad, top_pad, 0)).unsqueeze(0)

    start_time = time.time()
    pred_disp = test(imgL, imgR)
    print('time = %.2f' % (time.time() - start_time))

    if top_pad != 0 and right_pad != 0:
        img = pred_disp[top_pad:, :-right_pad]
    elif top_pad == 0 and right_pad != 0:
        img = pred_disp[:, :-right_pad]
    elif top_pad != 0 and right_pad == 0:
        img = pred_disp[top_pad:, :]
    else:
        img = pred_disp

    img = (img * 256).astype('uint16')
    img = Image.fromarray(img)
    img.save('./testImg/kitti2015/test.png')


if __name__ == '__main__':
    main()







