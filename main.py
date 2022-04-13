import argparse
import time
import torch
import torch.nn.functional as F
from torch import optim
from models.dispnet import DispNet
from utils.dataloader import sceneflowlist, SceneFlowdataLoader

# 命令行参数
parser = argparse.ArgumentParser(description='DispNet')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maximum disparity')
parser.add_argument('--model', default='dispnetc',
                    help='select model')
# parser.add_argument('--datapath', default='E:/KITTIStereo2015_data_scene_flow/training/',
parser.add_argument('--datapath', default='E:\scene_flow_dataset/',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=6,
                    help='batch_size')
parser.add_argument('--loadmodel', default=None,
                    help='load model')
parser.add_argument('--savemodel', default='./pretrained/',
                    help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()

# 启用GPU
args.cuda = args.no_cuda and torch.cuda.is_available()
if args.cuda:
    device = torch.device("gpu")
else:
    device = torch.device("cpu")
# 随机数设置
if args.cuda:
    torch.cuda.manual_seed(args.seed)
else:
    torch.manual_seed(args.seed)

# 1、数据迭代器
# KITTI 数据集
# all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = k2015.dataloader(args.datapath)
# trainloader = torch.utils.data.DataLoader(
#          KITTILoader.myImageFloder(all_left_img, all_right_img, all_left_disp, True),
#          batch_size=12, shuffle=True, num_workers=4, drop_last=False)
#
# testloader = torch.utils.data.DataLoader(
#          KITTILoader.myImageFloder(test_left_img, test_right_img, test_left_disp, False),
#          batch_size=8, shuffle=False, num_workers=4, drop_last=False)

# SceneFlow数据集
all_left_img, all_right_img, all_left_disp, \
test_left_img, test_right_img, test_left_disp = sceneflowlist.dataloader(args.datapath)
print('batch num:\t', len(all_left_disp) // 12)

trainloader = torch.utils.data.DataLoader(
         SceneFlowdataLoader.myImageFloder(all_left_img, all_right_img, all_left_disp, True),
         batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False)

testloader = torch.utils.data.DataLoader(
         SceneFlowdataLoader.myImageFloder(test_left_img, test_right_img, test_left_disp, False),
         batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)

# 2、新建/加载模型
if args.model == 'dispnets':
    model = DispNet.DispNetS()
elif args.model == 'dispnetc':
    model = DispNet.DispNetC(args.maxdisp)
else:
    print('no model')
# 多卡训练
# if args.cuda:
#     model = nn.DataParallel(model)
#     model.to(device)
# 加载预训练模型
if args.loadmodel is not None:
    print('Load pretrained model')
    pretrain_dict = torch.load(args.loadmodel)
    model.load_state_dict(pretrain_dict['state_dict'])
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

# 3、优化器
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))


# 4、训练函数
def train_dispnet(imgL, imgR, disp_L):
    model.train()
    imgL, imgR, disp_true = imgL.to(device), imgR.to(device), disp_L.to(device)
    # 视差图掩膜，过滤掉不在视差范围内的深度值
    mask = disp_true < args.maxdisp
    mask.detach_()  # 不计算梯度
    # 每次计算梯度前，将上一次梯度置零
    optimizer.zero_grad()

    if args.model in ['dispnets', 'dispnetc']:
        pre1, pre2, pre3, pre4, pre5, pre6 = model(imgL, imgR)
        pre1 = torch.squeeze(pre1, 1)
        pre2 = torch.squeeze(pre2, 1)
        pre3 = torch.squeeze(pre3, 1)
        pre4 = torch.squeeze(pre4, 1)
        pre5 = torch.squeeze(pre5, 1)
        pre6 = torch.squeeze(pre6, 1)

        loss =   0.32*F.smooth_l1_loss(pre1[mask], disp_true[mask], size_average=True) \
               + 0.16*F.smooth_l1_loss(pre2[mask], disp_true[mask], size_average=True) \
               + 0.08*F.smooth_l1_loss(pre3[mask], disp_true[mask], size_average=True) \
               + 0.04*F.smooth_l1_loss(pre4[mask], disp_true[mask], size_average=True) \
               + 0.02*F.smooth_l1_loss(pre5[mask], disp_true[mask], size_average=True) \
               + 0.01*F.smooth_l1_loss(pre6[mask], disp_true[mask], size_average=True)
    # 计算梯度
    loss.backward()
    # 更新权重
    optimizer.step()
    return loss.data


# 5、模型测试
def test(imgL, imgR, disp_true):
    model.eval()
    imgL, imgR, disp_true = imgL.to(device), imgR.to(device), disp_true.to(device)
    # 视差图掩膜
    mask = disp_true < 192
    if imgL.shape[2] % 64 != 0:
        times = imgL.shape[2] // 64
        top_pad = (times + 1) * 64 - imgL.shape[2]
    else:
        top_pad = 0

    if imgL.shape[3] % 64 != 0:
        times = imgL.shape[3] // 64
        right_pad = (times + 1) * 64 - imgL.shape[3]
    else:
        right_pad = 0

    imgL = F.pad(imgL, (0, right_pad, top_pad, 0))
    imgR = F.pad(imgR, (0, right_pad, top_pad, 0))

    with torch.no_grad():
        output3 = model(imgL, imgR)
        output3 = torch.squeeze(output3)

    if top_pad != 0:
        img = output3[:, top_pad:, :]
    else:
        img = output3
    if right_pad != 0:
        img = img[:, :, :-right_pad]
    else:
        img = img

    if len(disp_true[mask]) == 0:
        loss = 0
    else:
        # torch.mean(torch.abs(img[mask]-disp_true[mask]))  # end-point-error
        loss = F.l1_loss(img[mask], disp_true[mask])

    return loss.data.cpu()


# 6、学习率调整
def adjust_learning_rate(optimizer, epoch):
    lr = 0.001
    print('学习率：\t', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    start_full_time = time.time()
    for epoch in range(0, args.epochs):
        print('This is %d-th epoch' % epoch)
        total_train_loss = 0
        adjust_learning_rate(optimizer, epoch)

        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(trainloader):
            start_time = time.time()
            loss = train_dispnet(imgL_crop, imgR_crop, disp_crop_L)
            print('Iter %d training loss = %.3f , time = %.2f' % (batch_idx, loss, time.time() - start_time))
            total_train_loss += loss
            if batch_idx == 1:
                break
        print('epoch %d total training loss = %.3f' % (epoch, total_train_loss/len(trainloader)))
        break

       #SAVE
        savefilename = args.savemodel+'/checkpoint_'+str(epoch)+'.tar'
        torch.save({
             'epoch': epoch,
             'state_dict': model.state_dict(),
             'train_loss': total_train_loss/len(trainloader),
        }, savefilename)

    print('full training time = %.2f h' %((time.time() - start_full_time)/3600))
    #
    # #------------- TEST ------------------------------------------------------------
    # total_test_loss = 0
    # for batch_idx, (imgL, imgR, disp_L) in enumerate(testloader):
    #        test_loss = test(imgL, imgR, disp_L)
    #        print('Iter %d test loss = %.3f' %(batch_idx, test_loss))
    #        total_test_loss += test_loss
    #
    # print('total test loss = %.3f' %(total_test_loss/len(testloader)))
    # #----------------------------------------------------------------------------------
    # #SAVE test information
    # savefilename = args.savemodel+'testinformation.tar'
    # torch.save({
    #         'test_loss': total_test_loss/len(testloader),
    #     }, savefilename)


if __name__ == '__main__':
    main()
    print('End.')



