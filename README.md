# stereovision
## DispNet
### training
SceneFlow DataSet: raw shape 540x960

Dispnet scale: 64, high and width of image 为 64 整数倍

training: 按照网络架构，将维度变为 384x768 进行输入（imgL imgR dispL）

pre1-pr6: 在最后分别上采样到 384x768，以便计算 loss
### test
test: 540x960 -> 576x960 进行输入（imgL imgR）

pre1: 在最后上采样到 576x960 进行输出，再裁剪到 540x960 计算误差
### upsample mode
mode 0: deconv (FlowNet)

mode 1: deconv + iconv (Dispnet, smoother)

![image-20220416192547208](.\img\upsample.png)

mode 2: bilinear + iconv (待测试)

