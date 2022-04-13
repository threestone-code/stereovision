from torch import nn
from .feature_extraction import feature_extraction

def correlation(fea1, fea2):
    """全相关：不对特征图在通道维进行分组"""
    B, C, H, W = fea1.shape
    cost = (fea1 * fea2).mean(dim=1)
    assert cost.shape == (B, H, W)
    return cost
def groupwise_correlation(fea1, fea2, num_groups):
    """分组相关 计算"""
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost
def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    """分组相关 cost volume"""
    B, C, H, W = refimg_fea.shape
    #[B,G,D,H,W]
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],                                                           num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume


class BGNet(nn.Module):
    def __init__(self):
        super(BGNet, self).__init__()

        self.softmax = nn.Softmax(dim=1)
        # self.refinement_net = HourglassRefinement()
        self.feature_extraction = feature_extraction()
        self.coeffs_disparity_predictor = CoeffsPredictor()
        self.dres0 = nn.Sequential(convbn_3d_lrelu(44, 32, 3, 1, 1),
                                   convbn_3d_lrelu(32, 16, 3, 1, 1))
        self.guide = GuideNN()
        self.slice = Slice()
        self.weight_init()

    def forward(self, imgl, imgr):
        # 特征提取
        left_low_level_features_1, left_gwc_feature = self.feature_extraction(imgl)
        _, right_gwc_feature = self.feature_extraction(imgr)
        print('left_low_fea shape:\t', left_low_level_features_1.shape)
        print('feature map shape:\t', left_gwc_feature.shape)

        # 引导图
        guide = self.guide(left_low_level_features_1)  # [B,1,H,W]
        print('guide layer shape:\t', guide.shape)

        # 分组相关的 cost volume
        cost_volume = build_gwc_volume(left_gwc_feature, right_gwc_feature, 25, 44)
        print('gwc cost volume shape:\t', cost_volume.shape)

        # bg grid???????????????????????????
        cost_volume = self.dres0(cost_volume)
        print('bg shape:\t', cost_volume.shape)

        # 低分辨率的3D聚合（1/8 cost volume）
        

