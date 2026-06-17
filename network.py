import numpy as np
import torch
import torch.nn as nn
import time

from lib.models.spin import hmr_atten_14

import os.path as osp


import torch.nn.functional as F
from smplpytorch.pytorch.smpl_layer import SMPL_Layer


class BasePointNet(nn.Module):
    def __init__(self):
        super(BasePointNet, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=3,   out_channels=8,  kernel_size=1)
        self.cb1 = nn.BatchNorm1d(8)
        self.caf1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=8,  out_channels=16, kernel_size=1)
        self.cb2 = nn.BatchNorm1d(16)
        self.caf2 = nn.ReLU()

        self.conv3 = nn.Conv1d(in_channels=16, out_channels=24, kernel_size=1)
        self.cb3 = nn.BatchNorm1d(24)
        self.caf3 = nn.ReLU()

    def forward(self, in_mat):
        x = in_mat.transpose(1,2)

        x = self.caf1(self.cb1(self.conv1(x)))
        x = self.caf2(self.cb2(self.conv2(x)))
        x = self.caf3(self.cb3(self.conv3(x)))

        x = x.transpose(1,2)
        x = torch.cat((in_mat[:,:,:4], x), -1)

        return x

class GlobalPointNet(nn.Module):
    def __init__(self):
        super(GlobalPointNet, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=24+3,   out_channels=32,  kernel_size=1)
        self.cb1 = nn.BatchNorm1d(32)
        self.caf1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=32,  out_channels=48, kernel_size=1)
        self.cb2 = nn.BatchNorm1d(48)
        self.caf2 = nn.ReLU()

        self.conv3 = nn.Conv1d(in_channels=48, out_channels=64, kernel_size=1)
        self.cb3 = nn.BatchNorm1d(64)
        self.caf3 = nn.ReLU()

        self.attn=nn.Linear(64, 1)
        self.softmax=nn.Softmax(dim=1)

    def forward(self, x):
        x = x.transpose(1,2)
        #print("x:",x.shape)
        x = self.caf1(self.cb1(self.conv1(x)))
        x = self.caf2(self.cb2(self.conv2(x)))
        x = self.caf3(self.cb3(self.conv3(x)))

        x = x.transpose(1,2)

        attn_weights=self.softmax(self.attn(x))
        #print("attn_weights:", attn_weights.shape)
        #print("before atten:",x.shape)
        attn_vec=torch.sum(x*attn_weights, dim=1)
        #print("after atten:", attn_vec.shape)
        return attn_vec, attn_weights

class GlobalRNN_bidirectional(nn.Module):
    def __init__(self):
        super(GlobalRNN_bidirectional, self).__init__()
        self.rnn=nn.LSTM(input_size=64, hidden_size=64, num_layers=3, batch_first=True, dropout=0.1, bidirectional=True)
        self.fc1 = nn.Linear(128, 16)
        self.faf1 = nn.ReLU()
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x, h0, c0):
        g_vec, (hn, cn)=self.rnn(x, (h0, c0))
        g_loc=self.fc1(g_vec)
        g_loc=self.faf1(g_loc)
        g_loc=self.fc2(g_loc)
        return g_vec, g_loc, hn, cn


#Anchor Module
def AnchorInit(x_min=-0.3, x_max=0.3, x_interval=0.3, y_min=-0.3, y_max=0.3, y_interval=0.3, z_min=-1, z_max=1, z_interval=0.5):#[z_size, y_size, x_size, npoint] => [9,3,3,3]
    """
    Input:
        x,y,z min, max and sample interval
    Return:
        centroids: sampled controids [z_size, y_size, x_size, npoint] => [9,3,3,3]
    """
    x_size=round((x_max-x_min)/x_interval)+1
    y_size=round((y_max-y_min)/y_interval)+1
    z_size=round((z_max-z_min)/z_interval)+1
    device=torch.device('cuda:%d' % (0) if torch.cuda.is_available() else 'cpu')
    centroids = torch.zeros((z_size, y_size, x_size, 3), dtype=torch.float32).to(device)
    for z_no in range(z_size):
        for y_no in range(y_size):
            for x_no in range(x_size):
                lx=x_min+x_no*x_interval
                ly=y_min+y_no*y_interval
                lz=z_min+z_no*z_interval
                centroids[z_no, y_no, x_no, 0]=lx
                centroids[z_no, y_no, x_no, 1]=ly
                centroids[z_no, y_no, x_no, 2]=lz
    return centroids



class AnchorPointNet(nn.Module):
    def __init__(self):
        super(AnchorPointNet, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=24+3+3,   out_channels=32,  kernel_size=1)
        self.cb1 = nn.BatchNorm1d(32)
        self.caf1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=32,  out_channels=48, kernel_size=1)
        self.cb2 = nn.BatchNorm1d(48)
        self.caf2 = nn.ReLU()

        self.conv3 = nn.Conv1d(in_channels=48, out_channels=64, kernel_size=1)
        self.cb3 = nn.BatchNorm1d(64)
        self.caf3 = nn.ReLU()

        self.attn=nn.Linear(64, 1)
        self.softmax=nn.Softmax(dim=1)

    def forward(self, x):
        x = x.transpose(1,2)

        x = self.caf1(self.cb1(self.conv1(x)))
        x = self.caf2(self.cb2(self.conv2(x)))
        x = self.caf3(self.cb3(self.conv3(x))) #(Batch, feature, frame_point_number)

        x = x.transpose(1,2)

        attn_weights=self.softmax(self.attn(x))
        attn_vec=torch.sum(x*attn_weights, dim=1)
        return attn_vec, attn_weights

class AnchorVoxelNet(nn.Module):
    def __init__(self):
        super(AnchorVoxelNet, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=64, out_channels=96, kernel_size=(3, 3, 3), padding=(0,0,0))
        self.cb1 = nn.BatchNorm3d(96)
        self.caf1 = nn.ReLU()

        self.conv2 = nn.Conv3d(in_channels=96, out_channels=128, kernel_size=(2, 1, 1))
        self.cb2 = nn.BatchNorm3d(128)
        self.caf2 = nn.ReLU()

        self.conv3 = nn.Conv3d(in_channels=128, out_channels=64, kernel_size=(2, 1, 1))
        self.cb3 = nn.BatchNorm3d(64)
        self.caf3 = nn.ReLU()

    def forward(self, x):
        batch_size=x.size()[0]
        x=x.permute(0, 4, 1, 2, 3)

        x=self.caf1(self.cb1(self.conv1(x)))
        x=self.caf2(self.cb2(self.conv2(x)))
        x=self.caf3(self.cb3(self.conv3(x)))

        x=x.view(batch_size, 64)
        return x

class AnchorRNN(nn.Module):
    def __init__(self):
        super(AnchorRNN, self).__init__()
        self.rnn=nn.LSTM(input_size=64, hidden_size=64, num_layers=3, batch_first=True, dropout=0.1, bidirectional=False)

    def forward(self, x, h0, c0):
        a_vec, (hn, cn)=self.rnn(x, (h0, c0))
        return a_vec, hn, cn

class AnchorRNN_bidirectional(nn.Module):
    def __init__(self):
        super(AnchorRNN_bidirectional, self).__init__()
        self.rnn=nn.LSTM(input_size=64, hidden_size=64, num_layers=3, batch_first=True, dropout=0.1, bidirectional=True)

    def forward(self, x, h0, c0):
        a_vec, (hn, cn)=self.rnn(x)
        return a_vec, hn, cn


def AnchorGrouping_new(anchors, k, points, points_feature):
    """
    计算 anchors 与最近点之间的差异，并将结果与最近点的特征拼接。

    参数：
        anchors (torch.Tensor): 锚点的张量，形状为 (batch_size * length_size, num_anchors, 3)
        k (int): 最近点的数量
        points (torch.Tensor): 所有点的张量，形状为 (batch_size * length_size, num_points, 3)
        points_feature (torch.Tensor): 所有点的特征张量，形状为 (batch_size * length_size, num_points, feature_dim)

    返回：
        combined (torch.Tensor): 合并后的张量，形状为 (batch_size * length_size, num_anchors, k, 3 + 3 + feature_dim)
    """

    # 计算每个 anchor 到每个 point 的欧几里得距离
    dists = torch.norm(points.unsqueeze(1) - anchors.unsqueeze(2), dim=-1)  # (batch_size*length_size, num_anchors, num_points)

    # 对每个 anchor 找到最近的 k 个点
    _, indices = torch.topk(dists, k, dim=-1, largest=False)  # (batch_size*length_size, num_anchors, k)

    # 根据 indices 选择最近的 k 个点和对应的特征
    nearest_points = torch.gather(points.unsqueeze(1).expand(-1, anchors.size(1), -1, -1), 2, indices.unsqueeze(-1).expand(-1, -1, -1, points.size(-1)))  # (batch_size*length_size, num_anchors, k, 3)
    nearest_features = torch.gather(points_feature.unsqueeze(1).expand(-1, anchors.size(1), -1, -1), 2, indices.unsqueeze(-1).expand(-1, -1, -1, points_feature.size(-1)))  # (batch_size*length_size, num_anchors, k, feature_dim)

    # 计算 anchor 与最近点的坐标差异，并将结果拼接
    coord_diff = nearest_points - anchors.unsqueeze(2)  # (batch_size*length_size, num_anchors, k, 3)
    combined = torch.cat((anchors.unsqueeze(2).expand(-1, -1, k, -1), coord_diff, nearest_features), dim=-1)  # (batch_size*length_size, num_anchors, k, 3 + 3 + feature_dim)

    return combined


class AnchorModule_bidirectional(nn.Module):
    def __init__(self):
        super(AnchorModule_bidirectional, self).__init__()
        self.template_point=AnchorInit()
        self.z_size, self.y_size, self.x_size, _=self.template_point.shape
        #print(self.template_point.shape)
        self.anchor_size=self.z_size*self.y_size*self.x_size
        self.apointnet=AnchorPointNet()
        self.avoxel=AnchorVoxelNet()
        self.arnn=AnchorRNN_bidirectional()

    def forward(self, x, g_loc, h0, c0, batch_size, length_size, feature_size):
        g_loc=g_loc.view(batch_size*length_size, 1, 2).repeat(1,self.anchor_size,1)
        anchors=self.template_point.view(1, self.anchor_size, 3).repeat(batch_size*length_size, 1, 1)
        #print("anchors", anchors.shape)
        #print("anchors", anchors.device)
        #print("g_loc", g_loc.shape)

        anchors[:,:,:2]+=g_loc
        #draw3Dpose_frames_anchor(anchors, x)
        grouped_points=AnchorGrouping_new(anchors, k=8, points=x[..., :3], points_feature=x[..., 3:])
        #print("grouped_points:",grouped_points.shape)
        grouped_points=grouped_points.view(batch_size*length_size*self.anchor_size, 8, 3+feature_size)
        voxel_points, attn_weights=self.apointnet(grouped_points)
        voxel_points=voxel_points.view(batch_size*length_size, self.z_size, self.y_size, self.x_size, 64)
        voxel_vec=self.avoxel(voxel_points)
        voxel_vec=voxel_vec.view(batch_size, length_size, 64)
        a_vec, hn, cn=self.arnn(voxel_vec, h0, c0)
        return a_vec, attn_weights, hn, cn



class GlobalModule_bidirectional(nn.Module):
    def __init__(self):
        super(GlobalModule_bidirectional, self).__init__()
        self.gpointnet=GlobalPointNet()
        self.grnn=GlobalRNN_bidirectional()

    def forward(self, x, h0, c0,  batch_size, length_size):
        x, attn_weights=self.gpointnet(x)
        x=x.view(batch_size, length_size, 64)
        g_vec, g_loc, hn, cn=self.grnn(x, h0, c0)
        return g_vec, g_loc, attn_weights, hn, cn

class CombineModule_nosmpl_bidirectional(nn.Module):
    def __init__(self):
        super(CombineModule_nosmpl_bidirectional, self).__init__()
        self.fc1 = nn.Linear(256, 256)
        self.faf1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 24 * 3  )

    def forward(self, g_vec, a_vec, batch_size, length_size):
        x = torch.cat((g_vec, a_vec), -1)
        x = self.fc1(x)
        x = self.faf1(x)
        x = self.fc2(x)
        key_pre=x[:, :,:24*3].view(batch_size, length_size, 24, 3)
        #print("key_pre:",key_pre.shape)
        # translation vector

        return key_pre


class mmWaveModel_ti_Anchor_nosmpl_bidirectional_loc(nn.Module):
    def __init__(self):
        super(mmWaveModel_ti_Anchor_nosmpl_bidirectional_loc, self).__init__()
        self.module0 = BasePointNet()
        self.module1 = GlobalModule_bidirectional()
        self.module2 = AnchorModule_bidirectional()
        # self.module3 = CombineModule_nosmpl_bidirectional()

    def forward(self, x,h0, c0, batch_size,length_size):
        # print( "x:",x.size())
        out_feature_size = 24 + 3

        x = self.module0(x)

        g_vec, g_loc, global_weights, hn_g, cn_g = self.module1(x, h0, c0, batch_size, length_size)
        a_vec, anchor_weights, hn_a, cn_a = self.module2(x, g_loc, h0, c0, batch_size, length_size,
                                                         out_feature_size)

        # key_pre = self.module3(g_vec, a_vec, batch_size, length_size)

        g_loc = g_loc.view(batch_size*length_size,-1)
        #print("g_loc:", g_loc.shape)
        #v=F.normalize(v[0][0][:][0:1])
        return g_vec,a_vec,g_loc

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname))

class CombineModule_mid_modal(nn.Module):
    def __init__(self):
        super(CombineModule_mid_modal, self).__init__()
        self.fc1 = nn.Linear(256, 128)
        self.faf1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 24 * 3 )

    def forward(self, x, batch_size, length_size):
        #print("g_vec:",g_vec.shape)
        #print("a_vec:", a_vec.shape)
        #print("t_vec:", t_vec.shape)
        x = x.view(batch_size,length_size,-1)
        x = self.fc1(x)
        x = self.faf1(x)
        x = self.fc2(x)
        key_pre=x[:, :,:24*3].view(batch_size, length_size, 24, 3)

        return key_pre

class CombineModule_mid_modal_rgb(nn.Module):
    def __init__(self):
        super(CombineModule_mid_modal_rgb, self).__init__()
        self.fc1 = nn.Linear(256, 256)
        self.faf1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 24 * 3 )

    def forward(self, x, batch_size, length_size):
        #print("g_vec:",g_vec.shape)
        #print("a_vec:", a_vec.shape)
        #print("t_vec:", t_vec.shape)
        x = x.view(batch_size,length_size,-1)
        x = self.fc1(x)
        x = self.faf1(x)
        x = self.fc2(x)
        key_pre=x[:, :,:24*3].view(batch_size, length_size, 24, 3)

        return key_pre


class CombineModule_mid_modal_smpl(nn.Module):
    def __init__(self):
        super(CombineModule_mid_modal_smpl, self).__init__()


        self.smpl_layer = SMPL_Layer(
            center_idx=0,
            gender='male',
            model_root='./smplpytorch/native/models')


        self.fc1 = nn.Linear(256, 256)
        self.faf1 = nn.ReLU()
        self.fc_pose = nn.Linear(256, 72)
        self.fc_shape = nn.Linear(256, 10)
        self.fc_cam = nn.Linear(256, 3)
        self.fc_gender = nn.Linear(256, 2)

        # === 新增：生物特征预测头 ===
        # 共享的特征提取层，从256维特征中提取128维共享特征
        self.fc_shared = nn.Linear(256, 128)
        self.faf_shared = nn.ReLU()
        # 身高预测头 (输出1个值)
        self.fc_height = nn.Linear(128, 1)
        # BMI预测头 (输出1个值)
        self.fc_bmi = nn.Linear(128, 1)


    def forward(self, x, batch_size, length_size):
        #print("g_vec:",g_vec.shape)
        #print("a_vec:", a_vec.shape)
        #print("t_vec:", t_vec.shape)
        x = x.view(batch_size,length_size,-1)
        x = self.fc1(x)
        x = self.faf1(x)
        pose = self.fc_pose(x)  # (batch_size, 72)
        shape = self.fc_shape(x)  # (batch_size, 10)
        cam = self.fc_cam(x)  # (batch_size, 3)
        gender = self.fc_gender(x)
        pose = pose.view(-1, 72)
        shape = shape.view(-1, 10)
        cam = cam.view(-1, 3)
        gender = gender.view(-1, 2)

        # 3. 调用 SMPL 层生成 3D 网格和关节点
        vertices, joints = self.smpl_layer(
            th_pose_axisang=pose,
            th_betas=shape,
            th_trans=cam
        )

        x_global = x.mean(dim=1)

        # 通过共享层
        x_shared = self.faf_shared(self.fc_shared(x_global))  # (batch_size, 128)

        # 预测身高和BMI
        pred_height = self.fc_height(x_shared).squeeze(-1)  # (batch_size,)
        pred_bmi = self.fc_bmi(x_shared).squeeze(-1)  # (batch_size,)

        return {
            'vertices': vertices,  # (batch_size, 6890, 3)
            'joints': joints,  # (batch_size, 24, 3)
            'cam': cam,
            'gender': gender,

            # === 新增：返回预测的生物特征 ===
            'height': pred_height,  # (batch_size,)
            'bmi': pred_bmi,  # (batch_size,)
        }


#使用残差结构
class _NonLocalBlockND_2modules_pixelatten_res(nn.Module):
    def __init__(self, in_channels, inter_channels=None, selfrgb=1, sub_sample=False, bn_layer=True):
        super(_NonLocalBlockND_2modules_pixelatten_res, self).__init__()

        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.selfrgb = selfrgb

                # channel数减半，减少计算量
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if selfrgb==1:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2))
            bn = nn.BatchNorm2d
            conv_nd2 = nn.Conv1d
            max_pool_layer2 = nn.MaxPool1d(kernel_size=(2))
            bn2 = nn.BatchNorm1d
        elif selfrgb==0:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d
            conv_nd2 = nn.Conv2d
            max_pool_layer2 = nn.MaxPool2d(kernel_size=(2))
            bn2 = nn.BatchNorm2d


        # 定义1x1卷积形式的embeding层
        # 从上到下相当于Transformer里的q，k，v的embeding
        self.F_theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1,
                               stride=1, padding=0)

        self.F_phi = conv_nd2(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.F_g = conv_nd2(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        # self atten
        self.R_theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1,
                                stride=1,
                                padding=0)

        self.R_phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                              kernel_size=1, stride=1, padding=0)

        self.R_g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                            kernel_size=1, stride=1, padding=0)

        self.self_bnRelu = nn.Sequential(
            bn(self.in_channels),
            nn.ReLU(inplace=True),
        )

        self.mutual_bnRelu = nn.Sequential(
            bn2(self.in_channels),
            nn.ReLU(inplace=True),
        )

        # output embeding和Batch norm
        if bn_layer:
            self.F_W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.F_W[1].weight, 0)
            nn.init.constant_(self.F_W[1].bias, 0)

            self.R_W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.R_W[1].weight, 0)
            nn.init.constant_(self.R_W[1].bias, 0)

            # 拼接后进行映射
            self.C_W = nn.Sequential(
                conv_nd(in_channels=(self.in_channels) * 2, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.C_W[1].weight, 0)
            nn.init.constant_(self.C_W[1].bias, 0)

        else:
            self.F_W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.F_W.weight, 0)
            nn.init.constant_(self.F_W.bias, 0)
            self.R_W =conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.R_W[1].weight, 0)
            nn.init.constant_(self.R_W[1].bias, 0)

    def forward(self, self_fea, mutual_fea,return_nl_map=False):
            """
            :param x: (b, c, t, h, w)
            :param return_nl_map: if True return z, nl_map, else only return z.
            :return:
            """
            #print("self_fea:", self_fea.shape)
            selfNonLocal_fea = self.self_bnRelu(self_fea)
            mutualNonLocal_fea = self.mutual_bnRelu(mutual_fea)

            batch_size = selfNonLocal_fea.size(0)

            # using self feature to generate attention
            self_g_x = self.R_g(selfNonLocal_fea).view(batch_size, self.inter_channels, -1)
            #print("self_g_x:", self_g_x.shape)
            self_g_x = self_g_x.permute(0, 2, 1)
            # self_g_x = F.normalize(self_g_x, dim=2)
            # print("self_g_x:", torch.mean(self_g_x))
            self_theta_x = self.R_theta(selfNonLocal_fea).view(batch_size, self.inter_channels, -1)
            self_theta_x = self_theta_x.permute(0, 2, 1)
            self_phi_x = self.R_phi(selfNonLocal_fea).view(batch_size, self.inter_channels, -1)
            self_f = torch.matmul(self_theta_x, self_phi_x)
            self_f_div_C = F.softmax(self_f, dim=-1)
            #print("self_f_div_C:",self_f_div_C.shape)

            '''
            print("self_attention:")
            print("self_f_div_C:",self_f_div_C.shape)
            if self.selfrgb == 1:
                # 绘制attention map
                import matplotlib.pyplot as plt
                import seaborn as sns
                for i in range(1):
                    plt.figure(figsize=(12, 12))
                    plot = sns.heatmap(mutual_f_div_C[i][j ].cpu().detach().reshape(14,14).detach(), linewidths=0.8, annot=True, fmt=".3f")
                    # plt.pause(1.3)
                    # print(ax.lines)
                    plt.show()
'''
            self_y = torch.matmul(self_f_div_C, self_g_x)
            #print("self_f_div_C:", self_f_div_C.shape)
            #print("self_g_x:", self_g_x.shape)
            #print("self_y:", self_y.shape)
            self_y = self_y.permute(0, 2, 1).contiguous()
            #print("self_y:",self_y.shape)
            self_y = self_y.view(batch_size, self.inter_channels, *selfNonLocal_fea.size()[2:])
            #print("self_y:", self_y.shape)
            # 只映射最终结果，即self和mutual的结果相加后映射
            self_W_y = self.R_W(self_y)

            # using mutual feature to generate attention
            mutual_g_x = self.F_g(mutualNonLocal_fea).view(batch_size, self.inter_channels, -1)
            mutual_g_x = mutual_g_x.permute(0, 2, 1)
            # print("mutual_g_x:", mutual_g_x.shape)
            # mutual_g_x = F.normalize(mutual_g_x,dim=2)
            # print("mutual_g_x:",torch.mean(mutual_g_x))
            mutual_theta_x = self.F_theta(selfNonLocal_fea).view(batch_size, self.inter_channels, -1)
            mutual_theta_x = mutual_theta_x.permute(0, 2, 1)
            mutual_phi_x = self.F_phi(mutualNonLocal_fea).view(batch_size, self.inter_channels, -1)
            mutual_f = torch.matmul(mutual_theta_x, mutual_phi_x)
            mutual_f_div_C = F.softmax(mutual_f, dim=-1)
            '''
            print("mutual_attention:")
            print("mutual_f_div_C:", mutual_f_div_C.shape)
            if self.selfrgb==0:
                # 绘制attention map
                import matplotlib.pyplot as plt
                import seaborn as sns
                for i in range(1):
                    for j in range(1):
                        plt.figure(figsize=(12, 12))
                        #plot = sns.heatmap(mutual_f_div_C[i][j+30].cpu().detach().unsqueeze(1), linewidths=0.8, annot=True, fmt=".3f")
                        plot = sns.heatmap(mutual_f_div_C[i][j+10 ].cpu().detach().reshape(14,14), linewidths=0.8,
                                           annot=True, fmt=".3f")
                        # plt.pause(1.3)
                        # print(ax.lines)
                        plt.show()
            '''
            #print("mutual_f_div_C:", mutual_f_div_C.shape)
            #print("mutual_g_x:", mutual_g_x.shape)
            mutual_y = torch.matmul(mutual_f_div_C, mutual_g_x)
            mutual_y = mutual_y.permute(0, 2, 1).contiguous()
            mutual_y = mutual_y.view(batch_size, self.inter_channels, *selfNonLocal_fea.size()[2:])
            # 只映射最终结果，即self和mutual的结果相加后映射
            mutual_W_y = self.F_W(mutual_y)
            #print("mutual_W_y:", mutual_W_y.shape)
            # print("f_image:",f[0])
            '''
            #0502修改
            z = mutual_y + self_y
            z = self.F_W(z)
            '''
            # 0503修改，拼接做法
            z = torch.cat((self_W_y, mutual_W_y), dim=1)
            #0505修改：拼接后直接返回
            z = self.C_W(z)
            #print("z:", z.shape)
            z = z+self_fea
            if return_nl_map:
                return z, self_f_div_C, mutual_f_div_C
            return z,self_W_y,mutual_f_div_C


class   ImageNet(nn.Module):
    def __init__(self, feature_dim=128):
        super(ImageNet, self).__init__()

        # Modified for RGB input (3 channels)
        self.global_branch = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # Changed input channels to 3
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout(0.25),

            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout(0.25),

            # Block 3
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout(0.25),

            # Block 4
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout(0.25),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        # Dynamic FC layer (initialized in forward)
        self.fc = nn.Sequential(
            nn.Identity(),  # Placeholder, will be replaced
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=feature_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        # self.pose_estimator = PoseEstimator(feature_dim)
        self._fc_initialized = False

    def forward(self, x, batch_size =4,length_size=10):
        # Input shape: (batch, seq_len=20, 224, 224, 3)
        batch_size, seq_len = batch_size,length_size

        # Reshape for CNN: (batch * seq_len, 3, 224, 224)
        # x = x.permute(0, 1, 4, 2, 3)  # (batch, seq_len, 3, 224, 224)
        x = x.reshape(-1, 3, 224, 224)  # (batch * seq_len, 3, 224, 224)

        # CNN feature extraction
        x = self.global_branch(x)
        # x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        #
        # # Dynamic FC layer initialization
        # if not self._fc_initialized:
        #     fc_input_dim = x.shape[1]
        #     self.fc[0] = nn.Linear(fc_input_dim, 512).to(x.device)
        #     self._fc_initialized = True
        #
        # x = self.fc(x)
        #
        # # Reshape for LSTM: (batch, seq_len, 128)
        # x = x.view(batch_size, seq_len, -1)
        #
        # # LSTM processing
        # lstm_out, _ = self.lstm(x)  # (batch, seq_len, feature_dim * 2)

        # Pose estimation
        # pose = self.pose_estimator(lstm_out)  # (batch, seq_len, 24, 3)

        return x



class tianshun(nn.Module):
    def __init__(self, device2):
        super(tianshun, self).__init__()
        BASE_DATA_DIR = 'lib/models/pretrained/base_data'
        self.model_hmr_l = hmr_atten_14().to(device2)
        checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'),map_location=torch.device(device2))
        self.model_hmr_l.load_state_dict(checkpoint['model'], strict=False)

        self.model_ti = mmWaveModel_ti_Anchor_nosmpl_bidirectional_loc().to(device2)
        self.model_ti2 = mmWaveModel_ti_Anchor_nosmpl_bidirectional_loc().to(device2)

        self.bpointnet = BasePointNet()
        self.conv3 = nn.Conv1d(256 + 27, 256, 1)  # 27+64+64
        self.cb3 = nn.BatchNorm1d(256)
        self.caf3 = nn.ReLU()

        self.module1 = CombineModule_mid_modal()
        self.module2 = CombineModule_mid_modal()
        self.module3 = CombineModule_mid_modal()
        self.module4 = CombineModule_mid_modal()
        self.module5 = CombineModule_mid_modal()
        self.module6 = CombineModule_mid_modal()

        self.conv1 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1)
        self.cb1 = nn.BatchNorm2d(256)
        self.caf1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.cb2 = nn.BatchNorm1d(256)
        self.caf2 = nn.ReLU()

        self.conv4 = nn.Conv1d(in_channels=2048, out_channels=256, kernel_size=1)
        self.cb4 = nn.BatchNorm1d(256)
        self.caf4 = nn.ReLU()

        #mutual attention后的特征和无mutual attenttion的对齐
        self.conv5 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.cb5 = nn.BatchNorm1d(256)
        self.caf5 = nn.ReLU()
        self.conv6 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.cb6 = nn.BatchNorm1d(256)
        self.caf6 = nn.ReLU()

        # nln模块
        self.nl = _NonLocalBlockND_2modules_pixelatten_res(in_channels=256, selfrgb=1)
        self.nl2 = _NonLocalBlockND_2modules_pixelatten_res(in_channels=256, selfrgb=0)
        # 模态间attention
        self.attn1 = nn.Linear(256, 1)
        self.softmax1 = nn.Softmax(dim=1)

        self.attn2 = nn.Linear(256, 1)
        self.softmax2 = nn.Softmax(dim=1)

        self.avgpool = nn.AvgPool2d(14, stride=1)

        # 步态周期attention
        self.attn3 = nn.Linear(256, 1)
        self.softmax3 = nn.Softmax(dim=1)
        self.attn4 = nn.Linear(256, 1)
        self.softmax4 = nn.Softmax(dim=1)
        self.attn5 = nn.Linear(256, 1)
        self.softmax5 = nn.Softmax(dim=1)
        self.attn6 = nn.Linear(256, 1)
        self.softmax6 = nn.Softmax(dim=1)

    def forward(self, x_rgb,x_rgb_n, ti_p, ti_n, h0, c0, batch_size, length_size):
        # x_rgb：锚点样本RGB数据
        # x_rgb_n：负样本RGB数据，
        # ti_p：正样本雷达数据
        # ti_n：负样本雷达数据
        self_att=0

        feature_hmr_l,_ = self.model_hmr_l.feature_extractor(x_rgb)#锚点样本特征
        feature_hmr_l_n, _ = self.model_hmr_l.feature_extractor(x_rgb_n)#负样本特征

        rgb_l = self.caf1(self.cb1(self.conv1(feature_hmr_l)))#锚点 RGB特征    40X14X14X256（通道）
        rgb_l_n = self.caf1(self.cb1(self.conv1(feature_hmr_l_n)))#负样本 RGB 特征


        #feature_tcmr, _ = self.model_tcmr(feature_hmr)

        # 毫米波 正样本特征提取
        g_vec_h, a_vec_h, g_loc_p1 = self.model_ti(ti_p, h0, c0, batch_size, length_size)
        ti_h = torch.cat((g_vec_h, a_vec_h), dim=2)
        g_vec_l, a_vec_l, g_loc_p2 = self.model_ti2(ti_p, h0, c0, batch_size, length_size)
        ti_l = torch.cat((g_vec_l, a_vec_l), dim=2)

        #毫米波 负样本特征提取
        g_vec_h2, a_vec_h2,  g_loc_n1 = self.model_ti(ti_n, h0, c0, batch_size, length_size)
        ti_h2 = torch.cat((g_vec_h2, a_vec_h2), dim=2)
        g_vec_l2, a_vec_l2, g_loc_n2 = self.model_ti2(ti_n, h0, c0, batch_size, length_size)
        ti_l2 = torch.cat((g_vec_l2, a_vec_l2), dim=2)

        # rgb网络
        #f_tcmr = feature_tcmr.transpose(1, 2)
        #f_tcmr = feature_tcmr


        # mutual attention
        # nln模块
        ti_l = ti_l.view(batch_size, length_size, 256)#毫米波 正样本-帧特征
        n_pts = ti_p.size()[1] #64个点
        ti_l = ti_l.view(batch_size, length_size, 1, 256).repeat(1, 1, n_pts, 1)
        bpoint = self.bpointnet(ti_p)
        bpoint = bpoint.view(batch_size, length_size, n_pts, -1)
        bpoint = torch.cat([ti_l, bpoint], 3)
        bpoint = bpoint.view(batch_size * length_size, n_pts, -1)
        bpoint = bpoint.transpose(1, 2)
        ti_l_ma = self.caf3(self.cb3(self.conv3(bpoint)))#正样本 毫米波每个点特征 （40，256，64）

        ti_l2 = ti_l2.view(batch_size, length_size, 256) #毫米波 负样本-帧特征
        n_pts = ti_n.size()[1]
        ti_l2 = ti_l2.view(batch_size, length_size, 1, 256).repeat(1, 1, n_pts, 1)
        bpoint = self.bpointnet(ti_n)
        bpoint = bpoint.view(batch_size, length_size, n_pts, -1)
        bpoint = torch.cat([ti_l2, bpoint], 3)
        bpoint = bpoint.view(batch_size * length_size, n_pts, -1)
        bpoint = bpoint.transpose(1, 2)
        ti_l_ma_n = self.caf3(self.cb3(self.conv3(bpoint)))#负样本，毫米波每个点特征（40，256，64）

        #2nln,每个nln都计算self和mutual
        #print("rgb_nln:")
        #rgb_fusion 输出：融合毫米波特征后的RGB特征。
        rgb_fusion, rgb_self, mutual_f_div_C = self.nl(rgb_l, ti_l_ma)#rgb_l：锚点RGB 特征40X14X14X256（通道），     #ti_l_ma：正样本 毫米波每个点特征 （40，256，64）
        rgb_fusion_n, _, _ = self.nl(rgb_l_n, ti_l_ma_n)#rgb_l_n:负样本 RGB 特征 ；  ti_l_ma_n：负样本，毫米波特征
        #不同id融合结果
        rgb_fusion_n_dif, _, _ = self.nl(rgb_l_n, ti_l_ma)#rgb_l_n，负样本RGB特征；  ti_l_ma：正样本 毫米波每个点特征


        #print("mmwave_nln:")
        ti_fusion, ti_self ,_= self.nl2(ti_l_ma, rgb_l)#input：  正样本毫米波+锚点RGB
        ti_fusion_n, ti_self_n, _ = self.nl2(ti_l_ma_n, rgb_l_n)#负样本毫米波+负样本RGB
        ti_fusion_dif, _, _ = self.nl2(ti_l_ma, rgb_l_n)#正样本毫米波+负样本RGB


        rgb_fusion = self.avgpool(rgb_fusion)
        rgb_fusion = rgb_fusion.view(rgb_fusion.size(0), -1)#锚点RGB+正样本毫米波

        rgb_fusion_n = self.avgpool(rgb_fusion_n)#负样本RGB+负样本毫米波
        rgb_fusion_n = rgb_fusion_n.view(rgb_fusion_n.size(0), -1)

        rgb_fusion_n_dif = self.avgpool(rgb_fusion_n_dif)#负样本RGB+正样本毫米波
        rgb_fusion_n_dif = rgb_fusion_n_dif.view(rgb_fusion_n_dif.size(0), -1)

        if self_att == 1:
            rgb_self = self.avgpool(rgb_self)#自注意力后的RGB
        else:
            rgb_self = self.avgpool(rgb_l)

        rgb_self = rgb_self.view(rgb_self.size(0), -1)

        #【01】锚点RGB+正样本
        ti_fusion = ti_fusion.transpose(1, 2)#融合RGB特征后的毫米波-点特征

        if self_att == 1:
            ti_self = ti_self.transpose(1, 2)#自注意力后的毫米波特征
        else:
            ti_self = ti_l_ma.transpose(1, 2)

        attn_weights = self.softmax2(self.attn2(ti_fusion))
        ti_fusion = torch.sum(ti_fusion * attn_weights, dim=1)#毫米波-帧特征（正样本+锚点 RGB融合）

        attn_weights = self.softmax1(self.attn1(ti_self))
        ti_self = torch.sum(ti_self * attn_weights, dim=1)#毫米波帧特征（自注意力）

        #【02】负样本毫米波+负样本RGB
        ti_fusion_n = ti_fusion_n.transpose(1, 2)#毫米波-帧特征（负样本+负样本 RGB融合）
        if self_att == 1:
            ti_self_n = ti_self_n.transpose(1, 2)#负样本 毫米波自注意力
        else:
            ti_self_n = ti_l_ma_n.transpose(1, 2)

        attn_weights = self.softmax2(self.attn2(ti_fusion_n))
        ti_fusion_n = torch.sum(ti_fusion_n * attn_weights, dim=1)

        attn_weights = self.softmax1(self.attn1(ti_self_n))
        ti_self_n = torch.sum(ti_self_n * attn_weights, dim=1)

        #【03】正样本毫米波+负样本RGB
        ti_fusion_dif = ti_fusion_dif.transpose(1, 2)
        attn_weights = self.softmax2(self.attn2(ti_fusion_dif))
        ti_fusion_dif = torch.sum(ti_fusion_dif * attn_weights, dim=1)

        # reconstruction
        key_pre_rgb_self = self.module1(rgb_self, batch_size, length_size)#融合前 锚点 RGB重建的骨架；
        key_pre_ti_self = self.module2(ti_self, batch_size, length_size)#融合前 正样本 毫米波重建的骨架
        key_pre_ti_self_n = self.module2(ti_self_n, batch_size, length_size)#融合前 负样本毫米波重建的骨架

        key_pre_rgb = self.module3(rgb_fusion, batch_size, length_size)#融合后 锚点 RGB重建的骨架；
        key_pre_ti = self.module4(ti_fusion, batch_size, length_size)#融合后正样本 毫米波重建的骨架
        key_pre_ti2 = self.module4(ti_fusion_n, batch_size, length_size)#融合后 负样本毫米波with 负样本RGB重建的骨架

        #key_pre_ti2 = self.module6(ti_l2, batch_size, length_size)
        key_pre_rgb_self = key_pre_rgb_self.view(batch_size * length_size, 24, 3)
        key_pre_ti_self = key_pre_ti_self.view(batch_size * length_size, 24, 3)
        key_pre_ti_self_n = key_pre_ti_self_n.view(batch_size * length_size, 24, 3)
        key_pre_rgb = key_pre_rgb.view(batch_size * length_size, 24, 3)
        key_pre_ti = key_pre_ti.view(batch_size * length_size, 24, 3)
        key_pre_ti2 = key_pre_ti2.view(batch_size * length_size, 24, 3)

        # 正样本毫米波+锚点RGB
        ti_fusion = ti_fusion.view(batch_size, length_size, -1)

        # 毫米波-帧特征（负样本+负样本 RGB融合）
        ti_fusion_n = ti_fusion_n.view(batch_size, length_size, -1)

        # 正样本毫米波+负样本RGB
        ti_fusion_dif = ti_fusion_dif.view(batch_size, length_size, -1)

        #类似上面，只不过是融合毫米波的RGB特征
        rgb_fusion = rgb_fusion.view(batch_size, length_size, -1)
        rgb_fusion_n = rgb_fusion_n.view(batch_size, length_size, -1)
        rgb_fusion_n_dif = rgb_fusion_n_dif.view(batch_size, length_size, -1)


        # 总输出
        attn_weights_ti_h = self.softmax3(self.attn3(ti_h))
        attn_weights_ti_h2 = self.softmax3(self.attn3(ti_h2))
        attn_weights_ti_l = self.softmax4(self.attn4(ti_fusion))
        attn_weights_ti_l2 = self.softmax4(self.attn4(ti_fusion_n))
        attn_weights_ti_dif = self.softmax4(self.attn4(ti_fusion_dif))
        attn_weights_rgb_l = self.softmax5(self.attn5(rgb_fusion))
        attn_weights_rgb_l2 = self.softmax5(self.attn5(rgb_fusion_n))
        attn_weights_rgb_l_dif = self.softmax5(self.attn5(rgb_fusion_n_dif))

        #正样本（单模态）
        ti_h = torch.sum(ti_h * attn_weights_ti_h, dim=1)

        #正样本毫米波+锚点RGB
        ti_l = torch.sum(ti_fusion * attn_weights_ti_l, dim=1)

        #负样本 （单模态）
        ti_h2 = torch.sum(ti_h2 * attn_weights_ti_h2, dim=1)

        # 毫米波-帧特征（负样本+负样本 RGB融合）
        ti_l2 = torch.sum(ti_fusion_n * attn_weights_ti_l2, dim=1)

        # 正样本毫米波+负样本RGB
        ti_dif = torch.sum(ti_fusion_dif * attn_weights_ti_dif, dim=1)

        #RGB模态
        rgb_l = torch.sum(rgb_fusion * attn_weights_rgb_l, dim=1)#锚点RGB+正样本毫米波
        rgb_l2 = torch.sum(rgb_fusion_n * attn_weights_rgb_l2, dim=1)#（负样本毫米波+负样本 RGB融合）
        rgb_l_dif = torch.sum(rgb_fusion_n_dif * attn_weights_rgb_l_dif, dim=1)#正样本毫米波+负样本RGB



        ti_h = F.normalize(ti_h)#正样本（单模态）
        ti_l = F.normalize(ti_l) #正样本毫米波+锚点RGB

        ti_h2 = F.normalize(ti_h2)#负样本 （单模态）
        ti_l2 = F.normalize(ti_l2)# 毫米波-帧特征（负样本+负样本 RGB融合）


        rgb_l = F.normalize(rgb_l)#锚点RGB+正样本毫米波
        rgb_l2 = F.normalize(rgb_l2)#（负样本毫米波+负样本 RGB融合）（sum+norm）

        rgb_l_dif = F.normalize(rgb_l_dif)#正样本毫米波+负样本RGB（sum+norm）
        ti_dif = F.normalize(ti_dif)#正样本毫米波+负样本RGB
        # 在高低维特征norm前整体norm
        # output1 = rgb_l#锚点RGB+正样本毫米波（sum+norm）
        # output2 = ti_l#正样本毫米波+锚点RGB(sum+norm)
        # output3 = ti_l2# 毫米波-帧特征（负样本+负样本 RGB融合）(sum+norm)


        return rgb_l, ti_l, ti_l2, rgb_l_dif,ti_dif, key_pre_rgb, key_pre_ti, key_pre_ti2, key_pre_rgb_self,key_pre_ti_self,key_pre_ti_self_n, g_loc_p2, g_loc_n2, g_loc_p1, g_loc_n1
        #rgb_h：0，
        #ti_h：正样本毫米波（单模态）
        #ti_h2：负样本毫米波 （单模态）

        #key_pre_rgb： 融合后 锚点RGB with 正样本毫米波；
        #key_pre_ti：#融合后，正样本毫米波 with 锚点RGB融合）
        #key_pre_ti2：融合后，负样本毫米波 with 负样本 RGB融合）

        # output1 = rgb_l  # 融合后 锚点RGB with 正样本毫米波；（sum+norm）
        # output2 = ti_l  #融合后，正样本毫米波 with 锚点RGB融合）#正样本毫米波+锚点RGB(sum+norm)
        # output3 = ti_l2  #融合后，负样本毫米波 with 负样本 RGB融合）(sum+norm)

        #rgb_l, ti_l, ti_l2 同上
        #g_loc_p1, g_loc_p2, g_loc_n1, g_loc_n2,正样本，负样本毫米波的loc
        #key_pre_rgb_self,key_pre_ti_self,key_pre_ti_self_n，融合前
        #rgb_l2,（负样本毫米波+负样本 RGB融合）（sum+norm）

        # rgb_l_dif,#负样本RGB with 正样本毫米波（sum+norm）融合
        # ti_dif，#正样本毫米波 with 负样本RGB（sum+norm）融合

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname, map_location="cuda:0"))

class ruili(nn.Module):
    def __init__(self, device2):
        super(ruili, self).__init__()
        BASE_DATA_DIR = 'lib/models/pretrained/base_data'
        # self.model_hmr_l = ImageNet().to(device2)
        self.model_hmr_l = hmr_atten_14().to(device2)
        checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'),map_location=torch.device(device2))
        self.model_hmr_l.load_state_dict(checkpoint['model'], strict=False)

        self.model_ti  = mmWaveModel_ti_Anchor_nosmpl_bidirectional_loc().to(device2)
        self.model_ti2 = mmWaveModel_ti_Anchor_nosmpl_bidirectional_loc().to(device2)

        self.bpointnet = BasePointNet()
        self.conv3 = nn.Conv1d(256 + 27, 256, 1)  # 27+64+64
        self.cb3 = nn.BatchNorm1d(256)
        self.caf3 = nn.ReLU()

        self.module3 = CombineModule_mid_modal()
        self.module4 = CombineModule_mid_modal_rgb()
        self.module5 = CombineModule_mid_modal_rgb()
        self.module6 = CombineModule_mid_modal_rgb()

        self.conv1 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1)
        self.cb1 = nn.BatchNorm2d(256)
        self.caf1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.cb2 = nn.BatchNorm1d(256)
        self.caf2 = nn.ReLU()

        self.conv4 = nn.Conv1d(in_channels=2048, out_channels=256, kernel_size=1)
        self.cb4 = nn.BatchNorm1d(256)
        self.caf4 = nn.ReLU()

        #mutual attention后的特征和无mutual attenttion的对齐
        self.conv5 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.cb5 = nn.BatchNorm1d(256)
        self.caf5 = nn.ReLU()
        self.conv6 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.cb6 = nn.BatchNorm1d(256)
        self.caf6 = nn.ReLU()

        # nln模块
        self.nl = _NonLocalBlockND_2modules_pixelatten_res(in_channels=256, selfrgb=1)
        self.nl2 = _NonLocalBlockND_2modules_pixelatten_res(in_channels=256, selfrgb=0)
        # 模态间attention
        self.attn1 = nn.Linear(256, 1)
        self.softmax1 = nn.Softmax(dim=1)

        self.attn2 = nn.Linear(256, 1)
        self.softmax2 = nn.Softmax(dim=1)

        self.avgpool = nn.AvgPool2d(14, stride=1)

        # 步态周期attention
        self.attn3 = nn.Linear(256, 1)
        self.softmax3 = nn.Softmax(dim=1)
        self.attn4 = nn.Linear(256, 1)
        self.softmax4 = nn.Softmax(dim=1)
        self.attn5 = nn.Linear(256, 1)
        self.softmax5 = nn.Softmax(dim=1)
        self.attn6 = nn.Linear(256, 1)
        self.softmax6 = nn.Softmax(dim=1)

    def forward(self, x_rgb,x_rgb_n, ti_p, ti_n, h0, c0, batch_size, length_size):
        # x_rgb：锚点样本RGB数据
        # x_rgb_n：负样本RGB数据，
        # ti_p：正样本雷达数据
        # ti_n：负样本雷达数据

        #
        #视觉模态特征提取
        feature_hmr_l,_ = self.model_hmr_l.feature_extractor(x_rgb)#锚点样本特征
        feature_hmr_l_n, _ = self.model_hmr_l.feature_extractor(x_rgb_n)#负样本特征

        rgb_l = self.caf1(self.cb1(self.conv1(feature_hmr_l)))#锚点 RGB特征    40X14X14X256（通道）
        rgb_l_n = self.caf1(self.cb1(self.conv1(feature_hmr_l_n)))#负样本 RGB 特征



        # rgb_l = self.model_hmr_l(x_rgb)#锚点样本特征
        # rgb_l_n= self.model_hmr_l(x_rgb_n)#负样本特征

        # 毫米波 正样本特征提取
        g_vec_l, a_vec_l, g_loc_p2 = self.model_ti2(ti_p, h0, c0, batch_size, length_size)
        ti_l = torch.cat((g_vec_l, a_vec_l), dim=2)
        bpoint1 = self.bpointnet(ti_p)
        n_pts = ti_p.size()[1] #64个点
        ti_l = ti_l.view(batch_size*length_size, 1, 256).repeat(1, n_pts, 1)#4，10，64，256
        ti_l_ma = torch.cat([ti_l, bpoint1], 2)
        ti_l_ma = ti_l_ma.transpose(1, 2)
        ti_l_ma = self.caf3(self.cb3(self.conv3(ti_l_ma)))#正样本 毫米波每个点特征 （40，256，64）


        #毫米波 负样本特征提取
        g_vec_l2, a_vec_l2, g_loc_n2 = self.model_ti2(ti_n, h0, c0, batch_size, length_size)
        ti_l2 = torch.cat((g_vec_l2, a_vec_l2), dim=2)
        bpoint2 = self.bpointnet(ti_n)
        n_pts = ti_n.size()[1] #64个点
        ti_l2 = ti_l2.view(batch_size*length_size, 1, 256).repeat(1, n_pts, 1)#4，10，64，256
        ti_l_ma_n = torch.cat([ti_l2, bpoint2], 2)
        ti_l_ma_n = ti_l_ma_n.transpose(1, 2)
        ti_l_ma_n = self.caf3(self.cb3(self.conv3(ti_l_ma_n)))#正样本 毫米波每个点特征 （40，256，64）

        # 【nln模块计算】每个nln都计算self和mutual

        #【rgb_fusion】锚点RGB+正样本毫米波
        rgb_fusion, rgb_self, mutual_f_div_C = self.nl(rgb_l, ti_l_ma)
        rgb_fusion = self.avgpool(rgb_fusion)
        rgb_fusion = rgb_fusion.view(rgb_fusion.size(0), -1)#锚点RGB+正样本毫米波 40，256

        #【rgb_fusion_n_dif】负样本RGB+负样本毫米波
        rgb_fusion_n_dif, _, _ = self.nl(rgb_l_n, ti_l_ma)
        rgb_fusion_n_dif = self.avgpool(rgb_fusion_n_dif)#负样本RGB+正样本毫米波
        rgb_fusion_n_dif = rgb_fusion_n_dif.view(rgb_fusion_n_dif.size(0), -1)

        # 【ti_fusion】正样本毫米波+锚点RGB
        ti_fusion, ti_self ,_= self.nl2(ti_l_ma, rgb_l)
        ti_fusion = ti_fusion.transpose(1, 2)#融合RGB特征后的毫米波-点特征
        attn_weights = self.softmax2(self.attn2(ti_fusion))
        ti_fusion = torch.sum(ti_fusion * attn_weights, dim=1)#毫米波-帧特征（正样本+锚点 RGB融合）


        # 【ti_fusion_n】负样本毫米波+负样本RGB
        ti_fusion_n, ti_self_n, _ = self.nl2(ti_l_ma_n, rgb_l_n)
        ti_fusion_n = ti_fusion_n.transpose(1, 2)#毫米波-帧特征（负样本+负样本 RGB融合）
        attn_weights = self.softmax2(self.attn2(ti_fusion_n))
        ti_fusion_n = torch.sum(ti_fusion_n * attn_weights, dim=1)

        #【ti_fusion_dif】正样本毫米波+负样本RGB
        ti_fusion_dif, _, _ = self.nl2(ti_l_ma, rgb_l_n)
        ti_fusion_dif = ti_fusion_dif.transpose(1, 2)
        attn_weights = self.softmax2(self.attn2(ti_fusion_dif))
        ti_fusion_dif = torch.sum(ti_fusion_dif * attn_weights, dim=1)

        #rgb单模态
        rgb_self = self.avgpool(rgb_self)#自注意力后的RGB
        rgb_self = rgb_self.view(rgb_self.size(0), -1)

        #正样本毫米波
        ti_self = ti_self.transpose(1, 2)#自注意力后的毫米波特征
        attn_weights = self.softmax1(self.attn1(ti_self))
        ti_self = torch.sum(ti_self * attn_weights, dim=1)#毫米波帧特征（自注意力）

        #【02】负样本毫米波
        ti_self_n = ti_self_n.transpose(1, 2)#负样本 毫米波自注意力
        attn_weights = self.softmax1(self.attn1(ti_self_n))
        ti_self_n = torch.sum(ti_self_n * attn_weights, dim=1)


        # reconstruction
        key_pre_rgb_self = self.module6(rgb_self, batch_size, length_size)#融合前 锚点 RGB重建的骨架；
        key_pre_ti_self = self.module3(ti_self, batch_size, length_size)#融合前 正样本 毫米波重建的骨架
        key_pre_ti_self_n = self.module3(ti_self_n, batch_size, length_size)#融合前 负样本毫米波重建的骨架

        key_pre_rgb = self.module4(rgb_fusion, batch_size, length_size)#融合后 锚点RGB+正样本毫米波 重建的骨架；
        key_pre_ti = self.module4(ti_fusion, batch_size, length_size)#融合后 正样本毫米波+锚点RGB 重建的骨架
        key_pre_ti2 = self.module4(ti_fusion_n, batch_size, length_size)#融合后 负样本毫米波+负样本RGB 重建的骨架

        key_pre_rgb_self = key_pre_rgb_self.view(batch_size * length_size, 24, 3)
        key_pre_ti_self = key_pre_ti_self.view(batch_size * length_size, 24, 3)
        key_pre_ti_self_n = key_pre_ti_self_n.view(batch_size * length_size, 24, 3)

        key_pre_rgb = key_pre_rgb.view(batch_size * length_size, 24, 3)
        key_pre_ti = key_pre_ti.view(batch_size * length_size, 24, 3)
        key_pre_ti2 = key_pre_ti2.view(batch_size * length_size, 24, 3)

        #锚点RGB+正样本毫米波
        rgb_fusion = rgb_fusion.view(batch_size, length_size, -1)
        attn_weights_rgb_l = self.softmax5(self.attn5(rgb_fusion))
        rgb_l = torch.sum(rgb_fusion * attn_weights_rgb_l, dim=1)
        rgb_l = F.normalize(rgb_l)#锚点RGB+正样本毫米波

        # 正样本毫米波+锚点RGB
        ti_fusion = ti_fusion.view(batch_size, length_size, -1)
        attn_weights_ti_l = self.softmax4(self.attn4(ti_fusion))
        ti_l = torch.sum(ti_fusion * attn_weights_ti_l, dim=1)
        ti_l = F.normalize(ti_l) #正样本毫米波+锚点RGB

        # 负样本毫米波+负样本RGB
        ti_fusion_n = ti_fusion_n.view(batch_size, length_size, -1)
        attn_weights_ti_l2 = self.softmax4(self.attn4(ti_fusion_n))
        ti_l2 = torch.sum(ti_fusion_n * attn_weights_ti_l2, dim=1)
        ti_l2 = F.normalize(ti_l2)# 毫米波-帧特征（负样本+负样本 RGB融合）

        # 负样本RGB+负样本毫米波
        rgb_fusion_n_dif = rgb_fusion_n_dif.view(batch_size, length_size, -1)
        attn_weights_rgb_l_dif = self.softmax5(self.attn5(rgb_fusion_n_dif))
        rgb_l_dif = torch.sum(rgb_fusion_n_dif * attn_weights_rgb_l_dif, dim=1)
        rgb_l_dif = F.normalize(rgb_l_dif)

        # 正样本毫米波+负样本RGB
        ti_fusion_dif = ti_fusion_dif.view(batch_size, length_size, -1)
        attn_weights_ti_dif = self.softmax4(self.attn4(ti_fusion_dif))
        ti_dif = torch.sum(ti_fusion_dif * attn_weights_ti_dif, dim=1)
        ti_dif = F.normalize(ti_dif)#正样本毫米波+负样本RGB




        return rgb_l, ti_l, ti_l2, rgb_l_dif,ti_dif, key_pre_rgb, key_pre_ti, key_pre_ti2, key_pre_rgb_self,key_pre_ti_self,key_pre_ti_self_n, g_loc_p2, g_loc_n2

        # rgb_l  # 锚点RGB+正样本毫米波
        # ti_l   # 正样本毫米波+锚点RGB
        # ti_l2  # 负样本毫米波+负样本RGB
        #rgb_l_dif#负样本RGB+负样本毫米波
        # ti_dif # 正样本毫米波+负样本RGB

        # key_pre_rgb #融合后 锚点RGB+正样本毫米波 重建的骨架；
        # key_pre_ti  #融合后 正样本毫米波+锚点RGB 重建的骨架
        # key_pre_ti2 #融合后 负样本毫米波+负样本RGB 重建的骨架

        # key_pre_rgb_self  #融合前 锚点 RGB重建的骨架；
        # key_pre_ti_self   #融合前 正样本 毫米波重建的骨架
        # key_pre_ti_self_n #融合前 负样本毫米波重建的骨架





        #rgb_h：0，
        #ti_h：正样本毫米波（单模态）
        #ti_h2：负样本毫米波 （单模态）

        #key_pre_rgb： 融合后 锚点RGB with 正样本毫米波；
        #key_pre_ti：#融合后，正样本毫米波 with 锚点RGB融合）
        #key_pre_ti2：融合后，负样本毫米波 with 负样本 RGB融合）

        # output1 = rgb_l  # 融合后 锚点RGB with 正样本毫米波；（sum+norm）
        # output2 = ti_l  #融合后，正样本毫米波 with 锚点RGB融合）#正样本毫米波+锚点RGB(sum+norm)
        # output3 = ti_l2  #融合后，负样本毫米波 with 负样本 RGB融合）(sum+norm)

        #rgb_l, ti_l, ti_l2 同上
        #g_loc_p1, g_loc_p2, g_loc_n1, g_loc_n2,正样本，负样本毫米波的loc
        #key_pre_rgb_self,key_pre_ti_self,key_pre_ti_self_n，融合前
        #rgb_l2,（负样本毫米波+负样本 RGB融合）（sum+norm）

        # rgb_l_dif,#负样本RGB with 正样本毫米波（sum+norm）融合
        # ti_dif，#正样本毫米波 with 负样本RGB（sum+norm）融合

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname, map_location="cuda:0"))


class ruili_duanwu(nn.Module):
    def __init__(self, device2):
        super(ruili_duanwu, self).__init__()
        BASE_DATA_DIR = 'lib/models/pretrained/base_data'
        self.model_hmr_l = ImageNet().to(device2)
        # self.model_hmr_l = hmr_atten_14().to(device2)
        checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'),map_location=torch.device(device2))
        self.model_hmr_l.load_state_dict(checkpoint['model'], strict=False)

        self.model_ti  = mmWaveModel_ti_Anchor_nosmpl_bidirectional_loc().to(device2)
        self.model_ti2 = mmWaveModel_ti_Anchor_nosmpl_bidirectional_loc().to(device2)

        self.bpointnet = BasePointNet()
        self.conv3 = nn.Conv1d(256 + 27, 256, 1)  # 27+64+64
        self.cb3 = nn.BatchNorm1d(256)
        self.caf3 = nn.ReLU()

        self.module3 = CombineModule_mid_modal_smpl()  # 用于单模态毫米波
        self.module4 = CombineModule_mid_modal_smpl()  # 用于融合后的特征 (RGB+毫米波)
        self.module6 = CombineModule_mid_modal_smpl()  # 用于单模态RGB

        self.conv1 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1)
        self.cb1 = nn.BatchNorm2d(256)
        self.caf1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.cb2 = nn.BatchNorm1d(256)
        self.caf2 = nn.ReLU()

        self.conv4 = nn.Conv1d(in_channels=2048, out_channels=256, kernel_size=1)
        self.cb4 = nn.BatchNorm1d(256)
        self.caf4 = nn.ReLU()

        #mutual attention后的特征和无mutual attenttion的对齐
        self.conv5 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.cb5 = nn.BatchNorm1d(256)
        self.caf5 = nn.ReLU()
        self.conv6 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.cb6 = nn.BatchNorm1d(256)
        self.caf6 = nn.ReLU()

        # nln模块
        self.nl = _NonLocalBlockND_2modules_pixelatten_res(in_channels=256, selfrgb=1)
        self.nl2 = _NonLocalBlockND_2modules_pixelatten_res(in_channels=256, selfrgb=0)
        # 模态间attention
        self.attn1 = nn.Linear(256, 1)
        self.softmax1 = nn.Softmax(dim=1)

        self.attn2 = nn.Linear(256, 1)
        self.softmax2 = nn.Softmax(dim=1)

        self.avgpool = nn.AvgPool2d(14, stride=1)

        # 步态周期attention
        self.attn3 = nn.Linear(256, 1)
        self.softmax3 = nn.Softmax(dim=1)
        self.attn4 = nn.Linear(256, 1)
        self.softmax4 = nn.Softmax(dim=1)
        self.attn5 = nn.Linear(256, 1)
        self.softmax5 = nn.Softmax(dim=1)
        self.attn6 = nn.Linear(256, 1)
        self.softmax6 = nn.Softmax(dim=1)
    def forward(self, x_rgb,x_rgb_n, ti_p, ti_n, h0, c0, batch_size, length_size):
        # x_rgb：锚点样本RGB数据
        # x_rgb_n：负样本RGB数据，
        # ti_p：正样本雷达数据
        # ti_n：负样本雷达数据

        #
        #视觉模态特征提取
        # feature_hmr_l,_ = self.model_hmr_l.feature_extractor(x_rgb)#锚点样本特征
        # feature_hmr_l_n, _ = self.model_hmr_l.feature_extractor(x_rgb_n)#负样本特征

        # rgb_l = self.caf1(self.cb1(self.conv1(feature_hmr_l)))#锚点 RGB特征    40X14X14X256（通道）
        # rgb_l_n = self.caf1(self.cb1(self.conv1(feature_hmr_l_n)))#负样本 RGB 特征



        rgb_l = self.model_hmr_l(x_rgb)#锚点样本特征
        rgb_l_n= self.model_hmr_l(x_rgb_n)#负样本特征

        # 毫米波 正样本特征提取
        g_vec_l, a_vec_l, g_loc_p2 = self.model_ti2(ti_p, h0, c0, batch_size, length_size)
        ti_l = torch.cat((g_vec_l, a_vec_l), dim=2)
        bpoint1 = self.bpointnet(ti_p)
        n_pts = ti_p.size()[1] #64个点
        ti_l = ti_l.view(batch_size*length_size, 1, 256).repeat(1, n_pts, 1)#4，10，64，256
        ti_l_ma = torch.cat([ti_l, bpoint1], 2)
        ti_l_ma = ti_l_ma.transpose(1, 2)
        ti_l_ma = self.caf3(self.cb3(self.conv3(ti_l_ma)))#正样本 毫米波每个点特征 （40，256，64）


        #毫米波 负样本特征提取
        g_vec_l2, a_vec_l2, g_loc_n2 = self.model_ti2(ti_n, h0, c0, batch_size, length_size)
        ti_l2 = torch.cat((g_vec_l2, a_vec_l2), dim=2)
        bpoint2 = self.bpointnet(ti_n)
        n_pts = ti_n.size()[1] #64个点
        ti_l2 = ti_l2.view(batch_size*length_size, 1, 256).repeat(1, n_pts, 1)#4，10，64，256
        ti_l_ma_n = torch.cat([ti_l2, bpoint2], 2)
        ti_l_ma_n = ti_l_ma_n.transpose(1, 2)
        ti_l_ma_n = self.caf3(self.cb3(self.conv3(ti_l_ma_n)))#正样本 毫米波每个点特征 （40，256，64）

        # 【nln模块计算】每个nln都计算self和mutual

        #【rgb_fusion】锚点RGB+正样本毫米波
        rgb_fusion, rgb_self, mutual_f_div_C = self.nl(rgb_l, ti_l_ma)
        rgb_fusion = self.avgpool(rgb_fusion)
        rgb_fusion = rgb_fusion.view(rgb_fusion.size(0), -1)#锚点RGB+正样本毫米波 40，256

        #【rgb_fusion_n_dif】负样本RGB+正样本毫米波
        rgb_fusion_n_dif, _, _ = self.nl(rgb_l_n, ti_l_ma)
        rgb_fusion_n_dif = self.avgpool(rgb_fusion_n_dif)#负样本RGB+正样本毫米波
        rgb_fusion_n_dif = rgb_fusion_n_dif.view(rgb_fusion_n_dif.size(0), -1)

        # 【rgb_fusion_n】锚点RGB+负样本毫米波
        rgb_fusion_n, _, _ = self.nl(rgb_l, ti_l_ma_n)
        rgb_fusion_n = self.avgpool(rgb_fusion_n)#负样本RGB+正样本毫米波
        rgb_fusion_n = rgb_fusion_n.view(rgb_fusion_n.size(0), -1)

        # 【ti_fusion】正样本毫米波+锚点RGB
        ti_fusion, ti_self ,_= self.nl2(ti_l_ma, rgb_l)
        ti_fusion = ti_fusion.transpose(1, 2)#融合RGB特征后的毫米波-点特征
        attn_weights = self.softmax2(self.attn2(ti_fusion))
        ti_fusion = torch.sum(ti_fusion * attn_weights, dim=1)#毫米波-帧特征（正样本+锚点 RGB融合）


        # 【ti_fusion_n】负样本毫米波+负样本RGB
        ti_fusion_n, ti_self_n, _ = self.nl2(ti_l_ma_n, rgb_l_n)
        ti_fusion_n = ti_fusion_n.transpose(1, 2)#毫米波-帧特征（负样本+负样本 RGB融合）
        attn_weights = self.softmax2(self.attn2(ti_fusion_n))
        ti_fusion_n = torch.sum(ti_fusion_n * attn_weights, dim=1)

        #【ti_fusion_dif】正样本毫米波+负样本RGB
        ti_fusion_dif, _, _ = self.nl2(ti_l_ma, rgb_l_n)
        ti_fusion_dif = ti_fusion_dif.transpose(1, 2)
        attn_weights = self.softmax2(self.attn2(ti_fusion_dif))
        ti_fusion_dif = torch.sum(ti_fusion_dif * attn_weights, dim=1)

        #【ti_fusion_dif_n】负样本毫米波+锚点 RGB
        ti_fusion_dif_n, _, _ = self.nl2(ti_l_ma_n, rgb_l)
        ti_fusion_dif_n = ti_fusion_dif_n.transpose(1, 2)
        attn_weights = self.softmax2(self.attn2(ti_fusion_dif_n))
        ti_fusion_dif_n = torch.sum(ti_fusion_dif_n * attn_weights, dim=1)



        #rgb单模态
        rgb_self = self.avgpool(rgb_self)#自注意力后的RGB
        rgb_self = rgb_self.view(rgb_self.size(0), -1)

        #正样本毫米波
        ti_self = ti_self.transpose(1, 2)#自注意力后的毫米波特征
        attn_weights = self.softmax1(self.attn1(ti_self))
        ti_self = torch.sum(ti_self * attn_weights, dim=1)#毫米波帧特征（自注意力）

        #【02】负样本毫米波
        ti_self_n = ti_self_n.transpose(1, 2)#负样本 毫米波自注意力
        attn_weights = self.softmax1(self.attn1(ti_self_n))
        ti_self_n = torch.sum(ti_self_n * attn_weights, dim=1)


        # reconstruction
        output_rgb_self = self.module6(rgb_self, batch_size, length_size)  # 融合前 锚点 RGB
        output_ti_self = self.module3(ti_self, batch_size, length_size)  # 融合前 正样本 毫米波
        output_ti_self_n = self.module3(ti_self_n, batch_size, length_size)  # 融合前 负样本毫米波

        output_rgb = self.module4(rgb_fusion, batch_size, length_size)  # 融合后 锚点RGB+正样本毫米波
        output_ti = self.module4(ti_fusion, batch_size, length_size)  # 融合后 正样本毫米波+锚点RGB
        output_ti2 = self.module4(ti_fusion_n, batch_size, length_size)  # 融合后 负样本毫米波+负样本RGB

        # 从字典中提取关节点，用于计算关键点损失
        key_pre_rgb_self = output_rgb_self['joints'].view(batch_size * length_size, 24, 3)
        key_pre_ti_self = output_ti_self['joints'].view(batch_size * length_size, 24, 3)
        key_pre_ti_self_n = output_ti_self_n['joints'].view(batch_size * length_size, 24, 3)

        key_pre_rgb = output_rgb['joints'].view(batch_size * length_size, 24, 3)
        key_pre_ti = output_ti['joints'].view(batch_size * length_size, 24, 3)
        key_pre_ti2 = output_ti2['joints'].view(batch_size * length_size, 24, 3)

        key_pre_rgb_self = key_pre_rgb_self.view(batch_size * length_size, 24, 3)
        key_pre_ti_self = key_pre_ti_self.view(batch_size * length_size, 24, 3)
        key_pre_ti_self_n = key_pre_ti_self_n.view(batch_size * length_size, 24, 3)

        key_pre_rgb = key_pre_rgb.view(batch_size * length_size, 24, 3)
        key_pre_ti = key_pre_ti.view(batch_size * length_size, 24, 3)
        key_pre_ti2 = key_pre_ti2.view(batch_size * length_size, 24, 3)

        #锚点RGB+正样本毫米波
        rgb_fusion = rgb_fusion.view(batch_size, length_size, -1)
        attn_weights_rgb_l = self.softmax5(self.attn5(rgb_fusion))
        rgb_l = torch.sum(rgb_fusion * attn_weights_rgb_l, dim=1)
        rgb_l = F.normalize(rgb_l)#锚点RGB+正样本毫米波

        # 正样本毫米波+锚点RGB
        ti_fusion = ti_fusion.view(batch_size, length_size, -1)
        attn_weights_ti_l = self.softmax4(self.attn4(ti_fusion))
        ti_l = torch.sum(ti_fusion * attn_weights_ti_l, dim=1)
        ti_l = F.normalize(ti_l) #正样本毫米波+锚点RGB

        # 负样本毫米波+负样本RGB
        ti_fusion_n = ti_fusion_n.view(batch_size, length_size, -1)
        attn_weights_ti_l2 = self.softmax4(self.attn4(ti_fusion_n))
        ti_l2 = torch.sum(ti_fusion_n * attn_weights_ti_l2, dim=1)
        ti_l2 = F.normalize(ti_l2)# 毫米波-帧特征（负样本+负样本 RGB融合）

        # 负样本RGB+负样本毫米波
        rgb_fusion_n_dif = rgb_fusion_n_dif.view(batch_size, length_size, -1)
        attn_weights_rgb_l_dif = self.softmax5(self.attn5(rgb_fusion_n_dif))
        rgb_l_dif = torch.sum(rgb_fusion_n_dif * attn_weights_rgb_l_dif, dim=1)
        rgb_l_dif = F.normalize(rgb_l_dif)

        # 锚点RGB+负样本毫米波
        rgb_fusion_n = rgb_fusion_n.view(batch_size, length_size, -1)
        attn_weights_rgb_l_n = self.softmax5(self.attn5(rgb_fusion_n))
        rgb_l_n2 = torch.sum(rgb_fusion_n * attn_weights_rgb_l_n, dim=1)
        rgb_l_n2 = F.normalize(rgb_l_n2)

        # 正样本毫米波+负样本RGB
        ti_fusion_dif = ti_fusion_dif.view(batch_size, length_size, -1)
        attn_weights_ti_dif = self.softmax4(self.attn4(ti_fusion_dif))
        ti_dif = torch.sum(ti_fusion_dif * attn_weights_ti_dif, dim=1)
        ti_dif = F.normalize(ti_dif)#正样本毫米波+负样本RGB

        # 负样本毫米波+锚点RGB
        ti_fusion_dif_n = ti_fusion_dif_n.view(batch_size, length_size, -1)
        attn_weights_ti_dif_n = self.softmax4(self.attn4(ti_fusion_dif_n))
        ti_dif_n = torch.sum(ti_fusion_dif_n * attn_weights_ti_dif_n, dim=1)
        ti_dif_n = F.normalize(ti_dif_n)

        return (
            rgb_l, ti_l, ti_l2, rgb_l_dif, ti_dif,
            key_pre_rgb, key_pre_ti, key_pre_ti2,
            key_pre_rgb_self, key_pre_ti_self, key_pre_ti_self_n,
            g_loc_p2, g_loc_n2, ti_dif_n, rgb_l_n2,
            # 新增：返回 6 个生物特征字典
            output_rgb, output_ti, output_ti2,
            output_rgb_self, output_ti_self, output_ti_self_n
        )
        # rgb_l  # 锚点RGB+正样本毫米波
        # ti_l   # 正样本毫米波+锚点RGB
        # ti_l2  # 负样本毫米波+负样本RGB
        #rgb_l_dif#负样本RGB+负样本毫米波
        # ti_dif # 正样本毫米波+负样本RGB

        # key_pre_rgb #融合后 锚点RGB+正样本毫米波 重建的骨架；
        # key_pre_ti  #融合后 正样本毫米波+锚点RGB 重建的骨架
        # key_pre_ti2 #融合后 负样本毫米波+负样本RGB 重建的骨架

        # key_pre_rgb_self  #融合前 锚点 RGB重建的骨架；
        # key_pre_ti_self   #融合前 正样本 毫米波重建的骨架
        # key_pre_ti_self_n #融合前 负样本毫米波重建的骨架





        #rgb_h：0，
        #ti_h：正样本毫米波（单模态）
        #ti_h2：负样本毫米波 （单模态）

        #key_pre_rgb： 融合后 锚点RGB with 正样本毫米波；
        #key_pre_ti：#融合后，正样本毫米波 with 锚点RGB融合）
        #key_pre_ti2：融合后，负样本毫米波 with 负样本 RGB融合）

        # output1 = rgb_l  # 融合后 锚点RGB with 正样本毫米波；（sum+norm）
        # output2 = ti_l  #融合后，正样本毫米波 with 锚点RGB融合）#正样本毫米波+锚点RGB(sum+norm)
        # output3 = ti_l2  #融合后，负样本毫米波 with 负样本 RGB融合）(sum+norm)

        #rgb_l, ti_l, ti_l2 同上
        #g_loc_p1, g_loc_p2, g_loc_n1, g_loc_n2,正样本，负样本毫米波的loc
        #key_pre_rgb_self,key_pre_ti_self,key_pre_ti_self_n，融合前
        #rgb_l2,（负样本毫米波+负样本 RGB融合）（sum+norm）

        # rgb_l_dif,#负样本RGB with 正样本毫米波（sum+norm）融合
        # ti_dif，#正样本毫米波 with 负样本RGB（sum+norm）融合

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname, map_location="cuda:0"))



class ruili_duanwu_smpl(nn.Module):
    def __init__(self, device2):
        super(ruili_duanwu_smpl, self).__init__()
        BASE_DATA_DIR = 'lib/models/pretrained/base_data'
        # self.model_hmr_l = ImageNet().to(device2)
        self.model_hmr_l = hmr_atten_14().to(device2)
        checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'),map_location=torch.device(device2))
        self.model_hmr_l.load_state_dict(checkpoint['model'], strict=False)

        self.model_ti  = mmWaveModel_ti_Anchor_nosmpl_bidirectional_loc().to(device2)
        self.model_ti2 = mmWaveModel_ti_Anchor_nosmpl_bidirectional_loc().to(device2)

        self.bpointnet = BasePointNet()
        self.conv3 = nn.Conv1d(256 + 27, 256, 1)  # 27+64+64
        self.cb3 = nn.BatchNorm1d(256)
        self.caf3 = nn.ReLU()

        self.module3 = CombineModule_mid_modal_smpl()
        self.module4 = CombineModule_mid_modal_smpl()
        self.module6 = CombineModule_mid_modal_smpl()

        self.conv1 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1)
        self.cb1 = nn.BatchNorm2d(256)
        self.caf1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.cb2 = nn.BatchNorm1d(256)
        self.caf2 = nn.ReLU()

        self.conv4 = nn.Conv1d(in_channels=2048, out_channels=256, kernel_size=1)
        self.cb4 = nn.BatchNorm1d(256)
        self.caf4 = nn.ReLU()

        #mutual attention后的特征和无mutual attenttion的对齐
        self.conv5 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.cb5 = nn.BatchNorm1d(256)
        self.caf5 = nn.ReLU()
        self.conv6 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.cb6 = nn.BatchNorm1d(256)
        self.caf6 = nn.ReLU()

        # nln模块
        self.nl = _NonLocalBlockND_2modules_pixelatten_res(in_channels=256, selfrgb=1)
        self.nl2 = _NonLocalBlockND_2modules_pixelatten_res(in_channels=256, selfrgb=0)
        # 模态间attention
        self.attn1 = nn.Linear(256, 1)
        self.softmax1 = nn.Softmax(dim=1)

        self.attn2 = nn.Linear(256, 1)
        self.softmax2 = nn.Softmax(dim=1)

        self.avgpool = nn.AvgPool2d(14, stride=1)

        # 步态周期attention
        self.attn3 = nn.Linear(256, 1)
        self.softmax3 = nn.Softmax(dim=1)
        self.attn4 = nn.Linear(256, 1)
        self.softmax4 = nn.Softmax(dim=1)
        self.attn5 = nn.Linear(256, 1)
        self.softmax5 = nn.Softmax(dim=1)
        self.attn6 = nn.Linear(256, 1)
        self.softmax6 = nn.Softmax(dim=1)

    def forward(self, x_rgb,x_rgb_n, ti_p, ti_n, h0, c0, batch_size, length_size):
        # x_rgb：锚点样本RGB数据
        # x_rgb_n：负样本RGB数据，
        # ti_p：正样本雷达数据
        # ti_n：负样本雷达数据

        #
        #视觉模态特征提取
        feature_hmr_l,_ = self.model_hmr_l.feature_extractor(x_rgb)#锚点样本特征
        feature_hmr_l_n, _ = self.model_hmr_l.feature_extractor(x_rgb_n)#负样本特征

        rgb_l = self.caf1(self.cb1(self.conv1(feature_hmr_l)))#锚点 RGB特征    40X14X14X256（通道）
        rgb_l_n = self.caf1(self.cb1(self.conv1(feature_hmr_l_n)))#负样本 RGB 特征



        # rgb_l = self.model_hmr_l(x_rgb)#锚点样本特征
        # rgb_l_n= self.model_hmr_l(x_rgb_n)#负样本特征

        # 毫米波 正样本特征提取
        g_vec_l, a_vec_l, g_loc_p2 = self.model_ti2(ti_p, h0, c0, batch_size, length_size)
        ti_l = torch.cat((g_vec_l, a_vec_l), dim=2)
        bpoint1 = self.bpointnet(ti_p)
        n_pts = ti_p.size()[1] #64个点
        ti_l = ti_l.view(batch_size*length_size, 1, 256).repeat(1, n_pts, 1)#4，10，64，256
        ti_l_ma = torch.cat([ti_l, bpoint1], 2)
        ti_l_ma = ti_l_ma.transpose(1, 2)
        ti_l_ma = self.caf3(self.cb3(self.conv3(ti_l_ma)))#正样本 毫米波每个点特征 （40，256，64）


        #毫米波 负样本特征提取
        g_vec_l2, a_vec_l2, g_loc_n2 = self.model_ti2(ti_n, h0, c0, batch_size, length_size)
        ti_l2 = torch.cat((g_vec_l2, a_vec_l2), dim=2)
        bpoint2 = self.bpointnet(ti_n)
        n_pts = ti_n.size()[1] #64个点
        ti_l2 = ti_l2.view(batch_size*length_size, 1, 256).repeat(1, n_pts, 1)#4，10，64，256
        ti_l_ma_n = torch.cat([ti_l2, bpoint2], 2)
        ti_l_ma_n = ti_l_ma_n.transpose(1, 2)
        ti_l_ma_n = self.caf3(self.cb3(self.conv3(ti_l_ma_n)))#正样本 毫米波每个点特征 （40，256，64）

        # 【nln模块计算】每个nln都计算self和mutual

        #【rgb_fusion】锚点RGB+正样本毫米波
        rgb_fusion, rgb_self, mutual_f_div_C = self.nl(rgb_l, ti_l_ma)
        rgb_fusion = self.avgpool(rgb_fusion)
        rgb_fusion = rgb_fusion.view(rgb_fusion.size(0), -1)#锚点RGB+正样本毫米波 40，256

        #【rgb_fusion_n_dif】负样本RGB+正样本毫米波
        rgb_fusion_n_dif, _, _ = self.nl(rgb_l_n, ti_l_ma)
        rgb_fusion_n_dif = self.avgpool(rgb_fusion_n_dif)#负样本RGB+正样本毫米波
        rgb_fusion_n_dif = rgb_fusion_n_dif.view(rgb_fusion_n_dif.size(0), -1)

        # 【rgb_fusion_n】锚点RGB+负样本毫米波
        rgb_fusion_n, _, _ = self.nl(rgb_l, ti_l_ma_n)
        rgb_fusion_n = self.avgpool(rgb_fusion_n)#负样本RGB+正样本毫米波
        rgb_fusion_n = rgb_fusion_n.view(rgb_fusion_n.size(0), -1)

        # 【ti_fusion】正样本毫米波+锚点RGB
        ti_fusion, ti_self ,_= self.nl2(ti_l_ma, rgb_l)
        ti_fusion = ti_fusion.transpose(1, 2)#融合RGB特征后的毫米波-点特征
        attn_weights = self.softmax2(self.attn2(ti_fusion))
        ti_fusion = torch.sum(ti_fusion * attn_weights, dim=1)#毫米波-帧特征（正样本+锚点 RGB融合）


        # 【ti_fusion_n】负样本毫米波+负样本RGB
        ti_fusion_n, ti_self_n, _ = self.nl2(ti_l_ma_n, rgb_l_n)
        ti_fusion_n = ti_fusion_n.transpose(1, 2)#毫米波-帧特征（负样本+负样本 RGB融合）
        attn_weights = self.softmax2(self.attn2(ti_fusion_n))
        ti_fusion_n = torch.sum(ti_fusion_n * attn_weights, dim=1)

        #【ti_fusion_dif】正样本毫米波+负样本RGB
        ti_fusion_dif, _, _ = self.nl2(ti_l_ma, rgb_l_n)
        ti_fusion_dif = ti_fusion_dif.transpose(1, 2)
        attn_weights = self.softmax2(self.attn2(ti_fusion_dif))
        ti_fusion_dif = torch.sum(ti_fusion_dif * attn_weights, dim=1)

        #【ti_fusion_dif_n】负样本毫米波+锚点 RGB
        ti_fusion_dif_n, _, _ = self.nl2(ti_l_ma_n, rgb_l)
        ti_fusion_dif_n = ti_fusion_dif_n.transpose(1, 2)
        attn_weights = self.softmax2(self.attn2(ti_fusion_dif_n))
        ti_fusion_dif_n = torch.sum(ti_fusion_dif_n * attn_weights, dim=1)



        #rgb单模态
        rgb_self = self.avgpool(rgb_self)#自注意力后的RGB
        rgb_self = rgb_self.view(rgb_self.size(0), -1)

        #正样本毫米波
        ti_self = ti_self.transpose(1, 2)#自注意力后的毫米波特征
        attn_weights = self.softmax1(self.attn1(ti_self))
        ti_self = torch.sum(ti_self * attn_weights, dim=1)#毫米波帧特征（自注意力）

        #【02】负样本毫米波
        ti_self_n = ti_self_n.transpose(1, 2)#负样本 毫米波自注意力
        attn_weights = self.softmax1(self.attn1(ti_self_n))
        ti_self_n = torch.sum(ti_self_n * attn_weights, dim=1)


        # reconstruction
        output_rgb = self.module4(rgb_fusion, batch_size, length_size)
        output_ti = self.module4(ti_fusion, batch_size, length_size)
        output_ti2 = self.module4(ti_fusion_n, batch_size, length_size)

        output_rgb_self = self.module6(rgb_self, batch_size, length_size)
        output_ti_self = self.module3(ti_self, batch_size, length_size)
        output_ti_self_n = self.module3(ti_self_n, batch_size, length_size)

        # 从字典中提取骨架和生物特征
        key_pre_rgb = output_rgb['joints']  # 提取关节点
        key_pre_ti = output_ti['joints']
        key_pre_ti2 = output_ti2['joints']

        key_pre_rgb_self = output_rgb_self['joints']
        key_pre_ti_self = output_ti_self['joints']
        key_pre_ti_self_n = output_ti_self_n['joints']

        # key_pre_rgb_self = key_pre_rgb_self.view(batch_size * length_size, 24, 3)
        # key_pre_ti_self = key_pre_ti_self.view(batch_size * length_size, 24, 3)
        # key_pre_ti_self_n = key_pre_ti_self_n.view(batch_size * length_size, 24, 3)
        #
        # key_pre_rgb = key_pre_rgb.view(batch_size * length_size, 24, 3)
        # key_pre_ti = key_pre_ti.view(batch_size * length_size, 24, 3)
        # key_pre_ti2 = key_pre_ti2.view(batch_size * length_size, 24, 3)

        #锚点RGB+正样本毫米波
        rgb_fusion = rgb_fusion.view(batch_size, length_size, -1)
        attn_weights_rgb_l = self.softmax5(self.attn5(rgb_fusion))
        rgb_l = torch.sum(rgb_fusion * attn_weights_rgb_l, dim=1)
        rgb_l = F.normalize(rgb_l)#锚点RGB+正样本毫米波

        # 正样本毫米波+锚点RGB
        ti_fusion = ti_fusion.view(batch_size, length_size, -1)
        attn_weights_ti_l = self.softmax4(self.attn4(ti_fusion))
        ti_l = torch.sum(ti_fusion * attn_weights_ti_l, dim=1)
        ti_l = F.normalize(ti_l) #正样本毫米波+锚点RGB

        # 负样本毫米波+负样本RGB
        ti_fusion_n = ti_fusion_n.view(batch_size, length_size, -1)
        attn_weights_ti_l2 = self.softmax4(self.attn4(ti_fusion_n))
        ti_l2 = torch.sum(ti_fusion_n * attn_weights_ti_l2, dim=1)
        ti_l2 = F.normalize(ti_l2)# 毫米波-帧特征（负样本+负样本 RGB融合）

        # 负样本RGB+负样本毫米波
        rgb_fusion_n_dif = rgb_fusion_n_dif.view(batch_size, length_size, -1)
        attn_weights_rgb_l_dif = self.softmax5(self.attn5(rgb_fusion_n_dif))
        rgb_l_dif = torch.sum(rgb_fusion_n_dif * attn_weights_rgb_l_dif, dim=1)
        rgb_l_dif = F.normalize(rgb_l_dif)

        # 锚点RGB+负样本毫米波
        rgb_fusion_n = rgb_fusion_n.view(batch_size, length_size, -1)
        attn_weights_rgb_l_n = self.softmax5(self.attn5(rgb_fusion_n))
        rgb_l_n2 = torch.sum(rgb_fusion_n * attn_weights_rgb_l_n, dim=1)
        rgb_l_n2 = F.normalize(rgb_l_n2)

        # 正样本毫米波+负样本RGB
        ti_fusion_dif = ti_fusion_dif.view(batch_size, length_size, -1)
        attn_weights_ti_dif = self.softmax4(self.attn4(ti_fusion_dif))
        ti_dif = torch.sum(ti_fusion_dif * attn_weights_ti_dif, dim=1)
        ti_dif = F.normalize(ti_dif)#正样本毫米波+负样本RGB

        # 负样本毫米波+锚点RGB
        ti_fusion_dif_n = ti_fusion_dif_n.view(batch_size, length_size, -1)
        attn_weights_ti_dif_n = self.softmax4(self.attn4(ti_fusion_dif_n))
        ti_dif_n = torch.sum(ti_fusion_dif_n * attn_weights_ti_dif_n, dim=1)
        ti_dif_n = F.normalize(ti_dif_n)

        return (
            rgb_l, ti_l, ti_l2, rgb_l_dif, ti_dif,
            key_pre_rgb['joints'], key_pre_ti['joints'], key_pre_ti2['joints'],  # 提取关节点
            key_pre_rgb_self['joints'], key_pre_ti_self['joints'], key_pre_ti_self_n['joints'],  # 提取关节点
            g_loc_p2, g_loc_n2, ti_dif_n, rgb_l_n2,
            # === 新增：返回生物特征字典 ===
            key_pre_rgb, key_pre_ti, key_pre_ti2,
            key_pre_rgb_self, key_pre_ti_self, key_pre_ti_self_n
            # === 新增结束 ===
        )

        # rgb_l  # 锚点RGB+正样本毫米波
        # ti_l   # 正样本毫米波+锚点RGB
        # ti_l2  # 负样本毫米波+负样本RGB
        #rgb_l_dif#负样本RGB+负样本毫米波
        # ti_dif # 正样本毫米波+负样本RGB

        # key_pre_rgb #融合后 锚点RGB+正样本毫米波 重建的骨架；
        # key_pre_ti  #融合后 正样本毫米波+锚点RGB 重建的骨架
        # key_pre_ti2 #融合后 负样本毫米波+负样本RGB 重建的骨架

        # key_pre_rgb_self  #融合前 锚点 RGB重建的骨架；
        # key_pre_ti_self   #融合前 正样本 毫米波重建的骨架
        # key_pre_ti_self_n #融合前 负样本毫米波重建的骨架





        #rgb_h：0，
        #ti_h：正样本毫米波（单模态）
        #ti_h2：负样本毫米波 （单模态）

        #key_pre_rgb： 融合后 锚点RGB with 正样本毫米波；
        #key_pre_ti：#融合后，正样本毫米波 with 锚点RGB融合）
        #key_pre_ti2：融合后，负样本毫米波 with 负样本 RGB融合）

        # output1 = rgb_l  # 融合后 锚点RGB with 正样本毫米波；（sum+norm）
        # output2 = ti_l  #融合后，正样本毫米波 with 锚点RGB融合）#正样本毫米波+锚点RGB(sum+norm)
        # output3 = ti_l2  #融合后，负样本毫米波 with 负样本 RGB融合）(sum+norm)

        #rgb_l, ti_l, ti_l2 同上
        #g_loc_p1, g_loc_p2, g_loc_n1, g_loc_n2,正样本，负样本毫米波的loc
        #key_pre_rgb_self,key_pre_ti_self,key_pre_ti_self_n，融合前
        #rgb_l2,（负样本毫米波+负样本 RGB融合）（sum+norm）

        # rgb_l_dif,#负样本RGB with 正样本毫米波（sum+norm）融合
        # ti_dif，#正样本毫米波 with 负样本RGB（sum+norm）融合

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname, map_location="cuda:0"))


class ruili_duanwu_smpl_test1(nn.Module):
    def __init__(self, device2):
        super(ruili_duanwu_smpl_test1, self).__init__()
        BASE_DATA_DIR = 'lib/models/pretrained/base_data'
        # self.model_hmr_l = ImageNet().to(device2)
        self.model_hmr_l = hmr_atten_14().to(device2)
        checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'),map_location=torch.device(device2))
        self.model_hmr_l.load_state_dict(checkpoint['model'], strict=False)

        self.model_ti  = mmWaveModel_ti_Anchor_nosmpl_bidirectional_loc().to(device2)
        self.model_ti2 = mmWaveModel_ti_Anchor_nosmpl_bidirectional_loc().to(device2)

        self.bpointnet = BasePointNet()
        self.conv3 = nn.Conv1d(256 + 27, 256, 1)  # 27+64+64
        self.cb3 = nn.BatchNorm1d(256)
        self.caf3 = nn.ReLU()

        self.module3 = CombineModule_mid_modal_smpl()
        self.module4 = CombineModule_mid_modal_smpl()
        self.module6 = CombineModule_mid_modal_smpl()

        self.conv1 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1)
        self.cb1 = nn.BatchNorm2d(256)
        self.caf1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.cb2 = nn.BatchNorm1d(256)
        self.caf2 = nn.ReLU()

        self.conv4 = nn.Conv1d(in_channels=2048, out_channels=256, kernel_size=1)
        self.cb4 = nn.BatchNorm1d(256)
        self.caf4 = nn.ReLU()

        #mutual attention后的特征和无mutual attenttion的对齐
        self.conv5 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.cb5 = nn.BatchNorm1d(256)
        self.caf5 = nn.ReLU()
        self.conv6 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.cb6 = nn.BatchNorm1d(256)
        self.caf6 = nn.ReLU()

        # nln模块
        self.nl = _NonLocalBlockND_2modules_pixelatten_res(in_channels=256, selfrgb=1)
        self.nl2 = _NonLocalBlockND_2modules_pixelatten_res(in_channels=256, selfrgb=0)
        # 模态间attention
        self.attn1 = nn.Linear(256, 1)
        self.softmax1 = nn.Softmax(dim=1)

        self.attn2 = nn.Linear(256, 1)
        self.softmax2 = nn.Softmax(dim=1)

        self.avgpool = nn.AvgPool2d(14, stride=1)

        # 步态周期attention
        self.attn3 = nn.Linear(256, 1)
        self.softmax3 = nn.Softmax(dim=1)
        self.attn4 = nn.Linear(256, 1)
        self.softmax4 = nn.Softmax(dim=1)
        self.attn5 = nn.Linear(256, 1)
        self.softmax5 = nn.Softmax(dim=1)
        self.attn6 = nn.Linear(256, 1)
        self.softmax6 = nn.Softmax(dim=1)

    def forward(self, x_rgb,x_rgb_n, ti_p, ti_n, h0, c0, batch_size, length_size):
        # x_rgb：锚点样本RGB数据
        # x_rgb_n：负样本RGB数据，
        # ti_p：正样本雷达数据
        # ti_n：负样本雷达数据

        #
        #视觉模态特征提取
        feature_hmr_l,_ = self.model_hmr_l.feature_extractor(x_rgb)#锚点样本特征
        feature_hmr_l_n, _ = self.model_hmr_l.feature_extractor(x_rgb_n)#负样本特征

        rgb_l = self.caf1(self.cb1(self.conv1(feature_hmr_l)))#锚点 RGB特征    40X14X14X256（通道）
        rgb_l_n = self.caf1(self.cb1(self.conv1(feature_hmr_l_n)))#负样本 RGB 特征



        # rgb_l = self.model_hmr_l(x_rgb)#锚点样本特征
        # rgb_l_n= self.model_hmr_l(x_rgb_n)#负样本特征

        # 毫米波 正样本特征提取
        g_vec_l, a_vec_l, g_loc_p2 = self.model_ti2(ti_p, h0, c0, batch_size, length_size)
        ti_l = torch.cat((g_vec_l, a_vec_l), dim=2)
        bpoint1 = self.bpointnet(ti_p)
        n_pts = ti_p.size()[1] #64个点
        ti_l = ti_l.view(batch_size*length_size, 1, 256).repeat(1, n_pts, 1)#4，10，64，256
        ti_l_ma = torch.cat([ti_l, bpoint1], 2)
        ti_l_ma = ti_l_ma.transpose(1, 2)
        ti_l_ma = self.caf3(self.cb3(self.conv3(ti_l_ma)))#正样本 毫米波每个点特征 （40，256，64）


        #毫米波 负样本特征提取
        g_vec_l2, a_vec_l2, g_loc_n2 = self.model_ti2(ti_n, h0, c0, batch_size, length_size)
        ti_l2 = torch.cat((g_vec_l2, a_vec_l2), dim=2)
        bpoint2 = self.bpointnet(ti_n)
        n_pts = ti_n.size()[1] #64个点
        ti_l2 = ti_l2.view(batch_size*length_size, 1, 256).repeat(1, n_pts, 1)#4，10，64，256
        ti_l_ma_n = torch.cat([ti_l2, bpoint2], 2)
        ti_l_ma_n = ti_l_ma_n.transpose(1, 2)
        ti_l_ma_n = self.caf3(self.cb3(self.conv3(ti_l_ma_n)))#正样本 毫米波每个点特征 （40，256，64）

        # 【nln模块计算】每个nln都计算self和mutual

        #【rgb_fusion】锚点RGB+正样本毫米波
        rgb_fusion, rgb_self, mutual_f_div_C = self.nl(rgb_l, ti_l_ma)
        rgb_fusion = self.avgpool(rgb_fusion)
        rgb_fusion = rgb_fusion.view(rgb_fusion.size(0), -1)#锚点RGB+正样本毫米波 40，256

        #【rgb_fusion_n_dif】负样本RGB+正样本毫米波
        rgb_fusion_n_dif, _, _ = self.nl(rgb_l_n, ti_l_ma)
        rgb_fusion_n_dif = self.avgpool(rgb_fusion_n_dif)#负样本RGB+正样本毫米波
        rgb_fusion_n_dif = rgb_fusion_n_dif.view(rgb_fusion_n_dif.size(0), -1)

        # 【rgb_fusion_n】锚点RGB+负样本毫米波
        rgb_fusion_n, _, _ = self.nl(rgb_l, ti_l_ma_n)
        rgb_fusion_n = self.avgpool(rgb_fusion_n)#负样本RGB+正样本毫米波
        rgb_fusion_n = rgb_fusion_n.view(rgb_fusion_n.size(0), -1)

        # 【ti_fusion】正样本毫米波+锚点RGB
        ti_fusion, ti_self ,_= self.nl2(ti_l_ma, rgb_l)
        ti_fusion = ti_fusion.transpose(1, 2)#融合RGB特征后的毫米波-点特征
        attn_weights = self.softmax2(self.attn2(ti_fusion))
        ti_fusion = torch.sum(ti_fusion * attn_weights, dim=1)#毫米波-帧特征（正样本+锚点 RGB融合）


        # 【ti_fusion_n】负样本毫米波+负样本RGB
        ti_fusion_n, ti_self_n, _ = self.nl2(ti_l_ma_n, rgb_l_n)
        ti_fusion_n = ti_fusion_n.transpose(1, 2)#毫米波-帧特征（负样本+负样本 RGB融合）
        attn_weights = self.softmax2(self.attn2(ti_fusion_n))
        ti_fusion_n = torch.sum(ti_fusion_n * attn_weights, dim=1)

        #【ti_fusion_dif】正样本毫米波+负样本RGB
        ti_fusion_dif, _, _ = self.nl2(ti_l_ma, rgb_l_n)
        ti_fusion_dif = ti_fusion_dif.transpose(1, 2)
        attn_weights = self.softmax2(self.attn2(ti_fusion_dif))
        ti_fusion_dif = torch.sum(ti_fusion_dif * attn_weights, dim=1)

        #【ti_fusion_dif_n】负样本毫米波+锚点 RGB
        ti_fusion_dif_n, _, _ = self.nl2(ti_l_ma_n, rgb_l)
        ti_fusion_dif_n = ti_fusion_dif_n.transpose(1, 2)
        attn_weights = self.softmax2(self.attn2(ti_fusion_dif_n))
        ti_fusion_dif_n = torch.sum(ti_fusion_dif_n * attn_weights, dim=1)



        #rgb单模态
        rgb_self = self.avgpool(rgb_self)#自注意力后的RGB
        rgb_self = rgb_self.view(rgb_self.size(0), -1)

        #正样本毫米波
        ti_self = ti_self.transpose(1, 2)#自注意力后的毫米波特征
        attn_weights = self.softmax1(self.attn1(ti_self))
        ti_self = torch.sum(ti_self * attn_weights, dim=1)#毫米波帧特征（自注意力）

        #【02】负样本毫米波
        ti_self_n = ti_self_n.transpose(1, 2)#负样本 毫米波自注意力
        attn_weights = self.softmax1(self.attn1(ti_self_n))
        ti_self_n = torch.sum(ti_self_n * attn_weights, dim=1)


        # reconstruction
        key_pre_rgb_self = self.module6(rgb_self, batch_size, length_size)#融合前 锚点 RGB重建的骨架；
        key_pre_ti_self = self.module3(ti_self, batch_size, length_size)#融合前 正样本 毫米波重建的骨架
        key_pre_ti_self_n = self.module3(ti_self_n, batch_size, length_size)#融合前 负样本毫米波重建的骨架

        key_pre_rgb = self.module4(rgb_fusion, batch_size, length_size)#融合后 锚点RGB+正样本毫米波 重建的骨架；
        key_pre_ti = self.module4(ti_fusion, batch_size, length_size)#融合后 正样本毫米波+锚点RGB 重建的骨架
        key_pre_ti2 = self.module4(ti_fusion_n, batch_size, length_size)#融合后 负样本毫米波+负样本RGB 重建的骨架

        # key_pre_rgb_self = key_pre_rgb_self.view(batch_size * length_size, 24, 3)
        # key_pre_ti_self = key_pre_ti_self.view(batch_size * length_size, 24, 3)
        # key_pre_ti_self_n = key_pre_ti_self_n.view(batch_size * length_size, 24, 3)
        #
        # key_pre_rgb = key_pre_rgb.view(batch_size * length_size, 24, 3)
        # key_pre_ti = key_pre_ti.view(batch_size * length_size, 24, 3)
        # key_pre_ti2 = key_pre_ti2.view(batch_size * length_size, 24, 3)

        #锚点RGB+正样本毫米波
        rgb_fusion = rgb_fusion.view(batch_size, length_size, -1)
        attn_weights_rgb_l = self.softmax5(self.attn5(rgb_fusion))
        rgb_l = torch.sum(rgb_fusion * attn_weights_rgb_l, dim=1)
        rgb_l = F.normalize(rgb_l)#锚点RGB+正样本毫米波

        # 正样本毫米波+锚点RGB
        ti_fusion = ti_fusion.view(batch_size, length_size, -1)
        attn_weights_ti_l = self.softmax4(self.attn4(ti_fusion))
        ti_l = torch.sum(ti_fusion * attn_weights_ti_l, dim=1)
        ti_l = F.normalize(ti_l) #正样本毫米波+锚点RGB

        # 负样本毫米波+负样本RGB
        ti_fusion_n = ti_fusion_n.view(batch_size, length_size, -1)
        attn_weights_ti_l2 = self.softmax4(self.attn4(ti_fusion_n))
        ti_l2 = torch.sum(ti_fusion_n * attn_weights_ti_l2, dim=1)
        ti_l2 = F.normalize(ti_l2)# 毫米波-帧特征（负样本+负样本 RGB融合）

        # 负样本RGB+负样本毫米波
        rgb_fusion_n_dif = rgb_fusion_n_dif.view(batch_size, length_size, -1)
        attn_weights_rgb_l_dif = self.softmax5(self.attn5(rgb_fusion_n_dif))
        rgb_l_dif = torch.sum(rgb_fusion_n_dif * attn_weights_rgb_l_dif, dim=1)
        rgb_l_dif = F.normalize(rgb_l_dif)

        # 锚点RGB+负样本毫米波
        rgb_fusion_n = rgb_fusion_n.view(batch_size, length_size, -1)
        attn_weights_rgb_l_n = self.softmax5(self.attn5(rgb_fusion_n))
        rgb_l_n2 = torch.sum(rgb_fusion_n * attn_weights_rgb_l_n, dim=1)
        rgb_l_n2 = F.normalize(rgb_l_n2)

        # 正样本毫米波+负样本RGB
        ti_fusion_dif = ti_fusion_dif.view(batch_size, length_size, -1)
        attn_weights_ti_dif = self.softmax4(self.attn4(ti_fusion_dif))
        ti_dif = torch.sum(ti_fusion_dif * attn_weights_ti_dif, dim=1)
        ti_dif = F.normalize(ti_dif)#正样本毫米波+负样本RGB

        # 负样本毫米波+锚点RGB
        ti_fusion_dif_n = ti_fusion_dif_n.view(batch_size, length_size, -1)
        attn_weights_ti_dif_n = self.softmax4(self.attn4(ti_fusion_dif_n))
        ti_dif_n = torch.sum(ti_fusion_dif_n * attn_weights_ti_dif_n, dim=1)
        ti_dif_n = F.normalize(ti_dif_n)

        return rgb_l, ti_l, ti_l2, rgb_l_dif,ti_dif, key_pre_rgb, key_pre_ti, key_pre_ti2, key_pre_rgb_self,key_pre_ti_self,key_pre_ti_self_n, g_loc_p2, g_loc_n2,ti_dif_n,rgb_l_n2

        # rgb_l  # 锚点RGB+正样本毫米波
        # ti_l   # 正样本毫米波+锚点RGB
        # ti_l2  # 负样本毫米波+负样本RGB
        #rgb_l_dif#负样本RGB+负样本毫米波
        # ti_dif # 正样本毫米波+负样本RGB

        # key_pre_rgb #融合后 锚点RGB+正样本毫米波 重建的骨架；
        # key_pre_ti  #融合后 正样本毫米波+锚点RGB 重建的骨架
        # key_pre_ti2 #融合后 负样本毫米波+负样本RGB 重建的骨架

        # key_pre_rgb_self  #融合前 锚点 RGB重建的骨架；
        # key_pre_ti_self   #融合前 正样本 毫米波重建的骨架
        # key_pre_ti_self_n #融合前 负样本毫米波重建的骨架





        #rgb_h：0，
        #ti_h：正样本毫米波（单模态）
        #ti_h2：负样本毫米波 （单模态）

        #key_pre_rgb： 融合后 锚点RGB with 正样本毫米波；
        #key_pre_ti：#融合后，正样本毫米波 with 锚点RGB融合）
        #key_pre_ti2：融合后，负样本毫米波 with 负样本 RGB融合）

        # output1 = rgb_l  # 融合后 锚点RGB with 正样本毫米波；（sum+norm）
        # output2 = ti_l  #融合后，正样本毫米波 with 锚点RGB融合）#正样本毫米波+锚点RGB(sum+norm)
        # output3 = ti_l2  #融合后，负样本毫米波 with 负样本 RGB融合）(sum+norm)

        #rgb_l, ti_l, ti_l2 同上
        #g_loc_p1, g_loc_p2, g_loc_n1, g_loc_n2,正样本，负样本毫米波的loc
        #key_pre_rgb_self,key_pre_ti_self,key_pre_ti_self_n，融合前
        #rgb_l2,（负样本毫米波+负样本 RGB融合）（sum+norm）

        # rgb_l_dif,#负样本RGB with 正样本毫米波（sum+norm）融合
        # ti_dif，#正样本毫米波 with 负样本RGB（sum+norm）融合

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname, map_location="cuda:0"))


class ruili_smpl(nn.Module):
    def __init__(self, device2):
        super(ruili_smpl, self).__init__()
        BASE_DATA_DIR = 'lib/models/pretrained/base_data'
        # self.model_hmr_l = ImageNet().to(device2)
        self.model_hmr_l = hmr_atten_14().to(device2)
        checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'),map_location=torch.device(device2))
        self.model_hmr_l.load_state_dict(checkpoint['model'], strict=False)

        self.model_ti2 = mmWaveModel_ti_Anchor_nosmpl_bidirectional_loc().to(device2)
        self.bpointnet = BasePointNet()


        # self.bpointnet = BasePointNet()
        # self.conv3 = nn.Conv1d(256 + 27, 256, 1)  # 27+64+64
        # self.cb3 = nn.BatchNorm1d(256)
        # self.caf3 = nn.ReLU()
        self.module1 = CombineModule_mid_modal_smpl()
        self.module2 = CombineModule_mid_modal_smpl()
        self.module3 = CombineModule_mid_modal_smpl()
        self.module4 = CombineModule_mid_modal_smpl()
        self.module5 = CombineModule_mid_modal_smpl()
        self.module6 = CombineModule_mid_modal_smpl()

        # self.smpl_layer = SMPL_Layer(
        #     center_idx=0,
        #     gender='male',
        #     model_root='./smplpytorch/native/models')



        self.conv1 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1)
        self.cb1 = nn.BatchNorm2d(256)
        self.caf1 = nn.ReLU()

        # nln模块
        self.nl = _NonLocalBlockND_2modules_pixelatten_res(in_channels=256, selfrgb=1)
        self.nl2 = _NonLocalBlockND_2modules_pixelatten_res(in_channels=256, selfrgb=0)
        # 模态间attention
        self.attn1 = nn.Linear(256, 1)
        self.softmax1 = nn.Softmax(dim=1)

        self.attn2 = nn.Linear(256, 1)
        self.softmax2 = nn.Softmax(dim=1)

        self.avgpool = nn.AvgPool2d(14, stride=1)

        # 步态周期attention

        self.attn4 = nn.Linear(256, 1)
        self.softmax4 = nn.Softmax(dim=1)
        self.attn5 = nn.Linear(256, 1)
        self.softmax5 = nn.Softmax(dim=1)


    def forward(self, x_rgb,x_rgb_n, ti_p, ti_n, h0, c0, batch_size, length_size):
        # x_rgb：锚点样本RGB数据
        # x_rgb_n：负样本RGB数据，
        # ti_p：正样本雷达数据
        # ti_n：负样本雷达数据

        #
        #视觉模态特征提取
        feature_hmr_l,_ = self.model_hmr_l.feature_extractor(x_rgb)#锚点样本特征
        feature_hmr_l_n, _ = self.model_hmr_l.feature_extractor(x_rgb_n)#负样本特征
        rgb_l = self.caf1(self.cb1(self.conv1(feature_hmr_l)))#锚点 RGB特征    40X14X14X256（通道）
        rgb_l_n = self.caf1(self.cb1(self.conv1(feature_hmr_l_n)))#负样本 RGB 特征


        # 毫米波 正样本特征提取
        g_vec_l, a_vec_l, _, g_loc_p2 = self.model_ti2(ti_p, h0, c0, batch_size, length_size)
        ti_l = torch.cat((g_vec_l, a_vec_l), dim=2)

        # 简化后的代码：
        ti_l = ti_l.view(batch_size, length_size, 1, 256)  # 增加一个维度用于广播
        n_pts = ti_p.size()[1] #64个点
        bpoint = self.bpointnet(ti_p).view(batch_size, length_size, n_pts, -1)  # 处理点特征
        # 先拼接再扩展
        ti_l_ma = torch.cat([
            ti_l.expand(-1, -1, n_pts, -1),  # 将ti_l扩展到每个点
            bpoint  # 点特征
        ], dim=-1)  # 在最后一个维度拼接

        # 调整维度用于卷积
        ti_l_ma = ti_l_ma.view(batch_size * length_size, n_pts, -1).transpose(1, 2)
        ti_l_ma = self.caf3(self.cb3(self.conv3(ti_l_ma)))  # 最终处理
        #毫米波 负样本特征提取
        g_vec_l2, a_vec_l2, key_pre_ti2, g_loc_n2,bpoint2 = self.model_ti2(ti_n, h0, c0, batch_size, length_size)
        ti_l2 = torch.cat((g_vec_l2, a_vec_l2), dim=2)
        n_pts = ti_n.size()[1] #64个点
        ti_l2 = ti_l2.view(batch_size*length_size, 1, 226).repeat(1, n_pts, 1)#4，10，64，256
        ti_l_ma_n = torch.cat([ti_l2, bpoint2,ti_n], 2)
        ti_l_ma_n = ti_l_ma_n.transpose(1, 2)

        # 【nln模块计算】每个nln都计算self和mutual

        #【rgb_fusion】锚点RGB+正样本毫米波
        rgb_fusion, rgb_self, mutual_f_div_C = self.nl(rgb_l, ti_l_ma)
        rgb_fusion = self.avgpool(rgb_fusion)
        rgb_fusion = rgb_fusion.view(rgb_fusion.size(0), -1)#锚点RGB+正样本毫米波 40，256

        #【rgb_fusion_n_dif】负样本RGB+负样本毫米波
        rgb_fusion_n_dif, _, _ = self.nl(rgb_l_n, ti_l_ma_n)
        rgb_fusion_n_dif = self.avgpool(rgb_fusion_n_dif)#负样本RGB+正样本毫米波
        rgb_fusion_n_dif = rgb_fusion_n_dif.view(rgb_fusion_n_dif.size(0), -1)

        # 【ti_fusion】正样本毫米波+锚点RGB
        ti_fusion, ti_self ,_= self.nl2(ti_l_ma, rgb_l)
        ti_fusion = ti_fusion.transpose(1, 2)#融合RGB特征后的毫米波-点特征
        attn_weights = self.softmax2(self.attn2(ti_fusion))
        ti_fusion = torch.sum(ti_fusion * attn_weights, dim=1)#毫米波-帧特征（正样本+锚点 RGB融合）


        # 【ti_fusion_n】负样本毫米波+负样本RGB
        ti_fusion_n, ti_self_n, _ = self.nl2(ti_l_ma_n, rgb_l_n)
        ti_fusion_n = ti_fusion_n.transpose(1, 2)#毫米波-帧特征（负样本+负样本 RGB融合）
        attn_weights = self.softmax2(self.attn2(ti_fusion_n))
        ti_fusion_n = torch.sum(ti_fusion_n * attn_weights, dim=1)

        #【ti_fusion_dif】正样本毫米波+负样本RGB
        ti_fusion_dif, _, _ = self.nl2(ti_l_ma, rgb_l_n)
        ti_fusion_dif = ti_fusion_dif.transpose(1, 2)
        attn_weights = self.softmax2(self.attn2(ti_fusion_dif))
        ti_fusion_dif = torch.sum(ti_fusion_dif * attn_weights, dim=1)

        #rgb单模态
        rgb_self = self.avgpool(rgb_self)#自注意力后的RGB
        rgb_self = rgb_self.view(rgb_self.size(0), -1)

        #正样本毫米波
        ti_self = ti_self.transpose(1, 2)#自注意力后的毫米波特征
        attn_weights = self.softmax1(self.attn1(ti_self))
        ti_self = torch.sum(ti_self * attn_weights, dim=1)#毫米波帧特征（自注意力）

        #【02】负样本毫米波
        ti_self_n = ti_self_n.transpose(1, 2)#负样本 毫米波自注意力
        attn_weights = self.softmax1(self.attn1(ti_self_n))
        ti_self_n = torch.sum(ti_self_n * attn_weights, dim=1)


        # reconstruction
        key_pre_rgb_self = self.module1(rgb_self, batch_size, length_size)#融合前 锚点 RGB重建的骨架；
        # t=key_pre_rgb_self['cam']
        key_pre_ti_self = self.module2(ti_self, batch_size, length_size)#融合前 正样本 毫米波重建的骨架
        key_pre_ti_self_n = self.module3(ti_self_n, batch_size, length_size)#融合前 负样本毫米波重建的骨架

        key_pre_rgb = self.module4(rgb_fusion, batch_size, length_size)#融合后 锚点RGB+正样本毫米波 重建的骨架；
        key_pre_ti = self.module5(ti_fusion, batch_size, length_size)#融合后 正样本毫米波+锚点RGB 重建的骨架
        key_pre_ti2 = self.module6(ti_fusion_n, batch_size, length_size)#融合后 负样本毫米波+负样本RGB 重建的骨架

        key_pre_rgb_self = key_pre_rgb_self.view(batch_size * length_size, 24, 3)
        key_pre_ti_self = key_pre_ti_self.view(batch_size * length_size, 24, 3)
        key_pre_ti_self_n = key_pre_ti_self_n.view(batch_size * length_size, 24, 3)

        key_pre_rgb = key_pre_rgb.view(batch_size * length_size, 24, 3)
        key_pre_ti = key_pre_ti.view(batch_size * length_size, 24, 3)
        key_pre_ti2 = key_pre_ti2.view(batch_size * length_size, 24, 3)

        #锚点RGB+正样本毫米波
        rgb_fusion = rgb_fusion.view(batch_size, length_size, -1)
        attn_weights_rgb_l = self.softmax5(self.attn5(rgb_fusion))
        rgb_l = torch.sum(rgb_fusion * attn_weights_rgb_l, dim=1)
        rgb_l = F.normalize(rgb_l)#锚点RGB+正样本毫米波

        # 正样本毫米波+锚点RGB
        ti_fusion = ti_fusion.view(batch_size, length_size, -1)
        attn_weights_ti_l = self.softmax4(self.attn4(ti_fusion))
        ti_l = torch.sum(ti_fusion * attn_weights_ti_l, dim=1)
        ti_l = F.normalize(ti_l) #正样本毫米波+锚点RGB

        # 负样本毫米波+负样本RGB
        ti_fusion_n = ti_fusion_n.view(batch_size, length_size, -1)
        attn_weights_ti_l2 = self.softmax4(self.attn4(ti_fusion_n))
        ti_l2 = torch.sum(ti_fusion_n * attn_weights_ti_l2, dim=1)
        ti_l2 = F.normalize(ti_l2)# 毫米波-帧特征（负样本+负样本 RGB融合）

        # 负样本RGB+负样本毫米波
        rgb_fusion_n_dif = rgb_fusion_n_dif.view(batch_size, length_size, -1)
        attn_weights_rgb_l_dif = self.softmax5(self.attn5(rgb_fusion_n_dif))
        rgb_l_dif = torch.sum(rgb_fusion_n_dif * attn_weights_rgb_l_dif, dim=1)
        rgb_l_dif = F.normalize(rgb_l_dif)

        # 正样本毫米波+负样本RGB
        ti_fusion_dif = ti_fusion_dif.view(batch_size, length_size, -1)
        attn_weights_ti_dif = self.softmax4(self.attn4(ti_fusion_dif))
        ti_dif = torch.sum(ti_fusion_dif * attn_weights_ti_dif, dim=1)
        ti_dif = F.normalize(ti_dif)#正样本毫米波+负样本RGB


        return rgb_l, ti_l, ti_l2, rgb_l_dif, ti_dif, key_pre_rgb, key_pre_ti, key_pre_ti2, key_pre_rgb_self, key_pre_ti_self, key_pre_ti_self_n, g_loc_p2, g_loc_n2

        # rgb_l  # 锚点RGB+正样本毫米波
        # ti_l   # 正样本毫米波+锚点RGB
        # ti_l2  # 负样本毫米波+负样本RGB
        #rgb_l_dif#负样本RGB+负样本毫米波
        # ti_dif # 正样本毫米波+负样本RGB

        # key_pre_rgb #融合后 锚点RGB+正样本毫米波 重建的骨架；
        # key_pre_ti  #融合后 正样本毫米波+锚点RGB 重建的骨架
        # key_pre_ti2 #融合后 负样本毫米波+负样本RGB 重建的骨架

        # key_pre_rgb_self  #融合前 锚点 RGB重建的骨架；
        # key_pre_ti_self   #融合前 正样本 毫米波重建的骨架
        # key_pre_ti_self_n #融合前 负样本毫米波重建的骨架





        #rgb_h：0，
        #ti_h：正样本毫米波（单模态）
        #ti_h2：负样本毫米波 （单模态）

        #key_pre_rgb： 融合后 锚点RGB with 正样本毫米波；
        #key_pre_ti：#融合后，正样本毫米波 with 锚点RGB融合）
        #key_pre_ti2：融合后，负样本毫米波 with 负样本 RGB融合）

        # output1 = rgb_l  # 融合后 锚点RGB with 正样本毫米波；（sum+norm）
        # output2 = ti_l  #融合后，正样本毫米波 with 锚点RGB融合）#正样本毫米波+锚点RGB(sum+norm)
        # output3 = ti_l2  #融合后，负样本毫米波 with 负样本 RGB融合）(sum+norm)

        #rgb_l, ti_l, ti_l2 同上
        #g_loc_p1, g_loc_p2, g_loc_n1, g_loc_n2,正样本，负样本毫米波的loc
        #key_pre_rgb_self,key_pre_ti_self,key_pre_ti_self_n，融合前
        #rgb_l2,（负样本毫米波+负样本 RGB融合）（sum+norm）

        # rgb_l_dif,#负样本RGB with 正样本毫米波（sum+norm）融合
        # ti_dif，#正样本毫米波 with 负样本RGB（sum+norm）融合

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, pathname):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(pathname, map_location="cuda:0"))


