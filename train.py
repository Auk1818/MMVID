import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader
from losses import TripletLoss
# from network import smpl_tmc as mmwave_model
# from network import tianshun as mmwave_model
from network import ruili_duanwu as mmwave_model

from dataset_me_rgb3 import Sensys_SMPL as mmwave_data
# test_data = mmwave_data(train=True,issever=1)



from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


cudanum = 0
if torch.cuda.is_available():
    device = 'cuda:%d' % (cudanum)
    print("GPU is available")
else:
    device = 'cpu'

#s和t分开计算
iftrain = 1
ifallfusion = 1
import platform
if ('Windows' == platform.system()):
    issever = 1
else:
    issever = 1


ifrunonce = 1

#mutual attention是否监督self attention输出
self_atten = 1
num_epochs = 5000
learning_rate = 0.0002
#性能最优值
top1_best = 0
top5_best = 0
top10_best = 0

kprgb_best = 100
kpti1_best = 100
kpti2_best = 100
top1_best_test = 0
top5_best_test = 0
top10_best_test = 0

map_best_test = 0
# top1_best_test_l = 0
# top5_best_test_l = 0
# top10_best_test_l = 0
# map_best_test_l = 0

top1_best_test_h = 0
top5_best_test_h = 0
top10_best_test_h = 0
map_best_test_h = 0

kprgb_best_test = 100
kpti1_best_test = 100
kpti2_best_test = 100

batchsize = 4
batchsize2 =4

#38人只有20帧，所以+5
length_size=10

criterion_keypoints = nn.MSELoss(reduction='none').to(device)
writer1=SummaryWriter("C:/Users/lenovo/Desktop/mission/loss/ruili_pose_duanwu")
matname = './res/midmodal_allfusion_38_divded_duanwu.mat'

def save_models(epoch):
    path = "./log/ruili_pose_duanwu"
    import os
    if not os.path.exists(path):
        os.mkdir(path)
    torch.save(model.state_dict(), path + "/model_{}.pth".format(epoch))

model = mmwave_model(device).to(device)
# model.load('./log/final_model_38/model_{}.pth'.format('best'))


margin=0.3
loss_fn = TripletLoss(margin)

idloss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
test_data = mmwave_data(train=False,issever=issever)
test_loader = DataLoader(test_data, batch_size = batchsize2, shuffle = True,drop_last=True)#51×4

train_data = mmwave_data(issever=issever)
train_loader = DataLoader(train_data, batch_size=batchsize, shuffle=True, drop_last=True)#119*batch=4

loss_total=0
eval_loss_total=0

def evaluate(qf, ql, gf, gl):
    query = qf
    score=np.sum(np.square(query - gf),1)
    index = np.argsort(score)  # from small to large，返回相似度最高的索引
    query_index = np.argwhere(gl == ql)#表示图库中哪些图像的标签与查询图像的标签相同，返回为true的索引
    good_index = query_index
    junk_index = np.argwhere(gl == -1)#找出图库中标签为 -1 的图像的索引，不加入计算。
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp

def evaluate_l(qf, ql, gf, gl):
    query = qf
    score=np.sum(np.square(query - gf),1)
    index = np.argsort(score)  # from small to large
    query_index = np.argwhere(gl == ql)
    good_index = query_index
    junk_index = np.argwhere(gl == -1)
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp,score,index

def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:  # 没有任何一次匹配
        cmc[0] = -1
        return ap, cmc

    mask = np.in1d(index, junk_index, invert=True)#保留不在junk_index中的索引
    index = index[mask]#去除掉本身对应的

    ngood = len(good_index)
    mask = np.in1d(index, good_index)#保留在good_index中的索引
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()
    cmc[rows_good[0]:] = 1#将 cmc 中从第一个“好”的匹配结果开始的位置设置为 1。这表示从第一个“好”的匹配结果开始，CMC 曲线的值为 1。
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2
    return ap, cmc

for epoch in range(num_epochs):
    gallery_feature = []
    query_feature = []
    gallery_feature_kp = []
    query_feature_kp = []
    gallery_label = []
    query_label = []
    train_rgb_skeleton = []
    train_ti_skeleton = []
    train_rgb_gt = []
    train_ti_gt = []
    flag_single = 0
    flag_single_kp = 0
    flag_multi = 0
    if (epoch+1)%5==0:
        print("epoch: {}".format(epoch+1))

    if iftrain==1:
        training_loss = []
        # training_idloss = []
        training_loss_s1 = []
        training_loss_s2 = []
        training_loss_s3 = []
        training_loss_loc = []
        training_loss_s1_mpjpe = []
        training_loss_s2_mpjpe = []
        training_loss_s3_mpjpe = []
        training_loss_s1_mpjpe_self = []
        training_loss_s2_mpjpe_self = []
        training_loss_s1_mpjpe_t = []
        training_loss_s2_mpjpe_t = []
        training_loss_s3_mpjpe_t = []



        model.train()
        for batch_idx,data in tqdm(enumerate(train_loader)):
            # data=np.asarray(data)


            batch_size, seq_len, point_num, dim = data[1].shape
            #print("dim:",dim)
            data_ti_in = np.reshape(data[1], (batch_size * seq_len, point_num, dim))
            data_ti = data_ti_in.clone().to(device, dtype=torch.float32).squeeze()#data[1]：正样本雷达数据

            # print("ti2:",data[1].shape)
            batch_size, seq_len, point_num, dim = data[2].shape
            data_ti_in2 = np.reshape(data[2], (batch_size * seq_len, point_num, dim))
            data_ti2 = data_ti_in2.clone().to(device, dtype=torch.float32).squeeze()#data[2]：负样本雷达数据

            batch_size, seq_len, dim, x, y = data[0].shape
            data_rgb_in = np.reshape(data[0], (batch_size * seq_len, dim, x, y))
            data_rgb = data_rgb_in.clone().to(device, dtype=torch.float32).squeeze()#data[0]：锚点样本RGB数据

            batch_size, seq_len, dim, x, y = data[9].shape
            data_rgb_in2 = np.reshape(data[9], (batch_size * seq_len, dim,x,y))
            data_rgb2 = data_rgb_in2.clone().to(device, dtype=torch.float32).squeeze()#data[9]：负样本RGB数据

            batch_size, seq_len, point_num, dim = data[6].shape
            data_key_in = np.reshape(data[6], (batch_size * seq_len, point_num, dim))
            data_key = data_key_in.clone().to(device, dtype=torch.float32).squeeze()#data[6]：锚点样本的key数据

            data_key_in2 = np.reshape(data[7], (batch_size * seq_len, point_num, dim))
            data_key2 = data_key_in2.clone().to(device, dtype=torch.float32).squeeze()#data[7]：正样本key数据

            data_key_in3 = np.reshape(data[8], (batch_size * seq_len, point_num, dim))
            data_key3 = data_key_in3.clone().to(device, dtype=torch.float32).squeeze()#data[8]：负样本key数据
            data[3] = np.asarray(data[3], dtype=int)
            data[4] = np.asarray(data[4], dtype=int)
            data[5] = np.asarray(data[5], dtype=int)

            optimizer.zero_grad()
            h0 = torch.zeros((6, batchsize, 64), dtype=torch.float32, device=device)
            c0 = torch.zeros((6, batchsize, 64), dtype=torch.float32, device=device)
            #kp-reid-ptnet-lstm
            h1 = torch.zeros((3, batchsize, 64), dtype=torch.float32, device=device)
            c1 = torch.zeros((3, batchsize, 64), dtype=torch.float32, device=device)

            rgb_l, ti_l, ti_l2, rgb_l_dif,ti_dif, key_pre_rgb, key_pre_ti, key_pre_ti2, key_pre_rgb_self,key_pre_ti_self,key_pre_ti_self_n, g_loc_p2, g_loc_n2,ti_dif_n,rgb_l_n2,output_rgb,output_ti,output_ti2,output_rgb_self,output_ti_self,output_ti_self_n = \
                model(data_rgb, data_rgb2, data_ti, data_ti2, h0, c0, batchsize, length_size )
            #【input】
            #data_rgb：锚点样本RGB数据
            #data_rgb2：负样本RGB数据，
            #data_ti：正样本雷达数据
            #data_ti2：负样本雷达数据

            # ti_l2 = ti_dif_n #负样本 mmwave +锚点 RGB

            loss_metric_l = loss_fn(rgb_l, ti_l, ti_l2)
            #【锚  点】 锚点RGB+正样本毫米波
            #【正样本】 正样本毫米波+锚点RGB
            #【负样本】 负样本毫米波+负样本RGB


            loss_metric_l_dif1 = loss_fn(rgb_l_dif, ti_l2, ti_l)
            # 【锚  点】 负样本RGB+负样本毫米波
            # 【正样本】 负样本毫米波+负RGB
            # 【负样本】 正样本毫米波+锚点RGB


            # key_pre_rgb #融合后 锚点RGB+正样本毫米波 重建的骨架；
            # key_pre_ti  #融合后 正样本毫米波+锚点RGB 重建的骨架
            # key_pre_ti2 #融合后 负样本毫米波+负样本RGB 重建的骨架
            # key_pre_rgb_self  #融合前 锚点 RGB重建的骨架；
            # key_pre_ti_self   #融合前 正样本 毫米波重建的骨架
            # key_pre_ti_self_n #融合前 负样本毫米波重建的骨架
            loss_s1_part = torch.mean(torch.sqrt(torch.sum(torch.square(
                torch.cat((key_pre_rgb[:, :10], key_pre_rgb[:, 12:22]), dim=1) - torch.cat(
                    (data_key[:, :10], data_key[:, 12:22]),
                    dim=1)), dim=-1)), dim=0)

            loss_s2_part = torch.mean(torch.sqrt(torch.sum(torch.square(
                torch.cat((key_pre_ti[:, :10], key_pre_ti[:, 12:22]), dim=1) - torch.cat(
                    (data_key2[:, :10], data_key2[:, 12:22]),
                    dim=1)), dim=-1)), dim=0)
            loss_s3_part = torch.mean(torch.sqrt(torch.sum(torch.square(
                torch.cat((key_pre_ti2[:, :10], key_pre_ti2[:, 12:22]), dim=1) - torch.cat(
                    (data_key3[:, :10], data_key3[:, 12:22]),
                    dim=1)), dim=-1)), dim=0)
            loss_s1_mpjpe = torch.mean(loss_s1_part)
            loss_s2_mpjpe = torch.mean(loss_s2_part)
            loss_s3_mpjpe = torch.mean(loss_s3_part)



            loss_keypoint = (loss_s1_mpjpe+ loss_s2_mpjpe + loss_s3_mpjpe) / 3

            #单模态-自监督
            loss_s1_part_self = torch.mean(torch.sqrt(torch.sum(torch.square(
                torch.cat((key_pre_rgb_self[:, :10], key_pre_rgb_self[:, 12:22]), dim=1) - torch.cat(
                    (data_key[:, :10], data_key[:, 12:22]),
                    dim=1)), dim=-1)), dim=0)
            loss_s2_part_self = torch.mean(torch.sqrt(torch.sum(torch.square(
                torch.cat((key_pre_ti_self[:, :10], key_pre_ti_self[:, 12:22]), dim=1) - torch.cat(
                    (data_key2[:, :10], data_key2[:, 12:22]),
                    dim=1)), dim=-1)), dim=0)
            loss_s3_part_self = torch.mean(torch.sqrt(torch.sum(torch.square(
                torch.cat((key_pre_ti_self_n[:, :10], key_pre_ti_self_n[:, 12:22]), dim=1) - torch.cat(
                    (data_key3[:, :10], data_key3[:, 12:22]),
                    dim=1)), dim=-1)), dim=0)
            loss_s1_mpjpe_self = torch.mean(loss_s1_part_self)
            loss_s2_mpjpe_self = torch.mean(loss_s2_part_self)
            loss_s3_mpjpe_self = torch.mean(loss_s3_part_self)
            loss_keypoint =(loss_keypoint + (loss_s1_mpjpe_self+loss_s2_mpjpe_self+loss_s3_mpjpe_self)/3)/2

            #loc-loss
            # loss_loc_p1=torch.mean(torch.sqrt(torch.sum(torch.square(g_loc_p1-data_key2[:, 0,:2]), dim=-1)), dim=0)
            loss_loc_p2 = torch.mean(torch.sqrt(torch.sum(torch.square(g_loc_p2 - data_key2[:, 0,:2]), dim=-1)), dim=0)
            # loss_loc_n1 = torch.mean(torch.sqrt(torch.sum(torch.square(g_loc_n1 - data_key3[:, 0,:2]), dim=-1)), dim=0)
            loss_loc_n2 = torch.mean(torch.sqrt(torch.sum(torch.square(g_loc_n2 - data_key3[:, 0,:2]), dim=-1)), dim=0)
            loss_loc = (loss_loc_p2 + loss_loc_n2) / 2
            training_loss_loc.append(loss_loc.item())

            training_loss_s1_mpjpe.append(loss_s1_mpjpe.item())
            training_loss_s2_mpjpe.append(loss_s2_mpjpe.item())
            training_loss_s3_mpjpe.append(loss_s3_mpjpe.item())
            # loss_keypoint =(loss_keypoint + (loss_s1_mpjpe_self+loss_s2_mpjpe_self+loss_s3_mpjpe_self)/3)/2



            # self attention结果监督
            training_loss_s1_mpjpe_self.append(loss_s1_mpjpe_self.item())
            training_loss_s2_mpjpe_self.append(loss_s2_mpjpe_self.item())


            key_rgb_out = torch.cat((key_pre_rgb[:, :10], key_pre_rgb[:, 12:22]), dim=1).view(batch_size, length_size, -1)
            key_ti_out = torch.cat((key_pre_ti[:, :10], key_pre_ti[:, 12:22]), dim=1).view(batch_size,length_size,-1)
            key_ti2_out = torch.cat((key_pre_ti2[:, :10], key_pre_ti2[:, 12:22]), dim=1).view(batch_size, length_size, -1)

            key_rgb_out2 = torch.flatten(key_rgb_out, start_dim=1, end_dim=2)
            key_ti_out2 = torch.flatten(key_ti_out, start_dim=1, end_dim=2)
            key_ti2_out2 = torch.flatten(key_ti2_out, start_dim=1, end_dim=2)

            show_rgb = key_pre_rgb.cpu().detach()
            show_ti = key_pre_ti2.cpu().detach()
            show_gt_rgb = data_key.cpu().detach()
            show_gt_ti = data_key3.cpu().detach()

            # === 新增：计算生物特征损失 ===
            # 从模型输出字典中提取预测值
            pred_height_rgb = output_rgb['height']  # 锚点RGB+正样本毫米波 预测的身高
            pred_bmi_rgb = output_rgb['bmi']  # 锚点RGB+正样本毫米波 预测的BMI

            pred_height_ti = output_ti['height']  # 正样本毫米波+锚点RGB 预测的身高
            pred_bmi_ti = output_ti['bmi']  # 正样本毫米波+锚点RGB 预测的BMI

            pred_height_ti2 = output_ti2['height']  # 负样本毫米波+负样本RGB 预测的身高
            pred_bmi_ti2 = output_ti2['bmi']  # 负样本毫米波+负样本RGB 预测的BMI

            # 提取真实值 (根据你在 __getitem__ 中的顺序)
            # height_a, bmi_a, height_p, bmi_p, height_n, bmi_n
            true_height_a = torch.tensor(data[19], dtype=torch.float32, device=device)  # 锚点真实身高
            true_bmi_a = torch.tensor(data[20], dtype=torch.float32, device=device)  # 锚点真实BMI

            true_height_p = torch.tensor(data[21], dtype=torch.float32, device=device)  # 正样本真实身高
            true_bmi_p = torch.tensor(data[22], dtype=torch.float32, device=device)  # 正样本真实BMI

            true_height_n = torch.tensor(data[23], dtype=torch.float32, device=device)  # 负样本真实身高
            true_bmi_n = torch.tensor(data[24], dtype=torch.float32, device=device)  # 负样本真实BMI

            # 计算 MSE 损失
            loss_height_rgb = F.mse_loss(pred_height_rgb, true_height_a)
            loss_bmi_rgb = F.mse_loss(pred_bmi_rgb, true_bmi_a)

            loss_height_ti = F.mse_loss(pred_height_ti, true_height_p)
            loss_bmi_ti = F.mse_loss(pred_bmi_ti, true_bmi_p)

            loss_height_ti2 = F.mse_loss(pred_height_ti2, true_height_n)
            loss_bmi_ti2 = F.mse_loss(pred_bmi_ti2, true_bmi_n)

            # 平均损失
            loss_biometric = (
                                     loss_height_rgb + loss_bmi_rgb +
                                     loss_height_ti + loss_bmi_ti +
                                     loss_height_ti2 + loss_bmi_ti2
                             ) / 6.0

            a = 0.4  # keypoint loss weight
            b = 0.1  # biometric loss weight
            loss = a * loss_keypoint + loss_loc + a*loss_metric_l_dif1 +loss_metric_l + b * loss_biometric
            # loss = a * loss_keypoint + loss_loc + a*loss_metric_l_dif1+ loss_metric_l

            loss.backward()

            optimizer.step()
            training_loss.append(loss.item())

            for i in range(batchsize):#配合外面的epoch循环，把所有的样本都加入进来
                train_rgb_gt.append(data[6].cpu().detach().numpy()[i])#锚点样本的关键点数据。6
                train_ti_gt.append(data[7].cpu().detach().numpy()[i])#正样本的关键点数据。7
                train_rgb_skeleton.append(key_rgb_out.cpu().detach().numpy()[i])#融合后 锚点RGB with 正样本毫米波；#RGB关键点特征
                train_ti_skeleton.append(key_ti_out.cpu().detach().numpy()[i])##融合后，正样本毫米波 with 锚点RGB融合） #mmWave关键点特征

                gallery_feature.append(rgb_l.cpu().detach().numpy()[i])   # output1 = rgb_l  # 融合后 锚点RGB with 正样本毫米波；（sum+norm）
                gallery_feature_kp.append(key_rgb_out2.cpu().detach().numpy()[i])#融合后 锚点RGB with 正样本毫米波的骨架；
                gallery_label.append(np.asarray(data[3])[i]) #labelrgb：当前样本的标签

                query_feature.append(ti_l.cpu().detach().numpy()[i])#正样本毫米波+锚点RGB(sum+norm)
                query_feature_kp.append(key_ti_out2.cpu().detach().numpy()[i])##融合后，正样本毫米波 with 锚点RGB融合）
                query_label.append(np.asarray(data[4])[i])#正样本的标签

            #【输出】所有的
            #gallery_feature：【RGB模态】锚点RGB with 正样本毫米波
            #query_feature：【毫米波模态】正样本毫米波 with 锚点RGB

        training_loss = np.mean(training_loss)
        # training_idloss = np.mean(training_idloss)
        loss_total += training_loss

        ap1 = 0.0
        # ap1_kp = 0.0
        CMC1 = torch.IntTensor(len(gallery_label)).zero_()
        # CMC1_kp = torch.IntTensor(len(gallery_label)).zero_()

        # 测试标准
        for i in range(len(query_label)):
            ap_tmp, CMC_tmp = evaluate(query_feature[i], query_label[i], gallery_feature, gallery_label)

            if CMC_tmp[0] == -1:#标记位，-1代表没有任何一次匹配。
                continue
            #if query_label[i] < 15 or query_label[i] > 20:
            CMC1 = CMC1 + CMC_tmp
            ap1 += ap_tmp
            flag_single = flag_single + 1

        CMC1 = CMC1.float()
        CMC1 = CMC1 / flag_single  # average CMC
        CMC1 = np.asarray(CMC1)

        #CMC_kp
        # for i in range(len(query_label)):
        #     ap_tmp_kp, CMC_tmp_kp = evaluate(query_feature_kp[i], query_label[i], gallery_feature_kp, gallery_label)
        #     if CMC_tmp_kp[0] == -1:
        #         continue
        #     CMC1_kp = CMC1_kp + CMC_tmp_kp
        #     ap1_kp += ap_tmp_kp
        #     flag_single_kp = flag_single_kp + 1
        #
        # CMC1_kp = CMC1_kp.float()
        # CMC1_kp = CMC1_kp / flag_single_kp  # average CMC
        #
        # CMC1_kp = np.asarray(CMC1_kp)

        writer1.add_scalars('train_Rank-n_single',
                            {'rank1': torch.tensor(CMC1[0], dtype=float),
                             'rank2': torch.tensor(CMC1[1]),
                             'rank3': torch.tensor(CMC1[2]),
                             'rank4': torch.tensor(CMC1[3]),
                             'rank5': torch.tensor(CMC1[4]),
                             'rank6': torch.tensor(CMC1[5]),
                             'rank7': torch.tensor(CMC1[6]),
                             'rank8': torch.tensor(CMC1[7]),
                             'rank9': torch.tensor(CMC1[8]),
                             'rank10': torch.tensor(CMC1[9]),
                             'rank11': torch.tensor(CMC1[10]),
                             'rank12': torch.tensor(CMC1[11]),
                             'rank13': torch.tensor(CMC1[12]),
                             'rank14': torch.tensor(CMC1[13]),
                             'rank15': torch.tensor(CMC1[14])
                             }, epoch)

        writer1.add_scalar(tag='train_mAP_single', scalar_value=ap1 / flag_single, global_step=epoch)
        writer1.add_scalars('train_keypre',
                           {   'train_keypoint_rgb': np.mean(training_loss_s1_mpjpe),
                               'train_keypoint_ti1': np.mean(training_loss_s2_mpjpe),
                               'train_keypoint_ti2': np.mean(training_loss_s3_mpjpe),
                           }, epoch)

        writer1.add_scalars('train_keypre_self',
                            {'train_keypoint_rgb_self': np.mean(training_loss_s1_mpjpe_self),
                             'train_keypoint_ti_self': np.mean(training_loss_s2_mpjpe_self),
                             }, epoch)
        writer1.add_scalar('train_loc',np.mean(training_loss_loc),epoch)


        writer1.add_scalar(tag='train_loss', scalar_value=training_loss, global_step=epoch)
        #writer1.add_scalar(tag='train_idloss', scalar_value=training_idloss, global_step=epoch)
        #writer1.add_scalar(tag='train_mAP_multi', scalar_value=ap2 / flag_multi, global_step=epoch)

        # === 新增：在 TensorBoard 中记录生物特征损失 ===
        writer1.add_scalars('train_biometric',
                            {'train_height_loss': loss_height_rgb.item(),
                             'train_bmi_loss': loss_bmi_rgb.item(),
                             }, epoch)

        if (epoch + 1) % 5 == 0:
            print('train_loss mean after {} epochs: {}'.format((epoch + 1), loss_total / 5))
            print('train_top1: {}'.format(CMC1[0]))
            print('train_top5: {}'.format(CMC1[4]))
            print('train_mAP: {}'.format(ap1 / len(query_label)))
            loss_total = 0.


        # 更新最优解
        # top1_best = max(top1_best, CMC1[0])
        # top5_best = max(top1_best, CMC1[4])
        # top10_best = max(top1_best, CMC1[9])
        # kprgb_best = min(kprgb_best, np.mean(training_loss_s1_mpjpe))
        # kpti1_best = min(kpti1_best, np.mean(training_loss_s2_mpjpe))
        # kpti2_best = min(kpti2_best, np.mean(training_loss_s3_mpjpe))

    if (epoch+1)%5 != 0 and ifrunonce==1:#每训练10轮，测试一次
        continue

    #测试集
    model.eval()
    eval_loss = []
    eval_idloss = []
    eval_loss_s1 = []
    eval_loss_s2 = []
    eval_loss_s3 = []
    eval_loss_s1_self = []
    eval_loss_s2_self = []
    eval_loss_s3_self = []

    # loss_loc_p1 = []
    loss_loc_p2 = []
    # loss_loc_n1 = []
    loss_loc_n2 = []
    eval_loss_loc = []
    eval_loss_frame1 = []
    eval_loss_frame2 = []
    eval_loss_frame3 = []
    eval_loss_t_rgb = []
    eval_loss_t_ti = []
    eval_loss_t_ti2 = []
    gallery_feature = []
    gallery_feature_l = []
    gallery_feature_h = []
    query_feature = []
    query_feature_l = []
    query_feature_h = []
    gallery_feature_kp = []
    query_feature_kp = []
    gallery_label = []
    query_label = []
    gallery_label_l = []
    query_label_l = []
    query_feature_kp_singleti = []
    gallery_feature_kp_singleti = []
    query_feature_singleti = []
    gallery_feature_singleti = []
    gallery_label_singleti = []
    query_label_singleti = []
    eval_rgb_gt = []
    eval_ti_gt = []
    eval_rgb_skeleton = []
    eval_ti_skeleton = []
    flag_single = 0
    flag_single_l = 0
    flag_single_h = 0
    flag_single_kp = 0
    flag_single_singleti = 0
    flag_single_kp_singleti = 0
    flag_multi = 0
    correct_ti =0
    correct_rgb = 0
    eval_accu_flag = 0
    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(test_loader)):
            data = np.asarray(data)

            batch_size, seq_len, point_num, dim = data[1].shape
            data_ti_in = np.reshape(data[1], (batch_size * seq_len, point_num, dim))
            data_ti = torch.tensor(data_ti_in, dtype=torch.float32, device=device).squeeze()

            batch_size, seq_len, point_num, dim = data[2].shape
            data_ti_in2 = np.reshape(data[2], (batch_size * seq_len, point_num, dim))
            data_ti2 = torch.tensor(data_ti_in2, dtype=torch.float32, device=device).squeeze()

            data_ti_rand = torch.rand(batch_size*seq_len,64,3,device=device)

            batch_size, seq_len, dim, x, y = data[0].shape
            data_rgb_in = np.reshape(data[0], (batch_size * seq_len, dim, x, y))
            data_rgb = torch.tensor(data_rgb_in, dtype=torch.float32, device=device)

            batch_size, seq_len, dim, x, y = data[9].shape
            data_rgb_in2 = np.reshape(data[9], (batch_size * seq_len, dim, x, y))
            data_rgb2 = torch.tensor(data_rgb_in2, dtype=torch.float32, device=device)

            batch_size, seq_len, point_num, dim = data[6].shape
            data_key_in = np.reshape(data[6], (batch_size * seq_len, point_num, dim))
            data_key = torch.tensor(data_key_in, dtype=torch.float32, device=device)

            data_key_in2 = np.reshape(data[7], (batch_size * seq_len, point_num, dim))
            data_key2 = torch.tensor(data_key_in2, dtype=torch.float32, device=device)

            data_key_in3 = np.reshape(data[8], (batch_size * seq_len, point_num, dim))
            data_key3 = torch.tensor(data_key_in3, dtype=torch.float32, device=device)

            data[3] = np.asarray(data[3], dtype=int)
            data[4] = np.asarray(data[4], dtype=int)
            data[5] = np.asarray(data[5], dtype=int)

            h0 = torch.zeros((6, batchsize2, 64), dtype=torch.float32, device=device)
            c0 = torch.zeros((6, batchsize2, 64), dtype=torch.float32, device=device)

            h1 = torch.zeros((3, batchsize, 64), dtype=torch.float32, device=device)
            c1 = torch.zeros((3, batchsize, 64), dtype=torch.float32, device=device)



            (   rgb_l, ti_l, ti_l2, rgb_l_dif, ti_dif,
                key_pre_rgb, key_pre_ti, key_pre_ti2,
                key_pre_rgb_self, key_pre_ti_self, key_pre_ti_self_n,
                g_loc_p2, g_loc_n2, ti_dif_n, rgb_l_n2,
                output_rgb, output_ti, output_ti2,
                output_rgb_self, output_ti_self, output_ti_self_n
            ) = model(data_rgb, data_rgb2, data_ti, data_ti2, h0, c0, batchsize, length_size)


            loss_s1_part = torch.mean(torch.sqrt(torch.sum(torch.square(
                torch.cat((key_pre_rgb[:, :10], key_pre_rgb[:, 12:22]), dim=1) - torch.cat((data_key[:, :10], data_key[:, 12:22]),
                                                                                   dim=1)), dim=-1)), dim=0)
            loss_s2_part = torch.mean(torch.sqrt(torch.sum(torch.square(
                torch.cat((key_pre_ti[:, :10], key_pre_ti[:, 12:22]), dim=1) - torch.cat(
                    (data_key2[:, :10], data_key2[:, 12:22]),
                    dim=1)), dim=-1)), dim=0)
            loss_s3_part = torch.mean(torch.sqrt(torch.sum(torch.square(
                torch.cat((key_pre_ti2[:, :10], key_pre_ti2[:, 12:22]), dim=1) - torch.cat(
                    (data_key3[:, :10], data_key3[:, 12:22]),
                    dim=1)), dim=-1)), dim=0)

            loss_s1 = torch.mean(loss_s1_part)
            loss_s2 = torch.mean(loss_s2_part)
            loss_s3 = torch.mean(loss_s3_part)
            eval_loss_s1.append(loss_s1.item())
            eval_loss_s2.append(loss_s2.item())
            eval_loss_s3.append(loss_s3.item())
            # loc-loss
            # loss_loc_p1 = torch.mean(torch.sqrt(torch.sum(torch.square(g_loc_p1 - data_key2[:, 0,:2]), dim=-1)), dim=0)
            #print("loc_gt:",data_key2[0, 0])
            loss_loc_p2 = torch.mean(torch.sqrt(torch.sum(torch.square(g_loc_p2 - data_key2[:, 0,:2]), dim=-1)), dim=0)
            # loss_loc_n1 = torch.mean(torch.sqrt(torch.sum(torch.square(g_loc_n1 - data_key3[:, 0,:2]), dim=-1)), dim=0)
            loss_loc_n2 = torch.mean(torch.sqrt(torch.sum(torch.square(g_loc_n2 - data_key3[:, 0,:2]), dim=-1)), dim=0)
            loss_loc = (loss_loc_p2  + loss_loc_n2) / 2
            eval_loss_loc.append(loss_loc.item())

            # self attention结果监督
            loss_s1_part_self = torch.mean(torch.sqrt(torch.sum(torch.square(
                torch.cat((key_pre_rgb_self[:, :10], key_pre_rgb_self[:, 12:22]), dim=1) - torch.cat(
                    (data_key[:, :10], data_key[:, 12:22]),
                    dim=1)), dim=-1)), dim=0)
            loss_s2_part_self = torch.mean(torch.sqrt(torch.sum(torch.square(
                torch.cat((key_pre_ti_self[:, :10], key_pre_ti_self[:, 12:22]), dim=1) - torch.cat(
                    (data_key2[:, :10], data_key2[:, 12:22]),
                    dim=1)), dim=-1)), dim=0)
            loss_s3_part_self = torch.mean(torch.sqrt(torch.sum(torch.square(
                torch.cat((key_pre_ti_self_n[:, :10], key_pre_ti_self_n[:, 12:22]), dim=1) - torch.cat(
                    (data_key3[:, :10], data_key3[:, 12:22]),
                    dim=1)), dim=-1)), dim=0)

            loss_s1_mpjpe_self = torch.mean(loss_s1_part_self)
            loss_s2_mpjpe_self = torch.mean(loss_s2_part_self)
            loss_s3_mpjpe_self = torch.mean(loss_s3_part_self)

            eval_loss_s1_self.append(loss_s1_mpjpe_self.item())
            eval_loss_s2_self.append(loss_s2_mpjpe_self.item())
            eval_loss_s3_self.append(loss_s3_mpjpe_self.item())

            eval_loss_frame1.append(loss_s1_part.cpu().detach().numpy())
            eval_loss_frame2.append(loss_s2_part.cpu().detach().numpy())
            eval_loss_frame3.append(loss_s3_part.cpu().detach().numpy())


            loss_keypoint = 0.5*(loss_s1 + loss_s2 + loss_s3) / 3 #+ 0.5*loss_id
            eval_loss.append(loss_keypoint.item())

            key_rgb_out = torch.cat((key_pre_rgb[:, :10], key_pre_rgb[:, 12:22]), dim=1).view(batch_size,length_size,-1)
            key_ti_out = torch.cat((key_pre_ti[:, :10], key_pre_ti[:, 12:22]), dim=1).view(batch_size,length_size, -1)
            key_ti2_out =torch.cat((key_pre_ti2[:, :10], key_pre_ti2[:, 12:22]), dim=1).view(batch_size,length_size,-1)

            # print("key_ti_out1:", key_ti_out.shape)
            key_rgb_out2 = torch.flatten(key_rgb_out, start_dim=1, end_dim=2)#融合后 锚点RGB with 正样本毫米波；
            key_ti_out2 = torch.flatten(key_ti_out, start_dim=1, end_dim=2)
            key_ti2_out2 = torch.flatten(key_ti2_out, start_dim=1, end_dim=2)

            show_rgb = key_pre_rgb.cpu().detach()
            show_ti = key_pre_ti.cpu().detach()
            show_gt_rgb = data_key.cpu().detach()
            show_gt_ti = data_key2.cpu().detach()
            # draw3Dpose_frames(show_rgb,show_gt_rgb)
            #print(rgb_l==ti_l)
            for i in range(batchsize2):
                eval_rgb_gt.append(data[6].cpu().detach().numpy()[i])
                eval_ti_gt.append(data[7].cpu().detach().numpy()[i])
                eval_rgb_skeleton.append(key_rgb_out.cpu().detach().numpy()[i])
                eval_ti_skeleton.append(key_ti_out.cpu().detach().numpy()[i])
                gallery_feature.append(rgb_l.cpu().detach().numpy()[i])
                gallery_feature_kp.append(key_rgb_out2.cpu().detach().numpy()[i])
                gallery_feature_l.append(rgb_l.cpu().detach().numpy()[i])
                gallery_label.append(np.asarray(data[3])[i])
                gallery_label_l.append(np.asarray(data[3])[i])
                #全排列
                #gallery_feature.append(output4.cpu().detach().numpy()[i])
                #gallery_feature_h.append(rgb_h2.cpu().detach().numpy()[i])

                if ifallfusion == 1:
                    # rgb_l_dif = ti_p+rgb_n;ti_dif = ti_p+rgb_n
                    # anchor:rgb_l;p:ti_l;n:ti_dif
                    gallery_feature_l.append(rgb_l_dif.cpu().detach().numpy()[i])
                    gallery_label_l.append(np.asarray(data[5])[i])
                    query_feature_l.append(ti_dif.cpu().detach().numpy()[i])
                    query_label_l.append(np.asarray(data[3])[i])

                query_feature.append(ti_l.cpu().detach().numpy()[i])
                query_feature_l.append(ti_l.cpu().detach().numpy()[i])
                query_feature_kp.append(key_ti_out2.cpu().detach().numpy()[i])
                query_label.append(np.asarray(data[4])[i])
                query_label_l.append(np.asarray(data[4])[i])

            # print("train once")
    eval_loss = np.mean(eval_loss)

    # eval_loss_total += eval_loss


    # loss_total += loss.data.cpu().numpy()
    ap1 = 0.0
    # ap1_kp = 0.0
    ap1_l = 0.0
    ap1_kp_singleti = 0.0
    ap1_singleti = 0.0
    CMC1 = torch.IntTensor(len(gallery_label)-1).zero_()
    # CMC1_l = torch.IntTensor(len(gallery_label_l)-1).zero_()
    # CMC1_kp = torch.IntTensor(len(gallery_label)-1).zero_()

    # 测试标准
    for i in range(len(query_label)):

        clean_gallery_feature = np.delete(gallery_feature, i, axis=0)
        clean_gallery_label = np.delete(gallery_label, i, axis=0)
        ap_tmp, CMC_tmp = evaluate(query_feature[i], query_label[i], clean_gallery_feature, clean_gallery_label)
#目标A的第1条，gallery里面不存在
        if CMC_tmp[0] == -1:
            continue
        # if query_label[i] < 15 or query_label[i] > 20:
        CMC1 = CMC1 + CMC_tmp
        ap1 += ap_tmp
        flag_single = flag_single + 1

    CMC1 = CMC1.float()
    CMC1 = CMC1 / flag_single  # average CMC

    CMC1 = np.asarray(CMC1)
    # flag_single_l1 = 0
    # flag_single_l2 = 0
    # for i in range(len(query_label_l)//2):
    #     clean_gallery_feature_l = np.delete(gallery_feature_l, i*2, axis=0)
    #     clean_gallery_label = np.delete(gallery_label_l, i*2, axis=0)
    #     (ap_tmp, CMC_tmp),score,index = evaluate_l(query_feature_l[i*2], query_label_l[i*2], clean_gallery_feature_l, clean_gallery_label)
    #
    #     clean_gallery_feature_l = np.delete(gallery_feature_l, i*2+1, axis=0)
    #     clean_gallery_label = np.delete(gallery_label_l, i*2+1, axis=0)
    #     (ap_tmp2, CMC_tmp2), score2,index2 = evaluate_l(query_feature_l[i*2+1], query_label_l[i*2+1], clean_gallery_feature_l,
    #                                           clean_gallery_label)
    #
    #     #print(score[index[0]])
    #
    #     if CMC_tmp[0] == -1 or  CMC_tmp2[0] == -1:
    #         continue
    #     # if query_label[i] < 15 or query_label[i] > 20:
    #     if score[index[0]]< score2[index2[0]]:
    #         CMC1_l = CMC1_l + CMC_tmp
    #         ap1_l += ap_tmp
    #         flag_single_l = flag_single_l + 1
    #         flag_single_l1 = flag_single_l1 + 1
    #         #print(CMC_tmp)
    #     else:
    #         CMC1_l = CMC1_l + CMC_tmp2
    #         ap1_l += ap_tmp2
    #         flag_single_l = flag_single_l + 1
    #         flag_single_l2 = flag_single_l2 + 1
    #
    #
    # CMC1_l = CMC1_l.float()
    # CMC1_l = CMC1_l / flag_single_l  # average CMC
    #
    # CMC1_l = np.asarray(CMC1_l)

    # CMC_kp
    # for i in range(len(query_label)):
    #     clean_gallery_feature_kp = np.delete(gallery_feature_kp, i, axis=0)
    #     clean_gallery_label = np.delete(gallery_label, i, axis=0)
    #     ap_tmp_kp, CMC_tmp_kp = evaluate(query_feature_kp[i], query_label[i], clean_gallery_feature_kp, clean_gallery_label)
    #
    #     if CMC_tmp_kp[0] == -1:
    #         continue
    #     CMC1_kp = CMC1_kp + CMC_tmp_kp
    #     ap1_kp += ap_tmp_kp
    #     flag_single_kp = flag_single_kp + 1
    #
    # CMC1_kp = CMC1_kp.float()
    # CMC1_kp = CMC1_kp / flag_single_kp  # average CMC
    #
    # CMC1_kp = np.asarray(CMC1_kp)

    writer1.add_scalars('test_Rank-n_single',
                        {'rank1': torch.tensor(CMC1[0], dtype=float),
                         'rank2': torch.tensor(CMC1[1]),
                         'rank3': torch.tensor(CMC1[2]),
                         'rank4': torch.tensor(CMC1[3]),
                         'rank5': torch.tensor(CMC1[4]),
                         'rank6': torch.tensor(CMC1[5]),
                         'rank7': torch.tensor(CMC1[6]),
                         'rank8': torch.tensor(CMC1[7]),
                         'rank9': torch.tensor(CMC1[8]),
                         'rank10': torch.tensor(CMC1[9]),
                         'rank11': torch.tensor(CMC1[10]),
                         'rank12': torch.tensor(CMC1[11]),
                         'rank13': torch.tensor(CMC1[12]),
                         'rank14': torch.tensor(CMC1[13]),
                         'rank15': torch.tensor(CMC1[14])
                         }, epoch)


    if ifrunonce==1:
        print('eval_top1: {}'.format(CMC1[0]))
        print('eval_top3: {}'.format(CMC1[2]))
        print('eval_top5: {}'.format(CMC1[4]))
        print('eval_top10: {}\n\n'.format(CMC1[9]))
        # print('eval_mAP: {}'.format(ap1 / flag_single))
        # print('eval_top1_l: {}'.format(CMC1_l[0]))
        # print('eval_top5_l: {}'.format(CMC1_l[4]))
        # print('eval_mAP_single_l: {}'.format(ap1_l / flag_single_l))
        # print('eval_top1_kp: {}'.format(CMC1_kp[0]))
        # print('eval_top5_kp: {}'.format(CMC1_kp[4]))
        # print('eval_mAP_single_kp: {}'.format(ap1_kp / flag_single_kp))
        print('锚点RGB with 正样本mmWave: {} cm'.format(np.mean(eval_loss_s1)*100))
        print('正样本mmWave with 锚点RGB:: {} cm'.format(np.mean(eval_loss_s2)*100))
        print('负样本mmWave with 负样本RGB: {} cm'.format(np.mean(eval_loss_s3)*100))

        print('锚点RGB Self: {} cm'.format(np.mean(eval_loss_s1_self)*100))
        print('正样本 mmWave Self: {} cm'.format(np.mean(eval_loss_s2_self)*100))
        print('负样本 mmWave Self: {} cm'.format(np.mean(eval_loss_s3_self) * 100))
        print('定位误差: {} cm'.format(np.mean(eval_loss_loc)*100))

        # 只保存最优解
    if CMC1[0] > 0.6 and CMC1[0] > top1_best_test:
        save_models("best")
        print("当前最佳：epoch:{}".format(epoch))
        print('当前最佳：eval_top1: {}'.format(CMC1[0]))
        print('当前最佳：eval_top3: {}'.format(CMC1[2]))
        print('当前最佳：eval_top5: {}'.format(CMC1[4]))
        print('当前最佳：eval_top10: {}'.format(CMC1[9]))

        # 更新最优解
    top1_best_test = max(top1_best_test, CMC1[0])
    top5_best_test = max(top5_best_test, CMC1[4])
    top10_best_test = max(top10_best_test, CMC1[9])
    map_best_test = max(map_best_test, ap1 / flag_single)
    # top1_best_test_l = max(top1_best_test_l, CMC1_l[0])
    # top5_best_test_l = max(top5_best_test_l, CMC1_l[4])
    # top10_best_test_l = max(top10_best_test_l, CMC1_l[9])
    # map_best_test_l = max(map_best_test_l, ap1_l / flag_single_l)


    kprgb_best_test = min(kprgb_best_test, np.mean(eval_loss_s1))
    kpti1_best_test = min(kpti1_best_test, np.mean(eval_loss_s2))
    kpti2_best_test = min(kpti2_best_test, np.mean(eval_loss_s3))

    writer1.add_scalar(tag='eval_mAP_single', scalar_value=ap1 / flag_single, global_step=epoch)
    # writer1.add_scalar(tag='eval_mAP_single_l', scalar_value=ap1_l / flag_single_l, global_step=epoch)
    # writer1.add_scalar(tag='eval_idloss', scalar_value=np.mean(eval_idloss), global_step=epoch)

    #writer1.add_scalar(tag='eval_mAP_multi', scalar_value=ap2 / flag_multi, global_step=epoch)
    # if (epoch + 1) % 5 == 0:
    #     print('eval_top1: {}'.format(CMC1[0]))
    #     print('eval_top1: {}'.format(CMC1[2]))
    #     print('eval_top5: {}'.format(CMC1[4]))
    #     print('eval_top10: {}'.format(CMC1[9]))
    #     # print('eval_mAP: {}'.format(ap1 / len(query_label)))
    #     # print('eval_top1_kp: {}'.format(CMC1_kp[0]))
    #     # print('eval_top5_kp: {}'.format(CMC1_kp[4]))
    #     # print('eval_mAP_kp: {}'.format(ap1_kp / len(query_label)))
    #     eval_loss_total = 0.