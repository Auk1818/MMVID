import numpy as np
from torch.utils.data import Dataset
import os

class SiamesePC_38_rgb_full_midmodal_offlinetri_final_07train_zhengdui(Dataset):
    """
    训练：为每个样本随机创建一个正或负对
    Test：创建用于测试的固定对
    """
    def __init__(self, train=True, issever=0):
        self.train = train
        self.issever = issever
        if self.train:  # train
            self.data_ti_, self.data_label_,  self.data_key_ , self.data_rgb_ = self.dataRead()

            # 原始数据划分
            random_state = np.random.RandomState(1)  # 使用固定种子

            # 生成随机索引
            indices = np.arange(len(self.data_label_))
            random_state.shuffle(indices)

            # 使用相同的随机索引打乱所有数据
            self.data_ti_ = self.data_ti_[indices]
            self.data_label_ = self.data_label_[indices]
            self.data_key_ = self.data_key_[indices]
            self.data_rgb_ = self.data_rgb_[indices]

            # 构建 label 到索引的映射
            self.data_label_set = set(self.data_label_)
            self.label_to_indices_ = {label: np.where(np.asarray(self.data_label_) == label)[0]
                                      for label in self.data_label_set}

            #dict:34
            #人序号 1：[样本：1，20，680]

            self.data_ti = []
            self.data_ti1 = []
            self.data_ti2 = []
            self.data_kinect = []
            self.data_label = []
            self.data_label1 = []
            self.data_label2 = []
            self.data_index = []
            self.data_rgb = []
            self.data_6890 = []
            self.data_key = []
            self.data_key2 = []
            self.data_rgb = []
            self.data_rgb1 = []
            self.data_rgb2 = []

            # 原始数据采集方式
            for id in range(len(self.data_label_set)):
                idnum = self.label_to_indices_[list(self.data_label_set)[id]]
                for ii in range(len(idnum)):
                    if ii < len(idnum)*0.7:
                        self.data_ti.append(self.data_ti_[idnum[ii]])
                        self.data_label.append(self.data_label_[idnum[ii]])
                        self.data_rgb.append(self.data_rgb_[idnum[ii]])
                        self.data_key.append(self.data_key_[idnum[ii]])

            self.data_key2 = self.data_key

            self.data_ti = np.asarray(self.data_ti)
            self.data_label = np.asarray(self.data_label)
            self.data_rgb = np.asarray(self.data_rgb)
            self.data_key = np.asarray(self.data_key)

            print("train_ti:", self.data_ti.shape)
            print("train_label:", self.data_label.shape)
            print("train_key:", self.data_key.shape)
            print("train_rgb:", self.data_rgb.shape)

            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}

        else:  # test
            # read data
            self.data_ti, self.data_label, self.data_key, self.data_rgb = self.dataRead()

            # 原始数据分配方式
            self.data_label_set = set(self.data_label)
            random_state = np.random.RandomState(1)  # 打乱 每次打算之后顺序一样 伪随机
            random_state.shuffle(self.data_ti)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_label)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_rgb)
            random_state = np.random.RandomState(1)
            random_state.shuffle(self.data_key)
            self.label_to_indices = {label: np.where(np.asarray(self.data_label) == label)[0]
                                     for label in self.data_label_set}
            self.data_test_ti = []
            self.data_test_ti1 = []
            self.data_test_ti2 = []
            self.data_test_kinect = []
            self.data_test_label = []
            self.data_test_label1 = []
            self.data_test_label2 = []
            self.data_test_6890 = []
            self.data_test_key = []
            self.data_test_key2 = []
            self.data_test_rgb = []
            self.data_test_rgb1 = []
            self.data_test_rgb2 = []

            # 原始方式
            for id in range(len(self.data_label_set)):#34个人
                idnum = self.label_to_indices[list(self.data_label_set)[id]]#20，每个人的20次路线对应的样本编号,26号人对应的20组数据
                for ii in range(len(idnum)):#拿出30%做test
                    if ii > len(idnum)*0.7 or ii == len(idnum)*0.7:
                        self.data_test_ti.append(self.data_ti[idnum[ii]])
                        self.data_test_label.append(self.data_label[idnum[ii]])
                        self.data_test_rgb.append(self.data_rgb[idnum[ii]])
                        self.data_test_key.append(self.data_key[idnum[ii]])


            self.data_test_key2 = self.data_test_key

            self.data_test_ti = np.asarray(self.data_test_ti)
            self.data_test_label = np.asarray(self.data_test_label)
            self.data_test_key = np.asarray(self.data_test_key)
            self.data_test_rgb = np.asarray(self.data_test_rgb)
            print("test_ti:", self.data_test_ti.shape)
            print("test_label:", self.data_test_label.shape)
            print("test_key:", self.data_test_key.shape)
            print("test_rgb:", self.data_test_rgb.shape)

            self.label_to_indices_test = {label: np.where(np.asarray(self.data_test_label) == label)[0]
                                     for label in self.data_label_set}

    def __getitem__(self, index):
        if self.train:
            target = np.random.choice(2)
            # target = 1
            kinect_6890 = 0
            # 监督学习
            ti, labelti, rgb, labelrgb = self.data_ti[index], self.data_label[index], self.data_rgb[index], \
                                             self.data_label[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[labelti])

            negative_label = np.random.choice(list(self.data_label_set - set([labelti])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])

            ti_p = self.data_ti[positive_index]#与输入的ti来自同一个人的不同样本
            ti_n = self.data_ti[negative_index]#与输入的ti来自不同人的样本
            labelti_p = labelti#相同的样本
            labelti_n = negative_label
            key_a = self.data_key[index]
            #key_p = self.data_key[positive_index]

            # 测试同一对象两模态是否重建一致
            key_p = self.data_key[positive_index]
            key_n = self.data_key[negative_index]
            rgb_n =  self.data_rgb[negative_index]

            labelrgb_p = labelti
            labelrgb_n = negative_label
        else:  # test 固定的ti kinect 数据
            target = np.random.choice(2)
            ti, labelti, rgb, labelrgb = self.data_test_ti[index], self.data_test_label[index], \
                                             self.data_test_rgb[index], self.data_test_label[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices_test[labelti])
                if self.label_to_indices_test[labelti].size == 1:
                    break
            negative_label = np.random.choice(list(self.data_label_set - set([labelti])))
            negative_index = np.random.choice(self.label_to_indices_test[negative_label])
            # 测试同一对象两模态是否重建一致
            ti_p = self.data_test_ti[positive_index]

            ti_n = self.data_test_ti[negative_index]
            labelti_p = labelti
            labelti_n = negative_label
            key_a = self.data_test_key[index]
            key_p = self.data_test_key[positive_index]
            key_n = self.data_test_key[negative_index]
            rgb_n = self.data_test_rgb[negative_index]
        return rgb,ti_p,ti_n,labelrgb, labelti_p, labelti_n ,key_a,key_p,key_n , rgb_n

    # rgb：当前样本的RGB数据。
    # ti_p：正样本的毫米波雷达数据。
    # ti_n：负样本的毫米波雷达数据。
    # labelrgb：当前样本的标签。
    # labelti_p：正样本的标签。
    # labelti_n：负样本的标签。
    # key_a：锚点样本的关键点数据。
    # key_p：正样本的关键点数据。
    # key_n：负样本的关键点数据。
    # rgb_n：负样本的RGB数据。

    def __len__(self):
        if self.train:
            return len(self.data_ti)
        else:
            return len(self.data_test_ti)

    def dataRead(self):
        # 遍历文件夹
        list_all_ti = []
        list_all_kinect_key = []
        list_label_all = []  # 人名字
        list_all_rgb_full = []


        # 读取rgb数据
        if self.issever:
            list_all_ti = np.load(
                "C:/Users/lenovo/Desktop/mission/dataset/ruili34_nature/list_all_ti_3.npy")#680,20,64,5

            list_all_kinect_key = np.load(
                "C:/Users/lenovo/Desktop/mission/dataset/ruili34_nature/list_all_kinect_key_2.npy")#680,20,24,3
            list_label_all = np.load(
                "C:/Users/lenovo/Desktop/mission/dataset/ruili34_nature/list_label_all_2.npy")
            list_all_rgb_full = np.load(
                "C:/Users/lenovo/Desktop/mission/dataset/ruili34_nature/list_all_image_2.npy")
            #ti和kincet_key保持一致：是否calibration
            list_all_ti = list_all_ti[:,:,:,:3]

        print("data load end")
        print("length of key:", len(list_all_kinect_key))
        print("length of ti:", len(list_all_ti))
        print("length of rgb:", len(list_all_rgb_full))
        return list_all_ti, list_label_all, list_all_kinect_key, list_all_rgb_full


class Sensys_SMPL(Dataset):
    def __init__(self, train=True, issever=0):
        self.train = train
        self.issever = issever
        self.data_dir = "C:/Users/lenovo/Desktop/mission/dataset/ruili34_nature"

        # Load all data
        (self.data_ti, self.data_label, self.data_key,
         self.data_rgb, self.data_smpl_vert,
         self.data_smpl_joint, self.data_gender,
         # === 新增：加载生物特征数据 ===
         self.data_height, self.data_bmi
         ) = self._load_and_process_data()

        # Create label mappings
        self._create_label_mappings()

    def _load_and_process_data(self):
        """Load and process data for either train or test mode"""
        # Load raw data
        (data_ti, data_label, data_key,
         data_rgb, data_smpl_vert,
         data_smpl_joint, data_gender) = self.dataRead()

        print("=== Loading Biometric Traits ===")
        # 加载身高数据
        file_paths = [os.path.join(self.data_dir, f"list_all_height_{i}.npy") for i in range(1, 2)]
        arrays_to_concat_height = [np.load(file_path, allow_pickle=True) for file_path in file_paths]
        arrays_to_concat_height = [np.array(arr) if isinstance(arr, list) else arr for arr in arrays_to_concat_height]
        data_height = np.concatenate(arrays_to_concat_height, axis=0)

        # 加载BMI数据
        file_paths = [os.path.join(self.data_dir, f"list_all_bmi_{i}.npy") for i in range(1, 2)]
        arrays_to_concat_bmi = [np.load(file_path, allow_pickle=True) for file_path in file_paths]
        arrays_to_concat_bmi = [np.array(arr) if isinstance(arr, list) else arr for arr in arrays_to_concat_bmi]
        data_bmi = np.concatenate(arrays_to_concat_bmi, axis=0)

        print("height:", data_height.shape)  # 应为 (680,)
        print("bmi:", data_bmi.shape)  # 应为 (680,)

        # Shuffle all data consistently
        random_state = np.random.RandomState(1)
        indices = np.arange(len(data_label))
        random_state.shuffle(indices)

        # Apply shuffle to all data
        data_ti = data_ti[indices]
        data_label = data_label[indices]
        data_key = data_key[indices]
        data_rgb = data_rgb[indices]
        data_smpl_vert = data_smpl_vert[indices]
        data_smpl_joint = data_smpl_joint[indices]
        data_gender = data_gender[indices]

        data_height = data_height[indices]
        data_bmi = data_bmi[indices]

        # Split into train/test based on mode
        label_set = set(data_label)
        final_data = []

        for data_array in [data_ti, data_label, data_key, data_rgb,
                           data_smpl_vert, data_smpl_joint, data_gender,
                           data_height, data_bmi]:
            split_data = []
            for label in label_set:
                label_indices = np.where(data_label == label)[0]
                split_point = int(len(label_indices) * 0.7)

                if self.train:
                    split_data.append(data_array[label_indices[:split_point]])
                else:
                    split_data.append(data_array[label_indices[split_point:]])

            final_data.append(np.concatenate(split_data))

        return tuple(final_data)

    def _create_label_mappings(self):
        """Create label to indices mappings"""
        self.data_label_set = set(self.data_label)
        self.label_to_indices = {
            label: np.where(np.asarray(self.data_label) == label)[0]
            for label in self.data_label_set
        }
        #output:  self.label_to_indices
        # 对每个标签label，使用np.where找到所有等于该标签的样本的索引位置。
        # 最终生成一个字典，键是标签，值是对应的样本索引数组。

    def __getitem__(self, index):
        # Common logic for both train and test
        ti, labelti, rgb, labelrgb = (
            self.data_ti[index], self.data_label[index],
            self.data_rgb[index], self.data_label[index]
        )

        # Get positive sample (same class)
        positive_index = index
        while positive_index == index:
            positive_index = np.random.choice(self.label_to_indices[labelti])
            if not self.train and self.label_to_indices[labelti].size == 1:
                break

        # Get negative sample (different class)
        negative_label = np.random.choice(list(self.data_label_set - set([labelti])))
        negative_index = np.random.choice(self.label_to_indices[negative_label])

        # Prepare all return values
        return_values = [
            rgb, # rgb_a
            self.data_ti[positive_index],  # ti_p
            self.data_ti[negative_index],  # ti_n
            labelrgb, #labelti_a
            labelti,  # labelti_p
            negative_label,  # labelti_n
            self.data_key[index],  # key_a
            self.data_key[positive_index],  # key_p
            self.data_key[negative_index],  # key_n

            self.data_rgb[negative_index],  # rgb_n

            self.data_smpl_vert[index],  # vert_a
            self.data_smpl_vert[positive_index],  # vert_p
            self.data_smpl_vert[negative_index],  # vert_n

            self.data_smpl_joint[index],  # joint_a
            self.data_smpl_joint[positive_index],  # joint_p
            self.data_smpl_joint[negative_index],  # joint_n

            self.data_gender[index],  # gender_a
            self.data_gender[positive_index],  # gender_p
            self.data_gender[negative_index],  # gender_n

            # 锚点样本的生物特征
            self.data_height[index],
            self.data_bmi[index],
            # 正样本的生物特征
            self.data_height[positive_index],
            self.data_bmi[positive_index],
            # 负样本的生物特征
            self.data_height[negative_index],
            self.data_bmi[negative_index],

        ]

        return tuple(return_values)

        # return rgb,ti_p,ti_n,labelrgb, labelti_p, labelti_n ,key_a,key_p,key_n , rgb_n

        # rgb：当前样本的RGB数据。
        # ti_p：正样本的毫米波雷达数据。
        # ti_n：负样本的毫米波雷达数据。
        # labelrgb：当前样本的标签。
        # labelti_p：正样本的标签。
        # labelti_n：负样本的标签。
        # key_a：锚点样本的关键点数据。
        # key_p：正样本的关键点数据。
        # key_n：负样本的关键点数据。
        # rgb_n：负样本的RGB数据。


    def __len__(self):
        return len(self.data_ti)



    def dataRead(self):
        # 遍历文件夹
        list_all_ti = []
        list_all_kinect_key = []
        list_label_all = []  # 人名字
        list_all_rgb_full = []
        list_all_smpl_vert = []
        list_all_smpl_joint = []
        list_all_gender = []

        # 读取rgb数据
        if self.issever:
            # === 修正开始 ===
            # 第一步：先加载 list_all_ti 并处理时间帧
            file_paths = [os.path.join(self.data_dir, f"list_all_ti_{i}.npy") for i in range(1, 4)]
            arrays_to_concat_1 = [np.load(file_path, allow_pickle=True) for file_path in file_paths]
            arrays_to_concat_1 = [np.array(arr) if isinstance(arr, list) else arr for arr in arrays_to_concat_1]
            list_all_ti = np.concatenate(arrays_to_concat_1, axis=0)
            list_all_ti = list_all_ti[:, -10:]  # 取最后10帧

            # 第二步：加载 list_all_kinect_key
            file_paths = [os.path.join(self.data_dir, f"list_all_kinect_key_{i}.npy") for i in range(1, 4)]
            arrays_to_concat_2 = [np.load(file_path, allow_pickle=True) for file_path in file_paths]
            arrays_to_concat_2 = [np.array(arr) if isinstance(arr, list) else arr for arr in arrays_to_concat_2]
            list_all_kinect_key = np.concatenate(arrays_to_concat_2, axis=0)
            list_all_kinect_key = list_all_kinect_key[:, -10:]

            # 第三步：加载 list_label_all
            file_paths = [os.path.join(self.data_dir, f"list_label_all_{i}.npy") for i in range(1, 4)]
            arrays_to_concat_3 = [np.load(file_path, allow_pickle=True) for file_path in file_paths]
            arrays_to_concat_3 = [np.array(arr) if isinstance(arr, list) else arr for arr in arrays_to_concat_3]
            list_label_all = np.concatenate(arrays_to_concat_3, axis=0)
            list_label_all = [int(item) for item in list_label_all]
            list_label_all = np.asarray(list_label_all) - 1

            # 第四步：加载 list_all_rgb_full
            file_paths = [os.path.join(self.data_dir, f"list_all_image_{i}.npy") for i in range(1, 4)]
            arrays_to_concat_4 = [np.load(file_path, allow_pickle=True) for file_path in file_paths]
            arrays_to_concat_4 = [np.array(arr) if isinstance(arr, list) else arr for arr in arrays_to_concat_4]
            list_all_rgb_full = np.concatenate(arrays_to_concat_4, axis=0)
            list_all_rgb_full = list_all_rgb_full[:, -10:]
            list_all_rgb_full = np.transpose(list_all_rgb_full, (0, 1, 4, 2, 3))

            # 第五步：加载 list_all_smpl_vert
            file_paths = [os.path.join(self.data_dir, f"list_all_smpl_vert_{i}.npy") for i in range(1, 4)]
            arrays_to_concat_5 = [np.load(file_path, allow_pickle=True) for file_path in file_paths]
            arrays_to_concat_5 = [np.array(arr) if isinstance(arr, list) else arr for arr in arrays_to_concat_5]
            list_all_smpl_vert = np.concatenate(arrays_to_concat_5, axis=0)
            list_all_smpl_vert = list_all_smpl_vert[:, -10:]

            # 第六步：加载 list_all_smpl_joint
            file_paths = [os.path.join(self.data_dir, f"list_all_smpl_key_{i}.npy") for i in range(1, 4)]
            arrays_to_concat_6 = [np.load(file_path, allow_pickle=True) for file_path in file_paths]
            arrays_to_concat_6 = [np.array(arr) if isinstance(arr, list) else arr for arr in arrays_to_concat_6]
            list_all_smpl_joint = np.concatenate(arrays_to_concat_6, axis=0)
            list_all_smpl_joint = list_all_smpl_joint[:, -10:]

            # 第七步：加载 list_all_gender
            file_paths = [os.path.join(self.data_dir, f"list_all_gender_{i}.npy") for i in range(1, 4)]
            arrays_to_concat_7 = [np.load(file_path, allow_pickle=True) for file_path in file_paths]
            arrays_to_concat_7 = [np.array(arr) if isinstance(arr, list) else arr for arr in arrays_to_concat_7]
            list_all_gender = np.concatenate(arrays_to_concat_7, axis=0)

            # 最后，在所有数据都正确加载后，再进行通道校准（此时 list_all_ti 已是 ndarray）
            list_all_ti = list_all_ti[:, :, :, :3]
            list_all_smpl_vert = list_all_smpl_vert[:, :, :, :3]
            # === 修正结束 ===

        print("data load end")
        print("length of key:", list_all_kinect_key.shape)
        print("length of ti:", list_all_ti.shape)
        print("length of rgb:", list_all_rgb_full.shape)
        print("length of smpl vert:", list_all_smpl_vert.shape)
        print("length of smpl joint:", list_all_smpl_joint.shape)
        print("length of gender:", list_all_gender.shape)

        return (list_all_ti, list_label_all, list_all_kinect_key, list_all_rgb_full,
                list_all_smpl_vert, list_all_smpl_joint, list_all_gender)