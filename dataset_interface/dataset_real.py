import os
import sys
sys.path.append('/data/tanjingyang/demo/e2e/parkinge2e/dataset_interface/')
sys.path.append('/data/tanjingyang/demo/e2e/parkinge2e/')
from PIL import Image
from typing import List
import torchvision
import numpy as np
import torch.utils.data
import tqdm
import matplotlib.pyplot as plt
from utils.camera_utils import CameraInfoParser, ProcessImage, get_normalized_torch_image, get_torch_intrinsics_or_extrinsics
from utils.config import Configuration
from utils.trajectory_utils import TrajectoryInfoParser, tokenize_traj_point


class ParkingData(torch.utils.data.Dataset):
    def __init__(self, config: Configuration, is_train):
        super(ParkingData, self).__init__()
        self.cfg = config

        self.BOS_token = self.cfg.token_nums
        self.EOS_token = self.cfg.token_nums + self.cfg.append_token - 2
        self.PAD_token = self.cfg.token_nums + self.cfg.append_token - 1

        self.root_dir = self.cfg.data_dir
        self.is_train = is_train
        self.images_tag = ("rgb_front", "rgb_left", "rgb_right", "rgb_rear")

        self.intrinsic = {}
        self.extrinsic = {}
        self.images = {}
        self.bev = []
        for image_tag in self.images_tag:
            self.images[image_tag] = []

        self.task_index_list = []

        self.fuzzy_target_point = []
        self.traj_point = []
        self.traj_point_token = []
        # 用于avg-fc这个结构
        self.traj_point_normal = []
        self.target_point = []
        
        #用于统计GT的数据分布
        self.distribution_array = np.zeros(1203)
        self.create_gt_data()

    def __len__(self):
        return len(self.images["rgb_front"])

    def __getitem__(self, index):
        images, bev, intrinsics, extrinsics = self.process_camera(index)

        data = {}
        keys = ['image', 'bev','extrinsics', 'intrinsics', 'target_point', 'gt_traj_point', 'gt_traj_point_token', 'fuzzy_target_point', 'gt_traj_point_normal']
        for key in keys: 
            data[key] = []
        data['image'] = images
        data['bev'] = bev
        data['intrinsics'] = intrinsics
        data['extrinsics'] = extrinsics
        data["gt_traj_point"] = torch.from_numpy(np.array(self.traj_point[index]))
        data["gt_traj_point_normal"] = torch.from_numpy(np.array(self.traj_point_normal[index]))
        data['gt_traj_point_token'] = torch.from_numpy(np.array(self.traj_point_token[index]))
        data['target_point'] = torch.from_numpy(self.target_point[index])
        data["fuzzy_target_point"] = torch.from_numpy(self.fuzzy_target_point[index])

        return data

    def create_gt_data(self):
        all_tasks = self.get_all_tasks()    # 总共是n个场景
        padding_nums = 0
        traj_nums = 0
        invalid_traj_num = 0
        for task_index, task_path in tqdm.tqdm(enumerate(all_tasks)):  
            image_info_obj = CameraInfoParser(task_index, task_path)
            traje_info_obj = TrajectoryInfoParser(task_index, task_path)

            self.intrinsic[task_index] = image_info_obj.intrinsic
            self.extrinsic[task_index] = image_info_obj.extrinsic

            for ego_index in range(0, traje_info_obj.total_frames):  # 每个场景有几百帧（每帧包含front,back,left,right 4张图）
                ego_pose = traje_info_obj.get_trajectory_point(ego_index)   #ego_pose就是ego2world(x,y,z,yaw,pitch,roll) ego_pose就是世界坐标系下ego的位姿 换句话说json里面存的是世界坐标系下小车的坐标和位姿
                world2ego_mat = ego_pose.get_homogeneous_transformation().get_inverse_matrix()
                # ego坐标系下的 traj轨迹点坐标GT，共30组点
                predict_point_token_gt, predict_point_gt, pad_num = self.create_predict_point_gt(traje_info_obj, ego_index, world2ego_mat)
                # 泊车目标点转换到 ego坐标系下(fuzzy：加了噪声   parking_goal：没加噪声直接用的traj轨迹最后那个点)
                fuzzy_parking_goal, parking_goal = self.create_parking_goal_gt(traje_info_obj, world2ego_mat)

                # if ego_index == 0:
                #     print(f"task:{task_path}, parking_goal:{parking_goal}")
                #     if abs(parking_goal[1]) < 4.5:
                #         invalid_traj_num += 1
                #         break


                #统计GT的数据分布
                self.distribution_array[predict_point_token_gt] += 1

                # create image_path
                image_path = self.create_image_path_gt(task_path, ego_index)

                self.traj_point.append(predict_point_gt)    #每一帧对应的traj包含30个坐标点


                #用于avg-fc结构   归一化
                predict_point_gt = np.array(predict_point_gt)
                mid = np.empty_like(predict_point_gt)
                mid[::2] = (6.0 - predict_point_gt[1::2]) /12.0
                mid[1::2] = (6.0 - predict_point_gt[::2])/12.0
                
                predict_point_gt_normal = mid.tolist()
                self.traj_point_normal.append(predict_point_gt_normal)




                self.traj_point_token.append(predict_point_token_gt)
                self.target_point.append(parking_goal)
                self.fuzzy_target_point.append(fuzzy_parking_goal)
                for image_tag in self.images_tag:
                    self.images[image_tag].append(image_path[image_tag])

                # bev gt path
                bev_path = self.create_bev_path_gt(task_path, ego_index)
                self.bev.append(bev_path)

                self.task_index_list.append(task_index)

                if pad_num > 5:
                    padding_nums += 1

                traj_nums+=1
        print(f"park rate ={padding_nums/traj_nums}")
        # print(f"valid traj = {len(all_tasks) - invalid_traj_num}")

        x = np.arange(1203)
        plt.plot(x, self.distribution_array)
        plt.xlabel('Index')  # 横轴标签
        plt.ylabel('Values')               # 纵轴标签
        plt.title('Distribution Curve')    # 标题
        plt.grid()                         # 添加网格

        # 保存图像到文件
        if self.is_train:
            plt.savefig('distribution_curve.png', dpi=300, bbox_inches='tight')  # 保存为PNG文件
        plt.close()  # 关闭图形，以释放内存
        self.format_transform()

    def bev_preprocess(self, bev_path):
        if self.cfg.bev_input_backbone == 'vit':
            resize_len = 224
        else:
            resize_len = 256
        preprocess = torchvision.transforms.Compose([
            torchvision.transforms.Resize(resize_len),
            torchvision.transforms.CenterCrop(resize_len),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        color_jit = torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.3)
        bev_pil = Image.open(bev_path).convert("RGB")
        bev_pil = color_jit(bev_pil)
        bev_tensor = preprocess(bev_pil)

        return bev_tensor

    def process_camera(self, index):
        process_width, process_height = int(self.cfg.process_dim[0]), int(self.cfg.process_dim[1])
        images, intrinsics, extrinsics = [], [], []
        for image_tag in self.images_tag:
            image_path_list = self.images[image_tag]
            image_obj = ProcessImage(self.load_pil_image(image_path_list[index]), 
                                     self.intrinsic[self.task_index_list[index]][image_tag], 
                                     self.extrinsic[self.task_index_list[index]][image_tag], 
                                     target_size=(process_width, process_height))
            image_obj.color_aug()
            image_obj.resize_pil_image()
            images.append(get_normalized_torch_image(image_obj.resize_img))
            intrinsics.append(get_torch_intrinsics_or_extrinsics(image_obj.resize_intrinsics))
            extrinsics.append(get_torch_intrinsics_or_extrinsics(image_obj.extrinsics))
        
        #bev preprocess
        bev_tensor = self.bev_preprocess(self.bev[index])

        images = torch.cat(images, dim=0)
        intrinsics = torch.cat(intrinsics, dim=0)
        extrinsics = torch.cat(extrinsics, dim=0)

        return images, bev_tensor, intrinsics, extrinsics

    def create_predict_point_gt(self, traje_info_obj: TrajectoryInfoParser, ego_index: int, world2ego_mat: np.array) -> List[int]:
        predict_point, predict_point_token = [], []
        for predict_index in range(self.cfg.autoregressive_points):  # predict iteration
            predict_stride_index = self.get_clip_stride_index(predict_index = predict_index, 
                                                                start_index=ego_index, 
                                                                max_index=traje_info_obj.total_frames - 1, 
                                                                stride=self.cfg.traj_downsample_stride)         #index乘上一个stride(3)，意思就是间隔几个traj点取一个作为gt，一共取30个traj点，而不是取连续的
            predict_pose_in_world = traje_info_obj.get_trajectory_point(predict_stride_index)   #stride处理后的离散点，世界坐标系下
            predict_pose_in_ego = predict_pose_in_world.get_pose_in_ego(world2ego_mat)          #转换成ego坐标系
            progress = traje_info_obj.get_progress(predict_stride_index)
            predict_point.append([predict_pose_in_ego.x, predict_pose_in_ego.y])
            tokenize_ret = tokenize_traj_point(predict_pose_in_ego.x, predict_pose_in_ego.y,        #把真实的坐标搞成0-1200, progress没有用上
                                                progress, self.cfg.token_nums, self.cfg.xy_max)
            tokenize_ret_process = tokenize_ret[:2] if self.cfg.item_number == 2 else tokenize_ret
            predict_point_token.append(tokenize_ret_process)

            if predict_stride_index == traje_info_obj.total_frames - 1 or predict_index == self.cfg.autoregressive_points - 1:
                break

        predict_point_gt = [item for sublist in predict_point for item in sublist]
        append_pad_num = self.cfg.autoregressive_points * self.cfg.item_number - len(predict_point_gt)  #因为选取traj轨迹点的时候使用了stride和clip策略，所以可能要padding
        assert append_pad_num >= 0
        predict_point_gt = predict_point_gt + (append_pad_num // 2) * [predict_point_gt[-2], predict_point_gt[-1]]

        predict_point_token_gt = [item for sublist in predict_point_token for item in sublist]  # token是[x,y,x,y]的方式预测？
        predict_point_token_gt.insert(0, self.BOS_token)
        predict_point_token_gt.append(self.EOS_token)
        # predict_point_token_gt.append(self.PAD_token)
        predict_point_token_gt.append(self.PAD_token)
        append_pad_num = self.cfg.autoregressive_points * self.cfg.item_number + self.cfg.append_token - len(predict_point_token_gt)
        assert append_pad_num >= 0
        predict_point_token_gt = predict_point_token_gt + append_pad_num * [self.PAD_token]
        return predict_point_token_gt, predict_point_gt, append_pad_num

    def create_parking_goal_gt(self, traje_info_obj: TrajectoryInfoParser, world2ego_mat: np.array):
        candidate_target_pose_in_world = traje_info_obj.get_random_candidate_target_pose()
        candidate_target_pose_in_ego = candidate_target_pose_in_world.get_pose_in_ego(world2ego_mat)
        fuzzy_parking_goal = [candidate_target_pose_in_ego.x, candidate_target_pose_in_ego.y]

        target_pose_in_world = traje_info_obj.get_precise_target_pose()
        target_pose_in_ego = target_pose_in_world.get_pose_in_ego(world2ego_mat)
        parking_goal = [target_pose_in_ego.x, target_pose_in_ego.y]

        return fuzzy_parking_goal, parking_goal


    def create_image_path_gt(self, task_path, ego_index):
        filename = f"{str(ego_index).zfill(4)}.png"
        image_path = {}
        for image_tag in self.images_tag:
            image_path[image_tag] = os.path.join(task_path, image_tag, filename)
        return image_path

    def create_bev_path_gt(self, task_path, ego_index):
        filename = f"{str(ego_index).zfill(4)}.png"
        image_path = os.path.join(task_path, 'bev', filename)
        return image_path
    
    def get_all_tasks(self):
        all_tasks = []
        train_data_dir = os.path.join(self.root_dir, self.cfg.training_dir)
        val_data_dir = os.path.join(self.root_dir, self.cfg.validation_dir)
        data_dir = train_data_dir if self.is_train == 1 else val_data_dir
        for scene_item in os.listdir(data_dir):
            scene_path = os.path.join(data_dir, scene_item)
            for task_item in os.listdir(scene_path):
                task_path = os.path.join(scene_path, task_item)    
                all_tasks.append(task_path)
        return all_tasks

    def format_transform(self):
        self.traj_point = np.array(self.traj_point).astype(np.float32)
        self.traj_point_normal = np.array(self.traj_point_normal).astype(np.float32)
        self.traj_point_token = np.array(self.traj_point_token).astype(np.int64)
        self.target_point = np.array(self.target_point).astype(np.float32)
        self.fuzzy_target_point = np.array(self.fuzzy_target_point).astype(np.float32)
        for image_tag in self.images_tag:
            self.images[image_tag] = np.array(self.images[image_tag]).astype(np.string_)
        self.task_index_list = np.array(self.task_index_list).astype(np.int64)

    def get_clip_stride_index(self, predict_index, start_index, max_index, stride):
        return int(np.clip(start_index + stride * (1 + predict_index), 0, max_index))

    def load_pil_image(self, image_path):
        return Image.open(image_path).convert("RGB")
    
    def save_trajectorys(self):
        all_tasks = self.get_all_tasks()
        for idx in tqdm.tqdm(range(self.__len__())):
            task_path = all_tasks[self.task_index_list[idx]]
            traj_path = os.path.join(task_path, 'trajectory')
            if not os.path.exists(traj_path):
                os.mkdir(traj_path)

            out = {}
            out['target'] = self.target_point[idx]
            out['trajectory'] = self.traj_point[idx]

            # save
            img_file = str(self.images[self.images_tag[0]][idx])
            traj_file = os.path.join(traj_path, os.path.basename(img_file).split('.')[0]+'.npy')
            np.save(traj_file, out)

if __name__ == "__main__":
    from utils.config import get_train_config_obj
    # import ipdb
    # ipdb.set_trace()
    config_obj = get_train_config_obj("config/training_real.yaml")
    parking_dataset = ParkingData(config_obj, is_train=1)
    datas = parking_dataset[0]
    print(datas.keys())
    parking_dataset.save_trajectorys()