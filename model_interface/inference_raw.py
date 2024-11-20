from collections import OrderedDict
import numpy as np
import time
import cv2
from PIL import Image

import torch
import torchvision

from model_interface.model.parking_model_real import ParkingModelReal
from utils.camera_utils import CameraInfoParser, ProcessImage, get_normalized_torch_image, get_torch_intrinsics_or_extrinsics
from utils.config import InferenceConfiguration
from utils.pose_utils import PoseFlow, Pose, Point, Quaternion
from utils.traj_post_process import calculate_tangent, fitting_curve


class ParkingInferenceRaw:
    def __init__(self, inference_cfg: InferenceConfiguration):
        self.cfg = inference_cfg
        self.images_tag = ("rgb_front", "rgb_left", "rgb_right", "rgb_rear")
        self.images_fish_tag = ("rgb_front_fish", "rgb_left_fish", "rgb_right_fish", "rgb_rear_fish")
        # model init
        self.model = None
        self.device = None
        self.load_model(self.cfg.model_ckpt_path)
        print(f'inference model --- {self.cfg.model_ckpt_path}')
        
        # token
        self.BOS_token = self.cfg.train_meta_config.token_nums
        self.EOS_token = self.cfg.train_meta_config.token_nums + self.cfg.train_meta_config.append_token - 2
        
        # camera param
        camera_info_obj = CameraInfoParser(task_index=-1, parser_dir=self.cfg.cam_info_dir)
        self.intrinsic, self.extrinsic = camera_info_obj.intrinsic, camera_info_obj.extrinsic

    def predict(self, img_path_dict, bev_path, target_point: np.ndarray, is_raw:bool = True):
        images, bev_tensor, intrinsics, extrinsics = self.get_format_images(img_path_dict, bev_path)
        data = {
            "image": images,
            "bev":bev_tensor,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics
        }

        data["target_point"] = torch.from_numpy(target_point.astype(np.float32)[np.newaxis])  # shape [1, 2]
        data["fuzzy_target_point"] = data["target_point"]
        start_token = [self.BOS_token]
        data["gt_traj_point_token"] = torch.tensor([start_token], dtype=torch.int64)
        # 推理 
        self.model.eval()
        delta_predicts = self.inference(data)
        if is_raw:
            traj = np.array(delta_predicts)
            return traj
        
        # postprocess
        delta_predicts = fitting_curve(delta_predicts, num_points=self.cfg.train_meta_config.autoregressive_points, item_number=self.cfg.train_meta_config.item_number)
        traj_yaw_path = calculate_tangent(np.array(delta_predicts)[:, :2], mode="five_point")

        traj = []
        for (point_item, traj_yaw) in zip(delta_predicts, traj_yaw_path):
            if self.cfg.train_meta_config.item_number == 2:
                x, y = point_item
            elif self.cfg.train_meta_config.item_number == 3:
                x, y, progress_bar = point_item
                if abs(progress_bar) < 1 - self.cfg.progress_threshold:
                    break
            traj.append(self.get_posestamp_info(x, y, traj_yaw))
        return traj
        
    def inference(self, data):
        delta_predicts = []
        with torch.no_grad():
            if self.cfg.train_meta_config.decoder_method == "avg_fc":
                delta_predicts = self.inference_avg_fc(data)
            elif self.cfg.train_meta_config.decoder_method == "DETR":
                delta_predicts = self.inference_avg_fc(data)
            else:
                raise ValueError(f"Don't support decoder_method '{self.cfg.decoder_method}'!")
        delta_predicts = delta_predicts.tolist()
        return delta_predicts

    
    def inference_avg_fc(self, data):
        pred_traj_point = self.model.predict_avg_fc(data)
        pred_traj_point = pred_traj_point.view(30,2)
        return pred_traj_point
    
    def inference_DETR(self, data):
        pred_traj_point = self.model.predict_DETR(data)
        pred_traj_point = pred_traj_point.view(30,2)
        return pred_traj_point


    def get_posestamp_info(self, x, y, yaw):
        predict_pose = Pose()
        pose_flow_obj = PoseFlow(att_input=[yaw, 0, 0], type="euler", deg_or_rad="deg")
        quad = pose_flow_obj.get_quad()
        predict_pose.position = Point(x=x, y=y, z=0.0)
        predict_pose.orientation = Quaternion(x=quad.x, y=quad.y,z=quad.z, w=quad.w)
        return predict_pose

    def get_images(self, img_path):
        img_cv = cv2.imread(img_path)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_cv = img_cv.astype(np.float32) / 255
        img_tensor = torch.from_numpy(np.transpose(img_cv, ((2, 0, 1))))
        return img_tensor
    
    def load_pil_image(self, image_path):
        return Image.open(image_path).convert("RGB")

    def load_model(self, parking_pth_path):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = ParkingModelReal(self.cfg.train_meta_config)

        ckpt = torch.load(parking_pth_path, map_location='cpu')
        state_dict = OrderedDict([(k.replace('parking_model.', ''), v) for k, v in ckpt['state_dict'].items()])
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()


    def bev_preprocess(self, bev_path):
        if bev_path == None:
            return
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
        bev_pil = Image.open(bev_path).convert("RGB")
        bev_tensor = preprocess(bev_pil)
        return bev_tensor.unsqueeze(0)

    def get_format_images(self, img_path_dict, bev_path):
        process_width, process_height = int(self.cfg.train_meta_config.process_dim[0]), int(self.cfg.train_meta_config.process_dim[1])
        images, intrinsics, extrinsics = [], [], []
        for image_tag in self.images_tag:
            # pil_image = self.torch2pillow()(self.get_images(img_path_dict[image_tag]))
            pil_image = self.load_pil_image(img_path_dict[image_tag])
            image_obj = ProcessImage(pil_image, 
                                     self.intrinsic[image_tag], 
                                     self.extrinsic[image_tag], 
                                     target_size=(process_width, process_height))

            image_obj.resize_pil_image()
            images.append(get_normalized_torch_image(image_obj.resize_img))
            intrinsics.append(get_torch_intrinsics_or_extrinsics(image_obj.resize_intrinsics))
            extrinsics.append(get_torch_intrinsics_or_extrinsics(image_obj.extrinsics))

        images = torch.cat(images, dim=0).unsqueeze(0)
        intrinsics = torch.cat(intrinsics, dim=0).unsqueeze(0)
        extrinsics = torch.cat(extrinsics, dim=0).unsqueeze(0)
        bev_tensor = self.bev_preprocess(bev_path)
        return images, bev_tensor, intrinsics, extrinsics

    def torch2pillow(self):
        return torchvision.transforms.transforms.ToPILImage()

