# avm 
import numpy as np
import cv2
import math
class avm(object):
    def __init__(self):
        #小车尺寸
        car_h, car_w = 500, 200
        self.bev_width = 1600
        self.bev_height = 1600
        self.bev_center_x = self.bev_width/2
        self.bev_center_y = self.bev_height/2
        self.center2rear = 102

        self.car_left = int(self.bev_center_x-car_w/2)
        self.car_right =  int(self.bev_center_x+car_w/2)
        self.car_top = int(self.bev_center_y-car_h/2)
        self.car_bottom = int(self.bev_center_y+car_h/2)
        patch_w = self.car_left
        patch_h = self.car_top

        #右上角小块的mask
        patch_grids = np.meshgrid(np.arange(patch_w), np.arange(patch_h))
        patch_grids_x = patch_grids[0]
        patch_grids_y = patch_h - patch_grids[1]
        patch_grids_weight = patch_grids_y/(patch_grids_x+0.1)-np.tan(1.57/4)
        patch_grids_weight = np.clip(patch_grids_weight, 0,1)

        mask_front = np.zeros((self.bev_height,self.bev_width),dtype=np.float32)
        mask_front[:patch_h,:] = 1
        mask_front[:patch_h, self.car_right:self.car_right+ patch_w] = patch_grids_weight
        mask_front[:patch_h, :self.car_left] = patch_grids_weight[:, ::-1]
        mask_rear = mask_front[::-1, :]


        mask_right = np.zeros((self.bev_height,self.bev_width),dtype=np.float32)
        mask_right[:,self.car_right:] = 1
        mask_right[:patch_h, self.car_right:self.car_right+ patch_w] = 1-patch_grids_weight
        mask_right[self.car_bottom:self.bev_height, self.car_right:self.bev_width] = (1-patch_grids_weight)[::-1, :]
        mask_left = mask_right[:, ::-1]

        # mask_front[1:self.car_top+1, :] = mask_front[:self.car_top, :] 
        # mask_rear[self.car_bottom-1:self.bev_height-1, :] = mask_rear[self.car_bottom:, :] 


        self.masks = {
            'rgb_front':np.repeat(mask_front[:,:,np.newaxis], 3, axis=2),
            'rgb_rear':np.repeat(mask_rear[:,:,np.newaxis], 3, axis=2),
            'rgb_left':np.repeat(mask_left[:,:,np.newaxis], 3, axis=2),
            'rgb_right':np.repeat(mask_right[:,:,np.newaxis], 3, axis=2),

        }
        # cv2.imwrite('./dump/patch_f.jpg', mask_front*255)
        # cv2.imwrite('./dump/patch_b.jpg', mask_rear*255)
        # cv2.imwrite('./dump/patch_l.jpg', mask_left*255)
        # cv2.imwrite('./dump/patch_r.jpg', mask_right*255)


        #外参
        Rt_front = np.array([
        [0.99994552135467529,-0.0063513088971376419,0.0082848994061350822, 2.9067251682281494],
        [0.0060846279375255108,-0.29028576612472534,-0.95692068338394165, 143.10954284667969],
        [0.0084826871752738953,0.95691889524459839,-0.29023131728172302, -199.70960998535156 ],
        [0,0,0,1]
        ])
        Rt_rear = np.array([
        [-0.99986064434051514,0.0088267801329493523,-0.014169802889227867, -6.7475194931030273],
        [0.016682742163538933,0.49689623713493347,-0.8676496148109436, 207.89799499511719],
        [-0.00061763019766658545,-0.86776506900787354,-0.49697422981262207, -153.6607666015625],
            [0,0,0,1]
        ])

        Rt_left = np.array([
            [0.033330671489238739,0.9993520975112915,-0.013579367659986019, -63.609012603759766],
            [0.6595388650894165,-0.032201573252677917,-0.75098037719726562, 161.09248352050781],
            [-0.75093108415603638,0.016074558719992638,-0.66018491983413696, -0.64550012350082397],
            [0,0,0,1]
        ])

        Rt_right = np.array([
            [-0.01066849660128355,-0.99968576431274414,-0.022683475166559219, 64.157051086425781],
            [-0.69348704814910889,0.02373979240655899,-0.72007787227630615, 152.67385864257812],
            [0.72039008140563965,0.0080485483631491661,-0.69352239370346069, 12.809209823608398 ],
            [0,0,0,1]
        ])

        self.extrinsics = {
            'rgb_front':Rt_front,
            'rgb_rear':Rt_rear,
            'rgb_left':Rt_left,
            'rgb_right':Rt_right
        }

        #内参   用去畸变图做映射的话就改这个
        self.undis_intrinsic = np.array([
            [85, 0.0, 200],
            [0.0, 85, 150], 
            [0.0, 0.0, 1.0]
        ])
        #内参   用鱼眼做映射的话就改这个
        # self.undis_intrinsic = np.array([
        #     [30, 0.0, 200],
        #     [0.0, 30, 150], 
        #     [0.0, 0.0, 1.0]
        # ])

        self.maps = {}
        self.total_maps = {}

        
        self.images_tag = ("rgb_front", "rgb_rear", "rgb_left", "rgb_right")

        #用于计算total map
        camera_matrix = np.array([
            [327.586181640625, 0.0,640.0], 
            [0.0, 327.586181640625, 480.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        R = np.eye(3, dtype=np.float32)
        size = (400, 300)

        dist_coeffs = np.array([0.12333839000000001, -0.032352529999999997, 0.0081698799999999992, -0.0014338899999999999], dtype=np.float32)
        undis_mapx, undis_mapy = cv2.fisheye.initUndistortRectifyMap(camera_matrix, dist_coeffs, R, self.undis_intrinsic, size, cv2.CV_32FC1)
        
        

        for tag in self.images_tag:
            grids = np.meshgrid(np.arange(self.bev_width), np.arange(self.bev_height))
            grids = np.stack(grids, axis=2).astype(np.float32)
            grids = np.concatenate((grids, np.zeros((self.bev_height, self.bev_width, 1))), axis = 2)
            grids = np.concatenate((grids, np.ones((self.bev_height, self.bev_width, 1))), axis = 2)
            grids[:, :, 0] = (grids[:, :, 0] - self.bev_width/2.0)
            grids[:, :, 1] = -(grids[:, :, 1] - self.bev_height/2.0)
            grids = grids.reshape((-1, 4)).transpose(1, 0)
            coor_camera = np.matmul(self.extrinsics[tag], grids)
            coor_image = np.matmul(self.undis_intrinsic,coor_camera[:3, :])
            
            #齐次
            coor_image = coor_image.transpose(1, 0).reshape((self.bev_height, self.bev_width, 3))
            coor_image = coor_image[:, :, 0:2] / coor_image[:, :, 2:]
            self.maps[tag] = coor_image.astype(np.float32)

            #                       undis_mapx（畸变） self.maps（投影+旋转）
            map_total_x = cv2.remap(undis_mapx, self.maps[tag][:,:,0], self.maps[tag][:,:,1], cv2.INTER_LINEAR)
            map_total_y = cv2.remap(undis_mapy, self.maps[tag][:,:,0], self.maps[tag][:,:,1], cv2.INTER_LINEAR)
            self.total_maps[tag] = np.stack((map_total_x, map_total_y),axis=-1)
            

    def apply(self, img_path_dict:dict):
        out = np.zeros((self.bev_height, self.bev_width, 3), np.float32)
        for tag in self.images_tag:
            img = cv2.imread(img_path_dict[tag])
            xscale = 1
            yscale = 1
            #从去畸变图上查找像素
            bev = cv2.remap(img, self.maps[tag][:, :, 0] * xscale, self.maps[tag][:, :, 1] * yscale, cv2.INTER_LINEAR)
            # 从鱼眼上找像素
            # bev = cv2.remap(img, self.total_maps[tag][:, :, 0] * xscale, self.total_maps[tag][:, :, 1] * yscale, cv2.INTER_LINEAR)
            out += bev.astype(np.float32) * self.masks[tag]

        out[self.car_top-10 : self.car_bottom+10, self.car_left-10 : self.car_right+10] = 0
        return out

def save_frame1():
    # 打开视频文件
    video_path = 'gw_test_video/rgb_left.mp4'
    cap = cv2.VideoCapture(video_path)
    # 检查视频是否成功打开
    if not cap.isOpened():
        print("无法打开视频")
        exit()

    # 读取第一帧
    ret, frame = cap.read()

    # 检查是否成功读取到帧
    if ret:
        # 保存第一帧为图像文件
        cv2.imwrite('fish_left.jpg', frame)
    else:
        print("无法读取视频帧")

    # 释放视频资源
    cap.release()



def generate_bev(root_dir="e2e_dataset_gw/parking_twice/train/demo_bag"):
    avm_interface = avm()
    import os
    for filename in os.listdir(root_dir):
        folder_path = os.path.join(root_dir,filename)
        bev_path = os.path.join(folder_path, "bev")
        os.makedirs(bev_path, exist_ok=True)
        image_folder_dir = os.path.join(folder_path,"rgb_front")

        print(bev_path)
        for image_path in os.listdir(image_folder_dir):
            front_image_path = os.path.join(image_folder_dir,image_path)
            rear_image_path = front_image_path.replace("rgb_front","rgb_rear")
            right_image_path = front_image_path.replace("rgb_front","rgb_left")
            left_image_path = front_image_path.replace("rgb_front","rgb_left")
            image_name = front_image_path.split('/')[-1]
            img_path_dict = {"rgb_front": front_image_path,
                 "rgb_rear": rear_image_path,
                 "rgb_left": left_image_path,
                 "rgb_right": right_image_path}
            
            bev = avm_interface.apply(img_path_dict)
            bev = cv2.resize(bev,(400,400),interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(os.path.join(bev_path,image_name), bev)
    

if __name__ == "__main__":


    generate_bev()

    # save_frame1()
    # img_path_dict = {"rgb_front": "e2e_dataset_gw/package_150/val/demo_bag/20240710131120_right/rgb_front/0000.png",
    #              "rgb_rear": "e2e_dataset_gw/package_150/val/demo_bag/20240710131120_right/rgb_rear/0000.png",
    #              "rgb_left": "e2e_dataset_gw/package_150/val/demo_bag/20240710131120_right/rgb_left/0000.png",
    #              "rgb_right": "e2e_dataset_gw/package_150/val/demo_bag/20240710131120_right/rgb_right/0000.png",}
    # avm_interface = avm()
    # avm_interface.apply(img_path_dict)
    
    # img_path_dict = {"rgb_front": "fish_front.jpg",
    #              "rgb_rear": "fish_rear.jpg",
    #              "rgb_left": "fish_left.jpg",
    #              "rgb_right": "fish_right.jpg",}
    # avm_interface = avm()
    # avm_interface.apply(img_path_dict)