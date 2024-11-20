# PrakingTrajPredict
Trajectory Prediction in Simple Scenario based on LSS and Detr Decoder

#  Simple Parking Scenario Video  


https://github.com/user-attachments/assets/56be38e7-48df-48f6-833f-81a4546a5d46



## 1. 主要环境依赖
``` 
pytorch 2.1.2  
torchvision 0.16.2  
python 3.8.8
``` 
## 2. 仓库主要内容
1. 一套基于车身信号和avm的数据集生成方案（C++）
2. 一套avm可视化代码（python）
3. 一些基于ParkingE2E(https://github.com/qintonguav/ParkingE2E) 的魔改模型优化实验（python）
4. 开源pretrain model和训练数据

## 3. Train
``` 
CUDA_VISIBLE_DEVICES=1,2,3,4 python train.py
``` 
## 4. Test
``` 
python img_inference_avm --tasks_path=demo_bag
``` 
## 5. 训练数据内容说明

训练数据路径组成  
```
e2e_dataset/
├── train/ (训练集、验证集)
│   ├── demo_bag/ （由ros转出来的去畸变数据）
|       |——1708689773_right/    
|           |——rgb_front/       （undistorted image）
|           |——rgb_left/
|           |——rgb_right/
|           |——rgb_rear/
|           |——measurements/    （世界坐标系下ego的位姿roll,yaw,pitch,x,y,z, 世界坐标系定义为第一帧的坐标系）
|           |——parking_goal/    （target point）
|           |——camera_config_right_hand.json    （camera info）
``` 

### 5.1 坐标系
![image](https://github.com/user-attachments/assets/198edb78-e587-4646-a649-7c91c16f8b46)


### 5.2 camera info
json中以roll,yaw,pitch来记录相机的外参R，这里是根据右手定则算出的world2camera。这里的camera坐标系形态也是x为光轴,z垂直地面向上,y朝左手. 在训练代码中会用特定的矩阵将其转换成下图,然后进入到LSS模块。  
![image](https://github.com/user-attachments/assets/ff901b70-eb17-49eb-bece-7afb1bad2944)  
x,y,z为外参t, world watch camera position / coor camera2world  坐标系表征与坐标系转换区别：https://zhuanlan.zhihu.com/p/618604141  
内参：fov,image_h,image_w


## 6. 模型优化实验
### 6.1 轨迹预测头
1. 用DETR Decoder替换原有的轨迹点解码头(text decoder), 解决轨迹预测方向反向的问题(text decoder是串行迭代的流程,假如前面的预测误差对后面影响很大)
2. 用简单的avgpool + FC 直接预测轨迹点.

### 6.2 BEV Feature实验
1. 直接输入BEV Image, 并用EfficientNet, Resnet, SwinTransformer, VIT等backbone直接提取图像特征(替换掉将4张图象经LSS生成BEV Feature的方法)
实验结论：LSS + DETR Decoder的结构可以得到最优结果
   






