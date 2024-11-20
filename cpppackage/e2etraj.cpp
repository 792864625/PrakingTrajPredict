#include <opencv2\imgproc\types_c.h>

#include <fstream>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>
#include <cmath>


#include "jsoncpp-src-0.5.0/jsoncpp-src-0.5.0/include/json/json.h"
using namespace std;

enum class EParkDirection : int32_t { STOP = 0, FORWARD = 1, BACKWARD = 2 };
enum class EParkSteerDirection : int32_t { LEFT = 0, RIGHT = 1 };
enum class EParkGear : int32_t {
  P_GEAR = 0,
  D_GEAR = 1,
  N_GEAR = 2,
  R_GEAR = 3,
  M_GEAR = 4
};
typedef struct SParkCanInfo {
  float fl_wheel_spd = 0.0f;  // 左前轮轮速
  float fr_wheel_spd = 0.0f;  // 右前轮轮速
  float rl_wheel_spd = 0.0f;  // 左后轮轮速
  float rr_wheel_spd = 0.0f;  // 右后轮轮速
  int id = 0;                 // 当前帧号
  long long timestamp = 0;    // timestamp(ms),greenwich time
  EParkDirection fl_wheel_spd_direction = EParkDirection::STOP;  // 左前轮方向
  EParkDirection fr_wheel_spd_direction = EParkDirection::STOP;  // 右前轮方向
  EParkDirection rl_wheel_spd_direction = EParkDirection::STOP;  // 左后轮方向
  EParkDirection rr_wheel_spd_direction = EParkDirection::STOP;  // 右后轮方向
  EParkGear gear = EParkGear::P_GEAR;                            // 档位
  EParkSteerDirection steerwheel_direction =
      EParkSteerDirection::LEFT;  // 方向盘朝向
  float steerwheel_angle = 0.0f;  // 方向盘角度
} _PCI_;

struct SParkCanInfoRt {
  SParkCanInfo can_info;
  float delta_time = 0.0f;  // ms
};
SParkCanInfo cur_can_info_;
SParkCanInfo last_can_info_;
SParkCanInfoRt pre_can_info_rt_;
SParkCanInfoRt can_info_rt_;
float m_theta_cum = 0.0f;           //累计若干帧的转角theta，大于一定值才保存traj
float m_delta_x_pixel_cum = 0.f;
float m_delta_y_pixel_cum = 0.f;
float m_steering_ratio = 15.5f;
float m_forward_speed_coef = 1.025f;
float m_reverse_speed_coef = 1.025f;
float m_thres_speed = 3.0f;
float m_forward_speed_compensate = 0.12f;
float m_reverse_speed_compensate = 0.20f;
float m_wheel_base = 2800.0f;
float m_ppm = 75.0f;
std::vector<float> m_rt_matrix;
float m_chassis_displayment_cum = 0.0f;
float m_chassis_displayment_thres = 0.025f;

int m_bev_w = 640;
int m_bev_h = 840;
float m_car_h = 4771.0f;
float m_car_w_mm = 1895.0f;
float m_rear_overhang = 1023.0f;
float m_mmpp = 1000 / m_ppm;
float center2rear = (m_car_h / 2.0f - m_rear_overhang) / m_mmpp;
float m_center_x = m_bev_w / 2.0f;
float m_center_y = m_bev_h / 2.0f + center2rear;
cv::Mat frame_center = (cv::Mat_<double>(3, 1) << m_center_x, m_center_y, 1);

float m_segm_chassis_displayment = 0.0f;
float m_prev_chassis_displayment = 0.0f;
float m_curr_chassis_displayment = 0.0f;
cv::Mat m_HomoGlobal = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
cv::Mat m_PointGlobal;
std::vector<std::array<float, 6>> m_saved_traj;
std::vector<int> m_traj_index_vec;
float m_theta_total = 0;


//不用重置
cv::Mat m_mapx;
cv::Mat m_mapy;
void GetThetaTxy(const SParkCanInfoRt& pre_can_info_rt,
                 const SParkCanInfoRt& can_info_rt, float& delta_x,
                 float& delta_y, float& theta) {
  const SParkCanInfo& can_info = can_info_rt.can_info;
  const SParkCanInfo& pre_can_info = pre_can_info_rt.can_info;

  if (pre_can_info_rt.delta_time < 1 || pre_can_info_rt.delta_time > 1000) {
    return;
  }

  // compute angle
  // Steering wheel Angle Turn to tire Angle
  // 15.5 is the ratio of steering wheel angle to tire angle
  // Define right as positive and left as negative
  auto angle = fabs(pre_can_info.steerwheel_angle);
  if (pre_can_info.steerwheel_direction == EParkSteerDirection::LEFT) {
    angle = angle * -1.0f;
  }
  angle = angle / m_steering_ratio;  //轮胎转角（pre）

  const double DEG_TO_RAD = 0.017453292519943295769236907684886;
  float sinAlpha = static_cast<float>(sin(angle * DEG_TO_RAD));
  float cosAlpha = static_cast<float>(cos(angle * DEG_TO_RAD));

  // compute speed
  // If it's in reverse, the speed is negative,otherwise it's positive
  // 3.6 is the ratio of km/h to m/s
  float pre_wheel_speed =
      (pre_can_info.rl_wheel_spd + pre_can_info.rr_wheel_spd) *
      0.5f;  //后轮速度（pre）
  float cur_wheel_speed = (can_info.rl_wheel_spd + can_info.rr_wheel_spd) *
                          0.5f;  //后轮速度（curr）
  if (pre_can_info.rl_wheel_spd == 0.0f || pre_can_info.rr_wheel_spd == 0.0f) {
    pre_wheel_speed = 0.0f;
  }
  if (can_info.rl_wheel_spd == 0.0f || can_info.rr_wheel_spd == 0.0f) {
    cur_wheel_speed = 0.0f;
  }
  float wheel_speed =
      (pre_wheel_speed + cur_wheel_speed) * 0.5f / 3.6f;  //后轮速度 m/s
  if (pre_can_info.rl_wheel_spd_direction == EParkDirection::BACKWARD) {
    wheel_speed = wheel_speed * -1.0f;
  }

  float forward_speed_param = m_forward_speed_coef;
  float reverse_speed_param = m_reverse_speed_coef;
  if (cur_wheel_speed <
      m_thres_speed) {  //低速的时候添加一些速度补偿（拿到的轮速不准）
    if (can_info.rl_wheel_spd_direction == EParkDirection::BACKWARD) {
      reverse_speed_param += (m_thres_speed - cur_wheel_speed) / m_thres_speed *
                             m_reverse_speed_compensate;
    } else {
      forward_speed_param += (m_thres_speed - cur_wheel_speed) / m_thres_speed *
                             m_forward_speed_compensate;
    }
  }

  if (pre_can_info.rl_wheel_spd_direction == EParkDirection::STOP) {
    wheel_speed = 0.0f;
  } else {
    wheel_speed =
        (pre_can_info.rl_wheel_spd_direction == EParkDirection::BACKWARD)
            ? wheel_speed * reverse_speed_param
            : wheel_speed * forward_speed_param;
  }

  // compute theta/delta_x/delta_y
  auto delta_time = pre_can_info_rt.delta_time / 1000.0f;  // s
  theta = wheel_speed * delta_time * sinAlpha / cosAlpha /
          (m_wheel_base / 1000.0f);  // 知乎上那个公式

  delta_x = wheel_speed * delta_time * sin(theta) * m_ppm;
  delta_y = wheel_speed * delta_time * cos(theta) * m_ppm;
}

void EstimateMotion(const SParkCanInfoRt& pre_can_info_rt,
                    const SParkCanInfoRt& can_info_rt) {
  float delta_x = 0.0f;
  float delta_y = 0.0f;
  float theta = 0.0f;
  GetThetaTxy(pre_can_info_rt, can_info_rt, delta_x, delta_y, theta);
  m_theta_cum += theta;
  m_delta_x_pixel_cum += delta_x;
  m_delta_y_pixel_cum += delta_y;
}

void GetChassisParams(const SParkCanInfoRt& pre_can_info_rt,
                      const SParkCanInfoRt& can_info_rt,
                      const int frame_index) {
  const SParkCanInfo& can_info = can_info_rt.can_info;
  const SParkCanInfo& pre_can_info = pre_can_info_rt.can_info;

  EstimateMotion(pre_can_info_rt, can_info_rt);

  m_chassis_displayment_cum =
      sqrt((m_delta_x_pixel_cum / m_ppm) * (m_delta_x_pixel_cum / m_ppm) +
           (m_delta_y_pixel_cum / m_ppm) * (m_delta_y_pixel_cum / m_ppm));


  //判断两帧之间的位移大于阈值才记录轨迹点
  if ((pre_can_info.gear == EParkGear::R_GEAR &&
       can_info.gear != EParkGear::R_GEAR) ||
      (pre_can_info.gear != EParkGear::R_GEAR &&
       can_info.gear == EParkGear::R_GEAR) ||
      fabs(m_chassis_displayment_cum) > m_chassis_displayment_thres) {
    // compute bev pic rt matrix
    /*cout << "translation enough    theta = " << m_theta_cum
         << " x_pixel_cum = " << m_delta_x_pixel_cum
         << " y_pixel_cum = " << m_delta_y_pixel_cum << endl;*/
    float cos_theta_cum = cosf(m_theta_cum);
    float sin_theta_cum = sinf(m_theta_cum);

    m_rt_matrix[0] = cos_theta_cum;
    m_rt_matrix[1] = sin_theta_cum;
    m_rt_matrix[2] = (1 - cos_theta_cum) * m_center_x -
                     sin_theta_cum * m_center_y + m_delta_x_pixel_cum;
    m_rt_matrix[3] = -sin_theta_cum;
    m_rt_matrix[4] = cos_theta_cum;
    m_rt_matrix[5] = sin_theta_cum * m_center_x +
                     (1 - cos_theta_cum) * m_center_y + m_delta_y_pixel_cum;
    m_rt_matrix[6] = 0.0f;
    m_rt_matrix[7] = 0.0f;
    m_rt_matrix[8] = 1.0f;

    m_traj_index_vec.push_back(frame_index); //记录轨迹点对应的帧号
    // Homo表示当前帧->下一帧的坐标转换
    cv::Mat Homo =
        (cv::Mat_<double>(3, 3) << m_rt_matrix[0], m_rt_matrix[1],
         m_rt_matrix[2], m_rt_matrix[3], m_rt_matrix[4], m_rt_matrix[5],
         m_rt_matrix[6], m_rt_matrix[7], m_rt_matrix[8]);

    //cout << Homo << endl;

    //转换到第一帧坐标系下
    m_HomoGlobal = Homo * m_HomoGlobal;
    cv::Mat HomoGlobal;
    cv::invert(m_HomoGlobal, HomoGlobal);
    m_PointGlobal = HomoGlobal * frame_center;

    //处理和保存traj
    m_theta_total -= m_theta_cum;

    float x = m_PointGlobal.at<double>(0, 0) - m_center_x;
    float y = m_PointGlobal.at<double>(1, 0) - m_center_y;
    std::array<float, 6> vec = {-y / m_ppm, -x / m_ppm,
                                0,
                                m_theta_total/3.1415*180, //todo：要从弧度改成角度
                                0,
                                0};
    m_saved_traj.push_back(vec);


    //转换到当前帧坐标系下
    /*m_HomoGlobal = Homo * m_HomoGlobal;
    cv::Mat bbox_center = (cv::Mat_<double>(3, 1) << 225, 745, 1);
    m_PointGlobal = m_HomoGlobal * bbox_center;*/

    // reset cumulater value
    m_delta_x_pixel_cum = 0.f;
    m_delta_y_pixel_cum = 0.f;
    m_theta_cum = 0.f;
    m_segm_chassis_displayment = 0.0f;
  }

  if (fabs(m_prev_chassis_displayment) > m_chassis_displayment_thres) {
    m_prev_chassis_displayment = 0.f;
  }

  m_curr_chassis_displayment =
      m_chassis_displayment_cum - m_prev_chassis_displayment;
  m_prev_chassis_displayment = m_chassis_displayment_cum;

  return;
}

void SendCanData(const SParkCanInfo& can_info, const int frame_index) {
  last_can_info_ = cur_can_info_;
  cur_can_info_ = can_info;
  if (last_can_info_.timestamp > cur_can_info_.timestamp) {
    return;
  }

  if (last_can_info_.timestamp == 0) {
    return;
  }
  pre_can_info_rt_ = can_info_rt_;
  can_info_rt_.can_info = cur_can_info_;
  can_info_rt_.delta_time = cur_can_info_.timestamp - last_can_info_.timestamp;
  GetChassisParams(pre_can_info_rt_, can_info_rt_, frame_index);
}





void bev_first_frame(const std::string& video_path, cv::Mat& bev_show) {
  cv::VideoCapture cap(video_path);
  if (!cap.isOpened()) {
    std::cerr << "无法打开视频文件: " << video_path << std::endl;
    return;
  }
  cap >> bev_show;  // 从视频中读取一帧
  cap.release();
}

int frame_num(const std::string& video_path) {
  cv::VideoCapture cap(video_path);
  // 检查视频是否成功打开
  if (!cap.isOpened()) {
    std::cerr << "Error: Could not open video file " << video_path << std::endl;
    return -1;  // 返回 -1 表示错误
  }
  // 获取视频的帧数
  int frameCount = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
  return frameCount;
}

int fish_bev_frame_num_min(const std::string& root_path) {

  std::string video_path_f = root_path + "/front.mp4 ";
  int fish_front_num = frame_num(video_path_f);

  std::string video_path_back = root_path + "/back.mp4 ";
  int fish_back_num = frame_num(video_path_back);

  std::string video_path_l = root_path + "/left.mp4 ";
  int fish_left_num = frame_num(video_path_l);

  std::string video_path_r = root_path + "/right.mp4 ";
  int fish_right_num = frame_num(video_path_r);
  
  std::string video_path_bev = root_path + "/bev.mp4 ";
  int bev_num = frame_num(video_path_bev);

  int minValue = std::min(
      {fish_front_num, fish_back_num, fish_left_num, fish_right_num, bev_num});

  return minValue;
}
void fish_undis_dump(const std::string& video_path,
                     const std::vector<int>& traj_index_vec,
                     const std::string& dump_path);
void json_dump(const std::string& dump_json_path,
               const std::string& dump_json_path_goal,
               const std::vector<std::array<float, 6>>& saved_traj);
void e2e_traj(const std::string& root_path) {
  int bev_fish_min_num = fish_bev_frame_num_min(root_path);
  //读取bev视频帧
  std::string BevPath = root_path + "/bev.mp4 ";
  cv::Mat frame_show;
  bev_first_frame(BevPath, frame_show);
  // 创建 VideoCapture 对象
  cv::VideoCapture cap(BevPath);
  if (!cap.isOpened()) {
    std::cerr << "无法打开视频文件: " << BevPath << std::endl;
    return;
  }
  Json::Value config;
  Json::Reader reader;

  //计算每一条轨迹之前，先初始化各种参数，这些参数在循环中会改变
  m_rt_matrix.resize(9);
  m_rt_matrix[0] = 1.0f;
  m_rt_matrix[4] = 1.0f;
  m_rt_matrix[8] = 1.0f;
  m_HomoGlobal = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
  m_PointGlobal = frame_center.clone(); //深拷贝，因为后面m_PointGlobal会变。
  m_saved_traj.clear();
  m_traj_index_vec.clear();
  m_theta_total = 0.f;
  int start_index = 0;
  std::string json_path = root_path + "/front.json";
  std::ifstream ifs(json_path);  // Open Json
  
  if (!reader.parse(ifs, config)) {
    cout << "fail to parse json" << endl;
  } else {
    // int index = 0;
    cout << config["sensor_info"].size() << endl;
    const Json::Value& sensor_info_array = config["sensor_info"];
    for (Json::Value::ArrayIndex index = start_index; index < bev_fish_min_num;
         ++index) {
      auto sensorInfo = sensor_info_array[index];
      bool new_json = sensorInfo.size() == 12 ? true : false;
      float fr_speed =
          static_cast<float>(atof(sensorInfo[1].asString().c_str()));
      float fl_speed =
          static_cast<float>(atof(sensorInfo[2].asString().c_str()));
      float rr_speed =
          static_cast<float>(atof(sensorInfo[3].asString().c_str()));
      float rl_speed =
          static_cast<float>(atof(sensorInfo[4].asString().c_str()));

      int id = sensorInfo[9].asInt();
      float angle =
          static_cast<float>(atof(sensorInfo[5].asString().c_str())) * 10.f;
      int direction = sensorInfo[6].asInt();
      int gear = sensorInfo[7].asInt();
      int power = 2;
      if (new_json) {
        power = sensorInfo[8].asInt();
      }
      int timestamp_index = new_json ? 10 : 9;
      uint64_t curr_timestamp =
          (std::stoull(sensorInfo[timestamp_index].asString().c_str()));
      uint64_t prev_timestamp;
      if (id <= 2) {
        prev_timestamp = curr_timestamp - 33;
      } else {
        auto prev_sensorInfo = sensor_info_array[index - 1];
        prev_timestamp =
            (std::stoull(prev_sensorInfo[timestamp_index].asString().c_str()));
      }

      EParkDirection park_direction;
      EParkGear park_gear;
      EParkSteerDirection park_steer_direction;
      if (gear == 7) {
        park_direction = EParkDirection::BACKWARD;
        park_gear = EParkGear::R_GEAR;
      } else if (gear == 5) {
        park_direction = EParkDirection::FORWARD;
        park_gear = EParkGear::D_GEAR;
      } else {
        park_direction = EParkDirection::FORWARD;
        park_gear = EParkGear::P_GEAR;
      }
      if (direction == 0) {
        park_steer_direction = EParkSteerDirection::LEFT;
      } else {
        park_steer_direction = EParkSteerDirection::RIGHT;
      }

      SParkCanInfo park_can_info;
      park_can_info.fl_wheel_spd = fl_speed;
      park_can_info.fl_wheel_spd_direction = park_direction;
      park_can_info.fr_wheel_spd = fr_speed;
      park_can_info.fr_wheel_spd_direction = park_direction;
      park_can_info.rr_wheel_spd = rr_speed;
      park_can_info.rr_wheel_spd_direction = park_direction;
      park_can_info.rl_wheel_spd = rl_speed;
      park_can_info.rl_wheel_spd_direction = park_direction;
      park_can_info.id = id;
      park_can_info.timestamp = curr_timestamp;
      park_can_info.gear = park_gear;
      park_can_info.steerwheel_angle = angle;
      park_can_info.steerwheel_direction = park_steer_direction;
      SendCanData(park_can_info, index);
      int point_x = m_PointGlobal.at<double>(0, 0);
      int point_y = m_PointGlobal.at<double>(1, 0);

      // frame_show = bev_frames[index].clone();
      cv::Point pts((point_x), (point_y));
      cv::circle(frame_show, pts, 2, cv::Scalar(0, 255, 0), -1);  // 绿色点
    }

    cv::imwrite(root_path + "/bev_show.jpg", frame_show);

    std::vector<std::array<float, 6>> test_traj = m_saved_traj;
  }
  /*cout << "valid traj num = " << m_traj_index_vec.size() << endl;
  cout << "total traj num = " << bev_fish_min_num << endl;*/

  //存鱼眼图和json
  json_dump(root_path + "/measurements", root_path + "/parking_goal",  m_saved_traj);

  fish_undis_dump(root_path + "/front.mp4", m_traj_index_vec,
                  root_path + "/rgb_front");
  fish_undis_dump(root_path + "/back.mp4", m_traj_index_vec,
                  root_path + "/rgb_rear");
  fish_undis_dump(root_path + "/left.mp4", m_traj_index_vec,
                  root_path + "/rgb_left");
  fish_undis_dump(root_path + "/right.mp4", m_traj_index_vec,
                  root_path + "/rgb_right");

}
#include <direct.h>       // 文件夹操作

void fish_undis_dump(const std::string& video_path,
    const std::vector<int> &traj_index_vec, const std::string &dump_path) {
  _mkdir(dump_path.c_str());  // 创建文件夹
  std::string dump_path_fish = dump_path + "_fish";
  _mkdir(dump_path_fish.c_str());
    //存鱼眼图
  cv::VideoCapture cap_fish(video_path);
  if (!cap_fish.isOpened()) {
    std::cerr << "无法打开视频文件: " << std::endl;
    return;
  }
  // 读取视频帧
  int iter_traj = 0;
  int iter_fish_frame = 0;
  while (true) {
    cv::Mat frame;
    cap_fish >> frame;             
    if (frame.empty() || iter_traj == traj_index_vec.size()) break;      

    if (iter_fish_frame == traj_index_vec[iter_traj]) {
      std::ostringstream ss;
      ss << std::setw(4) << std::setfill('0') << iter_traj
         << ".png";  // 使用setw(4)指定宽度，setfill('0')填充0
      std::string filename = ss.str();

      cv::Mat test;
      cv::remap(frame, test, m_mapx, m_mapy, cv::INTER_LINEAR);

      //cv::imwrite(dump_path_fish + "/" + filename, frame);
      cv::imwrite(dump_path + "/" + filename, test);
        iter_traj++;
    }

    iter_fish_frame++;
  }
  cap_fish.release();
}


void json_dump(const std::string& dump_json_path,
               const std::string& dump_json_path_goal,
               const std::vector<std::array<float, 6>> &saved_traj) {
  _mkdir(dump_json_path.c_str());  // 创建文件夹
  for (size_t i = 0; i < saved_traj.size(); i++) {
    Json::Value root;  // 创建一个JSON值对象

    //创建一个对象
    Json::Value item;
    item["x"] = saved_traj[i][0];
    item["y"] = saved_traj[i][1];
    item["z"] = saved_traj[i][2];
    item["roll"] = saved_traj[i][4];
    item["yaw"] = saved_traj[i][3];
    item["pitch"] = saved_traj[i][5];

    // 将对象添加到根对象
    root = item;

    std::ostringstream ss;
    ss << std::setw(4) << std::setfill('0') << i
       << ".json";  // 使用setw(4)指定宽度，setfill('0')填充0
    std::string filename = ss.str();
    // 写入文件
    std::ofstream outfile(dump_json_path + "/" + filename);
    outfile << root.toStyledString();
    outfile.close();

  }

  _mkdir(dump_json_path_goal.c_str());
  Json::Value root;  // 创建一个JSON值对象
  //创建一个对象
  Json::Value item;
  item["x"] = saved_traj.back()[0];
  item["y"] = saved_traj.back()[1];
  item["z"] = saved_traj.back()[2];
  item["roll"] = saved_traj.back()[4];
  item["yaw"] = saved_traj.back()[3];
  item["pitch"] = saved_traj.back()[5];

  // 将对象添加到根对象
  root = item;
  std::ofstream outfile(dump_json_path_goal + "/" + "0001.json");
  outfile << root.toStyledString();
  outfile.close();

  
}



//文件夹数量
#undef UNICODE
#include <windows.h>
void ListFilesInDirectory(const std::string& directory,
                          std::vector<std::string> &data_path) {
  WIN32_FIND_DATA findFileData;
  HANDLE hFind = FindFirstFile((directory + "\\*").c_str(), &findFileData);
  if (hFind == INVALID_HANDLE_VALUE) {
    std::cout << "Error: No files found in directory." << std::endl;
    return;
  }
  do {
    const std::string fileName = findFileData.cFileName;
    if (fileName != "." && fileName != "..") {
      std::cout << fileName << std::endl;
      data_path.push_back(directory + fileName);
    }
  } while (FindNextFile(hFind, &findFileData) != 0);

  FindClose(hFind);
}
cv::Mat m_fish_intrinsic = (cv::Mat_<double>(3, 3) << 327.586181640625, 0.0,
                          640.0, 0.0, 327.586181640625, 480.0, 0.0, 0.0, 1.0);

cv::Mat m_undis_intrinsic =
    (cv::Mat_<double>(3, 3) << 85, 0.0, 200, 0.0, 85, 150, 0.0, 0.0, 1.0);

cv::Mat m_undis2fish_params =
    (cv::Mat_<double>(4, 1) << 0.12333839000000001, -0.032352529999999997,
     0.0081698799999999992, -0.0014338899999999999);
void undis_map() {
  
  cv::Mat R = cv::Mat::eye(3, 3, CV_32F);
  
  // cv::Mat m_undis_intrinsic;
  // m_fish_intrinsic.copyTo(m_undis_intrinsic);
  cv::fisheye::initUndistortRectifyMap(
      m_fish_intrinsic, m_undis2fish_params, R, m_undis_intrinsic,
      cv::Size(m_undis_intrinsic.at<double>(0, 2) * 2,
               m_undis_intrinsic.at<double>(1, 2) * 2),
      CV_32FC1, m_mapx, m_mapy);

  cv::Mat fish_intrin = m_fish_intrinsic;
  cv::Mat undis_intrin = m_undis_intrinsic;

  cv::Mat undis_params = m_undis2fish_params;

  cv::Mat mapx = m_mapx;
  cv::Mat mapy = m_mapy;
}

std::array<float, 6> cal_e2e_extrin(const Json::Value& R_bev,
                                    const Json::Value& t_bev) {
  double data_t[3] = {0};
  for (int i = 0; i < 3; i++) {
    data_t[i] = t_bev[i].asDouble();
  }
  double data_R[9] = {0};
  for (int i = 0; i < 9; i++) {
    data_R[i] = R_bev[i].asDouble();
  }
  cv::Mat R = cv::Mat(3, 3, CV_64F, data_R).clone();

  float pitch_test = asin(-R.at<double>(2, 1));
  float yaw_test = atan(R.at<double>(2, 0) / R.at<double>(2, 2));

  cv::Mat t = cv::Mat(3, 1, CV_64F, data_t).clone();
  cv::Mat world2cam = cv::Mat::eye(4, 4, CV_64F);  // 4x4 单位矩阵
  R.copyTo(world2cam(cv::Rect(0, 0, 3, 3)));
  t.copyTo(world2cam(cv::Rect(3, 0, 1, 3)));

  // ros2world
  cv::Mat ros2world = (cv::Mat_<double>(4, 4) << 0, -1, 0, 0, 1, 0, 0, 0, 0, 0,
                       1, 0, 0, 0, 0, 1);

  // cam -> camE2E
  cv::Mat cam2camE2E = (cv::Mat_<double>(4, 4) << 0, 0, 1, 0, -1, 0, 0, 0, 0,
                        -1, 0, 0, 0, 0, 0, 1);
  cv::Mat ros2camE2E = cam2camE2E * world2cam * ros2world;

  cv::Mat camE2E_2_ros;
  bool success = cv::invert(ros2camE2E, camE2E_2_ros);
  float pitch = asin(-ros2camE2E.at<double>(0, 2));
  float roll = atan(ros2camE2E.at<double>(1, 2) / ros2camE2E.at<double>(2, 2));
  float yaw = atan(ros2camE2E.at<double>(0, 1) / ros2camE2E.at<double>(0, 0));

  std::array<float, 6> arr = {
                              yaw/3.1415*180,
                              roll / 3.1415 * 180,
                              pitch/3.1415*180,
                              camE2E_2_ros.at<double>(0, 3)/100.f,
                              camE2E_2_ros.at<double>(1, 3) / 100.f,
                              camE2E_2_ros.at<double>(2, 3) / 100.f
  };
  return arr;
}

void dump_camera_info_json(const std::string& input_json_path,
                           const std::string& dump_json_path) {
  Json::Value config;
  Json::Reader reader;
  std::ifstream ifs(input_json_path);  // Open Json
  if (!reader.parse(ifs, config)) {
    cout << "fail to parse camera_info json" << endl;
  } else {
    const Json::Value& R_f = config["front"]["R_bev"];
    const Json::Value& t_f = config["front"]["t"];
    std::array<float, 6> front_extrin = cal_e2e_extrin(R_f, t_f);

    const Json::Value& R_b = config["back"]["R_bev"];
    const Json::Value& t_b = config["back"]["t"];
    std::array<float, 6> back_extrin = cal_e2e_extrin(R_b, t_b);
    back_extrin[0] =
        back_extrin[0] > 0 ? back_extrin[0] - 180 : back_extrin[0] + 180;

    //back_extrin[0] -= 180;  //back的yaw超过180°

    const Json::Value& R_l = config["left"]["R_bev"];
    const Json::Value& t_l = config["left"]["t"];
    std::array<float, 6> left_extrin = cal_e2e_extrin(R_l, t_l);

    const Json::Value& R_r = config["right"]["R_bev"];
    const Json::Value& t_r = config["right"]["t"];
    std::array<float, 6> right_extrin = cal_e2e_extrin(R_r, t_r);
   
    //只算了x方向的fov
    double f = m_undis_intrinsic.at<double>(0, 0);
    int width = m_undis_intrinsic.at<double>(0, 2) * 2;
    int height = m_undis_intrinsic.at<double>(1, 2) * 2;
    double fov = 2 * atan(width / (2 * f)) * 180.0 / CV_PI;



    Json::Value root;
    root["CAM_FRONT"]["extrinsics"]["yaw"] = front_extrin[0];
    root["CAM_FRONT"]["extrinsics"]["roll"] = front_extrin[1];
    root["CAM_FRONT"]["extrinsics"]["pitch"] = front_extrin[2];
    root["CAM_FRONT"]["extrinsics"]["x"] = front_extrin[3];
    root["CAM_FRONT"]["extrinsics"]["y"] = front_extrin[4];
    root["CAM_FRONT"]["extrinsics"]["z"] = front_extrin[5];
    root["CAM_FRONT"]["intrinsics"]["width"] = width;
    root["CAM_FRONT"]["intrinsics"]["height"] = height;
    root["CAM_FRONT"]["intrinsics"]["fov"] = fov;

    root["CAM_REAR"]["extrinsics"]["yaw"] = back_extrin[0];
    root["CAM_REAR"]["extrinsics"]["roll"] = back_extrin[1];
    root["CAM_REAR"]["extrinsics"]["pitch"] = back_extrin[2];
    root["CAM_REAR"]["extrinsics"]["x"] = back_extrin[3];
    root["CAM_REAR"]["extrinsics"]["y"] = back_extrin[4];
    root["CAM_REAR"]["extrinsics"]["z"] = back_extrin[5];
    root["CAM_REAR"]["intrinsics"]["width"] = width;
    root["CAM_REAR"]["intrinsics"]["height"] = height;
    root["CAM_REAR"]["intrinsics"]["fov"] = fov;

    root["CAM_LEFT"]["extrinsics"]["yaw"] = left_extrin[0];
    root["CAM_LEFT"]["extrinsics"]["roll"] = left_extrin[1];
    root["CAM_LEFT"]["extrinsics"]["pitch"] = left_extrin[2];
    root["CAM_LEFT"]["extrinsics"]["x"] = left_extrin[3];
    root["CAM_LEFT"]["extrinsics"]["y"] = left_extrin[4];
    root["CAM_LEFT"]["extrinsics"]["z"] = left_extrin[5];
    root["CAM_LEFT"]["intrinsics"]["width"] = width;
    root["CAM_LEFT"]["intrinsics"]["height"] = height;
    root["CAM_LEFT"]["intrinsics"]["fov"] = fov;

    root["CAM_RIGHT"]["extrinsics"]["yaw"] = right_extrin[0];
    root["CAM_RIGHT"]["extrinsics"]["roll"] = right_extrin[1];
    root["CAM_RIGHT"]["extrinsics"]["pitch"] = right_extrin[2];
    root["CAM_RIGHT"]["extrinsics"]["x"] = right_extrin[3];
    root["CAM_RIGHT"]["extrinsics"]["y"] = right_extrin[4];
    root["CAM_RIGHT"]["extrinsics"]["z"] = right_extrin[5];
    root["CAM_RIGHT"]["intrinsics"]["width"] = width;
    root["CAM_RIGHT"]["intrinsics"]["height"] = height;
    root["CAM_RIGHT"]["intrinsics"]["fov"] = fov;



    std::ofstream file_id;
    file_id.open(dump_json_path);
    // 写入文件
    std::ofstream outfile(dump_json_path);
    outfile << root.toStyledString();
    outfile.close();

  }
}


int main() { 

  std::string camera_info_json_path = "./b07_camera_infor.json";
  std::string root_dir = "./valid/";
  std::vector<std::string> data_path;
  ListFilesInDirectory(root_dir, data_path);
  undis_map();
  for (size_t i = 0; i < data_path.size(); ++i) {
    std::string camera_info_json_dump_path =
        data_path[i] + "/camera_config_right_hand.json";
    dump_camera_info_json(camera_info_json_path, camera_info_json_dump_path);

    e2e_traj(data_path[i]);
   
    cout << i << endl;
  }
}