import sys
import os

# 添加根目录到模块搜索路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.mujoco_viewer import BaseViewer
import mujoco,time,threading
import numpy as np
import pinocchio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools
import transformations as tf
from scipy.spatial.transform import Rotation
import torch
import cv2
import glfw

### 机械臂规划相关
import ikpy.chain
from pyroboplan.core.utils import (
    get_random_collision_free_state,
    extract_cartesian_poses,
)
from pyroboplan.models.piper import (
    load_models,
    add_self_collisions,
    add_object_collisions,
)
from pyroboplan.planning.rrt import RRTPlanner, RRTPlannerOptions
from pyroboplan.trajectory.trajectory_optimization import (
    CubicTrajectoryOptimization,
    CubicTrajectoryOptimizationOptions,
)
import math
from math import atan2, sqrt, acos, pi, sin, cos, atan
PI = math.pi
class BaseEnv(BaseViewer):
    def __init__(self, cfg):
        """
        path :       XML 模型路径(MJCF)
        distance :   相机距离
        azimuth :    水平旋转角度
        elevation :  俯视角度
        """
        super().__init__(cfg.path, 3, azimuth=180, elevation=-30)
        self.path = cfg.path

        if cfg["is_have_arm"] == True:
            self.my_chain = ikpy.chain.Chain.from_urdf_file("/home/ubuntu/piper_rrt_cubic/assets/piper_n.urdf")
            # 创建机械臂规划模型
            self.model_roboplan, self.collision_model, visual_model = load_models(use_sphere_collisions=True)
            if self.collision_model is None:
                raise ValueError("collision_model is None — collision model must be loaded before proceeding.")
            
            add_self_collisions(self.model_roboplan, self.collision_model)
            add_object_collisions(self.model_roboplan, self.collision_model, visual_model, inflation_radius=0.1)
            self.target_frame = "link6"
            np.set_printoptions(precision=3)
            self.distance_padding = 0.001
            self.index = 0
            self.l = 0.091 + 0.053   # joint4 → joint6 → 末端执行器
            # DH参数定义（单位：米/弧度）
            self.alpha = [0, -pi / 2, 0, pi / 2, -pi / 2, pi / 2]  # 扭转角
            self.a = [0, 0, 0.28503, -0.02198, 0, 0]  # 连杆长度
            self.d = [0.123, 0, 0, 0.25075, 0, 0.091]  # 连杆偏移
            self.theta_offset = [0, -172.2135102 * pi / 180, -102.7827493 * pi / 180, 0, 0, 0]  # 初始角度偏移


        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()

        self.episode_len = cfg["episode_len"]
        self.is_save_record_data = cfg["is_save_record_data"]
        self.camera_names = cfg["camera_names"]
        self.data_dict = {
            'observations': {
                'images': {cam_name: [] for cam_name in self.camera_names},
                'qpos': [],
                'actions': []
            }
        }
        self.step_number = 0
        self.goal_reached_count = 0

        # 打印当前场景 joint 和 body 信息
        self.print_all_joint_info()
        self.print_all_body_info()


    def close(self):
        super().close()
        if hasattr(self, "window") and self.window:
            glfw.destroy_window(self.window)
            glfw.terminate()
            self.window = None


    def print_all_joint_info(self):
        """
        打印模型中所有关节的名称、ID、范围限制和当前qpos值
        """
        print("\n=== 关节信息 ===")
        print(f"{'Joint Name':<20} {'Type':<15} {'Qpos Addr':<10} {'Range':<25} {'Current Value':<15}")
        print("-" * 90)
        
        for joint_id in range(self.model.njnt):
            # 获取关节名称
            name_addr = self.model.name_jntadr[joint_id]
            joint_name = self.model.names[name_addr:].split(b'\x00')[0].decode('utf-8')
            
            # 获取关节类型
            joint_type = self.model.jnt_type[joint_id]
            type_names = {
                0: "自由关节(6DOF)",
                1: "球关节(3DOF)", 
                2: "滑动关节",
                3: "铰链关节"
            }
            type_str = type_names.get(joint_type, "未知类型")
            
            # 获取qpos地址和范围
            qpos_addr = self.model.jnt_qposadr[joint_id]
            if self.model.jnt_limited[joint_id]:
                jnt_range = f"[{self.model.jnt_range[joint_id,0]:.2f}, {self.model.jnt_range[joint_id,1]:.2f}]"
            else:
                jnt_range = "无限制"
            
            # 获取当前值
            if joint_type == 0:  # 自由关节
                current_val = self.data.qpos[qpos_addr:qpos_addr+7]
            elif joint_type == 1:  # 球关节
                current_val = self.data.qpos[qpos_addr:qpos_addr+4]
            else:  # 滑动/铰链关节
                current_val = self.data.qpos[qpos_addr]
            
            print(f"{joint_name:<20} {type_str:<15} {qpos_addr:<10} {jnt_range:<25} {str(current_val):<15}")


    def print_all_body_info(self):
        """
        打印模型中所有body的名称、ID、位置和四元数姿态信息
        """
        print("\n=== Body 信息 ===")
        print(f"{'Body Name':<25} {'Body ID':<8} {'Position':<30} {'Quaternion':<35}")
        print("-" * 100)
        
        for body_id in range(self.model.nbody):
            # 获取body名称
            name_addr = self.model.name_bodyadr[body_id]
            body_name = self.model.names[name_addr:].split(b'\x00')[0].decode('utf-8')
            
            # 获取位置和四元数
            pos = self.data.body(body_id).xpos
            quat = self.data.body(body_id).xquat

            # cam_pos = self.data.cam_xpos
            
            print(f"{body_name:<25} {body_id:<8} {str(pos):<30} {str(quat):<35}")

    
    def _get_sensor_data(self, sensor_name: str):
        """
        通过 sensor 名称获取传感器数据
        Args:
            sensor_name     : sensor 名字
        Return:
            sensor_values   : sensor 值
        """
        sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
        if sensor_id == -1:
            raise ValueError(f"Sensor '{sensor_name}' not found in model!")
        start_idx = self.model.sensor_adr[sensor_id]
        dim = self.model.sensor_dim[sensor_id]
        sensor_values = self.data.sensordata[start_idx : start_idx + dim]
        return sensor_values


    # def _get_body_pose(self, body_name: str) -> np.ndarray:
    #     """
    #     通过 body 名称获取其位姿信息, 返回一个7维向量
    #         :param body_name: body名称字符串
    #         :return: 7维numpy数组, 格式为 [x, y, z, w, x, y, z]
    #         :raises ValueError: 如果找不到指定名称的body
    #     """
    #     body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    #     if body_id == -1:
    #         raise ValueError(f"未找到名为 '{body_name}' 的body")
    #
    #     # 提取位置和四元数并合并为一个7维向量
    #     position = np.array(self.data.body(body_id).xpos)  # [x, y, z]
    #     quaternion = np.array(self.data.body(body_id).xquat)  # [w, x, y, z]
    #
    #     return position, quaternion


    def _get_image_from_camera(self, w, h, camera_name):
        """
        通过 camera 名称获取其相机数据

        参数解释:
            w               :                   期望图像宽
            h               :                   期望图像高
            camera_name     :                   相机名称

        返回值:
            cv_image        :                   np.ndarray, OpenCV 格式的图像, shape 为 (h, w, 3)
        """

        # 指定渲染图像的大小和位置
        viewport = mujoco.MjrRect(0, 0, w, h)

        # 通过相机名称获取其在 MuJoCo 模型中的唯一 ID
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        self.camera.id = cam_id

        # 根据当前模型状态和相机设置，更新渲染场景
        mujoco.mjv_updateScene(
            self.model, self.data, mujoco.MjvOption(), 
            None, self.camera, mujoco.mjtCatBit.mjCAT_ALL, self.scene
        )

        # 将更新后的场景渲染到指定的视口
        mujoco.mjr_render(viewport, self.scene, self.context)

        # 从渲染结果中读取像素数据
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(rgb, None, viewport, self.context)

        # 将 MuJoCo 返回的 RGB 图像转换为 OpenCV 的 BGR 格式
        cv_image = cv2.cvtColor(np.flipud(rgb), cv2.COLOR_RGB2BGR)
        return cv_image

    def _get_observation(self):
        """
        获取当前环境中多个相机视角的观测图像
        Return:
            obs     :                            dict, 包含来自多个相机的图像观测, 键为相机名称, 值为对应的 np.ndarray 图像 (OpenCV 格式, shape 为 (480, 640, 3))
        """
        wrist_cam_image = self._get_image_from_camera(640, 480, "wrist")
        top_cam_image = self._get_image_from_camera(640, 480, "3rd")
        obs = {
            "wrist": wrist_cam_image,
            "3rd": top_cam_image
        }
        return obs


    def _set_original_state(self, qpos: np.ndarray, qvel: np.ndarray):
        """
        设置模拟器中的状态（关节位置和速度），并同步 forward。
        """
        assert qpos.shape == (self.model.nq,), f"Expected qpos shape ({self.model.nq},), got {qpos.shape}"
        assert qvel.shape == (self.model.nv,), f"Expected qvel shape ({self.model.nv},), got {qvel.shape}"

        self.data.qpos[:] = np.copy(qpos)
        self.data.qvel[:] = np.copy(qvel)

        if self.model.na == 0 and hasattr(self.data, "act"):
            # 如果没有 actuator，清空 act，否则可能会引发非法内存读写
            self.data.act[:] = 0

        self.cur_episode_done = False

        mujoco.mj_forward(self.model, self.data)

    #
    # def calc_arm_rrt_cubic_traj(
    #     self,
    #     cur_joints_state,
    #     target_joints_state
    # ):
    #     """
    #     计算机械臂在给定目标关节角度下的运动轨迹
    #
    #     参数:
    #         cur_joints_state:         当前机械臂的 6 关节状态
    #         target_joints_state:      机械臂目标 6 关节
    #     返回:
    #         path: 轨迹
    #     """
    #     q_start = cur_joints_state
    #     q_goal = target_joints_state
    #
    #     print(f"q_start : {q_start}")
    #
    #     # Search for a path
    #     options = RRTPlannerOptions(
    #         max_step_size=0.05,
    #         max_connection_dist=5.0,
    #         rrt_connect=False,
    #         bidirectional_rrt=True,
    #         rrt_star=True,
    #         max_rewire_dist=5.0,
    #         max_planning_time=20.0,
    #         fast_return=True,
    #         goal_biasing_probability=0.15,
    #         collision_distance_padding=0.01,
    #     )
    #     print(f"Planning a path...")
    #     planner = RRTPlanner(self.model_roboplan, self.collision_model, options=options)
    #     q_path = planner.plan(q_start, q_goal)
    #     if len(q_path) > 0:
    #         print(f"Got a path with {len(q_path)} waypoints")
    #     else:
    #         print("Failed to plan.")
    #
    #     # Perform trajectory optimization.
    #     dt = 0.025
    #     options = CubicTrajectoryOptimizationOptions(
    #         num_waypoints=len(q_path),
    #         samples_per_segment=7,
    #         min_segment_time=0.5,
    #         max_segment_time=10.0,
    #         min_vel=-1.5,
    #         max_vel=1.5,
    #         min_accel=-0.75,
    #         max_accel=0.75,
    #         min_jerk=-1.0,
    #         max_jerk=1.0,
    #         max_planning_time=30.0,
    #         check_collisions=True,
    #         min_collision_dist=self.distance_padding,
    #         collision_influence_dist=0.05,
    #         collision_avoidance_cost_weight=0.0,
    #         collision_link_list=[
    #             "ground_plane",
    #             "link6",
    #         ],
    #     )
    #     print("Optimizing the path...")
    #     optimizer = CubicTrajectoryOptimization(self.model_roboplan, self.collision_model, options)
    #     traj = optimizer.plan([q_path[0], q_path[-1]], init_path=q_path)
    #
    #     if traj is not None:
    #         print("Trajectory optimization successful")
    #         traj_gen = traj.generate(dt)

    #     if traj is not None:
    #         # my_q_vec = traj_gen[1]
    #         # print(f"path has {my_q_vec.shape[1]} points")
    #         # self.tforms = extract_cartesian_poses(self.model_roboplan, "link6", my_q_vec.T)
    #         #
    #         # positions = []
    #         # print(self.tforms[0].translation)
    #         # arm_base_name = "base_link"
    #         # arm_base_pos, arm_base_quat = self._get_body_pose(arm_base_name)
    #         #
    #         # R_world_to_base = Rotation.from_quat(arm_base_quat[[1, 2, 3, 0]]).inv()  # 注意 wxyz -> xyzw
    #         #
    #         #
    #         # my_tforms = [R_world_to_base.apply(t.translation - arm_base_pos) for t in self.tforms]
    #         #
    #         # # print(self.tforms[0].rotation)
    #         # self.handle.user_scn.ngeom = 0
    #         # i = 0
    #         # print(f"")
    #         #
    #         # for i,tform in enumerate(my_tforms):
    #         #     if i % 2 == 0:
    #         #         continue
    #         #     position = tform
    #         #
    #         #     mujoco.mjv_initGeom(
    #         #         self.handle.user_scn.geoms[i],
    #         #         type=mujoco.mjtGeom.mjGEOM_SPHERE,
    #         #         size=[0.005, 0, 0],
    #         #         pos=np.array([tform[0], tform[1], tform[2]]),
    #         #         mat=np.eye(3).flatten(),
    #         #         rgba=np.array([1, 0, 0, 1])
    #         #     )
    #         #     i += 1
    #         # self.handle.user_scn.ngeom = i
    #         # print(f"Added {i} spheres to the scene.")
    #         return traj_gen[1]
    #     else:
    #         return None
    #



    # def dh_transform(self, alpha, a, d, theta):
    #     """
    #     计算Denavit-Hartenberg标准参数的4x4齐次变换矩阵
    #
    #     参数:
    #     alpha (float): 连杆扭转角（绕x_(i-1)轴的旋转角，弧度）
    #     a (float): 连杆长度（沿x_(i-1)轴的平移量，米）
    #     d (float): 连杆偏移量（沿z_i轴的平移量，米）
    #     theta (float): 关节角（绕z_i轴的旋转角，弧度）
    #
    #     返回:
    #     np.ndarray: 4x4齐次变换矩阵
    #     """
    #     ct = np.cos(theta)
    #     st = np.sin(theta)
    #     ca = np.cos(alpha)
    #     sa = np.sin(alpha)
    #
    #     # 构建标准DH变换矩阵
    #     transform = np.array([
    #         [ct, -st, 0, a],
    #         [ca * st, ca * ct, -sa, -sa * d],
    #         [sa * st, sa * ct, ca, ca * d],
    #         [0, 0, 0, 1]
    #     ])
    #     return transform
    #
    # def forward_kinematics_sub(self, joints, end):
    #     # 0_T_end
    #     T_total = np.eye(4)
    #     for i in range(end):
    #         print("i alpha a d theta", self.alpha[i], self.a[i], self.d[i], self.theta_offset[i])
    #         T = self.dh_transform(self.alpha[i], self.a[i], self.d[i], self.theta_offset[i] + joints[i])
    #         T_total = T_total @ T
    #
    #     return T_total
    #
    # def rotation_matrix_to_euler(self,R):
    #     """从旋转矩阵计算欧拉角(ZYZ顺序)"""
    #     sin_theta = sqrt(R[2, 0] ** 2 + R[2, 1] ** 2)
    #     singular = sin_theta < 1e-6
    #
    #     if not singular:
    #         theta = atan2(sin_theta, R[2, 2])
    #         phi = atan2(R[1, 2] / sin(theta), R[0, 2] / sin(theta))
    #         psi = atan2(R[2, 1] / sin(theta), -R[2, 0] / sin(theta))
    #
    #     else:
    #         theta = 0
    #         phi = 0
    #         psi = atan2(-R[0, 1], R[0, 0])
    #
    #     return np.array([phi, theta, psi])
    #
    # def get_joint_tf(self, joint_idx, angle):
    #     """获取指定关节的变换矩阵"""
    #     transform = self.dh_transform(self.alpha[joint_idx], self.a[joint_idx], self.d[joint_idx], self.theta_offset[joint_idx] + angle)
    #     return transform
    #
    # def inverse_kinematics(self, T_target):
    #     """Pieper解法逆运动学求解"""
    #     # 计算 joint4 位置
    #     joint4_p = T_target @ np.array([0, 0, -self.l, 1], dtype=float)
    #     px, py, pz = joint4_p[0], joint4_p[1], joint4_p[2]
    #
    #     # 计算 link1 2 3 角度
    #     theta1 = atan2(py, px)
    #     T01 = self.dh_transform(self.alpha[0], self.a[0], self.d[0], theta1)
    #
    #     # Convert P05 to frame 1
    #     P15 = np.linalg.inv(T01) @ np.array([px, py, pz, 1])
    #     x1, z1 = P15[0], P15[2]
    #     a1, a2 = self.a[2], self.a[3]
    #     d1, d2 = self.d[2], self.d[3]
    #     l1 = sqrt(a1 ** 2 + d1 ** 2)
    #     l2 = sqrt(a2 ** 2 + d2 ** 2)
    #     l3 = sqrt(x1 ** 2 + z1 ** 2)
    #
    #     cos_phi3 = (l1 ** 2 + l2 ** 2 - l3 ** 2) / (2.0 * l1 * l2)
    #     if abs(cos_phi3) > 1:
    #         print("no ik solution, fail theta 3")
    #         return
    #     phi3 = acos(cos_phi3)
    #     print("phi3 is ", phi3 / PI * 180)
    #
    #     phi3 = atan2(sqrt(1 - cos_phi3 ** 2), cos_phi3)
    #     # print("phi3 is ", phi3 / PI * 180)
    #     gamma = atan2(abs(self.d[3]), abs(self.a[3]))
    #     # print("gamma is ", gamma / PI * 180)
    #     theta3 = -(gamma + phi3) - self.theta_offset[2]
    #
    #     cos_phi2 = (l1 ** 2 + l3 ** 2 - l2 ** 2) / (2 * l1 * l3)
    #     if abs(cos_phi2) > 1:
    #         print("no ik solution, fail theta 2")
    #     phi2 = acos(cos_phi2)
    #
    #     beta = atan(x1 / z1)
    #     if z1 > 0:
    #         theta2 = - (PI / 2 + phi2 - beta) - self.theta_offset[1]
    #     else:
    #         beta = atan(x1 / abs(z1))
    #         print("phi2", phi2 / 3.14 * 180)
    #         print("beta", beta / 3.14 * 180)
    #         theta2 = - (phi2 - (PI / 2 - beta)) - self.theta_offset[1]
    #
    #     q_sol = [theta1, theta2, theta3, 0, 0, 0]
    #
    #     # 计算link4 5 6 角度
    #     T03 = self.forward_kinematics_sub(q_sol, 3)
    #     R03 = T03[0:3, 0:3]
    #     R34d = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    #     T34d = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    #     R03d = R03 @ R34d
    #     R06 = T_target[0:3, 0:3]
    #     R36 = R03d.T @ R06
    #
    #     print("needed R36", R36)
    #     rx, ry, rz = self.rotation_matrix_to_euler(R36)
    #     q_sol = [theta1, theta2, theta3, rx, ry, rz]
    #     T34 = self.get_joint_tf(3, rx)
    #     T45 = self.get_joint_tf(4, ry)
    #     T56 = self.get_joint_tf(5, rz)
    #     T36 = T34d.T @ (T34 @ T45) @ T56
    #     print("REAL T36", T36)
    #
    #     return q_sol
    #
    # def rotation_matrix_to_quaternion(self,R):
    #     """将3x3旋转矩阵转换为四元数(w, x, y, z顺序)"""
    #     q = np.zeros(4)
    #     trace = np.trace(R)
    #
    #     if trace > 0:
    #         S = np.sqrt(trace + 1.0) * 2
    #         q[0] = 0.25 * S
    #         q[1] = (R[2, 1] - R[1, 2]) / S
    #         q[2] = (R[0, 2] - R[2, 0]) / S
    #         q[3] = (R[1, 0] - R[0, 1]) / S
    #     elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
    #         S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
    #         q[0] = (R[2, 1] - R[1, 2]) / S
    #         q[1] = 0.25 * S
    #         q[2] = (R[0, 1] + R[1, 0]) / S
    #         q[3] = (R[0, 2] + R[2, 0]) / S
    #     elif R[1, 1] > R[2, 2]:
    #         S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
    #         q[0] = (R[0, 2] - R[2, 0]) / S
    #         q[1] = (R[0, 1] + R[1, 0]) / S
    #         q[2] = 0.25 * S
    #         q[3] = (R[1, 2] + R[2, 1]) / S
    #     else:
    #         S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    #         q[0] = (R[1, 0] - R[0, 1]) / S
    #         q[1] = (R[0, 2] + R[2, 0]) / S
    #         q[2] = (R[1, 2] + R[2, 1]) / S
    #         q[3] = 0.25 * S
    #
    #     return q / np.linalg.norm(q)  # 归一化

    # def label_goal_pose(self, position, quat_wxyz):
    #     """
    #     设置目标位姿（位置 + 姿态）
    #
    #     Args:
    #         position: 目标的位置，(x, y, z)，类型为 numpy.ndarray 或 list。
    #         quat_wxyz: 目标的姿态，四元数 (w, x, y, z)，类型为 numpy.ndarray 或 list。
    #     """
    #     ## ====== 设置 target 的位姿 ======
    #     goal_body_name = "target"
    #     goal_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, goal_body_name)
    #
    #     if goal_body_id == -1:
    #         raise ValueError(f"Body named '{goal_body_name}' not found in the model.")
    #
    #     # 获取 joint ID 和 qpos 起始索引
    #     goal_joint_id = self.model.body_jntadr[goal_body_id]
    #     goal_qposadr = self.model.jnt_qposadr[goal_joint_id]
    #
    #     # 设置位姿
    #     if goal_qposadr + 7 <= self.model.nq:
    #         self.data.qpos[goal_qposadr     : goal_qposadr + 3] = position
    #         self.data.qpos[goal_qposadr + 3 : goal_qposadr + 7] = quat_wxyz
    #     else:
    #         print("[警告] target 的 qpos 索引越界或 joint 设置有误")
    # def run_before(self):
    #     """
    #     求解机械臂
    #
    #     Args:
    #         position: 目标的位置，(x, y, z)，类型为 numpy.ndarray 或 list。
    #         quat_wxyz: 目标的姿态，四元数 (w, x, y, z)，类型为 numpy.ndarray 或 list。
    #     """
    #     # step 1 : 设置机械臂、被抓物体的初始 pose
    #     self.init_state = self.data.qpos.copy()
    #     self.q_vec = None
    #
    #     q_start = np.zeros(6)
    #     if q_start is None:
    #         raise RuntimeError(" q_start is invalid... ")
    #
    #     # step 2 : 获取机械臂基座、被抓物体在世界坐标系下的位姿
    #     item_name = "apple"
    #     item_pos, item_quat = self._get_body_pose(item_name)
    #
    #     arm_base_name = "base_link"
    #     arm_base_pos, arm_base_quat = self._get_body_pose(arm_base_name)
    #     arm_base_quat_xyzw = np.roll(arm_base_quat, -1)
    #
    #     # step 3 : 将世界系下的点转换到base_link下的点
    #     # 世界坐标系下的目标item的变换矩阵，也是期望末端执行器达到的位置和姿态
    #     T_world_obj = np.eye(4)
    #
    #     # 计算目标item在world系下的变换矩阵T
    #     # rot_world_obj = np.eye(3)
    #     # T_world_obj[:3, :3] = Rotation.from_matrix(rot_world_obj).as_matrix()
    #     T_world_obj[:3, 3] = item_pos
    #
    #     # 计算world坐标系下baselink的变换矩阵
    #     T_world_base = np.eye(4)
    #     T_world_base[:3, :3] = Rotation.from_quat(arm_base_quat_xyzw).as_matrix()
    #     T_world_base[:3, 3] = arm_base_pos
    #
    #     # 求 T_world_base 的逆
    #     T_base_world = np.linalg.inv(T_world_base)
    #     # 将目标点的pose转换到base_link下
    #     T_base_obj = T_base_world @ T_world_obj
    #     # 自定义抓取的目标位姿
    #     T_base_obj[:3, :3] = np.array([[cos(135 / 180 * PI), 0, sin(135 / 180 * PI)],
    #                                         [0, 1, 0],
    #                                         [-sin(cos(135 / 180 * PI)), 0, cos(cos(135 / 180 * PI))]]
    #                                   ,dtype=float)
    #
    #     # step 4 : 调用IK求解关节角
    #     target_joints_state = self.inverse_kinematics(T_base_obj)
    #     target_joints_state_np = np.array(target_joints_state)
    #     print("q_goal:", target_joints_state)
    #
    #
    #     # 调试：用fk求解末端执行器位姿
    #     T_6_ee = np.eye(4)
    #     T_6_ee[:3, 3] = np.array([0, 0, 0.085],dtype=float)
    #
    #     T_base_6 = self.forward_kinematics_sub(target_joints_state, 6)
    #     T_total = T_world_base @ T_base_6 @ T_6_ee
    #     self.target_position = T_total[:3, 3]
    #     self.target_quat_wxyz = self.rotation_matrix_to_quaternion(T_total[:3, :3])
    #
    #     # 调试：抓取坐标轴可视化
    #     # self.label_goal_pose(self.target_position,self.target_quat_wxyz)
    #
    #
    #     # 调试：抓取点可视化
    #     # # 使用 self.handle.user_scn.ngeom 获取当前已添加的几何体数量
    #     # geom_id = self.handle.user_scn.ngeom
    #
    #     # # 添加一个小球可视化位置
    #     # mujoco.mjv_initGeom(
    #     #     self.handle.user_scn.geoms[geom_id],
    #     #     type=mujoco.mjtGeom.mjGEOM_SPHERE,
    #     #     size=[0.01, 0, 0],  # 球半径为 0.01
    #     #     pos=target_position,
    #     #     mat=np.eye(3).flatten(),  # 不旋转
    #     #     rgba=np.array([0, 1, 0, 1])  # 绿色
    #     # )
    #     # geom_id += 1
    #     # self.handle.user_scn.ngeom = geom_id
    #
    #
    #
    #
    #     #
    #     # # step 5 : 调用RRT规划轨迹位姿
    #     # path = self.calc_arm_rrt_cubic_traj(q_start, target_joints_state_np)
    #     #
    #     # self.q_vec = path
    #     #
    #     # if self.q_vec is None:
    #     #     raise RuntimeError(" planning path failed... ")



        

    def runFunc(self):
        pass