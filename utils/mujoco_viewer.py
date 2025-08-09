import mujoco
import mujoco.viewer
import glfw
import time
import cv2
import matplotlib.pyplot as plt

# 公式相关
import numpy as np
from scipy.spatial.transform import Rotation
import math
from math import atan2, sqrt, acos, pi, sin, cos, atan, asin
PI = math.pi
import random
import queue
import threading

# 机械臂规划相关
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
import logging
logger = logging.getLogger(__name__)

from scipy.signal import savgol_filter


class BaseViewer:
    """
    双臂机械臂仿真环境类
    支持双臂协调控制，物体抓取和放置任务
    """
    
    def __init__(self, cfg, distance=3, azimuth=0, elevation=-30):
        print(f"Model path: {cfg.path}")
        self.model = mujoco.MjModel.from_xml_path(cfg.path)
        self.data = mujoco.MjData(self.model)
        
        # 相机参数
        self.distance = distance
        self.azimuth = azimuth
        self.elevation = elevation
        
        # 初始化MuJoCo viewer
        self.handle = mujoco.viewer.launch_passive(self.model, self.data)
        self.handle.cam.distance = distance
        self.handle.cam.azimuth = azimuth
        self.handle.cam.elevation = elevation
        self.opt = mujoco.MjvOption()
        
        # 初始化离屏渲染
        self._init_offscreen_rendering()
        
        # 初始化双臂配置
        self._init_dual_arm_config(cfg)
        
        # 保存初始状态
        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()

        self.left_place_q = np.array([-0.262, 1.24, -0.728, 0.0, 0.0, 0.0])
        self.right_place_q = np.array([0.262, 1.24, -0.728, 0.0, 0.0, 0.0])
        
        # 任务相关参数
        self.episode_len = cfg.get("episode_len", 1000)
        self.is_save_record_data = cfg.get("is_save_record_data", False)
        self.camera_names = cfg.get("camera_names", ["wrist_cam"])
        
        # 数据记录
        self.data_dict = {
            'observations': {
                'images': {cam_name: [] for cam_name in self.camera_names},
                'qpos': [],
                'actions': []
            }
        }
        
        self.step_number = 0
        self.goal_reached_count = 0
        self.cur_episode_done = False
        
        # 双臂motor映射
        self._init_motor_mapping()
        
        # 状态队列
        self.sim_state_queue = queue.Queue(maxsize=1)
        
        # 物体和目标配置
        self.apple_name = "apple"
        self.banana_name = "banana"
        # 这里可能是 "board" 或 "desk"，统一用函数取
        self.board_candidates = ["board", "desk", "tray", "basket"]
        
        # 抓取位姿存储 - 这里您需要提供实际的抓取位姿数据（或在模型里放置 xxx_grasp_site）
        self.grasp_poses = {
            "apple": {"position": None, "quaternion": None},   # 如果你要手填，给数组即可
            "banana": {"position": None, "quaternion": None}
        }
        
        # 统计信息
        self.count = 0
        self.count_ik = 0
        self.phi = 0
        

        self._cache_mobile_base_handles()

        # 打印场景信息
        self.print_all_joint_info()
        self.print_all_body_info()

    # ------------ 传感器读取 ------------
    def _get_arm_joint_positions(self, arm: str, n_joints: int) -> np.ndarray:
        vals = []
        for i in range(1, n_joints + 1):
            name = f"{arm}_joint{i}_pos"
            v = self._get_sensor_data(name)  # 期望 dim=1
            vals.append(float(v[0]) if v.size == 1 else float(v.squeeze()))
        return np.asarray(vals, dtype=float)

    def _get_sensor_data(self, sensor_name: str):
        sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
        if sensor_id == -1:
            raise ValueError(f"Sensor '{sensor_name}' not found in model!")
        start_idx = self.model.sensor_adr[sensor_id]
        dim = self.model.sensor_dim[sensor_id]
        sensor_values = self.data.sensordata[start_idx : start_idx + dim]
        return sensor_values
        
    # ------------ 渲染 ------------
    def _init_offscreen_rendering(self):
        glfw.init()
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        self.window = glfw.create_window(640, 480, "offscreen", None, None)
        glfw.make_context_current(self.window)
        
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.scene = mujoco.MjvScene(self.model, maxgeom=1000)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.context)
        
        self.camera_pos = np.array([0, 0, 0])
        self.camera_mat = np.eye(3, dtype=np.float64)
        
    # ------------ 机械臂/模型配置 ------------
    def _init_dual_arm_config(self, cfg):
        if cfg.get("is_have_arm", False):
            self.model_roboplan, self.collision_model, visual_model = load_models(use_sphere_collisions=True)
            if self.collision_model is None:
                raise ValueError("collision_model is None — collision model must be loaded before proceeding.")
            
            add_self_collisions(self.model_roboplan, self.collision_model)
            add_object_collisions(self.model_roboplan, self.collision_model, visual_model, inflation_radius=0.1)
            
            np.set_printoptions(precision=3)
            self.distance_padding = 0.001
            self.l = 0.091 + 0.053  # joint4 → joint6 → 末端执行器
            
            # DH参数定义（单位：米/弧度）
            self.alpha = [0, -pi/2, 0, pi/2, -pi/2, pi/2]  # 扭转角
            self.a = [0, 0, 0.28503, -0.02198, 0, 0]  # 连杆长度
            self.d = [0.123, 0, 0, 0.25075, 0, 0.091]  # 连杆偏移
            self.theta_offset = [0, -172.2135102*pi/180, -102.7827493*pi/180, 0, 0, 0]
    
    def _init_motor_mapping(self):
        self.left_motors = {
            "left_joint_1.pos": 0.0,
            "left_joint_2.pos": 0.0,
            "left_joint_3.pos": 0.0,
            "left_joint_4.pos": 0.0,
            "left_joint_5.pos": 0.0,
            "left_joint_6.pos": 0.0,
            "left_gripper.pos": 0.0,
        }
        self.right_motors = {
            "right_joint_1.pos": 0.0,
            "right_joint_2.pos": 0.0,
            "right_joint_3.pos": 0.0,
            "right_joint_4.pos": 0.0,
            "right_joint_5.pos": 0.0,
            "right_joint_6.pos": 0.0,
            "right_gripper.pos": 0.0,
        }
    
    # ========================= 运动学相关 =========================
    def dh_transform(self, alpha, a, d, theta):
        """
        计算Denavit-Hartenberg标准参数的4x4齐次变换矩阵

        参数:
        alpha (float): 连杆扭转角（绕x_(i-1)轴的旋转角，弧度）
        a (float): 连杆长度（沿x_(i-1)轴的平移量，米）
        d (float): 连杆偏移量（沿z_i轴的平移量，米）
        theta (float): 关节角（绕z_i轴的旋转角，弧度）

        返回:
        np.ndarray: 4x4齐次变换矩阵
        """
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)

        # 构建标准DH变换矩阵
        transform = np.array([
            [ct, -st, 0, a],
            [ca * st, ca * ct, -sa, -sa * d],
            [sa * st, sa * ct, ca, ca * d],
            [0, 0, 0, 1]
        ])
        return transform
    
    def forward_kinematics_sub(self, joints, end):
        T_total = np.eye(4)
        for i in range(end):
            T = self.dh_transform(self.alpha[i], self.a[i], self.d[i], self.theta_offset[i] + joints[i])
            T_total = T_total @ T
        return T_total
    
    def rotation_matrix_to_euler(self, R):
        """
            从旋转矩阵计算欧拉角(ZYZ顺序)
            phi： joint4    range="-1.832 1.832"
            theta: joint5  range="-1.22 1.22"
            psi: joint6    range="-3.14 3.14"
        """
        psi = 0
        phi = atan2(-R[0, 1], R[1, 1])
        theta = atan2(-R[2, 0], R[2, 2])
        if ((phi > -1.832 and phi < 1.832) and (theta > -1.225 and theta < 1.225)
            and (psi > -3.14 and psi < 3.14)):
            return np.array([phi, theta, psi])
        else:
            return None
    
    def rotation_matrix_to_quaternion(self, R):
        q = np.zeros(4); trace = np.trace(R)
        if trace > 0:
            S = np.sqrt(trace + 1.0) * 2
            q[0] = 0.25 * S
            q[1] = (R[2, 1] - R[1, 2]) / S
            q[2] = (R[0, 2] - R[2, 0]) / S
            q[3] = (R[1, 0] - R[0, 1]) / S
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            q[0] = (R[2, 1] - R[1, 2]) / S
            q[1] = 0.25 * S
            q[2] = (R[0, 1] + R[1, 0]) / S
            q[3] = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            q[0] = (R[0, 2] - R[2, 0]) / S
            q[1] = (R[0, 1] + R[1, 0]) / S
            q[2] = 0.25 * S
            q[3] = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            q[0] = (R[1, 0] - R[0, 1]) / S
            q[1] = (R[0, 2] + R[2, 0]) / S
            q[2] = (R[1, 2] + R[2, 1]) / S
            q[3] = 0.25 * S
        return q / np.linalg.norm(q)
    
    def get_joint_tf(self, joint_idx, angle):
        """获取指定关节的变换矩阵"""
        transform = self.dh_transform(self.alpha[joint_idx], self.a[joint_idx], self.d[joint_idx], self.theta_offset[joint_idx] + angle)
        return transform
    
    def inverse_kinematics(self, T_base_target):
        """Pieper解法逆运动学求解"""
        # 计算 joint4 位置
        p_target_joint4 = np.array([0, 0, -self.l, 1], dtype=float)
        p_base_joint4 = T_base_target @ p_target_joint4
        px, py, pz = p_base_joint4[0], p_base_joint4[1], p_base_joint4[2]

        if px > 0:
            # 计算 link1 2 3 角度
            theta1 = atan2(py, px)
        else:
            if py > 0:
                theta1 = -PI + atan2(py, px)
            else:
                theta1 = PI + atan2(py, px)
            # theta1 = PI - atan2(py, px)

        T01 = self.dh_transform(self.alpha[0], self.a[0], self.d[0], theta1)

        # Convert P05 to frame 1
        P15 = np.linalg.inv(T01) @ np.array([px, py, pz, 1])
        x1, z1 = P15[0], P15[2]
        a1, a2 = self.a[2], self.a[3]
        d1, d2 = self.d[2], self.d[3]
        l1 = sqrt(a1 ** 2 + d1 ** 2)
        l2 = sqrt(a2 ** 2 + d2 ** 2)
        l3 = sqrt(x1 ** 2 + z1 ** 2)

        cos_phi3 = (l1 ** 2 + l2 ** 2 - l3 ** 2) / (2.0 * l1 * l2)
        if abs(cos_phi3) > 1:
            print("no ik solution, fail theta 3")
            # self.count_ik = self.count_ik + 1
            return None
        phi3 = acos(cos_phi3)
        # print("phi3 is ", phi3 / PI * 180)

        phi3 = atan2(sqrt(1 - cos_phi3 ** 2), cos_phi3)
        # print("phi3 is ", phi3 / PI * 180)
        gamma = atan2(abs(self.d[3]), abs(self.a[3]))
        # print("gamma is ", gamma / PI * 180)
        theta3 = -(gamma + phi3) - self.theta_offset[2]

        if (theta3 > 2 or theta3 < -2.967):
            print("no ik solution, fail theta 3")
            return None

        cos_phi2 = (l1 ** 2 + l3 ** 2 - l2 ** 2) / (2 * l1 * l3)
        if abs(cos_phi2) > 1:
            print("no ik solution, fail theta 2")
        phi2 = acos(cos_phi2)

        beta = atan(x1 / z1)
        if z1 > 0:
            theta2 = - (PI / 2 + phi2 - beta) - self.theta_offset[1]
        else:
            beta = atan(x1 / abs(z1))
            # print("phi2", phi2 / 3.14 * 180)
            # print("beta", beta / 3.14 * 180)
            theta2 = - (phi2 - (PI / 2 - beta)) - self.theta_offset[1]

        if (theta2 > 3.14 or theta2 < -2):
            print("no ik solution, fail theta 2")
            return None

        q_sol = [theta1, theta2, theta3, 0, 0, 0]

        # 计算link4 5 6 角度
        T03 = self.forward_kinematics_sub(q_sol, 3)
        R03 = T03[0:3, 0:3]
        R34d = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        T34d = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        R03d = R03 @ R34d
        R06 = T_base_target[0:3, 0:3]
        R36 = R03d.T @ R06

        # print("needed R36", R36)
        if self.rotation_matrix_to_euler(R36) is None:
            print("no ik solution, fail joint 4,5,6")
            # self.count_ik = self.count_ik + 1
            return None
        # 判断ry是否为0
        rx, ry, rz = self.rotation_matrix_to_euler(R36)

        q_sol = [theta1, theta2, theta3, rx, ry, rz]
        T34 = self.get_joint_tf(3, rx)
        T45 = self.get_joint_tf(4, ry)
        T56 = self.get_joint_tf(5, rz)
        T36 = T34d.T @ (T34 @ T45) @ T56
        # print("REAL T36", T36)

        return q_sol
    
    # ========================= 轨迹规划 =========================
    def slerp(self, q1, q2, t):
        q1, q2 = q1 / np.linalg.norm(q1), q2 / np.linalg.norm(q2)
        cos_half_theta = q1.dot(q2)
        if cos_half_theta < 0:
            q2 = -q2; cos_half_theta = -cos_half_theta
        if np.abs(cos_half_theta) >= 1.0:
            q1 = (1 - t) * q1 + t * q2
            return q1 / np.linalg.norm(q1)
        half_theta = np.arccos(cos_half_theta)
        sin_half_theta = np.sqrt(1.0 - cos_half_theta * cos_half_theta)
        if np.abs(sin_half_theta) < 0.001:
            return (1.0 - t) * q1 + t * q2
        ratioA = np.sin((1 - t) * half_theta) / sin_half_theta
        ratioB = np.sin(t * half_theta) / sin_half_theta
        return ratioA * q1 + ratioB * q2
    
    def calc_arm_rrt_cubic_traj(self, cur_joints_state, target_joints_state):
        """计算机械臂RRT+Cubic轨迹，返回形状 (N,6)"""
        q_start = np.asarray(cur_joints_state, dtype=float).reshape(-1)
        q_goal  = np.asarray(target_joints_state, dtype=float).reshape(-1)

        options = RRTPlannerOptions(
            max_step_size=0.05,
            max_connection_dist=5.0,
            rrt_connect=False,
            bidirectional_rrt=True,
            rrt_star=True,
            max_rewire_dist=5.0,
            max_planning_time=20.0,
            fast_return=True,
            goal_biasing_probability=0.15,
            collision_distance_padding=0.01,
        )

        print(f"Planning a path...")
        planner = RRTPlanner(self.model_roboplan, self.collision_model, options=options)
        q_path = planner.plan(q_start, q_goal)
        if q_path is None:
            print("RRT planning failed!")
            return None
        if len(q_path) > 0:
            print(f"Got a path with {len(q_path)} waypoints")

        # 轨迹优化
        dt = 0.0075
        opt = CubicTrajectoryOptimizationOptions(
            num_waypoints=len(q_path),
            samples_per_segment=7,
            min_segment_time=0.5,
            max_segment_time=10.0,
            min_vel=-1.5, max_vel=1.5,
            min_accel=-0.75, max_accel=0.75,
            min_jerk=-1.0, max_jerk=1.0,
            max_planning_time=30.0,
            check_collisions=True,
            min_collision_dist=self.distance_padding,
            collision_influence_dist=0.05,
            collision_avoidance_cost_weight=0.0,
            collision_link_list=["ground_plane", "link6"],
        )

        print("Optimizing the path...")
        optimizer = CubicTrajectoryOptimization(self.model_roboplan, self.collision_model, opt)
        traj = optimizer.plan([q_path[0], q_path[-1]], init_path=q_path)
        if traj is None:
            return None

        print("Trajectory optimization successful")
        # 关键：generate 的 q 通常是 (dof, N)，你需要转置成 (N, dof)
        traj_gen = traj.generate(dt)
        q = np.asarray(traj_gen[1])  # 通常 index 1 是关节位置
        if q.ndim != 2:
            raise RuntimeError(f"Unexpected traj q shape: {q.shape}")
        # 统一成 (N,6)
        if q.shape[0] == 6 and q.shape[1] > 6:
            q = q.T
        if q.shape[1] != 6:
            raise RuntimeError(f"RRT traj unexpected dof: {q.shape}")
        return q
    
    def _add_gripper_state(self, cur_joint_pos, gripper_state_num, is_grasp_open):
        grasp_control = 0.035 if is_grasp_open else 0.0
        joint_state = np.append(cur_joint_pos, grasp_control)
        path_total = np.tile(joint_state, (gripper_state_num, 1))
        return path_total
    
    def _plan_arm_traj(self, item_name, target_pose, cur_ee_pose, cur_base_link_pose, 
                      cur_link1_pose, slerp_num, cur_joint_pos, is_grasp_open):

        ## TODO 重新计算 grasp pose 的姿态
        # ===================================================================
        self.rho_apple = sqrt((target_pose[0] - cur_base_link_pose[0]) ** 2 + (target_pose[1] - cur_base_link_pose[1]) ** 2 )

        # 针对不同目标点使用不同插值曲线
        if item_name == "apple":
            if self.rho_apple <= 0.7 * np.sqrt(0.28745):
                t_target = 0.65
                # # 相机坐标系下控制点
                # pos_medium_ee_obj = (pos_wri_cam_ee + pos_cam_obj) / 2 + np.array([0, 0.2, 0])
                # 世界坐标系下控制点
                pos_world_ctrl = (cur_ee_pose[0:3] + target_pose[0:3]) / 2 + np.array([0, 0, 0.2])
            else:
                # high vision
                # t_target = 0.8
                # low vision
                t_target = 1
                # # 相机坐标系下控制点
                # pos_medium_ee_obj = (pos_wri_cam_ee + pos_cam_obj) / 2 + np.array([0, 0.1, 0])
                # 世界坐标系下控制点
                pos_world_ctrl = (cur_ee_pose[0:3] + target_pose[0:3]) / 2 + np.array([0, 0, 0.15])
        elif item_name == "banana":
            if self.rho_apple <= 0.7 * np.sqrt(0.28745):
                t_target = 0.65
                # # 相机坐标系下控制点
                # pos_medium_ee_obj = (pos_wri_cam_ee + pos_cam_obj) / 2 + np.array([0, 0.2, 0])
                # 世界坐标系下控制点
                pos_world_ctrl = (cur_ee_pose[0:3] + target_pose[0:3]) / 2 + np.array([0, 0, 0.2])
            else:
                # high vision
                # t_target = 0.8
                # low vision
                t_target = 1
                # # 相机坐标系下控制点
                # pos_medium_ee_obj = (pos_wri_cam_ee + pos_cam_obj) / 2 + np.array([0, 0.1, 0])
                # 世界坐标系下控制点
                pos_world_ctrl = (cur_ee_pose[0:3] + target_pose[0:3]) / 2 + np.array([0, 0, 0.15])

        # # 对末端点在相机坐标系下pos_wri_cam_ee和目标位置pos_cam_obj进行插值
        # # 插值比例 t 从 0 到 1
        # t_values = np.linspace(0, 1, num=150)  # 生成10个点
        # # 方案一： 位置线性插值
        # # points_cam_ee2obj = np.array([
        # #                 (1 - t) * pos_wri_cam_ee + t * pos_cam_obj
        # #                 for t in t_values])
        # # 方案二： 贝塞尔曲线插值
        # # 世界坐标系下位置插值
        # points_world_ee2obj = np.array([
        #     (1 - t) ** 2 * pos_world_ee + 2 * (1 - t) * t * pos_world_ctrl +
        #     t ** 2 * pos_world_obj
        #     for t in t_values
        # ])

        # 计算目标点切向量
        B_1_dot_target = (2 * (1 - t_target) * (pos_world_ctrl - cur_ee_pose[0:3])
                          + 2 * t_target * (target_pose[0:3] - pos_world_ctrl))
        target_point_tangent = PI / 2 + atan2((- B_1_dot_target[2]),
                                              sqrt(B_1_dot_target[0] ** 2 + B_1_dot_target[1] ** 2))
        

        T_world_obj = np.eye(4)
        T_world_obj_sta1 = np.eye(4)
        T_world_obj_sta2 = np.eye(4)

        # TODO:自定义抓取位姿
        # 重点在于抓取位姿随着一阶段的转动发生了改变
        # 绕z轴旋转theta_link1_obj
        dx_link1_obj = target_pose[0] - cur_link1_pose[0]
        dy_link1_obj = target_pose[1] - cur_link1_pose[1]
        theta_link1_obj = np.arctan2(dy_link1_obj, dx_link1_obj)
        T_world_obj_sta1[:3, :3] = np.array([[cos(theta_link1_obj), -sin(theta_link1_obj), 0],
                                             [sin(theta_link1_obj), cos(theta_link1_obj), 0],
                                             [0, 0, 1]]
                                            , dtype=float)
        # 再绕y轴转贝塞尔曲线的切线方向
        T_world_obj_sta2[:3, :3] = np.array([[cos(target_point_tangent), 0, sin(target_point_tangent)],
                                             [0, 1, 0],
                                             [-sin(target_point_tangent), 0, cos(target_point_tangent)]]
                                            , dtype=float)

        T_world_obj = T_world_obj_sta1 @ T_world_obj_sta2
        T_world_obj[:3, 3] = target_pose[0:3]

        target_quat = self.rotation_matrix_to_quaternion((T_world_obj)[:3, :3])
        target_pose[3:] = target_quat
        # ===================================================================

        cur_ee_pos = cur_ee_pose[0:3]
        cur_ee_quat = cur_ee_pose[3:7]
        cur_base_link_pos = cur_base_link_pose[0:3]
        cur_base_link_quat = cur_base_link_pose[3:7]
        target_pos = target_pose[0:3]
        target_quat = target_pose[3:7]
        ee_to_obj_dis = sqrt((target_pos[0] - cur_base_link_pos[0])**2 + 
                             (target_pos[1] - cur_base_link_pos[1])**2)
        if ee_to_obj_dis <= 0.7 * np.sqrt(0.28745):
            pos_world_ctrl = (cur_ee_pos + target_pos) / 2 + np.array([0, 0, 0.2])
        else:
            pos_world_ctrl = (cur_ee_pos + target_pos) / 2 + np.array([0, 0, 0.15])
        t_values = np.linspace(0, 1, num=slerp_num)
        points_world_ee2obj = np.array([
            (1-t)**2 * cur_ee_pos + 2*(1-t)*t * pos_world_ctrl + t**2 * target_pos
            for t in t_values
        ])
        quats_world_ee2obj = np.array([
            self.slerp(cur_ee_quat, target_quat, t) for t in t_values
        ])
        cur_base_link_quat_xyzw = np.roll(cur_base_link_quat, -1)
        T_world_base = np.eye(4)
        T_world_base[:3, :3] = Rotation.from_quat(cur_base_link_quat_xyzw).as_matrix()
        T_world_base[:3, 3] = cur_base_link_pos
        T_base_world = np.linalg.inv(T_world_base)
        path_total = np.zeros((1, 7))
        path_total[0, :6] = cur_joint_pos
        path_total[0, 6] = 0.035 if is_grasp_open else 0
        grasp_control = 0.035 if is_grasp_open else 0

        if item_name == "apple":
            drop_lens = 15

        if item_name == "banana":
            drop_lens = 0

        for idx in range(len(quats_world_ee2obj)- drop_lens):
            quat_world_xyzw = np.roll(quats_world_ee2obj[idx], -1)
            T_world = np.eye(4)
            T_world[:3, :3] = Rotation.from_quat(quat_world_xyzw).as_matrix()
            T_world[:3, 3] = points_world_ee2obj[idx][:3]
            T_base = T_base_world @ T_world
            target_joints_state = self.inverse_kinematics(T_base)
            if target_joints_state is None:
                continue
            target_np = np.array([[*target_joints_state, grasp_control]])
            path_total = np.vstack([path_total, target_np])
        return path_total
    
    # ======= 新增：通用工具 =======
    def _pose_to_T_base(self, base_pose_wxyz7, target_pose_wxyz7):
        """
        根据 base_link 世界位姿和目标世界位姿，得到 T_base_target（4x4）
        pose = [x,y,z,w,x,y,z] (wxyz)
        """
        base_pos = base_pose_wxyz7[:3]
        base_quat_wxyz = base_pose_wxyz7[3:7]
        tgt_pos = target_pose_wxyz7[:3]
        tgt_quat_wxyz = target_pose_wxyz7[3:7]
        # world->base
        T_world_base = np.eye(4)
        T_world_base[:3, :3] = Rotation.from_quat(np.roll(base_quat_wxyz, -1)).as_matrix()
        T_world_base[:3, 3] = base_pos
        T_base_world = np.linalg.inv(T_world_base)
        # world->target
        T_world_tgt = np.eye(4)
        T_world_tgt[:3, :3] = Rotation.from_quat(np.roll(tgt_quat_wxyz, -1)).as_matrix()
        T_world_tgt[:3, 3] = tgt_pos
        # base->target
        return T_base_world @ T_world_tgt

    def _pad_with_gripper(self, q_path, is_open):
        """把 (N,6) 或 (6,N) 的关节序列，变成 (N,7)（最后一列为夹爪）"""
        q = np.asarray(q_path, dtype=float)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        # 统一成 (N,6)
        if q.shape == (6,):
            q = q.reshape(1, 6)
        elif q.shape[0] == 6 and q.shape[1] != 6:
            q = q.T
        if q.shape[1] != 6:
            raise RuntimeError(f"_pad_with_gripper expects (...,6), got {q.shape}")
        grip = 0.035 if is_open else 0.0
        gcol = np.full((q.shape[0], 1), grip, dtype=float)
        return np.hstack([q, gcol])

    def _get_body_pose_any(self, names):
        """多个候选名里找第一个存在的 body，并返回位姿和实际名字"""
        for name in names:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid != -1:
                pos = np.array(self.data.body(bid).xpos)
                quat = np.array(self.data.body(bid).xquat)
                return pos, quat, name
        available = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
                     for i in range(self.model.nbody)]
        raise ValueError(f"找不到候选 body {names}，可用 body 前 20 个: {available[:20]}")

    # ========================= 环境控制相关 =========================
    def set_goal_pose(self, goal_body_name, position, quat_wxyz):
        goal_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, goal_body_name)
        if goal_body_id == -1:
            raise ValueError(f"Body named '{goal_body_name}' not found in the model.")
        goal_joint_id = self.model.body_jntadr[goal_body_id]
        if goal_joint_id == -1:
            raise ValueError(f"Body '{goal_body_name}' 没有关节，不能设定位姿")
        goal_qposadr = self.model.jnt_qposadr[goal_joint_id]
        self.data.qpos[goal_qposadr: goal_qposadr + 3] = position
        self.data.qpos[goal_qposadr + 3: goal_qposadr + 7] = quat_wxyz
    
    def _get_body_pose(self, body_name: str):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise ValueError(f"未找到名为 '{body_name}' 的body")
        position = np.array(self.data.body(body_id).xpos)
        quaternion = np.array(self.data.body(body_id).xquat)
        return position, quaternion
    
    def _get_site_pose(self, site_name: str):
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id == -1:
            raise ValueError(f"未找到名为 {site_name} 的site")
        position = np.array(self.data.site(site_id).xpos)
        xmat = np.array(self.data.site(site_id).xmat)
        quaternion = np.zeros(4)
        mujoco.mju_mat2Quat(quaternion, xmat)  # [w, x, y, z]
        return position, quaternion
    
    def _get_obj_grasp_pose(self, obj_name):
        """
        最好在 XML 里给物体放置 site: <site name="apple_grasp_site" .../>
        我会先找 grasp site；找不到就看 self.grasp_poses；再不行用物体位姿+上移 5cm
        """
        # 1) site
        grasp_site_name = f"{obj_name}_grasp_pose"
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, grasp_site_name)
        if site_id != -1:
            pos = np.array(self.data.site(site_id).xpos)
            xmat = np.array(self.data.site(site_id).xmat)
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, xmat)
            return pos, quat
        # 2) 用户预置
        if obj_name in self.grasp_poses and self.grasp_poses[obj_name]["position"] is not None:
            return np.array(self.grasp_poses[obj_name]["position"]), np.array(self.grasp_poses[obj_name]["quaternion"])
        # 3) 兜底：物体位姿上移 5cm
        obj_pos, obj_quat = self._get_body_pose(obj_name)
        grasp_pos = obj_pos.copy(); grasp_pos[2] += 0.05
        return grasp_pos, obj_quat
    
    def get_image_pos_R_from_camera(self, w, h, camera_name):
        viewport = mujoco.MjrRect(0, 0, w, h)
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        self.camera.fixedcamid = cam_id
        self.camera_pos = self.data.cam_xpos[cam_id]
        self.camera_mat = self.data.cam_xmat[cam_id].reshape((3, 3))
        mujoco.mjv_updateScene(
            self.model, self.data, mujoco.MjvOption(),
            None, self.camera, mujoco.mjtCatBit.mjCAT_ALL, self.scene
        )
        mujoco.mjr_render(viewport, self.scene, self.context)
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(rgb, None, viewport, self.context)
        cv_image = cv2.cvtColor(np.flipud(rgb), cv2.COLOR_RGB2BGR)
        return cv_image
    
    # ========================= 双臂轨迹对齐 =========================
    def _as_Nx7(self, seg, who):
        arr = np.asarray(seg, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        # 可能有人返回 (7, N)
        if arr.shape[0] == 7 and arr.shape[1] != 7:
            arr = arr.T
        if arr.shape[1] != 7:
            raise RuntimeError(f"{who} segment must be (N,7), got {arr.shape}")
        return arr

    def _align_dual_arm_trajs(self, left_arm_full_traj, right_arm_full_traj):
        """对左右臂多阶段轨迹进行阶段补齐、轨迹点数对齐"""
        # 先都变成 Nx7
        left_arm_full_traj  = [self._as_Nx7(s, "left")  for s in left_arm_full_traj]
        right_arm_full_traj = [self._as_Nx7(s, "right") for s in right_arm_full_traj]

        # ……下面保留你原来的阶段补齐&点数对齐逻辑……
        len_left = len(left_arm_full_traj)
        len_right = len(right_arm_full_traj)
        if len_left < len_right and len_left > 0:
            last_point = left_arm_full_traj[-1][-1]
            for i in range(len_right - len_left):
                pad_len = right_arm_full_traj[len_left + i].shape[0]
                left_arm_full_traj.append(np.tile(last_point, (pad_len, 1)))
        elif len_right < len_left and len_right > 0:
            last_point = right_arm_full_traj[-1][-1]
            for i in range(len_left - len_right):
                pad_len = left_arm_full_traj[len_right + i].shape[0]
                right_arm_full_traj.append(np.tile(last_point, (pad_len, 1)))

        dual_arm_full_traj = []
        for left_stage, right_stage in zip(left_arm_full_traj, right_arm_full_traj):
            left_len = left_stage.shape[0]
            right_len = right_stage.shape[0]
            max_len = max(left_len, right_len)
            if left_len < max_len:
                left_stage = np.vstack([left_stage, np.repeat(left_stage[-1][np.newaxis, :], max_len - left_len, axis=0)])
            if right_len < max_len:
                right_stage = np.vstack([right_stage, np.repeat(right_stage[-1][np.newaxis, :], max_len - right_len, axis=0)])
            # 拼成 14 列
            combined_stage = np.hstack([left_stage, right_stage])
            dual_arm_full_traj.append(combined_stage)

        return np.vstack(dual_arm_full_traj) if dual_arm_full_traj else np.array([])
    
    def set_place_joint_targets(self, left_q=None, right_q=None):
        """
        在 run_before() 之前调用，用于告诉我二阶段的目标关节角（6个关节，不含夹爪）。
        """
        if left_q is not None:
            self.left_place_q = np.asarray(left_q, dtype=float).reshape(6,)
        if right_q is not None:
            self.right_place_q = np.asarray(right_q, dtype=float).reshape(6,)

    def _pad_with_gripper(self, q_traj, is_open):
        """
        q_traj: (N,6) 只含6个关节
        返回:   (N,7) 在最后一列加夹爪目标（开=0.035 / 闭=0.0）
        """
        g = 0.035 if is_open else 0.0
        gcol = np.full((q_traj.shape[0], 1), g, dtype=float)
        return np.hstack([q_traj, gcol])
    
    # ========================= 关键修改：左右臂完整轨迹 =========================
    def cal_left_arm_full_traj(self, first_stage_start_ee_pose, first_stage_start_base_link_pose,
                           first_stage_start_link1_pose, first_stage_start_joint_pos,
                           _unused1, _unused2, _unused3):
        """
        左臂:
        阶段1 抓苹果:         plan_arm_traj + 关夹爪
        阶段2 搬运至放置位:    RRT+Cubic (从阶段1末点到 预设的 left_place_q)，途中爪保持闭合
                            到达后增加开夹爪的保持段
        """
        full_path = []
        try:
            # ---------- 阶段1：去抓苹果（IK路径，夹爪打开） ----------
            obj_grasp_pos, obj_grasp_quat = self._get_obj_grasp_pose(self.apple_name)
            grasp_pose = np.concatenate([obj_grasp_pos, obj_grasp_quat])

            print(f"Left grasp pose: {grasp_pose}")

            first_path = self._plan_arm_traj(
                "apple",
                grasp_pose,
                first_stage_start_ee_pose,
                first_stage_start_base_link_pose,
                first_stage_start_link1_pose,
                slerp_num=150,
                cur_joint_pos=first_stage_start_joint_pos,
                is_grasp_open=True
            )
            if first_path is None or len(first_path) == 0:
                print("Left Stage1 plan failed")
                return []

            # 夹爪闭合保持
            first_path_last_q = first_path[-1][:6]
            first_path_grip_close = self._add_gripper_state(first_path_last_q, 40, is_grasp_open=False)

            # ---------- 阶段2：RRT+Cubic 到预设“放置关节角”，途中爪保持闭合 ----------
            if not hasattr(self, "left_place_q"):
                print("[Left] place target q not set, call set_place_joint_targets(...) first.")
                full_path.extend([first_path, first_path_grip_close])
                return full_path

            q_start = first_path_grip_close[-1][:6]   # 从阶段1（含关爪保持段）末点出发
            q_goal  = self.left_place_q               # 你预先给定的6关节放置目标

            q_rrt = self.calc_arm_rrt_cubic_traj(q_start, q_goal)
            if q_rrt is None or len(q_rrt) == 0:
                print("Left Stage2 RRT failed")
                full_path.extend([first_path, first_path_grip_close])
                return full_path

            # RRT轨迹补夹爪（闭合）
            second_path_joint7 = self._pad_with_gripper(q_rrt, is_open=False)

            # 到达后开夹爪保持
            last_q2 = second_path_joint7[-1][:6]
            second_path_grip_open = self._add_gripper_state(last_q2, 20, is_grasp_open=True)

            # 合并
            full_path.extend([first_path, first_path_grip_close, second_path_joint7, second_path_grip_open])
            return full_path

        except Exception as e:
            print(f"Error in left arm trajectory: {e}")
            return []
    
    def cal_right_arm_full_traj(self, first_stage_start_ee_pose, first_stage_start_base_link_pose,
                            first_stage_start_link1_pose, first_stage_start_joint_pos,
                            _unused1, _unused2, _unused3):
        """
        右臂:
        阶段1 抓香蕉:         plan_arm_traj + 关夹爪
        阶段2 搬运至放置位:    RRT+Cubic 到预设 right_place_q，途中爪保持闭合
                            到达后增加开夹爪的保持段
        """
        full_path = []
        try:
            # ---------- 阶段1 ----------
            obj_grasp_pos, obj_grasp_quat = self._get_obj_grasp_pose(self.banana_name)
            grasp_pose = np.concatenate([obj_grasp_pos, obj_grasp_quat])

            first_path = self._plan_arm_traj(
                "banana",
                grasp_pose,
                first_stage_start_ee_pose,
                first_stage_start_base_link_pose,
                first_stage_start_link1_pose,
                slerp_num=150,
                cur_joint_pos=first_stage_start_joint_pos,
                is_grasp_open=True
            )
            if first_path is None or len(first_path) == 0:
                print("Right Stage1 plan failed")
                return []
            
            print(f"right first_path length: {len(first_path)}")

            # 夹爪闭合保持
            first_path_last_q = first_path[-1][:6]
            first_path_grip_close = self._add_gripper_state(first_path_last_q, 40, is_grasp_open=False)

            print(f"right first_path_grip_close length: {len(first_path_grip_close)}")

            # ---------- 阶段2 ----------
            if not hasattr(self, "right_place_q"):
                print("[Right] place target q not set, call set_place_joint_targets(...) first.")
                full_path.extend([first_path, first_path_grip_close])
                return full_path

            q_start = first_path_grip_close[-1][:6]
            q_goal  = self.right_place_q

            q_rrt = self.calc_arm_rrt_cubic_traj(q_start, q_goal)
            if q_rrt is None or len(q_rrt) == 0:
                print("Right Stage2 RRT failed")
                full_path.extend([first_path, first_path_grip_close])
                return full_path

            second_path_joint7 = self._pad_with_gripper(q_rrt, is_open=False)

            last_q2 = second_path_joint7[-1][:6]
            second_path_grip_open = self._add_gripper_state(last_q2, 20, is_grasp_open=True)

            full_path.extend([first_path, first_path_grip_close, second_path_joint7, second_path_grip_open])
            for i, seg in enumerate(full_path, start=1):
                try:
                    print(f"Segment {i}: {len(seg)} waypoints")
                except Exception:
                    print(f"Segment {i}: type={type(seg)}")
            return full_path

        except Exception as e:
            print(f"Error in right arm trajectory: {e}")
            return []
    
    # ========================= 计算双臂协调轨迹 =========================
    def cal_dual_arm_traj(self):
        try:
            # 左臂当前状态
            left_ee_pos, left_ee_quat = self._get_site_pose("left_ee")
            left_base_pos, left_base_quat = self._get_body_pose("left_base_link")
            left_link1_pos, left_link1_quat = self._get_body_pose("left_link1")
            left_ee_pose = np.concatenate([left_ee_pos, left_ee_quat])
            left_base_pose = np.concatenate([left_base_pos, left_base_quat])
            left_link1_pose = np.concatenate([left_link1_pos, left_link1_quat])
            left_joint_pos = self._get_arm_joint_positions("left", 8)[:6]
            
            # 右臂当前状态
            right_ee_pos, right_ee_quat = self._get_site_pose("right_ee")
            right_base_pos, right_base_quat = self._get_body_pose("right_base_link")
            right_link1_pos, right_link1_quat = self._get_body_pose("right_link1")
            right_ee_pose = np.concatenate([right_ee_pos, right_ee_quat])
            right_base_pose = np.concatenate([right_base_pos, right_base_quat])
            right_link1_pose = np.concatenate([right_link1_pos, right_link1_quat])
            right_joint_pos = self._get_arm_joint_positions("right", 8)[:6]

            # 规划左臂
            left_arm_full_traj = self.cal_left_arm_full_traj(
                left_ee_pose, left_base_pose, left_link1_pose, left_joint_pos,
                None, None, None
            )
            # 规划右臂
            right_arm_full_traj = self.cal_right_arm_full_traj(
                right_ee_pose, right_base_pose, right_link1_pose, right_joint_pos,
                None, None, None
            )
            if not left_arm_full_traj or not right_arm_full_traj:
                print("One arm failed to plan.")
                return None
            
            path_total = self._align_dual_arm_trajs(left_arm_full_traj, right_arm_full_traj)
            return path_total
        except Exception as e:
            print(f"Error in dual arm trajectory planning: {e}")
            return None
    
    # ========================= 环境重置和运行 =========================
    def reset_objects_random_position(
        self,
        obj_names,
        margin=0.15,
        min_sep=0.10,
        z_offset=0.03,
        keep_orientation=True,
        corner_clearance=0.12,      # 角落避让半径
        banana_edge_margin=0.10,    # 香蕉额外边缘留白（在 margin 基础上再加）
        banana_board_clearance=0.05 # 香蕉额外远离板子的距离
    ):
        """
        只重置指定物体到桌面上；若存在 board，则避免把物体放到 board 的占用区域内。
        额外规则：
        - 苹果只出现在桌子左半部分（y>0）
        - 香蕉只出现在桌子右半部分（y<0），并且更远离桌边和 board
        - 都不得出现在桌子的边边角角（四角有圆形禁区 + 四边留 margin）
        """
        try:
            # 标准化传参
            obj_list = [obj_names] if isinstance(obj_names, str) else list(obj_names)

            # 桌面参数（与XML一致）
            desk_pos, _ = self._get_body_pose("desk")  # [0.7, 0, 0.73]
            half_x, half_y, half_z = 0.3, 0.6, 0.01115
            table_top_z = desk_pos[2] + half_z

            # board（可选）
            board_exists = False
            board_xy_half = (0.08357354000000003, 0.12273938750000006)
            board_pad = 0.01
            try:
                board_pos, _ = self._get_body_pose("board")
                board_exists = True
            except Exception:
                board_exists = False

            # 基础可放置区域（四边留 margin）
            x_min_base = desk_pos[0] - half_x + margin
            x_max_base = desk_pos[0] + half_x - margin
            y_min_base = desk_pos[1] - half_y + margin
            y_max_base = desk_pos[1] + half_y - margin

            # 角落禁区：四个角的圆形清除区中心
            corners = [
                (desk_pos[0] - half_x + margin, desk_pos[1] - half_y + margin),
                (desk_pos[0] - half_x + margin, desk_pos[1] + half_y - margin),
                (desk_pos[0] + half_x - margin, desk_pos[1] - half_y + margin),
                (desk_pos[0] + half_x - margin, desk_pos[1] + half_y - margin),
            ]

            # 已放位置 & 当前姿态
            placed_xy = {}
            name2quat = {}
            for name in obj_list:
                _, q = self._get_body_pose(name)
                name2quat[name] = q.copy()

            def inside_board(x, y, extra=0.0):
                if not board_exists:
                    return False
                bx, by = board_pos[0], board_pos[1]
                hx = board_xy_half[0] + board_pad + extra
                hy = board_xy_half[1] + board_pad + extra
                return (abs(x - bx) <= hx) and (abs(y - by) <= hy)

            def far_from_others(x, y):
                for _, (ox, oy) in placed_xy.items():
                    if ((x - ox) ** 2 + (y - oy) ** 2) ** 0.5 < min_sep:
                        return False
                return True

            def away_from_corners(x, y):
                for cx, cy in corners:
                    if ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5 < corner_clearance:
                        return False
                return True

            # 逐个物体放置
            for name in obj_list:
                is_apple = "apple" in name.lower()
                is_banana = "banana" in name.lower()

                # x/y 采样范围：基础矩形范围
                x_min, x_max = x_min_base, x_max_base
                y_min, y_max = y_min_base, y_max_base

                # 半区限制
                if is_apple:
                    # 左半边（y>0），稍微离 0 远一点，避免贴中线
                    y_min = max(y_min, 0.10)
                elif is_banana:
                    # 右半边（y<0），离 0 更远 + 额外边缘收紧
                    y_max = min(y_max, -0.10)
                    # 再在四边基础 margin 上，加更大的留白
                    x_min = x_min_base + banana_edge_margin
                    x_max = x_max_base - banana_edge_margin
                    y_min = y_min_base + banana_edge_margin
                    y_max = y_max - banana_edge_margin  # y_max 已经是负数，这里再缩小绝对值

                    # 防止范围被挤没了
                    if x_min >= x_max or y_min >= y_max:
                        raise RuntimeError("Banana sampling area collapsed; reduce banana_edge_margin or margin.")

                attempts = 0
                found = False
                while attempts < 400:
                    x = np.random.uniform(x_min, x_max)
                    y = np.random.uniform(y_min, y_max)

                    # 板子禁区：香蕉再加更大 clearance
                    extra_board = banana_board_clearance if is_banana else 0.0
                    if inside_board(x, y, extra=extra_board):
                        attempts += 1
                        continue

                    # 角落禁区 + 物体间距
                    if (not away_from_corners(x, y)) or (not far_from_others(x, y)):
                        attempts += 1
                        continue

                    # 通过约束，设置姿态（保持原姿态 or 统一朝向）
                    z = table_top_z + z_offset
                    quat = name2quat[name] if keep_orientation else np.array([1.0, 0.0, 0.0, 0.0])

                    self.set_goal_pose(name, np.array([x, y, z]), quat)
                    placed_xy[name] = (x, y)
                    found = True
                    break

                if not found:
                    half_hint = "left half" if is_apple else ("right half (tighter)" if is_banana else "whole desk")
                    raise RuntimeError(f"Failed to place '{name}' on the {half_hint} after many attempts.")

            mujoco.mj_forward(self.model, self.data)

            for k, (x, y) in placed_xy.items():
                print(f"{k} placed at: [{x:.3f}, {y:.3f}, {table_top_z + z_offset:.3f}]")
            return True

        except Exception as e:
            print(f"Error in resetting object positions: {e}")
            return False


        

    def _cache_mobile_base_handles(self):
        """在 __init__ 里调用一次：缓存 mobile_ai 自由关节和四个轮子关节的地址/默认值"""
        # 1) 根 body: mobile_ai（有 <freejoint/>）
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "mobile_ai")
        if body_id == -1:
            raise ValueError("Body 'mobile_ai' not found! 请确认 XML 名称一致。")

        jadr = self.model.body_jntadr[body_id]
        if jadr < 0:
            raise ValueError("mobile_ai 没有关节，检查是否真的有 <freejoint/>")

        root_jid = jadr  # 该 body 的第一个关节就是 freejoint
        if self.model.jnt_type[root_jid] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("mobile_ai 的第一个关节不是 freejoint，XML 结构可能改了。")

        root_qposadr = self.model.jnt_qposadr[root_jid]  # 7 维 (x,y,z, qw,qx,qy,qz)
        root_qveladr = self.model.jnt_dofadr[root_jid]   # 6 维自由度

        # 缓存默认位姿（用模型加载后的初态作为“初始值”）
        self._root_freejoint = {
            "jid": root_jid,
            "qposadr": root_qposadr,
            "qveladr": root_qveladr,
            "qpos0": self.data.qpos[root_qposadr:root_qposadr+7].copy(),  # (3 pos + 4 quat)
        }

        # 2) 四个轮子关节
        wheel_joint_names = [
            ("left_wheel_joint",  "hinge"),
            ("right_wheel_joint", "hinge"),
            ("front_wheel_joint", "ball"),
            ("back_wheel_joint",  "ball"),
        ]
        self._wheel_joints = {}

        for name, jtype in wheel_joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid == -1:
                raise ValueError(f"Joint '{name}' not found! 请确认名称与 XML 一致。")

            qposadr = self.model.jnt_qposadr[jid]
            dofadr  = self.model.jnt_dofadr[jid]
            if jtype == "hinge":
                qpos_dim = 1  # 角度
                dof_dim  = 1
            elif jtype == "ball":
                qpos_dim = 4  # 四元数
                dof_dim  = 3
            else:
                raise ValueError(f"不支持的轮子关节类型: {jtype}")

            self._wheel_joints[name] = {
                "jid": jid,
                "type": jtype,
                "qposadr": qposadr,
                "qveladr": dofadr,
                "qpos_dim": qpos_dim,
                "dof_dim": dof_dim,
            }

    def _set_freejoint_pose(self, body_name: str, pos, quat_wxyz):
        """按 body 的 freejoint 设置 3 平移 + 4 四元数（wxyz）"""
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid == -1:
            raise ValueError(f"Body '{body_name}' not found!")
        jadr = self.model.body_jntadr[bid]
        if jadr < 0:
            raise ValueError(f"Body '{body_name}' 没有关联关节。")
        jid = jadr
        if self.model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f"Body '{body_name}' 第一个关节不是 freejoint。")

        qposadr = self.model.jnt_qposadr[jid]
        self.data.qpos[qposadr:qposadr+3] = np.asarray(pos, dtype=float)
        self.data.qpos[qposadr+3:qposadr+7] = np.asarray(quat_wxyz, dtype=float)

    def reset_mobile_base_to_initial(self):
        """
        将 mobile_ai（freejoint）恢复到加载时的初始位姿；
        将 4 个轮子关节恢复到 0（hinge=0, ball=单位四元数）；
        并清零对应速度。
        """
        if not hasattr(self, "_root_freejoint") or not hasattr(self, "_wheel_joints"):
            # 防呆：如果忘了调用缓存函数，就临时建一下
            self._cache_mobile_base_handles()

        # 1) 根 freejoint pose 恢复
        root = self._root_freejoint
        self.data.qpos[root["qposadr"]:root["qposadr"]+7] = root["qpos0"]
        self.data.qvel[root["qveladr"]:root["qveladr"]+6] = 0.0

        # 2) 四个轮子置零
        for name, meta in self._wheel_joints.items():
            qa = meta["qposadr"]
            da = meta["qveladr"]
            if meta["type"] == "hinge":
                self.data.qpos[qa] = 0.0
                self.data.qvel[da] = 0.0
            else:  # ball
                # 单位四元数
                self.data.qpos[qa:qa+4] = np.array([1.0, 0.0, 0.0, 0.0])
                self.data.qvel[da:da+3] = 0.0

        # 推前向
        mujoco.mj_forward(self.model, self.data)
        print("Mobile base reset: pose -> initial, wheels -> zero.")
    
    def _set_joint_positions_by_name(self, joint_names, joint_values, clamp_to_range=True):
        """
        根据关节名列表设置 qpos，支持范围裁剪。
        joint_names: list[str]
        joint_values: list[float] (同长度)
        """
        assert len(joint_names) == len(joint_values), "joint_names 和 joint_values 长度不一致"
        for name, val in zip(joint_names, joint_values):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid == -1:
                raise ValueError(f"关节 '{name}' 未找到！")
            qadr = self.model.jnt_qposadr[jid]
            # 只处理一维（铰链/滑动），球关节/自由关节不在这儿用
            if self.model.jnt_type[jid] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                v = float(val)
                if clamp_to_range and self.model.jnt_limited[jid]:
                    lo, hi = self.model.jnt_range[jid]
                    v = max(min(v, hi), lo)
                self.data.qpos[qadr] = v
            else:
                raise ValueError(f"关节 '{name}' 不是单自由度关节（类型={self.model.jnt_type[jid]}），不支持这里设置。")


    def reset_dual_arms_to_zero(self, left_init=None, right_init=None, default_gripper_open=0.035):
        """
            将左右臂关节设置为给定初始角度并前向更新。
            - left_init/right_init: list/tuple，长度可为 6 或 8（单位：弧度/米，按模型定义）
            * 长度 6：只给臂 1~6 关节；夹爪自动设为开口 [+g, -g]（左臂）/ [+g, -g]（右臂）
            * 长度 8：包含指 7、8，完全按你给的值写
            - default_gripper_open: 若只给 6 关节时，指7/8设为 +g / -g
        """
        try:
            # 左臂目标
            if left_init is None:
                left_init = [0.0] * 6
            left_init = list(left_init)
            if len(left_init) == 6:
                # 左指：left_joint7 ∈ [0, 0.035]，left_joint8 ∈ [-0.035, 0]
                left_init += [default_gripper_open, -default_gripper_open]
            elif len(left_init) != 8:
                raise ValueError("left_init 必须是长度 6 或 8")

            # 右臂目标
            if right_init is None:
                right_init = [0.0] * 6
            right_init = list(right_init)
            if len(right_init) == 6:
                # 右指：right_joint7 ∈ [0, 0.035]，right_joint8 ∈ [-0.035, 0]
                right_init += [default_gripper_open, -default_gripper_open]
            elif len(right_init) != 8:
                raise ValueError("right_init 必须是长度 6 或 8")

            # 写入左臂
            left_joint_names = [f"left_joint{i}" for i in range(1, 9)]
            self._set_joint_positions_by_name(left_joint_names, left_init, clamp_to_range=True)

            # 写入右臂
            right_joint_names = [f"right_joint{i}" for i in range(1, 9)]
            self._set_joint_positions_by_name(right_joint_names, right_init, clamp_to_range=True)

            # 清速度并前向
            self.data.qvel[:] = 0
            mujoco.mj_forward(self.model, self.data)

            print("Dual arms reset to given initial joint angles.")
            return True

        except Exception as e:
            print(f"Error in resetting dual arms: {e}")
            return False
    
    def reset(self):

        print("Resetting environment...")
        left_home  = [0.0, 0.958, -0.485, 0.0, 0.0, 0.0, 0.035, -0.035]   # 8个：含夹爪
        right_home = [0.0, 0.958, -0.485, 0.0, 0.0, 0.0, 0.035, -0.035]   # 8个：含夹爪
        if not self.reset_dual_arms_to_zero(left_home, right_home):
            return False
        obj_names = ["apple", "banana"]
        if not self.reset_objects_random_position(obj_names):
            return False
        self.handle.user_scn.ngeom = 0
        mujoco.mj_forward(self.model, self.data)
        self.sync()
        time.sleep(0.1)
        self.cur_episode_done = False
        self.step_number = 0
        self.goal_reached_count = 0
        print("Environment reset complete")
        return True
    
    def cal_dual_arm_traj_and_cache(self):
        print("Planning dual arm trajectory...")
        self.path_total = self.cal_dual_arm_traj()
        if self.path_total is None or len(self.path_total) == 0:
            print("Failed to plan dual arm trajectory")
            return False
        print(f"Successfully planned trajectory with {len(self.path_total)} waypoints")
        return True

    def run_before(self):
        # 重置环境
        if not self.reset():
            return False
        # 计算双臂轨迹并缓存
        return self.cal_dual_arm_traj_and_cache()
    
    def run_loop(self, steps_per_waypoint=5, base_ctrl=(0.0, 0.0)):
        """
        - steps_per_waypoint: 每个轨迹点仿真步数
        - base_ctrl: (v_left, v_right) 或者你底盘前两个控制量的期望值
        """
        if not hasattr(self, 'path_total') or self.path_total is None:
            print("No trajectory available. Please call run_before() first.")
            return False

        if self.data.ctrl is None or self.data.ctrl.size < 16:
            print(f"Unexpected ctrl size: {self.data.ctrl.size}, expect >= 16 (2 base + 14 arms).")
            return False

        print("Executing dual arm trajectory...")
        self.cur_episode_done = False

        try:
            n = len(self.path_total)
            for idx, waypoint in enumerate(self.path_total):
                wp = np.asarray(waypoint, dtype=float).reshape(-1)
                if wp.size < 14:
                    print(f"Waypoint {idx} dim={wp.size} < 14, skip")
                    continue
                if not np.all(np.isfinite(wp[:14])):
                    print(f"Waypoint {idx} contains NaN/Inf, skip")
                    continue

                # 前2个控制量：底盘
                self.data.ctrl[0] = float(base_ctrl[0])
                self.data.ctrl[1] = float(base_ctrl[1])

                # 后14个控制量：左臂7 + 右臂7
                # 假设顺序就是你规划输出的顺序：left(7) + right(7)
                self.data.ctrl[2:16] = wp[:14]

                # 多步积分以执行该控制
                for _ in range(int(steps_per_waypoint)):
                    mujoco.mj_step(self.model, self.data)
                    self.sync()
                    time.sleep(0.002)

                self.step_number += 1
                if self.step_number % 100 == 0:
                    print(f"Executed {self.step_number}/{n} waypoints")

            self.cur_episode_done = True
            print("Dual arm trajectory execution completed")
            return True

        except Exception as e:
            print(f"Error during trajectory execution: {e}")
            return False

    
    # ========================= 辅助和调试函数 =========================
    def print_all_body_info(self):
        print("\n=== Body 信息 ===")
        print(f"{'Body Name':<25} {'Body ID':<8} {'Position':<30} {'Quaternion':<35}")
        print("-" * 100)
        for body_id in range(self.model.nbody):
            try:
                name_addr = self.model.name_bodyadr[body_id]
                body_name = self.model.names[name_addr:].split(b'\x00')[0].decode('utf-8')
                pos = self.data.body(body_id).xpos
                quat = self.data.body(body_id).xquat
                print(f"{body_name:<25} {body_id:<8} {str(pos):<30} {str(quat):<35}")
            except Exception as e:
                print(f"Error reading body {body_id}: {e}")
    
    def print_all_joint_info(self):
        print("\n=== 关节信息 ===")
        print(f"{'Joint Name':<20} {'Type':<15} {'Qpos Addr':<10} {'Range':<25} {'Current Value':<15}")
        print("-" * 90)
        for joint_id in range(self.model.njnt):
            try:
                name_addr = self.model.name_jntadr[joint_id]
                joint_name = self.model.names[name_addr:].split(b'\x00')[0].decode('utf-8')
                joint_type = self.model.jnt_type[joint_id]
                type_names = {0: "自由关节(6DOF)", 1: "球关节(3DOF)", 2: "滑动关节", 3: "铰链关节"}
                type_str = type_names.get(joint_type, "未知类型")
                qpos_addr = self.model.jnt_qposadr[joint_id]
                if self.model.jnt_limited[joint_id]:
                    jnt_range = f"[{self.model.jnt_range[joint_id, 0]:.2f}, {self.model.jnt_range[joint_id, 1]:.2f}]"
                else:
                    jnt_range = "无限制"
                if joint_type == 0:
                    current_val = self.data.qpos[qpos_addr:qpos_addr + 7]
                elif joint_type == 1:
                    current_val = self.data.qpos[qpos_addr:qpos_addr + 4]
                else:
                    current_val = self.data.qpos[qpos_addr]
                print(f"{joint_name:<20} {type_str:<15} {qpos_addr:<10} {jnt_range:<25} {str(current_val):<15}")
            except Exception as e:
                print(f"Error reading joint {joint_id}: {e}")
    
    def is_running(self):
        return self.handle.is_running()
    
    def sync(self):
        self.handle.sync()
    
    def close(self):
        if hasattr(self, 'handle'):
            self.handle.close()
        if hasattr(self, 'window'):
            glfw.destroy_window(self.window)
        glfw.terminate()
        print("Environment closed")
