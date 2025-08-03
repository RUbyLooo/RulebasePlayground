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
    参数解释：
        cfg.path：路径或文件名，通常是模型文件或场景文件路径。
        3：一般表示自由度（DOF）数量或维度，可能指 3D 空间。
        azimuth=180：方位角（水平旋转角度），单位为度，设置为 180°。
        elevation=-30：俯仰角（垂直旋转角度），单位为度，设置为 -30°（向下倾斜）。
    """
    def __init__(self, cfg, distance=3, azimuth=0, elevation=-30):
        print(f"model_path : {cfg.path}")
        self.model = mujoco.MjModel.from_xml_path(cfg.path)

        # 初始化上下文、场景等渲染资源
        self.data = mujoco.MjData(self.model)
        self.distance = distance
        self.azimuth = azimuth
        self.elevation = elevation

        # 开一个物理引擎的线程
        self.handle = mujoco.viewer.launch_passive(self.model, self.data)
        self.handle.cam.distance = distance
        self.handle.cam.azimuth = azimuth
        self.handle.cam.elevation = elevation
        self.opt = mujoco.MjvOption()

        # ✅ 正确使用 glfw（模块调用，不加 self.）
        glfw.init()
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        self.window = glfw.create_window(640, 480, "offscreen", None, None)
        glfw.make_context_current(self.window)

        # mujoco 相机数据相关
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED 
        self.scene = mujoco.MjvScene(self.model, maxgeom=1000)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.context)

        self.camera_pos = np.array([0, 0, 0])
        self.camera_mat = np.eye(3, dtype=np.float64)
        self.cur_episode_done = False

        # 加载机械臂相关代码
        self.target_position = np.array([0, 0, 0])
        self.target_quat_wxyz = np.array([0, 0, 0, 0])
        self.target_quat_xyzw = np.roll(self.target_quat_wxyz, -1)
        self.item_name = "apple"
        # self.item_name = "banana"

        # 用一个全局变量保存目标位姿，方便随时取用
        self.target_position, self.target_quat_wxyz = self._get_body_pose(self.item_name)
        # 针对banana
        if self.item_name == "banana":
            # 重新计算位置
            self.target_position += np.array([0, 0, -0.02])
            # 重新计算姿态
            # 先获得baselink的世界坐标
        # self.target_position += np.array([0, 0, 0.005])
        self.target_quat_xyzw = np.roll(self.target_quat_wxyz, -1)

        # 保存目标初始化半径
        self.rho = 0.0

        # 统计planning和ik失败率
        self.count = 0
        self.count_ik = 0

        self.path = cfg.path

        if (cfg["is_have_arm"] == True):
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
            self.l = 0.091 + 0.053  # joint4 → joint6 → 末端执行器
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

        # motor映射
        # 创建 motor 映射
        self.motors = {
            "joint_1.pos": 0.0,
            "joint_2.pos": 0.0,
            "joint_3.pos": 0.0,
            "joint_4.pos": 0.0,
            "joint_5.pos": 0.0,
            "joint_6.pos": 0.0,
            "gripper.pos": 0.0,
        }
        self.sim_state_queue = queue.Queue(maxsize=1)

        self.mjstep_thread = threading.Thread(target=self.mujoco_step)
        # self.mjstep_thread.start()


        self.phi = 0


        # 打印当前场景 joint 和 body 信息
        self.print_all_joint_info()
        self.print_all_body_info()

    # -------------------------平滑处理-------------------------
    # 定义平滑函数（使用移动平均法或更高级的Savitzky-Golay滤波器）
    def smooth_waveform(self,waveform, window_size=51):
        # 确保窗口大小是奇数
        window_size = window_size if window_size % 2 != 0 else window_size + 1
        # 使用Savitzky-Golay滤波器进行平滑
        return savgol_filter(waveform, window_length=window_size, polyorder=2)

    # ------------------------正逆运动学------------------------

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
        # 0_T_end
        T_total = np.eye(4)
        for i in range(end):
            # print("i alpha a d theta", self.alpha[i], self.a[i], self.d[i], self.theta_offset[i])
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
        sin_theta = sqrt(R[2, 0] ** 2 + R[2, 1] ** 2)
        singular = sin_theta < 1e-6

        if not singular:
            # theta = atan2(sin_theta, R[2, 2])
            theta = asin(R[0, 2])
            # theta = acos(R[0, 0] / R[0, 1])
            # phi = atan2(R[1, 2] / sin(theta), R[0, 2] / sin(theta))
            phi = 0
            # psi = atan2(R[2, 1] / sin(theta), -R[2, 0] / sin(theta))
            psi = atan2(R[1, 0],R[1, 1])
            # print("phi, theta, psi",[phi, theta, psi])
            if ((phi > -1.832 and phi < 1.832) and (theta > -1.22 and theta < 1.22)
            and (psi > -3.14 and psi < 3.14)):
                self.phi = phi
                return np.array([phi, theta, psi])
            #
            # theta2 = -theta
            # # phi2 = atan2(R[1, 2] / sin(theta2), R[0, 2] / sin(theta2))
            # phi2 = 0
            # # psi2 = atan2(R[2, 1] / sin(theta2), -R[2, 0] / sin(theta2))
            # psi2 = atan2(R[1, 0], R[1, 1])
            # # print("phi2, theta2, psi2",[phi2, theta2, psi2])
            # if ((phi2 > -1.832 and phi2 < 1.832) and (theta2 > -1.22 and theta2 < 1.22)
            # and (psi2 > -3.14 and psi2 < 3.14)):
            #     self.phi = phi2
            #     return np.array([phi2, theta2, psi2])
            else:
                return None

        else:
            theta = 0
            phi = self.phi
            psi = atan2(-R[0, 1], R[0, 0])
            if ((phi > -1.832 and phi < 1.832) and (theta > -1.22 and theta < 1.22)
                    and (psi > -3.14 and psi < 3.14)):
                self.phi = phi
                return np.array([phi, theta, psi])
            else:
                return None

    def rotation_matrix_to_quaternion(self,R):
        """将3x3旋转矩阵转换为四元数(w, x, y, z顺序)"""
        q = np.zeros(4)
        trace = np.trace(R)

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

        return q / np.linalg.norm(q)  # 归一化
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

        # 计算 link1 2 3 角度
        theta1 = atan2(py, px)
        if (theta1 == PI):
            theta1 = 0
            self.theta1 = theta1
        else:
            # TODO
            theta1 = self.theta1
            # print("no ik solution, fail theta 1")
            # return None

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

    # -------------------------轨迹规划-------------------------
    def slerp(self, q1, q2, t):
        """
        四元数球面线性插值
        参数:
        q1, q2: 起始和结束四元数
        t: 插值参数，取值范围[0, 1]
        返回:
        插值结果的四元数
        """
        # 确保两个四元数标准化
        q1, q2 = q1 / np.linalg.norm(q1), q2 / np.linalg.norm(q2)
        # 计算两个四元数之间的余弦值
        cos_half_theta = q1.dot(q2)
        # 如果点积是负的，则通过改变q2的符号来最小化角度
        if cos_half_theta < 0:
            q2 = -q2
            cos_half_theta = -cos_half_theta
        # 如果两个四元数过于接近，则使用线性插值而不是球面插值
        if np.abs(cos_half_theta) >= 1.0:
            q1 = (1 - t) * q1 + t * q2
            return q1 / np.linalg.norm(q1)
        # 计算半角的正弦值
        half_theta = np.arccos(cos_half_theta)
        sin_half_theta = np.sqrt(1.0 - cos_half_theta * cos_half_theta)
        # 如果半角的正弦值接近0，使用线性插值
        if np.abs(sin_half_theta) < 0.001:
            return (1.0 - t) * q1 + t * q2
        ratioA = np.sin((1 - t) * half_theta) / sin_half_theta
        ratioB = np.sin(t * half_theta) / sin_half_theta
        # 返回插值结果
        result = ratioA * q1 + ratioB * q2
        return result
    def calc_arm_rrt_cubic_traj(
            self,
            cur_joints_state,
            target_joints_state
    ):
        """
        计算机械臂在给定目标关节角度下的运动轨迹

        参数:
            cur_joints_state:         当前机械臂的 6 关节状态
            target_joints_state:      机械臂目标 6 关节
        返回:
            path: 轨迹
        """
        q_start = cur_joints_state
        q_goal = target_joints_state

        # print(f"q_start : {q_start}")
        # print(f"q_goal : {q_goal}")

        # Search for a path
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
        # print(f"q_path : {q_path}")
        if q_path is None:
            return None
        if len(q_path) > 0:
            print(f"Got a path with {len(q_path)} waypoints")

        # Perform trajectory optimization.
        dt = 0.025
        options = CubicTrajectoryOptimizationOptions(
            num_waypoints=len(q_path),
            samples_per_segment=7,
            min_segment_time=0.5,
            max_segment_time=10.0,
            min_vel=-1.5,
            max_vel=1.5,
            min_accel=-0.75,
            max_accel=0.75,
            min_jerk=-1.0,
            max_jerk=1.0,
            max_planning_time=30.0,
            check_collisions=True,
            min_collision_dist=self.distance_padding,
            collision_influence_dist=0.05,
            collision_avoidance_cost_weight=0.0,
            collision_link_list=[
                "ground_plane",
                "link6",
            ],
        )
        print("Optimizing the path...")
        optimizer = CubicTrajectoryOptimization(self.model_roboplan, self.collision_model, options)
        traj = optimizer.plan([q_path[0], q_path[-1]], init_path=q_path)

        if traj is not None:
            print("Trajectory optimization successful")
            traj_gen = traj.generate(dt)
            return traj_gen[1]
        else:
            return None

    # ---------------------MuJoCo环境渲染---------------------
    def set_goal_pose(self, goal_body_name, position, quat_wxyz):
        """
        设置目标位姿（位置 + 姿态）

        参数解释:
            position: 目标的位置，(x, y, z)，类型为 numpy.ndarray 或 list。
            quat_wxyz: 目标的姿态，四元数 (w, x, y, z)，类型为 numpy.ndarray 或 list。
        """
        # 设置 target 的位姿
        # goal_body_name = "target"
        goal_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, goal_body_name)

        if goal_body_id == -1:
            raise ValueError(f"Body named '{goal_body_name}' not found in the model.")

        # 获取 joint ID 和 qpos 起始索引
        goal_joint_id = self.model.body_jntadr[goal_body_id]
        goal_qposadr = self.model.jnt_qposadr[goal_joint_id]

        # 设置位姿
        if goal_qposadr + 7 <= self.model.nq:
            self.data.qpos[goal_qposadr: goal_qposadr + 3] = position
            self.data.qpos[goal_qposadr + 3: goal_qposadr + 7] = quat_wxyz
        else:
            print("[警告] target 的 qpos 索引越界或 joint 设置有误")

    def _get_body_pose(self, body_name: str) -> np.ndarray:
        """
        通过 body 名称获取其位姿信息, 返回一个7维向量
            :param body_name: body名称字符串
            :return: 7维numpy数组, 格式为 [x, y, z, w, x, y, z]
            :raises ValueError: 如果找不到指定名称的body
        """
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise ValueError(f"未找到名为 '{body_name}' 的body")

        # 提取位置和四元数并合并为一个7维向量
        position = np.array(self.data.body(body_id).xpos)  # [x, y, z]
        quaternion = np.array(self.data.body(body_id).xquat)  # [w, x, y, z]

        return position, quaternion

    def get_image_pos_R_from_camera(self, w, h, camera_name):

        # 新建一个矩形区域，(0, 0, w, h) 表示从左上角 (0, 0) 开始，宽 w，高 h
        viewport = mujoco.MjrRect(0, 0, w, h)
        # 查找相机 ID
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        # 设置 fixedcamid 表示当前要渲染的固定相机 ID
        self.camera.fixedcamid = cam_id

        # 获取相机位姿，以及四元数表示
        self.camera_pos = self.data.cam_xpos[cam_id]
        self.camera_mat = self.data.cam_xmat[cam_id].reshape((3, 3))

        # 构建渲染场景数据到 self.scene 中
        mujoco.mjv_updateScene(
            self.model, self.data, mujoco.MjvOption(),
            None, self.camera, mujoco.mjtCatBit.mjCAT_ALL, self.scene
        )
        mujoco.mjr_render(viewport, self.scene, self.context)
        # 创建一个空的 RGB 图像数组，大小为 h x w x 3
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        # 将渲染结果读入 rgb 数组中，第二个参数是深度图(这里传 None 表示不需要)
        mujoco.mjr_readPixels(rgb, None, viewport, self.context)
        # 将 RGB 转换为 BGR，适配 OpenCV 默认格式
        cv_image = cv2.cvtColor(np.flipud(rgb), cv2.COLOR_RGB2BGR)
        return cv_image

    # ---------------------MuJoCo环境运行---------------------
    # mj_step单独线程
    def mujoco_step(self):
        while True:
            mujoco.mj_step(self.model, self.data)
            # 调用一次 get obs 获得最新的 sim data
            # self._set_sim_data(target_joints_state_np)
            # 更新一下队列中的 sim data
            self.sync()
            time.sleep(0.002)
    def close(self):
        # self.running = False
        self.planning_thread.join()
        print("Planning stopped.")

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
                jnt_range = f"[{self.model.jnt_range[joint_id, 0]:.2f}, {self.model.jnt_range[joint_id, 1]:.2f}]"
            else:
                jnt_range = "无限制"

            # 获取当前值
            if joint_type == 0:  # 自由关节
                current_val = self.data.qpos[qpos_addr:qpos_addr + 7]
            elif joint_type == 1:  # 球关节
                current_val = self.data.qpos[qpos_addr:qpos_addr + 4]
            else:  # 滑动/铰链关节
                current_val = self.data.qpos[qpos_addr]

            print(f"{joint_name:<20} {type_str:<15} {qpos_addr:<10} {jnt_range:<25} {str(current_val):<15}")


    def is_running(self):
        return self.handle.is_running()

    def sync(self):
        self.handle.sync()

    @property
    def cam(self):
        return self.handle.cam

    def calculate_camera_matrix(self, fov, height, width):
        """
        计算相机内参矩阵 K。

        参数:
        fovy (float): 相机的视场角，单位为度。
        height (int): 图像高度，单位为像素。
        width (int): 图像宽度，单位为像素。

        返回:
        K (numpy.array): 相机内参矩阵。
        """
        # 将视场角转换为弧度
        fov_rad = np.radians(fov)

        # 计算焦距（以像素为单位）
        f = height / (2 * np.tan(fov_rad / 2))

        # 主点（图像中心）
        cx = (width - 1) / 2
        cy = (height - 1) / 2

        # 内参矩阵 K
        K = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 1, 1]
        ])

        return K

    @property
    def viewport(self):
        return self.handle.viewportW

    def reset(self):
        """
        重置目标位置,机械臂关节回零。
        Returns
        -------
        """
        while True:
            # ----------------------------随机目标位置----------------------------
            center_x = -0.25
            center_y = 0
            radius = np.sqrt(0.28745) # 0.53619m
            theta = np.random.uniform(-np.pi / 2, np.pi / 2)
            # # theta = -1.3
            rho = radius * np.random.uniform(0.56, 1)
            # # 近处点
            # rho = radius * np.random.uniform(0.56, 0.7)

            # 远处点
            # # rho = radius * np.random.uniform(0.7, 1)
            # rho = 0.3002664
            # theta = 1.1322677361482247

            # 越过桌子的情况
            if (theta < -PI / 9 and theta > -PI / 6) or (theta > PI / 9 and theta < PI / 6):
                rho = radius * np.random.uniform(0.45 / radius, 1)
            if (theta < PI / 9 and theta > -PI / 9) :
                rho = radius * np.random.uniform(0.75, 1)

            self.rho = rho
            print("rho", rho)
            print("theta", theta)
            x_world_target = rho * np.cos(theta) + center_x
            y_world_target = rho * np.sin(theta) + center_y

            # 随机设置位置
            self.target_position[0] = x_world_target
            self.target_position[1] = y_world_target

            print("self.target_position",self.target_position)

            # 获得篮子的位置
            board_position, _= self._get_body_pose("board")

            # 如果苹果初始化在了篮子，continue重新设置一遍
            if ((abs(board_position[0] - self.target_position[0]) < 0.1) and
                    (abs(board_position[1] - self.target_position[1]) < 0.18)):
                continue
            # 如果苹果没有初始化在篮子里，break一次成功
            else:
                # 随机夹爪位置
                # 随机一个球体,定义抓取位置
                x_sphere_center = self.target_position[0]
                y_sphere_center = self.target_position[1]
                z_sphere_center = self.target_position[2]
                radius_sphere = 0.0883
                theta_sphere = np.random.uniform(0, 2 * np.pi)
                # 抓取上方的点
                phi_sphere = np.random.uniform(0, np.pi / 2)
                # rho_sphere = radius * np.random.uniform(0.3, 1)
                piper_position = np.zeros(3)
                piper_position[0] = radius_sphere * sin(phi_sphere) * cos(theta_sphere) + x_sphere_center
                piper_position[1] = radius_sphere * sin(phi_sphere) * sin(theta_sphere) + y_sphere_center
                piper_position[2] = radius_sphere * cos(phi_sphere) + z_sphere_center

                # 在世界系下定义抓取姿态
                # 绕y轴旋转90度
                T_world_obj_y_90 = np.eye(4)
                T_world_obj_y_90[:3, :3] = np.array([[cos(90 / 180 * PI), 0, sin(90 / 180 * PI)],
                                                     [0, 1, 0],
                                                     [-sin(90 / 180 * PI), 0, cos(90 / 180 * PI)]]
                                                    , dtype=float)

                # 绕x轴旋转
                dx_piper_target = abs(piper_position[0] - self.target_position[0])
                dy_piper_target = abs(piper_position[1] - self.target_position[1])

                theta_piper_target = 0.0

                # piper在世界坐标系第一象限
                if (((piper_position[0] - self.target_position[0]) > 0) and
                        ((piper_position[1] - self.target_position[1]) > 0)):
                    theta_piper_target = PI + atan2(dx_piper_target, dy_piper_target) - PI / 2
                # piper在世界坐标系第二象限
                if (((piper_position[0] - self.target_position[0]) < 0) and
                        ((piper_position[1] - self.target_position[1]) > 0)):
                    theta_piper_target = PI - atan2(dx_piper_target, dy_piper_target) - PI / 2
                # piper在世界坐标系第三象限
                if (((piper_position[0] - self.target_position[0]) < 0) and
                        ((piper_position[1] - self.target_position[1]) < 0)):
                    theta_piper_target = atan2(dx_piper_target, dy_piper_target) - PI / 2
                # piper在世界坐标系第四象限
                if (((piper_position[0] - self.target_position[0]) > 0) and
                        ((piper_position[1] - self.target_position[1]) < 0)):
                    theta_piper_target = 2 * PI - atan2(dx_piper_target, dy_piper_target) - PI / 2

                # # 计算绕x轴旋转矩阵
                T_y_90_x_theta = np.eye(4)
                T_y_90_x_theta[:3, :3] = np.array([[1, 0, 0],
                                                     [0, cos(theta_piper_target), -sin(theta_piper_target)],
                                                     [0, sin(theta_piper_target), cos(theta_piper_target)]],
                                                     dtype=float)

                # T_world_piper = np.eye(4)
                # T_world_piper = T_world_obj_y_90 @ T_y_90_x_theta
                # piper_quat_wxyz = self.rotation_matrix_to_quaternion(T_world_piper[:3, :3])

                # 绕y轴旋转90-phi度
                T_x_theta_y = np.eye(4)
                T_x_theta_y[:3, :3] = np.array([[cos(PI /2 - phi_sphere), 0, sin(PI /2 - phi_sphere)],
                                                     [0, 1, 0],
                                                     [-sin(PI /2 - phi_sphere), 0, cos(PI /2 - phi_sphere)]]
                                                    , dtype=float)

                T_world_piper = np.eye(4)
                T_world_piper = T_world_obj_y_90 @ T_y_90_x_theta @ T_x_theta_y
                piper_quat_wxyz = self.rotation_matrix_to_quaternion(T_world_piper[:3, :3])

                # 传入piper位姿
                # self.set_goal_pose("target", piper_position, piper_quat_wxyz)
                # 判断目标是不是香蕉
                if self.item_name == "banana":
                    # 获得baselink在世界系下的位姿
                    arm_base_pos, _ = self._get_body_pose("base_link")

                    # 计算香蕉x，y的位置差
                    dx_base_obj = self.target_position[0] - arm_base_pos[0]
                    dy_base_obj = self.target_position[1] - arm_base_pos[1]

                    # 计算香蕉姿态绕z轴旋转的夹角
                    if (dy_base_obj > 0 and dx_base_obj > 0):
                        # # 正常情况
                        # theta_banana = -1 * (random.choice([0, 1]) * PI + atan2(dx_base_obj, dy_base_obj) )
                        # # 调试：各种异常情况
                        theta_banana = -1 *  atan2(dx_base_obj, dy_base_obj)
                        # 绕z轴旋转theta_banana
                        R_banana = np.array([[cos(theta_banana), -sin(theta_banana), 0],
                                                     [sin(theta_banana), cos(theta_banana), 0],
                                                     [0, 0, 1]]
                                                    , dtype=float)
                        # 转成四元数
                        self.target_quat_wxyz  = self.rotation_matrix_to_quaternion(R_banana)
                        # self.set_goal_pose(self.item_name, self.target_position, banana_quat_wxyz)

                    if (dy_base_obj < 0 and dx_base_obj > 0):
                        # theta_banana = -1 * (random.choice([0, 1]) * PI + PI - atan2(abs(dx_base_obj), abs(dy_base_obj)))
                        theta_banana = atan2(abs(dx_base_obj), abs(dy_base_obj))
                        # 绕z轴旋转theta_banana
                        R_banana = np.array([[cos(theta_banana), -sin(theta_banana), 0],
                                                     [sin(theta_banana), cos(theta_banana), 0],
                                                     [0, 0, 1]]
                                                    , dtype=float)
                        # 转成四元数
                        self.target_quat_wxyz  = self.rotation_matrix_to_quaternion(R_banana)
                        print("theta_banana",theta_banana)
                        # banana_quat_wxyz = np.array([-0.7071, 0, 0, 0.7071])
                        # self.set_goal_pose(self.item_name, self.target_position, banana_quat_wxyz)
                self.set_goal_pose(self.item_name, self.target_position, self.target_quat_wxyz )

                # ----------------------------传入目标物体位姿----------------------------
                # self.set_goal_pose("apple", self.target_position, self.target_quat_wxyz)
                # for i range(3):
                # self.set_goal_pose("apple", self.target_position, np.array([1, 0, 0, 0]))
                # self.set_goal_pose("banana", self.target_position, np.array([1, 0, 0, 0]))
                # self.set_goal_pose(self.item_name, self.target_position, np.array([1, 0, 0, 0]))

                # ------------------------传入机械臂初始状态qpos------------------------
                # self.data.qpos[:6] = np.zeros(6)
                # self.data.qpos[6] = 0
                # self.data.qpos[7] = 0
                # 仿真同步
                self.handle.user_scn.ngeom = 0

                mujoco.mj_forward(self.model, self.data)
                self.sync()
                time.sleep(0.002)

                break

    # 这个函数是每走一个 mj step 就获得一个最新的 sim data
    def _get_sim_data(self, joints_state):

        # Read arm position
        start = time.perf_counter()
        # ==== 左臂 ====
        joint_state = joints_state
        self.motors["joint_1.pos"] = round(joints_state[0][0], 8)
        self.motors["joint_2.pos"] = round(joints_state[0][1], 8)
        self.motors["joint_3.pos"] = round(joints_state[0][2], 8)
        self.motors["joint_4.pos"] = round(joints_state[0][3], 8)
        self.motors["joint_5.pos"] = round(joints_state[0][4], 8)
        self.motors["joint_6.pos"] = round(joints_state[0][5], 8)
        # gripper_raw = self.piper.GetArmGripperMsgs().gripper_state.grippers_angle
        self.motors["gripper.pos"] = round(joints_state[0][6] / 0.035, 8)
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")
        obs_dict = self.motors.copy()
        # 获取相机图片
        start = time.perf_counter()
        obs_dict["wrist_cam"] = self.get_image_pos_R_from_camera(640, 480, "wrist_cam")
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read wrist_cam: {dt_ms:.1f}ms")
        # wrist_cam_image = self.get_image_pos_R_from_camera(640, 480, "wrist_cam")
        return obs_dict

    # 这个函数是把 sim data 往队列里传
    def _set_sim_data(self, joints_state):
        sim_data = self._get_sim_data(joints_state)
        try:
            self.sim_state_queue.put_nowait(sim_data)
        except queue.Full:
            try:
                _ = self.sim_state_queue.get_nowait()
                self.sim_state_queue.put_nowait(sim_data)
            except queue.Empty:
                pass

    # 这个函数是获取 sim data 队列中最新的元素
    def get_observation(self):
        try:
            # 尝试从队列中获取数据
            sim_data = self.sim_state_queue.get_nowait()
            return sim_data
        except queue.Empty:
            # 如果队列为空，处理异常
            pass

    def run_before(self):
        """
        求解机械臂IK

        参数解释:
            position: 目标的位置，(x, y, z)，类型为 numpy.ndarray 或 list。
            quat_wxyz: 目标的姿态，四元数 (w, x, y, z)，类型为 numpy.ndarray 或 list。
        """
        # -----------------step 1 : 获取body在世界坐标系下的位姿-----------------
        self.init_state = self.data.qpos.copy()
        self.q_vec = None

        q_start = np.zeros(6)
        if q_start is None:
            raise RuntimeError(" q_start is invalid... ")
        # mujoco返回的是[w, x, y, z]顺序的
        arm_base_name = "base_link"
        arm_base_pos, arm_base_quat = self._get_body_pose(arm_base_name)
        arm_base_quat_xyzw = np.roll(arm_base_quat, -1)

        arm_link1_name = "link1"
        arm_link1_pos, arm_link1_quat = self._get_body_pose(arm_link1_name)
        arm_link1_quat_xyzw = np.roll(arm_link1_quat, -1)

        arm_link6_name = "link6"
        arm_link6_pos, arm_link6_quat = self._get_body_pose(arm_link6_name)
        arm_link6_quat_xyzw = np.roll(arm_link6_quat, -1)

        # -----------------------step 2 : 设置初始抓取位置----------------------
        T_world_link1 = np.eye(4)
        T_world_link1[:3, :3] = Rotation.from_quat(arm_link1_quat_xyzw).as_matrix()
        T_world_link1[:3, 3] = arm_link1_pos

        # 计算joint1转动的夹角
        dx_link1_obj = self.target_position[0] - arm_link1_pos[0]
        dy_link1_obj = self.target_position[1] - arm_link1_pos[1]
        theta_link1_obj = np.arctan2(dy_link1_obj, dx_link1_obj)
        self.theta1 = theta_link1_obj

        # # 控制joint1转动到能看到目标且视野大的位置
        # # 理想夹角 joint3：-0.661；joint5：1.22；gripper：0.035
        # target_joints_state_sta1 = np.array([theta_link1_obj, 0, -0.661, 0, 1.22, 0]
        #                                     , dtype=float)

        # 抬高视角，想要看到板子
        target_joints_state_sta1 = np.array([0, 0, -0.661, 0, 1.22, 0]
                                            , dtype=float)

        # # 测试最大工作半径
        # self.data.qpos[1] = 2.51
        # self.data.qpos[2] = -2.06
        # self.data.qpos[4] = 0.647

        # 调用RRT规划轨迹位姿，从初始位置0到初始可以看到板子的抓取位姿
        path = self.calc_arm_rrt_cubic_traj(q_start, target_joints_state_sta1)
        # 新建一个用来保存全部waypoint的数组
        path_total = np.array([[0,0,0,0,0,0,0]])

        if path is None:
            self.count = self.count + 1
            raise RuntimeError(" planning path failed... ")

        # 一阶段mujoco同步执行可视化
        index = 1

        while True:
            if index >= path.shape[1] - 1:
                # self.cur_episode_done = True
                break

            for i in range(3):
                target_joints_state = path[:6, index]
                target_joints_state_np = np.array([
                    [target_joints_state[0], target_joints_state[1], target_joints_state[2],
                     target_joints_state[3], target_joints_state[4], target_joints_state[5],
                     0.035]])
                path_total = np.vstack([path_total, target_joints_state_np])
                # self.data.ctrl[:7] = target_joints_state_np
                # mujoco.mj_step(self.model, self.data)
                # # 调用一次 get obs 获得最新的 sim data
                # # self._set_sim_data(target_joints_state_np)
                # # 更新一下队列中的 sim data
                # self.sync()
                # time.sleep(0.002)

                # # 获取当前腕部相机图片，并将腕部相机位置保存在全局变量中
                # wrist_cam_image = self.get_image_pos_R_from_camera(640, 480, "wrist_cam")
                # cv2.imshow("MuJoCo Camera", wrist_cam_image)
                # key = cv2.waitKey(1)
                # # Press esc or 'q' to close the image window
                # if key & 0xFF == ord('q') or key == 27:
                #     cv2.destroyAllWindows()

            index += 1

        # self.get_observation()
        # print("self.get_observation()",self.get_observation())
        # print("self.get_observation()", self.get_observation())

        # 转动joint1，想要看到苹果
        target_joints_state_sta2 = np.array([theta_link1_obj, 0, -0.661, 0, 1.22, 0]
                                            , dtype=float)

        # # 测试最大工作半径
        # self.data.qpos[1] = 2.51
        # self.data.qpos[2] = -2.06
        # self.data.qpos[4] = 0.647

        # 调用RRT规划轨迹位姿，从可以看到板子的抓取位姿到可以看到苹果的抓取位姿
        path = self.calc_arm_rrt_cubic_traj(target_joints_state_sta1, target_joints_state_sta2)

        if path is None:
            self.count = self.count + 1
            raise RuntimeError(" planning path failed... ")

        # 访问最后一行
        last_point_sta2 = np.array(path[:, -1], dtype=float)

        # 一阶段mujoco同步执行可视化
        index = 0

        while True:
            if index >= path.shape[1] - 1:
                # self.cur_episode_done = True
                break

            for i in range(3):
                target_joints_state = path[:6, index]
                target_joints_state_np = np.array([
                    [target_joints_state[0], target_joints_state[1], target_joints_state[2],
                     target_joints_state[3], target_joints_state[4], target_joints_state[5],
                     0.035]])
                path_total = np.vstack([path_total, target_joints_state_np])
                # self.data.ctrl[:7] = target_joints_state_np
                # mujoco.mj_step(self.model, self.data)
                # # self._set_sim_data(target_joints_state_np)
                # self.sync()
                # time.sleep(0.002)

            index += 1

        # 获取当前腕部相机图片，并将腕部相机位置保存在全局变量中
        # wrist_cam_image = self.get_image_pos_R_from_camera(640, 480, "wrist_cam")
        # cv2.imshow("MuJoCo Camera", wrist_cam_image)
        # # cv2.waitKey(1000)
        # cv2.destroyAllWindows()

        # ------------------step 3 : 二阶段计算平滑的抓取轨迹------------------
        # # 获取世界坐标系下的相机位姿
        # T_world_wri_cam = np.eye(4)
        # T_world_wri_cam[:3, :3] = self.camera_mat
        # T_world_wri_cam[:3, 3] = self.camera_pos
        #
        # self.wri_cam_quat_wxyz = self.rotation_matrix_to_quaternion(self.camera_mat)
        #
        # # 获取目标点在相机坐标系下的位姿
        # T_wri_cam_world = np.linalg.inv(T_world_wri_cam)
        #
        # T_wri_cam_obj = np.eye(4)
        T_world_obj = np.eye(4)
        T_world_obj[:3, :3] = Rotation.from_quat(self.target_quat_xyzw).as_matrix()
        T_world_obj[:3, 3] = self.target_position
        pos_world_obj = self.target_position

        # T_wri_cam_obj = T_wri_cam_world @ T_world_obj
        #
        # pos_cam_obj = T_wri_cam_obj[:3, 3]
        # pos_cam_obj_norm = pos_cam_obj / pos_cam_obj[2]
        #
        # # 获取相机内参矩阵
        # camera_matrix = self.calculate_camera_matrix(58, 480, 640)
        # # print("camera_center", camera_matrix[0][2], camera_matrix[1][2])
        #
        # # 获取像素坐标系下的目标点
        # pos_img_obj = np.zeros(3)
        # pos_img_obj = camera_matrix @ pos_cam_obj_norm

        # # 因为h是480对应y，w是640对应x，判断图像系下是否存在这个点
        # if (pos_img_obj[0] > 640 or pos_img_obj[0] < 0) or (pos_img_obj[1] > 480
        #                                                     or pos_img_obj[1] < 0):
        #     print("Can not found object, planning fail")
        #     return

        # cv2.imshow("MuJoCo Camera", wrist_cam_image)
        # cv2.waitKey(5000)
        # cv2.destroyAllWindows()

        # 直接在3D的相机坐标系下进行位置插值
        # 获取相机坐标系下末端执行器的位置与姿态
        T_6_ee = np.eye(4)
        T_world_link6 = np.eye(4)
        T_6_ee[:3, 3] = np.array([0, 0, 0.053], dtype=float)

        # arm_link6_name = "link6"
        # arm_link6_pos, arm_link6_quat = self._get_body_pose(arm_link6_name)
        # arm_link6_quat_xyzw = np.roll(arm_link6_quat, -1)
        # T_world_link6[:3, :3] = Rotation.from_quat(arm_link6_quat_xyzw).as_matrix()
        # T_world_link6[:3, 3] = arm_link6_pos
        # # T_wri_cam_ee = T_wri_cam_world @ T_world_link6 @ T_6_ee
        # T_world_ee = T_world_link6 @ T_6_ee
        #
        arm_base_name = "base_link"
        arm_base_pos, arm_base_quat = self._get_body_pose(arm_base_name)
        arm_base_quat_xyzw = np.roll(arm_base_quat, -1)
        # 计算world坐标系下baselink的变换矩阵
        T_world_base = np.eye(4)
        T_world_base[:3, :3] = Rotation.from_quat(arm_base_quat_xyzw).as_matrix()
        T_world_base[:3, 3] = arm_base_pos


        T_base_link6 = self.forward_kinematics_sub([theta_link1_obj, 0, -0.661, 0, 1.22, 0], 6)
        # R06 = T06[0:3, 0:3]
        T_world_link6 = T_world_base @ T_base_link6
        T_world_ee = T_world_link6 @ T_6_ee
        pos_world_ee = T_world_ee[:3, 3]
        # pos_wri_cam_ee = T_wri_cam_ee[:3, 3]

        # 针对不同目标点使用不同插值曲线
        if self.item_name == "apple":
            if self.rho <= 0.7 * np.sqrt(0.28745):
                t_target = 0.65
                # # 相机坐标系下控制点
                # pos_medium_ee_obj = (pos_wri_cam_ee + pos_cam_obj) / 2 + np.array([0, 0.2, 0])
                # 世界坐标系下控制点
                pos_world_ctrl = (pos_world_ee + pos_world_obj) / 2 + np.array([0, 0, 0.2])
            else:
                t_target = 0.8
                # # 相机坐标系下控制点
                # pos_medium_ee_obj = (pos_wri_cam_ee + pos_cam_obj) / 2 + np.array([0, 0.1, 0])
                # 世界坐标系下控制点
                pos_world_ctrl = (pos_world_ee + pos_world_obj) / 2 + np.array([0, 0, 0.1])

        # if self.item_name == "banana":
        #     if self.rho <= 0.7 * np.sqrt(0.28745):
        #         t_target = 1
        #
        #         # 控制点在3/4点
        #         pos_medium_ee_obj = (3 * pos_cam_obj + pos_wri_cam_ee) / 4 + np.array([0, 0.2, 0])
        #     else:
        #         # 目标点初始化在远处
        #         t_target = 1
        #
        #         # 控制点在3/4点
        #         pos_medium_ee_obj = (3 * pos_cam_obj + pos_wri_cam_ee) / 4 + np.array([0, 0.15, 0])

        # 对末端点在相机坐标系下pos_wri_cam_ee和目标位置pos_cam_obj进行插值
        # 插值比例 t 从 0 到 1
        t_values = np.linspace(0, 1, num=150)  # 生成10个点
        # 方案一： 位置线性插值
        # points_cam_ee2obj = np.array([
        #                 (1 - t) * pos_wri_cam_ee + t * pos_cam_obj
        #                 for t in t_values])
        # 方案二： 贝塞尔曲线插值
        # 相机坐标系下位置插值
        # points_cam_ee2obj = np.array([
        #     (1 - t) ** 2 * pos_wri_cam_ee + 2 * (1 - t) * t * pos_medium_ee_obj +
        #     t ** 2 * pos_cam_obj
        #     for t in t_values
        # ])
        # 世界坐标系下位置插值
        points_world_ee2obj = np.array([
            (1 - t) ** 2 * pos_world_ee + 2 * (1 - t) * t * pos_world_ctrl +
            t ** 2 * pos_world_obj
            for t in t_values
        ])
        #
        # # 世界系下贝塞尔曲线起点
        # pos_world_ee = (T_world_wri_cam @ np.array([pos_wri_cam_ee[0],
        #                 pos_wri_cam_ee[1], pos_wri_cam_ee[2], 1]))[:3]
        # # 世界系下贝塞尔曲线终点
        # pos_world_obj = (T_world_wri_cam @ np.array([pos_cam_obj[0],
        #                 pos_cam_obj[1], pos_cam_obj[2], 1]))[:3]
        # # 世界系下贝塞尔曲线控制点
        # pos_world_medium = (T_world_wri_cam @ np.array([pos_medium_ee_obj[0],
        #                 pos_medium_ee_obj[1], pos_medium_ee_obj[2], 1]))[:3]
        # 计算目标点切向量
        B_1_dot_target = (2 * (1 - t_target) * (pos_world_ctrl - pos_world_ee)
                        + 2 * t_target *(pos_world_obj - pos_world_ctrl))
        target_point_tangent = PI / 2 + atan2((- B_1_dot_target[2]),
                        sqrt(B_1_dot_target[0] ** 2 + B_1_dot_target[1] ** 2))

        # # 将相机坐标系下的插值点转换回世界坐标系下
        # points_world_ee2obj = np.array([
        #     T_world_wri_cam @ np.array([p[0], p[1], p[2], 1])
        #     for p in points_cam_ee2obj])

        # 二阶段计算抓取位姿
        arm_base_name = "base_link"
        arm_base_pos, arm_base_quat = self._get_body_pose(arm_base_name)
        arm_base_quat_xyzw = np.roll(arm_base_quat, -1)
        # 计算world坐标系下baselink的变换矩阵
        T_world_base = np.eye(4)
        T_world_base[:3, :3] = Rotation.from_quat(arm_base_quat_xyzw).as_matrix()
        T_world_base[:3, 3] = arm_base_pos

        # 求 T_world_base 的逆
        T_base_world = np.linalg.inv(T_world_base)

        T_world_obj = np.eye(4)
        T_world_obj_sta1 = np.eye(4)
        T_world_obj_sta2 = np.eye(4)

        # TODO:自定义抓取位姿
        # 重点在于抓取位姿随着一阶段的转动发生了改变
        # 绕z轴旋转theta_link1_obj
        T_world_obj_sta1[:3, :3] = np.array([[cos(theta_link1_obj), -sin(theta_link1_obj), 0],
                                             [sin(theta_link1_obj), cos(theta_link1_obj), 0],
                                             [0, 0, 1]]
                                            , dtype=float)
        # 再绕y轴转贝塞尔曲线的切线方向
        T_world_obj_sta2[:3, :3] = np.array([[cos(target_point_tangent), 0, sin(target_point_tangent)],
                                             [0, 1, 0],
                                             [-sin(target_point_tangent), 0, cos(target_point_tangent)]]
                                            , dtype=float)
        # option：给定抓取姿态
        # T_world_obj_sta2[:3, :3] = np.array([[cos(125 / 180 * PI), 0, sin(125 / 180 * PI)],
        #                                      [0, 1, 0],
        #                                      [-sin(125 / 180 * PI), 0, cos(125 / 180 * PI)]]
        #                                     , dtype=float)
        T_world_obj = T_world_obj_sta1 @ T_world_obj_sta2
        T_world_obj[:3, 3] = self.target_position
        # 将目标点的pose转换到base_link下
        T_base_obj = T_base_world @ T_world_obj

        # 姿态四元数插值
        # 获取世界坐标系下末端姿态四元数
        world_ee_quat = self.rotation_matrix_to_quaternion((T_world_link6 @ T_6_ee)[:3, :3])

        # 获取世界坐标系下抓取位姿四元数
        world_obj_quat = self.rotation_matrix_to_quaternion((T_world_obj)[:3, :3])

        # slerp插值
        quats_world_ee2obj = np.array([
            self.slerp(world_ee_quat, world_obj_quat, t)
            for t in t_values
        ])

        # TODO：最终抓取位置
        if self.item_name == "apple":
            drop_lens = 15
        # 香蕉容易滑动
        if self.item_name == "banana":
            drop_lens = 30

        for index_1 in range(len(quats_world_ee2obj) - drop_lens):

            # mujoco.mjv_initGeom(
            #     self.handle.user_scn.geoms[index_1],
            #     type=mujoco.mjtGeom.mjGEOM_SPHERE,
            #     size=[0.005, 0, 0],
            #     pos=np.array([points_world_ee2obj[index_1][0], points_world_ee2obj[index_1][1],
            #                   points_world_ee2obj[index_1][2]]),
            #     mat=np.eye(3).flatten(),
            #     rgba=np.array([1, 0, 0, 1])
            # )
            # self.handle.user_scn.ngeom += 1
            #
            # # self.set_goal_pose("target", points_world_ee2obj[index_1][:3],
            # #                        quats_world_ee2obj[index_1])
            # # 调试：world_obj_quat
            # self.set_goal_pose("target", points_world_ee2obj[index_1][:3],
            #                                           world_obj_quat)

            quat_world_ee2obj_xyzw = np.roll(quats_world_ee2obj[index_1], -1)
            # 计算world坐标系下插值点的变换矩阵
            T_world_ee2obj = np.eye(4)
            T_world_ee2obj[:3, :3] = Rotation.from_quat(quat_world_ee2obj_xyzw).as_matrix()
            T_world_ee2obj[:3, 3] = points_world_ee2obj[index_1][:3]
            T_base_ee2obj = T_base_world @ T_world_ee2obj
            target_joints_state = self.inverse_kinematics(T_base_ee2obj)
            # TODO: 如果关节超限
            if target_joints_state is None:
                print("at index:", index_1)
                continue
            # print("new index:", index_1)
            for i in range(5):
                target_joints_state_np = np.array([
                    [target_joints_state[0], target_joints_state[1], target_joints_state[2],
                     target_joints_state[3], target_joints_state[4], target_joints_state[5],
                     0.035]])
                path_total = np.vstack([path_total, target_joints_state_np])
                if index_1 > 75:
                    path_total = np.vstack([path_total, target_joints_state_np])
                    path_total = np.vstack([path_total, target_joints_state_np])
                # self.data.ctrl[:7] = target_joints_state_np
                # # # 调试：world_obj_quat
                # # self.set_goal_pose("target", points_world_ee2obj[index_1][:3],
                # #                    world_obj_quat)
                # mujoco.mj_step(self.model, self.data)
                # # self._set_sim_data(target_joints_state_np)
                # self.sync()
                # time.sleep(0.002)

        # TODO：到达目标位姿，夹爪闭合抓取
        for j in range(1000):
            target_joints_state_np = np.array([
                                               [target_joints_state[0],target_joints_state[1],target_joints_state[2]
                                               ,target_joints_state[3],target_joints_state[4],target_joints_state[5],
                                               - (0.035 / 1000) * j + 0.035]])
            # print(target_joints_state_np)
            path_total = np.vstack([path_total, target_joints_state_np])
            # self.data.ctrl[:7] = target_joints_state_np
            # mujoco.mj_step(self.model, self.data)
            # # self._set_sim_data(target_joints_state_np)
            # self.sync()
            # time.sleep(0.002)

        # TODO:抓取成功后，回放之前规划的位姿
        latest_points_world_ee2obj = points_world_ee2obj[:len(points_world_ee2obj) - drop_lens][::-1]
        latest_quats_world_ee2obj = quats_world_ee2obj[:len(quats_world_ee2obj) - drop_lens][::-1]

        for index_1 in range(len(latest_quats_world_ee2obj) - 40 ):
            # 清除可视化点
            self.handle.user_scn.ngeom  = 0
            latest_quat_world_ee2obj_xyzw = np.roll(latest_quats_world_ee2obj[index_1], -1)
            # 计算world坐标系下baselink的变换矩阵
            T_world_ee2obj = np.eye(4)
            T_world_ee2obj[:3, :3] = Rotation.from_quat(latest_quat_world_ee2obj_xyzw).as_matrix()
            T_world_ee2obj[:3, 3] = latest_points_world_ee2obj[index_1][:3]
            T_base_ee2obj = T_base_world @ T_world_ee2obj
            target_joints_state = self.inverse_kinematics(T_base_ee2obj)
            if target_joints_state is None:
                print("at index:", index_1)
                continue
            target_joints_state_np_sub = np.array(target_joints_state)


            # 苹果不容易滑动，只用range3
            if self.item_name == "apple":
                integration_time = 10
            # 香蕉容易滑动
            if self.item_name == "banana":
                integration_time = 25

            for i in range(integration_time):
                target_joints_state_np = np.array([
                    [target_joints_state[0], target_joints_state[1], target_joints_state[2],
                     target_joints_state[3], target_joints_state[4], target_joints_state[5],
                     0]])
                path_total = np.vstack([path_total, target_joints_state_np])
                # self.data.ctrl[:7] = target_joints_state_np
                #
                # mujoco.mj_step(self.model, self.data)
                # # self._set_sim_data(target_joints_state_np)
                # self.sync()
                # time.sleep(0.002)

        # TODO:调用RRT规划轨迹位姿，转回起始位姿最终理想关节角
        target_joints_state_sta_end = np.array([0, 1.19, -0.661, 0, 0.781, 0]
                                                    , dtype=float)

        for i in range(30):
            path = self.calc_arm_rrt_cubic_traj(target_joints_state_np_sub, target_joints_state_sta_end)
            if path is None:
                self.count = self.count + 1
                print(" planning path failed... ")
                continue
                # raise RuntimeError(" planning path failed... ")
            else:
                break

        index = 0
        while True:
            if index >= path.shape[1] - 1:
                # self.cur_episode_done = True
                break

            # 苹果不容易滑动，只用range3
            if self.item_name == "apple":
                integration_time = 3
            # 香蕉容易滑动
            if self.item_name == "banana":
                integration_time = 5

            for i in range(integration_time):
                target_joints_state = path[:6, index]
                target_joints_state_np = np.array([
                    [target_joints_state[0], target_joints_state[1], target_joints_state[2],
                     target_joints_state[3], target_joints_state[4], target_joints_state[5],
                     0]])
                path_total = np.vstack([path_total, target_joints_state_np])
                # self.data.ctrl[:7] = target_joints_state_np
                # mujoco.mj_step(self.model, self.data)
                # # self._set_sim_data(target_joints_state_np)
                # self.sync()
                # time.sleep(0.002)

            index += 1

        # TODO:张开夹爪
        for j in range(1000):
            if j % 2 == 0:
                continue
            target_joints_state_np = np.array([[target_joints_state_sta_end[0], target_joints_state_sta_end[1],
                                               target_joints_state_sta_end[2], target_joints_state_sta_end[3],
                                               target_joints_state_sta_end[4], target_joints_state_sta_end[5],
                                               (0.035 / 1000) * j]])
            path_total = np.vstack([path_total, target_joints_state_np])
            # self.data.ctrl[:7] = target_joints_state_np
            # mujoco.mj_step(self.model, self.data)
            # # self._set_sim_data(target_joints_state_np)
            # self.sync()
            # time.sleep(0.002)

        # TODO:等待1秒
        for j in range(500):
            target_joints_state_np = np.array([[target_joints_state_sta_end[0], target_joints_state_sta_end[1],
                                               target_joints_state_sta_end[2], target_joints_state_sta_end[3],
                                               target_joints_state_sta_end[4], target_joints_state_sta_end[5],
                                               0.035 ]])
            path_total = np.vstack([path_total, target_joints_state_np])
            # self.data.ctrl[:7] = target_joints_state_np
            # mujoco.mj_step(self.model, self.data)
            # # self._set_sim_data(target_joints_state_np)
            # self.sync()
            # time.sleep(0.002)

        # 遍历每一列并进行平滑处理
        for i in range(path_total.shape[1]):
            # 应用平滑处理
            path_total[:, i] = self.smooth_waveform(path_total[:, i])

        # 绘制所有平滑后的波形并保存为一个 PNG 文件
        plt.figure(figsize=(15, 8))  # 设置较大图像大小以便容纳所有波形
        for i in range(path_total.shape[1]):
            plt.plot(path_total[:, i], label=f'Smoothed Waveform {i + 1}')  # 绘制平滑后的波形

        plt.title('Smoothed Combined Waveforms')  # 设置合并后的标题
        plt.xlabel('Index')  # 设置 x 轴标签
        plt.ylabel('Value')  # 设置 y 轴标签
        plt.legend()  # 添加图例
        plt.grid(True)  # 添加网格
        plt.tight_layout()  # 自动调整子图参数，使之填充整个图表区域
        plt.savefig('smoothed_combined_waveforms.png')  # 保存为 PNG 文件
        plt.close()  # 关闭当前图像窗口


        # # 遍历每一列，绘制并保存为 PNG 文件
        # for i in range(path_total.shape[1]):
        #     plt.figure(figsize=(10, 4))  # 设置图像大小
        #     plt.plot(path_total[:, i], label=f'Waveform {i + 1}')  # 绘制第 i 列的波形
        #     plt.title(f'Waveform {i + 1}')  # 设置标题
        #     plt.xlabel('Index')  # 设置 x 轴标签
        #     plt.ylabel('Value')  # 设置 y 轴标签
        #     plt.legend()  # 添加图例
        #     plt.grid(True)  # 添加网格
        #     plt.savefig(f'waveform_{i + 1}.png')  # 保存为 PNG 文件
        #     plt.close()  # 关闭当前图像窗口
        #
        # print("所有波形已保存为 PNG 文件。")
        # 绘制所有波形并保存为一个 PNG 文件
        plt.figure(figsize=(15, 8))  # 设置较大图像大小以便容纳所有波形
        for i in range(path_total.shape[1]):
            plt.plot(path_total[:, i], label=f'Waveform {i + 1}')  # 绘制所有列的波形

        plt.title('Combined Waveforms')  # 设置合并后的标题
        plt.xlabel('Index')  # 设置 x 轴标签
        plt.ylabel('Value')  # 设置 y 轴标签
        plt.legend()  # 添加图例
        plt.grid(True)  # 添加网格
        plt.tight_layout()  # 自动调整子图参数，使之填充整个图表区域
        plt.savefig('combined_waveforms.png')  # 保存为 PNG 文件
        plt.close()  # 关闭当前图像窗口


        # 调试：回放拼接waypoint的
        for index in range(path_total.shape[0]):
            # if index % 16 == 0 :
            #     self.data.ctrl[:7] = path_total[index, :]
            self.data.ctrl[:7] = path_total[index, :]
            mujoco.mj_step(self.model, self.data)
            self.sync()
            time.sleep(0.002)
        #
        # time.sleep(300)
        # -------------------------------对比实验： RRT成功率-------------------------------
        # # step 4 : 调用IK求解关节角
        # target_joints_state = self.inverse_kinematics(T_base_obj)
        # target_joints_state_np = np.array(target_joints_state)
        # print("q_goal:", target_joints_state)
        #
        # if target_joints_state is None:
        #     self.count_ik = self.count_ik + 1
        #     return
        #
        # # 调用RRT规划轨迹位姿，从初始位置0到预定义位姿
        # path = self.calc_arm_rrt_cubic_traj(target_joints_state_sta1, target_joints_state_np)
        #
        # if path is None:
        #     self.count = self.count + 1
        #     return
        #     # raise RuntimeError(" planning path failed... ")
        #
        # index = 0
        #
        # while True:
        #     if index >= path.shape[1] - 1:
        #         # self.cur_episode_done = True
        #         break
        #
        #     self.data.qpos[:6] = path[:6, index]
        #     self.data.qpos[6] = 0.035
        #     self.data.qpos[7] = -0.035
        #
        #     mujoco.mj_step(self.model, self.data)
        #     self.sync()
        #     time.sleep(0.002)
        #
        #     index += 1  # 索引递增，准备下一帧


    def run_loop(self):
        self.cur_episode_done = False

        # 随机初始化目标位姿
        self.reset()

        # 求解规划位置的关节角,运动机械臂
        self.run_before()
        self.cur_episode_done = True
