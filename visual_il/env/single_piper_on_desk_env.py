from scipy.spatial.transform import Rotation as R
import copy
# import pygpg
from mujoco import MjModel, MjData, mjtObj
from termcolor import cprint
import threading
from base_env import BaseEnv
from easydict import EasyDict
import sys
import os
# 添加根目录到模块搜索路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.mujoco_viewer import BaseViewer


# class SingleArmEnv(BaseEnv):
class SingleArmEnv(BaseViewer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.path = cfg.path
        self.action = None
        self.count = 0


    def unnormalizar_gripper(self,gripper_action ):
        return gripper_action * 0.035


    # def run_func(self):
    #     if self.q_vec is None:
    #         raise ValueError(f" planning path does not exist... ")
    #
    #     # print(f"self.index : {self.index}")
    #     # print(f"self.q_vec.shape[1] : {self.q_vec.shape[1]}")
    #     # 控制
    #     # self.data.ctrl[:6] = self.q_vec[:6, self.index]
    #     self.data.qpos[:6] = self.q_vec
    #     self.count = self.count + 1
    #     if self.count > 10:
    #         self.index += 1
    #         self.count = 0
    #     # if self.index >= self.q_vec.shape[1] - 1:
    #     if self.index >= 1:
    #         self.cur_episode_done = True
    #         self.index = 0


def main():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.abspath(os.path.join(script_dir, '../..', 'model_assets', 'piper_on_desk', 'scene.xml'))

    cfg = EasyDict({
        "path": model_path,
        "is_have_arm": True,
        "episode_len": 20,
        "is_save_record_data": True,
        "camera_names": ["3rd_camera", "wrist_cam"],
        "env_name": "SingleArmEnv",
        "obj_list": ["desk","apple","banana"]
    })
    env = SingleArmEnv(cfg)

    # 记录数据
    for i in range(cfg["episode_len"]):
        print(f"iiii : {i}")
        env.run_loop()

    print("planning fail count:",env.count)
    print("ik fail count:", env.count_ik)

    env.mjstep_thread.join()


if __name__ == "__main__":
    main()
