import open3d as o3d
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
class DualArmViewer(BaseViewer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.path = cfg.path
        self.action = None
        self.count = 0


    def unnormalizar_gripper(self,gripper_action ):
        return gripper_action * 0.035

def main():
    """主函数示例"""
    # 配置参数
    cfg = EasyDict({
        "path": "/home/ubuntu/Documents/nn_coding/RulebasePlayground_nn/model_assets/mobile_ai_robot/scene.xml",
        "is_have_arm": True,
        "episode_len": 100,
        "is_save_record_data": False,
        "camera_names": ["3rd", "wrist_cam_left", "wrist_cam_right"],
    })
    
    # 创建环境
    dual_arm_env = DualArmViewer(cfg)
    
    try:
        for i in range(cfg["episode_len"]):
            # 运行前准备（包含环境重置和轨迹规划）
            if dual_arm_env.run_before():
                # 执行双臂协调运动
                dual_arm_env.run_loop()
            else:
                print("Failed to prepare for execution")
            
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error in main execution: {e}")
    finally:
        dual_arm_env.close()


if __name__ == "__main__":
    main()