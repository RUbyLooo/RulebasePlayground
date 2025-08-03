import mujoco
import numpy as np
import glfw
import cv2
# import mediapy as media
#
# resolution = (640, 480)
# # 创建OpenGL上下文（离屏渲染）
# glfw.init()
# glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
# window = glfw.create_window(resolution[0], resolution[1], "Offscreen", None, None)
# glfw.make_context_current(window)
#
# model = mujoco.MjModel.from_xml_path('/home/ubuntu/mujoco_il_rl/model_assets/piper_on_desk/scene.xml')
# data = mujoco.MjData(model)
# scene = mujoco.MjvScene(model, maxgeom=10000)
# context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
#
# # 设置相机参数
# camera_name = "wrist_camera"
# camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
# camera = mujoco.MjvCamera()
# # 使用固定相机
# camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
# # 设置相机为跟踪模式
# # camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
# if camera_id != -1:
#     print("camera_id", camera_id)
#     camera.fixedcamid = camera_id
#
# # 创建帧缓冲对象
# framebuffer = mujoco.MjrRect(0, 0, resolution[0], resolution[1])
# mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, context)
#
#
#
#
# while True:
#     mujoco.mj_step(model, data)
#     tracking_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "apple")
#     camera.trackbodyid = tracking_body_id
#     camera.distance = 1  # 相机与目标的距离
#     camera.azimuth = 0  # 水平方位角（度）
#     camera.elevation = -90  # 俯仰角（度）
#     viewport = mujoco.MjrRect(0, 0, resolution[0], resolution[1])
#     mujoco.mjv_updateScene(model, data, mujoco.MjvOption(),
#                            mujoco.MjvPerturb(), camera,
#                            mujoco.mjtCatBit.mjCAT_ALL, scene)
#     mujoco.mjr_render(viewport, scene, context)
#     rgb = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
#     mujoco.mjr_readPixels(rgb, None, viewport, context)
#     # 转换颜色空间 (OpenCV使用BGR格式)
#     bgr = cv2.cvtColor(np.flipud(rgb), cv2.COLOR_RGB2BGR)
#     cv2.imshow('MuJoCo Camera Output', bgr)
#     # cam_position, cam_quaternion = get_camera_pos_ori("wrist_camera")
#     # 获取相机 ID
#     camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
#     if camera_id == -1:
#         raise ValueError(f"未找到名为 '{camera_name}' 的相机")
#
#     # 位置：直接从相机数据中获取
#     cam_position = np.array(data.cam(camera_id).xpos)  # shape (3,)
#
#     # 方向：从相机矩阵转换为四元数
#     xmat = np.array(data.cam(camera_id).xmat)  # shape (9,)
#     cam_quaternion = np.zeros(4)
#     mujoco.mju_mat2Quat(cam_quaternion, xmat)  # [w, x, y, z]
#     print("Camera Position: ", cam_position)
#     print("Camera`s Quaternion: ", cam_quaternion)
#     if cv2.waitKey(1) == 27:
#         break
#
# cv2.imwrite('debug_output.png', bgr)
# cv2.destroyAllWindows()
# glfw.terminate()
# del context
# del scene

# xml = """
# <mujoco>
#   <worldbody>
#     <light name="top" pos="0 0 1"/>
#     <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
#     <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
#   </worldbody>
# </mujoco>
# """
# model = mujoco.MjModel.from_xml_string(xml)
model = mujoco.MjModel.from_xml_path("/home/ubuntu/mujoco_il_rl/model_assets/piper_on_desk/scene.xml")
data = mujoco.MjData(model)
height = 480
width = 640

# with mujoco.Renderer(model, height, width) as renderer:
#   mujoco.mj_forward(model, data)
#   renderer.update_scene(data, camera="wrist_cam")
#
#
#   media.show_image(renderer.render())
# with mujoco.Renderer(model, height, width) as renderer:
#   mujoco.mj_forward(model, data)
#   renderer.update_scene(data, camera="wrist_cam")
#   img = renderer.render()
#   cv_image = cv2.cvtColor(np.flipud(img), cv2.COLOR_RGB2BGR)
#   cv2.imshow('image', cv_image)
#   cv2.waitKey(0)
#   cv2.destroyAllWindows()
with mujoco.Renderer(model, height, width) as renderer:
  mujoco.mj_forward(model, data)
  renderer.update_scene(data, camera="wrist_cam")
  img = renderer.render()
  # 垂直翻转图像
  # img_flipped = np.flipud(img)
  # 转换为 BGR 格式
  cv_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
  # 显示图像
  cv2.imshow('image', cv_image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()