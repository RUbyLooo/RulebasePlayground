# import os
# import time
# import threading
#

# def fun(n):
#     start = time.time()
#     my_thread_name = threading.current_thread().name  # 获取当前线程名称
#     print('%s开始运行...' % my_thread_name)
#     time.sleep(n)
#     my_thread_id = threading.current_thread().ident  # 获取当前线程id
#     print('当前线程为：{}，线程id为：{}，所在进程为：{}'.format(my_thread_name, my_thread_id, os.getpid()))
#     print('%s线程运行结束，耗时%ds...' % (my_thread_name, time.time() - start))
#
#
# t1 = time.time()
# # 创建3个线程
# for i in range(1, 4):
#     t = threading.Thread(target=fun, name='线程%s' % i, args=(i,))
#     t.start()
#
#
# main_thread_name = threading.current_thread().name  # 获取当前线程名称
# main_thread_id = threading.current_thread().ident  # 获取当前线程id
# print('主线程为：{}，线程id为：{}，所在进程为：{}'.format(main_thread_name, main_thread_id, os.getpid()))
#
# print("一共耗时%ds" % (time.time() - t1))
#
# # 线程1开始运行...
# # 线程2开始运行...
# # 线程3开始运行...
# # 主线程为：MainThread，线程id为：8637730304，所在进程为：19493
# # 一共耗时0s
# # 当前线程为：线程1，线程id为：13005955072，所在进程为：19493
# # 线程1线程运行结束，耗时1s...
# # 当前线程为：线程2，线程id为：13022744576，所在进程为：19493
# # 线程2线程运行结束，耗时2s...
# # 当前线程为：线程3，线程id为：13039534080，所在进程为：19493
# # 线程3线程运行结束，耗时3s...

# import threading
# import queue
# import time
#
# def producer(q):
#     for i in range(5):
#         print(f"生产者生产了产品{i}")
#         q.put(i)
#         time.sleep(1)
#     q.put(None)  # 结束信号
#
# def consumer(q):
#     while True:
#         item = q.get()
#         if item is None:
#             break
#         print(f"消费者消费了产品{item}")
#         q.task_done()
#
# q = queue.Queue()
# prod = threading.Thread(target=producer, args=(q,))
# cons = threading.Thread(target=consumer, args=(q,))
#
# prod.start()
# cons.start()
#
# prod.join()
# cons.join()
# print("生产消费结束")
#
# import threading
# import time
#
# class BaseClass:
#     def __init__(self):
#         self.planning_thread = threading.Thread(target=self.run_loop)
#
#     def run_loop(self):
#         for i in range(5):
#             print(f"Running loop iteration {i}")
#             time.sleep(1)
#
# class SubClass(BaseClass):
#     def __init__(self):
#         super().__init__()  # 调用父类的构造函数
#
#     def start_planning(self):
#         self.planning_thread.start()  # 启动线程
#
# # 示例运行
# if __name__ == "__main__":
#     sub_instance = SubClass()
#     sub_instance.start_planning()
#     time.sleep(6)  # 等待线程完成

# import threading
# import logging
# import time
# from dataclasses import asdict, dataclass
# from pathlib import Path
# from pprint import pformat
#
# # Include your existing imports here...
# # from lerobot.cameras import CameraConfig
# # from lerobot.configs import parser
# # from lerobot.utils.control_utils import init_keyboard_listener
# # from lerobot.utils.robot_utils import busy_wait
#
# # Your DatasetRecordConfig, RecordConfig, and other functions as already defined...
#
# def record_in_thread(cfg: RecordConfig):
#     """
#     Start the record function in a separate thread to run concurrently.
#     """
#     thread = threading.Thread(target=record, args=(cfg,))
#     thread.start()
#     return thread
#
#
# # Example usage
# if __name__ == "__main__":
#     # Assume you have a valid RecordConfig object
#     cfg = RecordConfig(
#         robot=your_robot_config,
#         dataset=your_dataset_config,
#         teleop=None,
#         policy=None,
#         display_data=True
#     )
#
#     # Start the recording function in a separate thread
#     record_thread = record_in_thread(cfg)
#
#     # Now you can do other things in the main thread, like monitoring or logging.
#     print("Recording started in a separate thread.")
#
#     # If you need to wait for the thread to finish (optional):
#     record_thread.join()
#
#     print("Recording completed.")
