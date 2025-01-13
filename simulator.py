# 导入所需的库和模块
import os  # 用于操作文件系统
import sys  # 用于访问与Python解释器密切相关的变量和函数
import time  # 用于时间相关的函数
import mujoco  # 用于MuJoCo仿真
import mujoco.viewer as viewer  # 用于MuJoCo的可视化
from functools import partial  # 用于部分函数应用
import limxsdk  # 导入limxsdk库
import limxsdk.robot.Rate as Rate  # 用于控制循环频率
import limxsdk.robot.Robot as Robot  # 用于机器人操作
import limxsdk.robot.RobotType as RobotType  # 用于定义机器人类型
import limxsdk.datatypes as datatypes  # 用于定义数据类型

# 定义SimulatorMujoco类，用于MuJoCo仿真
class SimulatorMujoco:
    def __init__(self, asset_path, joint_sensor_names, robot): 
        # 初始化机器人对象和关节传感器名称
        self.robot = robot
        self.joint_sensor_names = joint_sensor_names
        self.joint_num = len(joint_sensor_names)
        
        # 从指定的XML资产路径加载MuJoCo模型和数据
        self.mujoco_model = mujoco.MjModel.from_xml_path(asset_path)  # 加载MuJoCo模型
        self.mujoco_data = mujoco.MjData(self.mujoco_model)  # 初始化MuJoCo数据
        
        # 以被动模式启动MuJoCo查看器，并设置自定义设置
        self.viewer = viewer.launch_passive(self.mujoco_model, self.mujoco_data, key_callback=self.key_callback, show_left_ui=True, show_right_ui=True)
        self.viewer.cam.distance = 10  # 设置相机距离
        self.viewer.cam.elevation = -20  # 设置相机仰角
    
        # 获取仿真时间步长和计算帧率
        self.dt = self.mujoco_model.opt.timestep  # 获取仿真时间步长
        self.fps = 1 / self.dt  # 计算帧率

        # 使用默认值初始化机器人命令数据
        self.robot_cmd = datatypes.RobotCmd()
        self.robot_cmd.mode = [0. for x in range(0, self.joint_num)]  # 初始化模式列表
        self.robot_cmd.q = [0. for x in range(0, self.joint_num)]  # 初始化关节位置列表
        self.robot_cmd.dq = [0. for x in range(0, self.joint_num)]  # 初始化关节速度列表
        self.robot_cmd.tau = [0. for x in range(0, self.joint_num)]  # 初始化关节扭矩列表
        self.robot_cmd.Kp = [0. for x in range(0, self.joint_num)]  # 初始化比例增益列表
        self.robot_cmd.Kd = [0. for x in range(0, self.joint_num)]  # 初始化微分增益列表

        # 使用默认值初始化机器人状态数据
        self.robot_state = datatypes.RobotState()
        self.robot_state.tau = [0. for x in range(0, self.joint_num)]  # 初始化关节扭矩列表
        self.robot_state.q = [0. for x in range(0, self.joint_num)]  # 初始化关节位置列表
        self.robot_state.dq = [0. for x in range(0, self.joint_num)]  # 初始化关节速度列表

        # 初始化IMU数据结构
        self.imu_data = datatypes.ImuData()

        # 设置接收仿真模式下机器人命令的回调函数
        self.robotCmdCallbackPartial = partial(self.robotCmdCallback)  # 部分应用回调函数
        self.robot.subscribeRobotCmdForSim(self.robotCmdCallbackPartial)  # 订阅机器人命令

    # 接收机器人命令数据的回调函数
    def robotCmdCallback(self, robot_cmd: datatypes.RobotCmd):
        self.robot_cmd = robot_cmd  # 更新机器人命令数据

    # MuJoCo查看器中按键事件的回调函数（当前无操作）
    def key_callback(self, keycode):
        pass

    # 运行仿真
    def run(self):
        frame_count = 0
        self.rate = Rate(self.fps)  # 根据帧率设置更新频率
        while self.viewer.is_running():    
            # 执行MuJoCo物理仿真步骤
            mujoco.mj_step(self.mujoco_model, self.mujoco_data)

            # 从仿真中更新机器人状态数据
            for i in range(self.joint_num):
                self.robot_state.q[i] = self.mujoco_data.qpos[i + 7]  # 更新关节位置
                self.robot_state.dq[i] = self.mujoco_data.qvel[i + 6]  # 更新关节速度
                self.robot_state.tau[i] = self.mujoco_data.ctrl[i]  # 更新关节扭矩

                # 根据接收到的机器人命令数据向机器人应用控制命令
                self.mujoco_data.ctrl[i] = (self.robot_cmd.Kp[i] * (self.robot_cmd.q[i] - self.robot_state.q[i]) + 
                                            self.robot_cmd.Kd[i] * (self.robot_cmd.dq[i] - self.robot_state.dq[i]) + 
                                            self.robot_cmd.tau[i])  # 计算控制输入
        
            # 设置当前机器人状态的时间戳并发布
            self.robot_state.stamp = time.time_ns()  # 设置时间戳
            self.robot.publishRobotStateForSim(self.robot_state)  # 发布机器人状态

            # 从仿真中提取IMU数据（方向、陀螺仪和加速度）
            imu_quat_id = mujoco.mj_name2id(self.mujoco_model, mujoco.mjtObj.mjOBJ_SENSOR, "quat")  # 获取四元组传感器ID
            self.imu_data.quat[0] = self.mujoco_data.sensordata[self.mujoco_model.sensor_adr[imu_quat_id] + 0]  # 更新四元组数据
            self.imu_data.quat[1] = self.mujoco_data.sensordata[self.mujoco_model.sensor_adr[imu_quat_id] + 1]
            self.imu_data.quat[2] = self.mujoco_data.sensordata[self.mujoco_model.sensor_adr[imu_quat_id] + 2]
            self.imu_data.quat[3] = self.mujoco_data.sensordata[self.mujoco_model.sensor_adr[imu_quat_id] + 3]

            imu_gyro_id = mujoco.mj_name2id(self.mujoco_model, mujoco.mjtObj.mjOBJ_SENSOR, "gyro")  # 获取陀螺仪传感器ID
            self.imu_data.gyro[0] = self.mujoco_data.sensordata[self.mujoco_model.sensor_adr[imu_gyro_id] + 0]  # 更新陀螺仪数据
            self.imu_data.gyro[1] = self.mujoco_data.sensordata[self.mujoco_model.sensor_adr[imu_gyro_id] + 1]
            self.imu_data.gyro[2] = self.mujoco_data.sensordata[self.mujoco_model.sensor_adr[imu_gyro_id] + 2]

            imu_acc_id = mujoco.mj_name2id(self.mujoco_model, mujoco.mjtObj.mjOBJ_SENSOR, "acc")  # 获取加速度传感器ID
            self.imu_data.acc[0] = self.mujoco_data.sensordata[self.mujoco_model.sensor_adr[imu_acc_id] + 0]  # 更新加速度数据
            self.imu_data.acc[1] = self.mujoco_data.sensordata[self.mujoco_model.sensor_adr[imu_acc_id] + 1]
            self.imu_data.acc[2] = self.mujoco_data.sensordata[self.mujoco_model.sensor_adr[imu_acc_id] + 2]

            # 设置当前IMU数据的时间戳并发布
            self.imu_data.stamp = time.time_ns()  # 设置时间戳
            self.robot.publishImuDataForSim(self.imu_data)  # 发布IMU数据

            # 每20帧同步一次查看器，以获得更平滑的可视化效果
            if frame_count % 20 == 0:
                self.viewer.sync()

            frame_count += 1
            self.rate.sleep()  # 按照正确的速率维持仿真循环

# 程序的主入口
if __name__ == '__main__': 
    robot_type = os.getenv("ROBOT_TYPE")  # 获取环境变量中的机器人类型

    # 如果没有设置ROBOT_TYPE环境变量，则退出程序并显示错误信息
    if not robot_type:
        print("Error: Please set the ROBOT_TYPE using 'export ROBOT_TYPE=<robot_type>'.")
        sys.exit(1)

    # 创建指定类型的机器人实例
    robot = Robot(RobotType.PointFoot, True)

    # 使用指定的IP地址初始化机器人
    if not robot.init("127.0.0.1"):
        sys.exit()
    script_dir = os.path.dirname(os.path.abspath(__file__))  # 获取脚本所在目录

    # 根据机器人类型定义机器人模型XML文件的路径
    model_path = f'{script_dir}/robot_description/{robot_type}/xml/robot.xml'

    # 如果模型文件不存在，则退出程序并显示错误信息
    if not os.path.exists(model_path):
        print(f"Error: The file {model_path} does not exist. Please ensure the ROBOT_TYPE is set correctly.")
        sys.exit(1)

    print(f"*** Model File Loaded: robot_description/{robot_type}/xml/robot.xml ***")  # 打印模型文件加载信息

    # 根据机器人类型定义使用的关节传感器名称
    if robot_type.startswith("WF"):
        joint_sensor_names = [
            "abad_L_Joint", "hip_L_Joint", "knee_L_Joint", "wheel_L_Joint", "abad_R_Joint", "hip_R_Joint", "knee_R_Joint", "wheel_R_Joint"
        ]
    elif robot_type.startswith("SF"):
        joint_sensor_names = [
            "abad_L_Joint", "hip_L_Joint", "knee_L_Joint", "ankle_L_Joint", "abad_R_Joint", "hip_R_Joint", "knee_R_Joint", "ankle_R_Joint"
        ]
    else:
        joint_sensor_names = [
            "abad_L_Joint", "hip_L_Joint", "knee_L_Joint", "abad_R_Joint", "hip_R_Joint", "knee_R_Joint"
        ]

    # 创建并运行MuJoCo仿真实例
    simulator = SimulatorMujoco(model_path, joint_sensor_names, robot)
    simulator.run()  # 开始仿真