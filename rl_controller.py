import os  # 导入操作系统相关的模块
import sys  # 导入系统相关的模块
import copy  # 导入深拷贝模块
import numpy as np  # 导入NumPy库
import yaml  # 导入YAML解析库
import time  # 导入时间模块
import onnxruntime as ort  # 导入ONNX运行时库
from scipy.spatial.transform import Rotation as R  # 导入SciPy库中的旋转模块
from functools import partial  # 导入functools库中的partial函数
import limxsdk  # 导入limxsdk库
import limxsdk.robot.Rate as Rate  # 导入limxsdk库中的Rate模块
import limxsdk.robot.Robot as Robot  # 导入limxsdk库中的Robot模块
import limxsdk.robot.RobotType as RobotType  # 导入limxsdk库中的RobotType模块
import limxsdk.datatypes as datatypes  # 导入limxsdk库中的datatypes模块
import threading
from pynput import keyboard
import time

class PointfootController:
    def __init__(self, model_dir, robot, robot_type, start_controller):
        # 初始化机器人和类型信息
        self.robot = robot
        self.robot_type = robot_type
        self.is_point_foot = self.robot_type.startswith("PF")  # 判断是否为点足型机器人
        self.is_wheel_foot = self.robot_type.startswith("WF")  # 判断是否为轮足型机器人
        self.is_sole_foot = self.robot_type.startswith("SF")  # 判断是否为单足型机器人

        # 根据机器人类型加载配置和模型文件路径
        self.config_file = f'{model_dir}/{self.robot_type}/params.yaml'
        self.model_file = f'{model_dir}/{self.robot_type}/policy/policy.onnx'

        # 从YAML文件加载配置设置
        self.load_config(self.config_file)

        # 加载ONNX模型并设置输入和输出名称
        self.policy_session = ort.InferenceSession(self.model_file)  # 使用ONNX运行时加载模型
        self.policy_input_names = [self.policy_session.get_inputs()[0].name]  # 获取输入层名称
        self.policy_output_names = [self.policy_session.get_outputs()[0].name]  # 获取输出层名称

        # 准备机器人命令结构，设置默认值：mode，q，dq，tau，Kp，Kd
        self.robot_cmd = datatypes.RobotCmd()  # 创建机器人命令数据结构
        self.robot_cmd.mode = [0. for x in range(0, self.joint_num)]  # 设置模式为0
        self.robot_cmd.q = [0. for x in range(0, self.joint_num)]  # 设置关节角度为0
        self.robot_cmd.dq = [0. for x in range(0, self.joint_num)]  # 设置关节速度为0
        self.robot_cmd.tau = [0. for x in range(0, self.joint_num)]  # 设置关节扭矩为0
        self.robot_cmd.Kp = [self.control_cfg['stiffness'] for x in range(0, self.joint_num)]  # 设置关节刚度
        self.robot_cmd.Kd = [self.control_cfg['damping'] for x in range(0, self.joint_num)]  # 设置关节阻尼

        # 准备机器人状态结构
        self.robot_state = datatypes.RobotState()  # 创建机器人状态数据结构
        self.robot_state.tau = [0. for x in range(0, self.joint_num)]  # 设置关节扭矩为0
        self.robot_state.q = [0. for x in range(0, self.joint_num)]  # 设置关节角度为0
        self.robot_state.dq = [0. for x in range(0, self.joint_num)]  # 设置关节速度为0
        self.robot_state_tmp = copy.deepcopy(self.robot_state)  # 复制机器人状态数据结构

        # 初始化IMU（惯性测量单元）数据结构
        self.imu_data = datatypes.ImuData()  # 创建IMU数据结构
        self.imu_data.quat[0] = 0  # 设置IMU数据的四元数值
        self.imu_data.quat[1] = 0  # 设置IMU数据的四元数值
        self.imu_data.quat[2] = 0  # 设置IMU数据的四元数值
        self.imu_data.quat[3] = 1  # 设置IMU数据的四元数值
        self.imu_data_tmp = copy.deepcopy(self.imu_data)  # 复制IMU数据结构

        # 设置回调函数以接收更新后的机器人状态数据
        self.robot_state_callback_partial = partial(self.robot_state_callback)  # 机器人状态函数封装
        self.robot.subscribeRobotState(self.robot_state_callback_partial)  # 订阅机器人状态数据

        # 设置回调函数以接收更新后的IMU数据
        self.imu_data_callback_partial = partial(self.imu_data_callback)  # IMU数据函数封装
        self.robot.subscribeImuData(self.imu_data_callback_partial)  # 订阅IMU数据

        # 设置回调函数以接收更新后的传感器数据
        self.sensor_joy_callback_partial = partial(self.keyboard_control_callback)  # 手柄数据函数封装
        self.robot.subscribeSensorJoy(self.sensor_joy_callback_partial)  # 订阅传感器数据

        # 设置回调函数以接收诊断数据
        #self.robot_diagnostic_callback_partial = partial(self.robot_diagnostic_callback)  # 部分函数封装
        #self.robot.subscribeDiagnosticValue(self.robot_diagnostic_callback_partial)  # 订阅诊断数据

        # 初始化校准状态为-1，表示尚未完成校准
        self.calibration_state = -1

        # 启动控制器的标志
        self.start_controller = start_controller

    # 从YAML文件加载配置
    def load_config(self, config_file):
        with open(config_file, 'r') as f:  # 打开YAML配置文件
            config = yaml.safe_load(f)  # 解析YAML文件

        # 将配置参数赋值给控制器变量
        self.joint_names = config['PointfootCfg']['joint_names']  # 获取关节名称
        self.init_state = config['PointfootCfg']['init_state']['default_joint_angle']  # 获取默认关节角度
        self.stand_duration = config['PointfootCfg']['stand_mode']['stand_duration']  # 获取站立模式的持续时间
        self.control_cfg = config['PointfootCfg']['control']  # 获取控制配置
        self.rl_cfg = config['PointfootCfg']['normalization']  # 获取归一化配置
        self.obs_scales = config['PointfootCfg']['normalization']['obs_scales']  # 获取观察量的缩放比例
        self.actions_size = config['PointfootCfg']['size']['actions_size']  # 获取动作的大小
        self.observations_size = config['PointfootCfg']['size']['observations_size']  # 获取观察值的大小
        self.imu_orientation_offset = np.array(list(config['PointfootCfg']['imu_orientation_offset'].values()))  # 获取IMU的方向偏移
        self.user_cmd_cfg = config['PointfootCfg']['user_cmd_scales']  # 获取用户命令的缩放比例
        self.loop_frequency = config['PointfootCfg']['loop_frequency']  # 获取控制循环频率
        
        # 初始化动作、观察和命令的变量
        self.actions = np.zeros(self.actions_size)  # 初始化动作数组
        self.observations = np.zeros(self.observations_size)  # 初始化观察值数组
        self.last_actions = np.zeros(self.actions_size)  # 初始化上一次的动作数组
        self.commands = np.zeros(3)  # 初始化机器人命令数组（例如线速度、旋转）
        self.scaled_commands = np.zeros(3)  # 初始化缩放后的命令数组
        self.base_lin_vel = np.zeros(3)  # 初始化机器人基座线速度
        self.base_position = np.zeros(3)  # 初始化机器人基座位置
        self.loop_count = 0  # 初始化控制循环计数
        self.stand_percent = 0  # 初始化机器人站立模式的时间百分比
        self.policy_session = None  # 初始化ONNX模型会话
        self.joint_num = len(self.joint_names)  # 获取关节数量
        self.commands = np.zeros(3)  # 初始化命令数组

        if self.is_wheel_foot:  # 如果是轮足型机器人
          self.joint_pos_idxs = config['PointfootCfg']['size']['jointpos_idxs']  # 获取关节位置索引
          self.wheel_joint_damping = config['PointfootCfg']['control']['wheel_joint_damping']  # 获取轮子关节的阻尼值
          self.wheel_joint_torque_limit = config['PointfootCfg']['control']['wheel_joint_torque_limit']  # 获取轮子关节的扭矩限制

        # 初始化关节角度，根据初始配置设置
        self.init_joint_angles = np.zeros(len(self.joint_names))  # 初始化关节角度数组
        for i in range(len(self.joint_names)):  # 遍历每个关节
            self.init_joint_angles[i] = self.init_state[self.joint_names[i]]  # 设置初始关节角度
        
        # 设置初始模式为“STAND”
        self.mode = "STAND"  # 设置机器人初始模式为站立模式

    # Main control loop
    def run(self):
        # 等待直到控制器启动
        while not self.start_controller:
            time.sleep(1)

        # 初始化默认的站立姿势关节角度
        self.default_joint_angles = np.array([0.0] * len(self.joint_names))
        self.stand_percent += 1 / (self.stand_duration * self.loop_frequency)  # 逐步增加站立百分比
        self.mode = "STAND"  # 设置模式为站立
        self.loop_count = 0  # 初始化循环计数器

        # 根据配置中的频率设置循环速率
        rate = Rate(self.loop_frequency)
        while self.start_controller:
            self.update()  # 更新机器人状态
            rate.sleep()  # 以指定频率睡眠

        # 重置机器人命令的值，确保退出循环时安全停止
        self.robot_cmd.q = [0. for x in range(0, self.joint_num)]  # 重置关节位置
        self.robot_cmd.dq = [0. for x in range(0, self.joint_num)]  # 重置关节速度
        self.robot_cmd.tau = [0. for x in range(0, self.joint_num)]  # 重置关节力矩
        self.robot_cmd.Kp = [0. for x in range(0, self.joint_num)]  # 重置刚度
        self.robot_cmd.Kd = [1.0 for x in range(0, self.joint_num)]  # 设置阻尼为默认值
        self.robot.publishRobotCmd(self.robot_cmd)  # 发布机器人命令
        time.sleep(1)  # 等待1秒

    # Handle the stand mode for smoothly transitioning the robot into standing
    def handle_stand_mode(self):
        if self.stand_percent < 1:
            # 如果还没有完全站立，逐步插值关节角度
            for j in range(len(self.joint_names)):
                # 在站立模式下，关节角度在初始和默认值之间插值
                pos_des = self.default_joint_angles[j] * (1 - self.stand_percent) + self.init_state[self.joint_names[j]] * self.stand_percent
                self.set_joint_command(j, pos_des, 0, 0, self.control_cfg['stiffness'], self.control_cfg['damping'])
            # 随着时间的推移，逐渐增加站立百分比
            self.stand_percent += 1 / (self.stand_duration * self.loop_frequency)
        else:
            # 站立完成后切换到行走模式
            self.mode = "WALK"

    # Handle the walk mode where the robot moves based on computed actions
    def handle_walk_mode(self):
        # 更新临时机器人状态和IMU数据
        self.robot_state_tmp = copy.deepcopy(self.robot_state)
        self.imu_data_tmp = copy.deepcopy(self.imu_data)

        # 每隔'decimation'次迭代执行一次动作
        if self.loop_count % self.control_cfg['decimation'] == 0:
            self.compute_observation()  # 计算观察值
            self.compute_actions()  # 计算强化学习模型的动作
            # 限制动作在预设的范围内
            action_min = -self.rl_cfg['clip_scales']['clip_actions']
            action_max = self.rl_cfg['clip_scales']['clip_actions']
            self.actions = np.clip(self.actions, action_min, action_max)  # 将动作裁剪到指定范围

        # 遍历所有关节并设置命令
        joint_pos = np.array(self.robot_state_tmp.q)  # 获取关节位置
        joint_vel = np.array(self.robot_state_tmp.dq)  # 获取关节速度

        for i in range(len(joint_pos)):
            if self.is_point_foot or (i + 1) % 4 != 0:
                # 计算每个关节的动作限制
                action_min = (joint_pos[i] - self.init_joint_angles[i] +
                            (self.control_cfg['damping'] * joint_vel[i] - self.control_cfg['user_torque_limit']) /
                            self.control_cfg['stiffness'])
                action_max = (joint_pos[i] - self.init_joint_angles[i] +
                            (self.control_cfg['damping'] * joint_vel[i] + self.control_cfg['user_torque_limit']) /
                            self.control_cfg['stiffness'])

                # 限制动作在计算的最小和最大值之间
                self.actions[i] = max(action_min / self.control_cfg['action_scale_pos'],
                                    min(action_max / self.control_cfg['action_scale_pos'], self.actions[i]))

                # 计算目标关节位置并设置
                pos_des = self.actions[i] * self.control_cfg['action_scale_pos'] + self.init_joint_angles[i]
                self.set_joint_command(i, pos_des, 0, 0, self.control_cfg['stiffness'], self.control_cfg['damping'])

                # 保存上次的动作以备参考
                self.last_actions[i] = self.actions[i]
            elif self.is_wheel_foot:
                # 如果是轮式脚，计算轮式关节的动作限制
                action_min = joint_vel[i] - self.wheel_joint_torque_limit / self.wheel_joint_damping
                action_max = joint_vel[i] + self.wheel_joint_torque_limit / self.wheel_joint_damping
                self.last_actions[i] = self.actions[i]
                self.actions[i] = max(action_min / self.wheel_joint_damping,
                                    min(action_max / self.wheel_joint_damping, self.actions[i]))
                velocity_des = self.actions[i] * self.wheel_joint_damping
                self.set_joint_command(i, 0, velocity_des, 0, 0, self.wheel_joint_damping)

    # Compute the observation based on robot's state and IMU data
    def compute_observation(self):
        # 将IMU的四元数姿态转换为欧拉角（ZYX顺序）
        imu_orientation = np.array(self.imu_data_tmp.quat)
        q_wi = R.from_quat(imu_orientation).as_euler('zyx')  # 四元数转换为欧拉角
        inverse_rot = R.from_euler('zyx', q_wi).inv().as_matrix()  # 获取反向旋转矩阵

        # 将重力向量（指向下方）投影到身体坐标系中
        gravity_vector = np.array([0, 0, -1])  # 世界坐标系中的重力向量（z轴向下）
        projected_gravity = np.dot(inverse_rot, gravity_vector)  # 将重力向量转换到身体坐标系

        # 从IMU数据中获取基础角速度
        base_ang_vel = np.array(self.imu_data_tmp.gyro)
        # 应用IMU姿态的偏移修正（使用欧拉角）
        rot = R.from_euler('zyx', self.imu_orientation_offset).as_matrix()  # 偏移修正的旋转矩阵
        base_ang_vel = np.dot(rot, base_ang_vel)  # 将角速度应用偏移修正
        projected_gravity = np.dot(rot, projected_gravity)  # 将投影的重力向量应用偏移修正

        # 获取机器人的关节位置和速度
        joint_positions = np.array(self.robot_state_tmp.q)
        joint_velocities = np.array(self.robot_state_tmp.dq)

        # 获取上次应用的动作
        actions = np.array(self.last_actions)

        # 创建一个命令缩放矩阵，用于线性速度和角速度
        command_scaler = np.diag([
            self.user_cmd_cfg['lin_vel_x'],  # x方向的线性速度缩放因子
            self.user_cmd_cfg['lin_vel_y'],  # y方向的线性速度缩放因子
            self.user_cmd_cfg['ang_vel_yaw']  # 偏航角（角速度）缩放因子
        ])

        # 将命令缩放应用到输入的速度命令
        scaled_commands = np.dot(command_scaler, self.commands)

        # 填充观察值向量
        joint_pos_value = (joint_positions - self.init_joint_angles) * self.obs_scales['dof_pos']

        # 在轮式脚的情况下，关节位置不包括轮速，需要去掉某些索引
        if self.is_wheel_foot:
            joint_pos_input = np.array([joint_pos_value[idx] for idx in self.joint_pos_idxs])
        else:
            joint_pos_input = joint_pos_value

        # 创建包含各种状态变量的观察值向量：
        # - 基础角速度（缩放后）
        # - 投影的重力向量
        # - 关节位置（与初始角度的差异，已缩放）
        # - 关节速度（已缩放）
        # - 上次应用的动作
        # - 缩放后的命令输入
        obs = np.concatenate([
            base_ang_vel * self.obs_scales['ang_vel'],  # 缩放后的基础角速度
            projected_gravity,  # 身体坐标系中的投影重力向量
            joint_pos_input,  # 缩放后的关节位置
            joint_velocities * self.obs_scales['dof_vel'],  # 缩放后的关节速度
            actions,  # 上次应用的动作
            scaled_commands  # 缩放后的用户输入的速度命令
        ])
        
        # 将观察值裁剪到指定的范围内，确保稳定性
        self.observations = np.clip(
            obs, 
            -self.rl_cfg['clip_scales']['clip_observations'],  # 裁剪的下限
            self.rl_cfg['clip_scales']['clip_observations']  # 裁剪的上限
        )
    
    def compute_actions(self):
        """
        基于当前观察结果计算动作。
        """
        # 将观察结果拼接为一个张量，并转换为float32类型
        input_tensor = np.concatenate([self.observations], axis=0)
        input_tensor = input_tensor.astype(np.float32)
        
        # 创建一个字典，包含传入策略会话的输入数据
        inputs = {self.policy_input_names[0]: input_tensor}
        
        # 运行策略会话，获取输出
        output = self.policy_session.run(self.policy_output_names, inputs)
        
        # 将输出展平并存储为动作
        self.actions = np.array(output).flatten()

    def set_joint_command(self, joint_index, q, dq, tau, kp, kd):
        """
        发送命令来配置特定关节的状态。
        该方法更新关节的期望位置、速度、扭矩以及控制增益。
        请将此实现替换为与硬件的实际通信逻辑。

        参数：
        joint_index (int): 要控制的关节索引。
        q (float): 期望关节位置，通常为弧度或角度。
        dq (float): 期望关节速度，通常为弧度/秒或角度/秒。
        tau (float): 期望关节扭矩，通常为牛顿·米（Nm）。
        kp (float): 位置控制的比例增益。
        kd (float): 速度控制的微分增益。
        """
        self.robot_cmd.q[joint_index] = q
        self.robot_cmd.dq[joint_index] = dq
        self.robot_cmd.tau[joint_index] = tau
        self.robot_cmd.Kp[joint_index] = kp
        self.robot_cmd.Kd[joint_index] = kd

    def update(self):
        """
        根据当前模式更新机器人的状态并发布机器人命令。
        """
        if self.mode == "STAND":
            self.handle_stand_mode()  # 处理站立模式
        elif self.mode == "WALK":
            self.handle_walk_mode()  # 处理行走模式
        
        # 增加循环计数
        self.loop_count += 1

        # 发布机器人命令
        self.robot.publishRobotCmd(self.robot_cmd)

    # 回调函数，用于接收机器人命令数据
    def robot_state_callback(self, robot_state: datatypes.RobotState):
        """
        回调函数，通过接收的最新机器人状态数据更新机器人状态。
        
        参数：
        robot_state (datatypes.RobotState): 当前机器人的状态。
        """
        self.robot_state = robot_state

    # 回调函数，用于接收IMU数据
    def imu_data_callback(self, imu_data: datatypes.ImuData):
        """
        回调函数,通过接收的IMU数据更新IMU信息。
        
        参数：
        imu_data (datatypes.ImuData): 包含时间戳、加速度、陀螺仪和四元数的IMU数据。
        """
        self.imu_data.stamp = imu_data.stamp
        self.imu_data.acc = imu_data.acc
        self.imu_data.gyro = imu_data.gyro
        
        # 旋转四元数值
        self.imu_data.quat[0] = imu_data.quat[1]
        self.imu_data.quat[1] = imu_data.quat[2]
        self.imu_data.quat[2] = imu_data.quat[3]
        self.imu_data.quat[3] = imu_data.quat[0]

    # 回调函数，用于接收传感器手柄数据
    def sensor_joy_callback(self, sensor_joy: datatypes.SensorJoy):
        # 检查机器人是否处于校准状态，并且L1按钮（按钮索引为4）和Y按钮（按钮索引为3）被按下
        if not self.start_controller and self.calibration_state == 0 and sensor_joy.buttons[4] == 1 and sensor_joy.buttons[3] == 1:
            print(f"L1 + Y: start_controller...")  # 按下L1 + Y按钮，启动控制器
            self.start_controller = True

        # 检查是否按下L1（按钮索引为4）和X（按钮索引为2）按钮来停止控制器
        if self.start_controller and sensor_joy.buttons[4] == 1 and sensor_joy.buttons[2] == 1:
            print(f"L1 + X: stop_controller...")  # 按下L1 + X按钮，停止控制器
            self.start_controller = False

        # 获取手柄的线性和角速度输入
        linear_x  = sensor_joy.axes[1]
        linear_y  = sensor_joy.axes[0]
        angular_z = sensor_joy.axes[2]

        # 限制线性和角速度的范围
        linear_x  = 1.0 if linear_x > 1.0 else (-1.0 if linear_x < -1.0 else linear_x)
        linear_y  = 1.0 if linear_y > 1.0 else (-1.0 if linear_y < -1.0 else linear_y)
        angular_z = 1.0 if angular_z > 1.0 else (-1.0 if angular_z < -1.0 else angular_z)

        # 更新控制命令
        self.commands[0] = linear_x * 0.5
        self.commands[1] = linear_y * 0.5
        self.commands[2] = angular_z * 0.5
    
    # 键盘输入读取
    def keyboard_control_callback(self):
   
        def on_press(key):
        # 检查按键并更新命令
            if key == keyboard.Key.f1:
                self.commands[0] = min(1.0, self.commands[0] + 0.1)
                print(f"linear x value add {self.commands[0]}")
            elif key == keyboard.Key.f2:
                self.commands[0] = max(-1.0, self.commands[0] - 0.1)
                print(f"linear x value sub {self.commands[0]}")
            elif key == keyboard.Key.f3:
                self.commands[1] = min(1.0, self.commands[1] + 0.1)
                print(f"linear y value add {self.commands[1]}")
            elif key == keyboard.Key.f4:
                self.commands[1] = max(-1.0, self.commands[1] - 0.1)
                print(f"linear y value sub {self.commands[1]}")
            elif key == keyboard.Key.space:
            # 重置命令
                self.commands = np.zeros(3)

        # 启动键盘监听
        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        print("keyboard controller start.")

    # Callback function for receiving diagnostic data
    def robot_diagnostic_callback(self, diagnostic_value: datatypes.DiagnosticValue):
      # Check if the received diagnostic data is related to calibration.
      if diagnostic_value.name == "calibration":
        print(f"Calibration state: {diagnostic_value.code}")
        self.calibration_state = diagnostic_value.code

if __name__ == '__main__':
    # 从环境变量获取机器人类型
    robot_type = os.getenv("ROBOT_TYPE")
    
    # 检查是否设置了ROBOT_TYPE环境变量，否则退出并报错
    if not robot_type:
        print("Error: Please set the ROBOT_TYPE using 'export ROBOT_TYPE=<robot_type>'.")
        sys.exit(1)

    # 创建指定类型的机器人实例
    robot = Robot(RobotType.PointFoot)

    # 默认的机器人IP地址
    robot_ip = "127.0.0.1"
    
    # 如果命令行参数提供了机器人IP，则使用命令行参数中的IP
    if len(sys.argv) > 1:
        robot_ip = sys.argv[1]

    # 使用提供的IP地址初始化机器人
    if not robot.init(robot_ip):
        sys.exit()

    # 仿真模式下运行
    start_controller = robot_ip == "127.0.0.1"

    # 创建并运行PointfootController
    controller = PointfootController(f'{os.path.dirname(os.path.abspath(__file__))}/policy', robot, robot_type, start_controller)
    threading.Thread(target=controller.keyboard_control_callback).start()
    threading.Thread(target=controller.run).start()
