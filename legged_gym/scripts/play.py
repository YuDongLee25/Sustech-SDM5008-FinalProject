# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR  # 导入 LEGGED_GYM_ROOT_DIR 变量（路径）
import os  # 导入 os 模块，提供与操作系统交互的功能

import isaacgym  # 导入 isaacgym 模块，Isaac Gym 提供的模拟环境
from legged_gym.envs import *  # 导入 legged_gym 环境中的所有环境类
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger  # 导入工具函数和类
from export_policy_as_onnx import *  # 导入导出策略为 ONNX 格式的相关函数

import numpy as np  # 导入 numpy，用于数值计算
import torch  # 导入 PyTorch，深度学习框架


def play(args):  # 定义训练函数
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)  # 根据任务名称获取环境配置和训练配置
    # 覆盖一些用于测试的参数
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)  # 设置环境数量为 1
    env_cfg.terrain.num_rows = 1  # 设置地形的行数
    env_cfg.terrain.num_cols = 1  # 设置地形的列数
    env_cfg.terrain.curriculum = False  # 禁用地形的课程学习（训练时逐步增加难度）
    env_cfg.noise.add_noise = False  # 禁用噪声
    env_cfg.domain_rand.randomize_friction = False  # 禁用摩擦力随机化
    env_cfg.domain_rand.push_robots = True  # 禁用推动机器人

    # 准备环境
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)  # 创建环境
    obs = env.get_observations()  # 获取环境的初始观测值
    # 加载策略
    train_cfg.runner.resume = True  # 设置恢复训练
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)  # 创建PPO算法训练器
    policy = ppo_runner.get_inference_policy(device=env.device)  # 获取推理策略

    # 导出策略为 JIT 模块（用于从 C++ 运行）
    if EXPORT_POLICY:  # 如果需要导出策略
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')  # 设置导出路径
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)  # 导出为 JIT 脚本
        print('Exported policy as jit script to: ', path)  # 打印导出路径
        export_policy_as_onnx(args)  # 导出为 ONNX 格式

    logger = Logger(env.dt)  # 创建日志记录器
    robot_index = 0  # 设置使用的机器人索引
    joint_index = 1  # 设置用于记录的关节索引
    stop_state_log = 100  # 设置停止记录状态的步数
    stop_rew_log = env.max_episode_length + 1  # 设置停止记录奖励的步数
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)  # 设置相机位置
    camera_vel = np.array([1., 1., 0.])  # 设置相机速度
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)  # 设置相机方向
    img_idx = 0  # 设置图片索引

    for i in range(10 * int(env.max_episode_length)):  # 进行多个训练步骤
        actions = policy(obs.detach())  # 使用策略根据当前观测值计算动作
        obs, _, rews, dones, infos = env.step(actions.detach())  # 在环境中执行动作并获取新的观测、奖励等信息
        if RECORD_FRAMES:  # 如果需要记录帧
            if i % 2:  # 每两步记录一帧
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")  # 设置文件保存路径
                env.gym.write_viewer_image_to_file(env.viewer, filename)  # 将当前视图保存为图片
                img_idx += 1  # 图片索引递增
        if MOVE_CAMERA:  # 如果需要移动相机
            camera_position += camera_vel * env.dt  # 更新相机位置
            env.set_camera(camera_position, camera_position + camera_direction)  # 设置相机位置和方向

        if i < stop_state_log:  # 如果步数小于停止记录状态的步数
            logger.log_states(  # 记录状态
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,  # 目标关节位置
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),  # 当前关节位置
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),  # 当前关节速度
                    'dof_torque': env.torques[robot_index, joint_index].item(),  # 当前关节扭矩
                    'command_x': env.commands[robot_index, 0].item(),  # 当前命令的 x 轴值
                    'command_y': env.commands[robot_index, 1].item(),  # 当前命令的 y 轴值
                    'command_yaw': env.commands[robot_index, 2].item(),  # 当前命令的偏航角
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),  # 基座的线速度 x
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),  # 基座的线速度 y
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),  # 基座的线速度 z
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),  # 基座的角速度
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()  # 机器人接触力
                }
            )
        elif i == stop_state_log:  # 如果步数等于停止记录状态的步数
            logger.plot_states()  # 绘制状态图
        if 0 < i < stop_rew_log:  # 如果步数在奖励记录范围内
            if infos["episode"]:  # 如果有 episode 信息
                num_episodes = torch.sum(env.reset_buf).item()  # 获取已重置的回合数
                if num_episodes > 0:  # 如果有回合被重置
                    logger.log_rewards(infos["episode"], num_episodes)  # 记录奖励
        elif i == stop_rew_log:  # 如果步数等于停止记录奖励的步数
            logger.print_rewards()  # 打印奖励

if __name__ == '__main__':
    EXPORT_POLICY = True  # 设置是否导出策略
    RECORD_FRAMES = False  # 设置是否记录帧
    MOVE_CAMERA = False  # 设置是否移动相机
    args = get_args()  # 获取命令行参数
    play(args)  # 执行训练
