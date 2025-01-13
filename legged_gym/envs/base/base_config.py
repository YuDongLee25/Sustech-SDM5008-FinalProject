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

import inspect  # 导入 inspect 模块，用于检查对象的类型等信息

class BaseConfig:
    def __init__(self) -> None:
        """ 初始化所有成员类，递归调用。忽略所有以'__'开头的名称（内置方法）。"""
        # 调用 init_member_classes 方法初始化实例成员
        self.init_member_classes(self)
    
    @staticmethod
    def init_member_classes(obj):
        # 遍历对象的所有属性名称
        for key in dir(obj):
            # 忽略内置属性（名称以'__'开头）
            # if key.startswith("__"):
            if key == "__class__":  # 特别地，跳过'__class__'属性
                continue
            # 获取对应的属性对象
            var = getattr(obj, key)
            # 检查属性是否是类
            if inspect.isclass(var):
                # 如果是类，实例化该类
                i_var = var()
                # 将属性设置为该类的实例而不是类型
                setattr(obj, key, i_var)
                # 递归地初始化该属性成员
                BaseConfig.init_member_classes(i_var)