import gym
from gym import spaces
import numpy as np
import tensorflow as tf
class MREEnv(gym.Env):
    def __init__(self):
        self.f = None
        self.m1 = 15.0  # 主系统质量（单位：Kg）
        self.f1 = 5.8   #主系统固有频率
        self.k1 = 4 * np.pi ** 2 * self.f1 ** 2 * self.m1 # 主系统刚度（单位：N/m）
        self.m2 = 7.0  # 隔振器质量（单位：Kg）
        self.mu = self.m2 / self.m1
        self.F = 50.0  # 激振力幅值（单位：N）
        self.R1 = 0.01  # 磁流变弹性体内径
        self.R2 = 0.015  # 磁流变弹性体外径
        self.l = 0.01  # 磁流变弹性体高度
        self.S = 2 * np.pi * (self.R1 + self.R2) * self.l  # 磁流变弹性体剪切面积
        self.H = 0.005  # 磁流变弹性体厚度
        self.xi = 0.2
        self.action_space = spaces.Box(low=np.array([0.0]), high=np.array([3.0]))  # 电流范围
        self.state = None

    def reset(self):
        self.f_re = np.random.randint(10, 20, size=(1,)) #假设外界激励为不同频率的耦合-
        self.state = np.random.uniform(low=0.0, high=10.0, size=(1,)) #初始状态：主系统的位移、速度、加速度变化
        return self.state, self.f_re

    def fourie(self, f_re): #将激励频率进行傅里叶变换，分解出不同的正弦波供神经网络识别
        t = np.linspace(0, 1, 500, endpoint=False)
        x = 0.0
        #length = len(f_re)
        #for i in range(length):
        x = x + np.sin(2 * np.pi * f_re * t)
        x = tf.constant(x)
        # 将时域信号转换为频域信号
        fft_signal = tf.signal.rfft(x)
        # 将复数结果转换为幅度谱
        magnitude = tf.abs(fft_signal)
        # 将幅度谱归一化到[0, 1]范围
        magnitude = magnitude / tf.reduce_max(magnitude)
        input_data = tf.reshape(magnitude, (1, 251))
        return input_data

    def stp(self, action, f_re):    #接受电流值action和外界的激励频率
        self.x = 0.0
        self.f_ex = f_re
        action = np.clip(action, self.action_space.low, self.action_space.high) #规范电流值，将电流值限制到0~3
        self.G = (0.74 + action * 1.077) * 0.1 * 1000000 # 磁流变弹性体的弹性模量
        self.k = (self.S / self.H) * self.G #磁流变弹性体的刚度
        self.f2 = np.sqrt((self.G * self.S) / (self.m2 * self.H)) / (2 * np.pi) #磁流变弹性体的频率
        self.c2 = 100 #磁流变弹性体的阻尼
        for f in f_re: #将三个不同频率下对系统振幅响应值进行复数累加，得到总的激励响应
            self.omega = 2 * np.pi * f  # 外界激振角频率(单位：rad/s)
            self.z_ome = (self.k1 - self.m1 * self.omega ** 2) * (self.k - self.m2 * self.omega ** 2) - self.k * self.m2 * self.omega ** 2 + self.c2 * self.omega * (self.k1 - self.m1 * self.omega ** 2 - self.m2 * self.omega ** 2) * 1j
            self.x += (self.k - self.m2 * self.omega ** 2 + self.c2 * self.omega * 1j) * self.F / self.z_ome
        next_state = np.abs(self.x) * 1000 #下一状态
        reward = -np.sum(next_state) - np.sum(0.1 * np.abs(self.f_ex - self.f2)) #计算本回合经验
        done_min = self.calculate_min(self.f_ex) #计算振动的最小值
        done = False
        if np.abs(np.sum(next_state) - done_min) < 0.005: #计算振动幅值是否满足要求
            done = True
        self.state = next_state
        return self.state, reward, done, self.f_ex, self.f2

    def calculate_min(self, fre_cal): #计算震动的最小值
        min_list = []
        calcu_min = 0.0
        freq_start = 10
        freq_end = 20
        freq_step = 0.1  # 每间隔1Hz生成一组f_re
        f_range = np.arange(freq_start, freq_end + 1, freq_step)
        for f_start in f_range:
            x1 = 0.0
            f_re = fre_cal
            m2 = 7.0
            f2 = f_start
            k2 = 4 * np.pi ** 2 * f2 ** 2 * m2
            c2 = 100
            for f in f_re:
                omega = 2 * np.pi * f  # 外界激振角频率(单位：rad/s)
                z_ome = (self.k1 - self.m1 * omega ** 2) * (
                        k2 - m2 * omega ** 2) - k2 * m2 * omega ** 2 + c2 * omega * (
                                self.k1 - self.m1 * omega ** 2 - m2 * omega ** 2) * 1j
                x1 += (k2 - m2 * omega ** 2 + c2 * omega * 1j) * self.F / z_ome
            amplitude = abs(x1) * 1000
            min_list.append(amplitude)
            calcu_min = min(min_list)
        return calcu_min