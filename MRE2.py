import gym
from gym import spaces
import numpy as np
import tensorflow as tf
class MREEnv(gym.Env):
    def __init__(self):
        self.f = None
        self.m1 = 15.0 
        self.f1 = 5.8   
        self.k1 = 4 * np.pi ** 2 * self.f1 ** 2 * self.m1 
        self.m2 = 7.0  
        self.mu = self.m2 / self.m1
        self.F = 50.0 
        self.R1 = 0.01  
        self.R2 = 0.015 
        self.l = 0.01  
        self.S = 2 * np.pi * (self.R1 + self.R2) * self.l  
        self.H = 0.005  
        self.xi = 0.2
        self.action_space = spaces.Box(low=np.array([0.0]), high=np.array([3.0])) 
        self.state = None

    def reset(self):
        self.f_re = np.random.uniform(10, 20, size=(1,)) 
        self.state = np.random.uniform(low=0.0, high=10.0, size=(1,)) 
        return self.state, self.f_re

    def fourie(self, f_re):
        t = np.linspace(0, 1, 500, endpoint=False)
        x = 0.0
        x = x + np.sin(2 * np.pi * f_re * t)
        x = tf.constant(x)
        fft_signal = tf.signal.rfft(x)
        magnitude = tf.abs(fft_signal)
        magnitude = magnitude / tf.reduce_max(magnitude)
        input_data = tf.reshape(magnitude, (1, 251))
        return input_data

    def stp(self, action, f_re):    
        self.x = 0.0
        self.f_ex = f_re
        action = np.clip(action, self.action_space.low, self.action_space.high) 
        self.G = (0.74 + action * 1.077) * 0.1 * 1000000
        self.k = (self.S / self.H) * self.G 
        self.f2 = np.sqrt((self.G * self.S) / (self.m2 * self.H)) / (2 * np.pi) 
        self.c2 = 100 
        for f in f_re: 
            self.omega = 2 * np.pi * f  
            self.z_ome = (self.k1 - self.m1 * self.omega ** 2) * (self.k - self.m2 * self.omega ** 2) - self.k * self.m2 * self.omega ** 2 + self.c2 * self.omega * (self.k1 - self.m1 * self.omega ** 2 - self.m2 * self.omega ** 2) * 1j
            self.x += (self.k - self.m2 * self.omega ** 2 + self.c2 * self.omega * 1j) * self.F / self.z_ome
        next_state = np.abs(self.x) * 1000 
        reward = -np.sum(next_state) - np.sum(0.1 * np.abs(self.f_ex - self.f2)) 
        done_min = self.calculate_min(self.f_ex) 
        done = False
        if np.abs(np.sum(next_state) - done_min) < 0.005: 
            done = True
        self.state = next_state
        return self.state, reward, done, self.f_ex, self.f2

    def calculate_min(self, fre_cal):
        min_list = []
        calcu_min = 0.0
        freq_start = 10
        freq_end = 20
        freq_step = 0.1  
        f_range = np.arange(freq_start, freq_end + 1, freq_step)
        for f_start in f_range:
            x1 = 0.0
            f_re = fre_cal
            m2 = 7.0
            f2 = f_start
            k2 = 4 * np.pi ** 2 * f2 ** 2 * m2
            c2 = 100
            for f in f_re:
                omega = 2 * np.pi * f 
                z_ome = (self.k1 - self.m1 * omega ** 2) * (
                        k2 - m2 * omega ** 2) - k2 * m2 * omega ** 2 + c2 * omega * (
                                self.k1 - self.m1 * omega ** 2 - m2 * omega ** 2) * 1j
                x1 += (k2 - m2 * omega ** 2 + c2 * omega * 1j) * self.F / z_ome
            amplitude = abs(x1) * 1000
            min_list.append(amplitude)
            calcu_min = min(min_list)
        return calcu_min
