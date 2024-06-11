from MRE import MREEnv
from TD3 import Agent
import numpy as np
env = MREEnv()
agent = Agent(state_size=1, action_size=1, f_size=1)
class verify:
    def verify_conclude(self, state):
        i = 0.0
        R1 = 0.01 
        R2 = 0.015  
        l = 0.01  
        S = 2 * np.pi * (R1 + R2) * l 
        H = 0.005 
        while i < 4:
            print('Num:' + str(i))
            # m1 = 15.0 
            # m2 = 7.0
            # f1 = 5.8  
            # k1 = 4 * np.pi ** 2 * f1 ** 2 * m1
            # F = 50
            m1 = 15.0 + np.random.uniform(-3, 3)
            m2 = 7.0 + np.random.uniform(-1.4, 1.4)
            f1 = 5.8 + np.random.uniform(-1.16, 1.16) 
            k1 = 4 * np.pi ** 2 * f1 ** 2 * m1
            print('m1 = ', m1)
            print('k1 = ', k1)
            print('m2 = ', m2)
            j = 0.0
            x = 0.0
            state, _ = env.reset()
            X1 = []
            X_min = []
            X_sub = []
            Action = []
            while j < 1.0 :
                F = 50 + np.random.uniform(-10, 10)
                f_ex = np.random.uniform(10, 20, size=(1,))
                f_four = env.fourie(f_ex)
                action = agent.act(state, 200, f_four)
                G = (0.74 + action * 1.077) * 0.1 * 1000000  
                k = (S / H) * G
                c2 = 100
                for f in f_ex:
                    omega = 2 * np.pi * f  
                    z_ome = (k1 - m1 * omega ** 2) * (
                                k - m2 * omega ** 2) - k * m2 * omega ** 2 + c2 * omega * (
                                             k1 - m1 * omega ** 2 - m2 * omega ** 2) * 1j
                    x = (k - m2 * omega ** 2 + c2 * omega * 1j) * F / z_ome
                next_state = np.abs(x) * 1000
                x1 = np.sum(next_state)
                x_min = env.calculate_min(f_ex)
                x_sub = np.abs(x1 - x_min)
                X1.append(x1)
                X_min.append(x_min)
                X_sub.append(x_sub)
                action1 = np.sum(action)
                Action.append(action1)
                state = next_state
                j = j + 0.002
            i += 1
