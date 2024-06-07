from MRE import MREEnv
from TD3 import Agent
import numpy as np
env = MREEnv()
agent = Agent(state_size=1, action_size=1, f_size=1)
class verify:
    def verify_conclude(self, state):
        i = 0.0
        R1 = 0.01  # 磁流变弹性体内径
        R2 = 0.015  # 磁流变弹性体外径
        l = 0.01  # 磁流变弹性体高度
        S = 2 * np.pi * (R1 + R2) * l  # 磁流变弹性体剪切面积
        H = 0.005  # 磁流变弹性体厚度
        while i < 4:
            print('第' + str(i) + '次')
            # m1 = 15.0  # 主系统质量（单位：Kg）
            # m2 = 7.0
            # f1 = 5.8  # 主系统固有频率
            # k1 = 4 * np.pi ** 2 * f1 ** 2 * m1
            # F = 50
            m1 = 15.0 + np.random.uniform(-3, 3) # 主系统质量（单位：Kg）
            m2 = 7.0 + np.random.uniform(-1.4, 1.4)
            f1 = 5.8 + np.random.uniform(-1.16, 1.16) # 主系统固有频率
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
                G = (0.74 + action * 1.077) * 0.1 * 1000000  # 磁流变弹性体的弹性模量(单位：MPa）(1MPa=1N/mm²,1MPa=10^6 N/m²）
                k = (S / H) * G
                c2 = 100
                for f in f_ex:
                    omega = 2 * np.pi * f  # 外界激振角频率(单位：rad/s)
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
            with open(r'C:\Users\17702\Desktop\修稿仿真数据\半主动系统振动幅值{}.txt'.format(i), 'w') as f:
                pass
            np.savetxt(r'C:\Users\17702\Desktop\修稿仿真数据\半主动系统振动幅值{}.txt'.format(i), np.vstack([X1]).T, delimiter=',')
            with open(r'C:\Users\17702\Desktop\修稿仿真数据\半主动系统振动最小值{}.txt'.format(i), 'w') as f1:
                pass
            np.savetxt(r'C:\Users\17702\Desktop\修稿仿真数据\半主动系统振动最小值{}.txt'.format(i), np.vstack([X_min]).T, delimiter=',')

            with open(r'C:\Users\17702\Desktop\修稿仿真数据\电流变化{}.txt'.format(i), 'w') as f2:
                pass
            np.savetxt(r'C:\Users\17702\Desktop\修稿仿真数据\电流变化{}.txt'.format(i), np.vstack([Action]).T, delimiter=',')
            with open(r'C:\Users\17702\Desktop\修稿仿真数据\振动差值{}.txt'.format(i), 'w') as f3:
                pass
            np.savetxt(r'C:\Users\17702\Desktop\修稿仿真数据\振动差值{}.txt'.format(i), np.vstack([X_sub]).T, delimiter=',')
            i += 1