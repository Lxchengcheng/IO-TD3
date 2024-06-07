from operator import itemgetter
import numpy as np
from Agentag import Agent
from MRE2 import MREEnv
env = MREEnv()
class immune:
    def clone_selection(self, n_clones):
        bad = []
        clones = []
        memory = []
        index = np.random.choice(len(Agent.memory), 5, replace=False)
        memory1 = itemgetter(*index)(Agent.memory)
        affinity = [getter[2] for getter in memory1]
        action = [getter[1] for getter in memory1]
        reward_ave = np.mean(affinity)
        index = np.array(index)
        index.sort()
        for i in reversed(index):  
            Agent.memory.pop(i)
        for i in range(len(affinity)):
            if affinity[i] >= reward_ave:
                for j in range(n_clones):
                    clones.append(action[i])
                    memory.append(memory1[i])
            else:
                bad.append(memory1[i])
        if len(bad) > 1:
            weight = [getter[2] for getter in bad]
            indies = np.argmin(weight)
            bad.pop(indies)
        return clones, reward_ave, memory, bad
    
    def mutation(self, clones, memory, bad, reward_ave, mutation_rate, mutation_range):
        n_mutation = 1
        max_reward = reward_ave
        a = 0
        great = []
        immune_pool = []
        for i in range(n_mutation):
            for j in range(len(clones)):
                if np.random.rand() < mutation_rate:
                    clones[j] = clones[j] + np.random.uniform(-mutation_range, mutation_range, size=clones[i].shape)
                    state_m, _, _, _, _, f1_m, _ = memory[j]
                    next_state, reward, done, _, _ = env.stp(clones[j], f1_m)
                    if reward >= max_reward:
                        max_reward = reward
                        a = j
                if (j + 1) % 10 == 0:
                    action = clones[a]
                    state_m, _, _, _, f, f1_m, _ = memory[a]
                    next_state, reward, done, _, _ = env.stp(clones[a], f1_m)
                    great.append((state_m, action, reward, next_state, f, f1_m, done))
                    max_reward = reward_ave
        immune_pool.extend(great)
        immune_pool.extend(bad)
        memory.extend(immune_pool)
