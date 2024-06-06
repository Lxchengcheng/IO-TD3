from MRE2 import MREEnv
from Agentag import Agent
import numpy as np
from Immune import immune
import datetime
import matplotlib.pyplot as plt
env = MREEnv()
imm = immune()
agent = Agent(state_size=1, action_size=1, f_size=1)
episodes = 50 #训练回合数
n_clones = 10  #克隆数量
mutation_rate = 0.7  #变异率
mutation_range = 0.5  #变异范围
batch_size = 32  #小样本数目
max_steps = 100  #最大回合数
scores = []  #奖励数组
states1 = []  #状态数组
episode_times = []
action_dim = env.action_space.shape[0]
current_time = datetime.datetime.now()
start_time = current_time.timestamp()
for episode in range(episodes):
    current_time = datetime.datetime.now()
    episode_time = current_time.timestamp()
    episode_time = episode_time - start_time
    print("episode=", episode, "episode_time=", episode_time)
    score = 0.0 #初始化奖励
    states = 0.0 #初始化状态
    f_MRE = 0.0 #初始化MRE频率h
    f_ex = 0.0  #初始化外界激励频率
    state, f_re = env.reset() #初始化状态、外界激励频率0
    print('f_ex=', f_re)
    f_four = env.fourie(f_re)
    for step in range(max_steps): #开始训练神经网络
        action = agent.act(state, episode ,f_four) ##初始化动作
        next_state, reward, done, f_ex, f_MRE = env.stp(action,f_re) #获取下一状态值，当前奖励、外界激励频率和MRE的频率
        agent.remember(state, action, reward, next_state, f_four, f_re, done) #将上述值存入经验池供神经网络训练
        agent.replay(batch_size, step) #将经验供给神经网络进行学习
        state = next_state #替换当前状态
        score += reward #奖励累计
        if done: #判断本次回合是否训练完毕
            print(done)
            print('action=',np.array(action))
            print('step=',step)
            break
        if (step+1) % 10 == 0: #每10个step进行免疫训练
            clone, reward_ave, memory, bad = imm.clone_selection(n_clones)
            imm.mutation(clone, memory, bad, reward_ave=reward_ave, mutation_rate=mutation_rate, mutation_range=mutation_range)
        if step % 20 == 0:
            print('f_MRE=', f_MRE)
    scores.append(score)
    episode_times.append(episode_time)
    print("Episode:", episode + 1, "Score:", score, "f_ex:", f_ex, "f_MRE:", f_MRE, )
current_time = datetime.datetime.now()
end_time = current_time.timestamp()
training_time = end_time - start_time
print("训练所花费的时间为:", training_time)

plt.plot(scores)
plt.xlabel('Episode')
plt.ylabel('Score')
plt.show()

