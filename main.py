from MRE import MREEnv
from TD3 import Agent
#from DoubleQ import Agent
#from DDPG import Agent
#from DQN import Agent
import numpy as np
from Immune import immune
import datetime
import matplotlib.pyplot as plt
from Verify import verify
env = MREEnv()
imm = immune()
agent = Agent(state_size=1, action_size=1, f_size=1)
episodes = 50 
n_clones = 10  
mutation_rate = 0.7  
mutation_range = 0.5  
batch_size = 32  
max_steps = 100 
scores = []  
states1 = [] 
episode_times = []
action_dim = env.action_space.shape[0]
current_time = datetime.datetime.now()
start_time = current_time.timestamp()
for episode in range(episodes):
    current_time = datetime.datetime.now()
    episode_time = current_time.timestamp()
    episode_time = episode_time - start_time
    print("episode=", episode, "episode_time=", episode_time)
    score = 0.0
    states = 0.0 
    f_MRE = 0.0 
    f_ex = 0.0 
    state, f_re = env.reset() 
    print('f_ex=', f_re)
    f_four = env.fourie(f_re)
    for step in range(max_steps): 
        action = agent.act(state, episode ,f_four) 
        next_state, reward, done, f_ex, f_MRE = env.stp(action,f_re) 
        agent.remember(state, action, reward, next_state, f_four, f_re, done)
        agent.replay(batch_size, step)
        state = next_state 
        score += reward 
        if done: 
            print(done)
            print('action=',np.array(action))
            print('step=',step)
            break
        # if (step+1) % 10 == 0: 
        #     clone, reward_ave, memory, bad = imm.clone_selection(n_clones)
        #     imm.mutation(clone, memory, bad, reward_ave=reward_ave, mutation_rate=mutation_rate, mutation_range=mutation_range)
        if step % 20 == 0:
            print('f_MRE=', f_MRE)
    scores.append(score)
    episode_times.append(episode_time)
    print("Episode:", episode + 1, "Score:", score, "f_ex:", f_ex, "f_MRE:", f_MRE, )
current_time = datetime.datetime.now()
end_time = current_time.timestamp()
training_time = end_time - start_time
print("Time:", training_time)

state = np.random.uniform(low=0.0, high=10.0, size=(1,))
ver.verify_conclude(state)

plt.plot(scores)
plt.xlabel('Episode')
plt.ylabel('Score')
plt.show()

