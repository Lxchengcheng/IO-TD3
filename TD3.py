from operator import itemgetter
import numpy as np
from scipy.stats import truncnorm
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
import random
class Agent:
    memory = []
    def __init__(self, state_size, action_size, f_size, max_memory_size=5000, gamma=0.8, tau=0.001, actor_lr=0.001, critic_lr=0.001): 
        self.action_size = action_size
        self.state_size = state_size
        self.f_size = f_size
        self.max_memory_size = max_memory_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.tau = tau
        self.max_current = 3.0
        self.index = 0
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.weight = []
        self.action = []
        self.actor = self.build_actor()
        self.target_actor = self.build_actor()
        self.critic1 = self.build_critic()
        self.critic2 = self.build_critic()
        self.target1_critic = self.build_critic()
        self.target2_critic = self.build_critic()
        self.actor_optimizer = Adam(learning_rate=self.actor_lr)
        self.critic_optimizer = Adam(learning_rate=self.critic_lr)
    def build_actor(self): 
        input_actor1 = Input(shape=(1,)) 
        input_actor2 = Input(shape=(251,)) 
        x1 = Dense(20, activation='relu',kernel_initializer='glorot_uniform')(input_actor1) 
        x2 = Dense(20, activation='relu', kernel_initializer='glorot_uniform')(input_actor2)
        x = concatenate([x1, x2], axis=1) 
        x = Dense(20, activation='relu',kernel_initializer='glorot_uniform')(x)
        x = Dropout(0.1)(x)
        x = Dense(20, activation='relu', kernel_initializer='glorot_uniform')(x)
        x = Dropout(0.1)(x)
        x = Dense(10, activation='relu',kernel_initializer='glorot_uniform')(x)
        x = Dense(8, activation='relu',kernel_initializer='glorot_uniform')(x)
        output_actor = Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform')(x)
        output_actor = tf.multiply(output_actor, tf.constant(self.max_current))
        model_actor = Model(inputs=[input_actor1,input_actor2], outputs=output_actor)
        model_actor.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.actor_lr))
        return model_actor
    def build_critic(self): 
        input_critic1 = Input(shape=(self.state_size,))
        input_critic2 = Input(shape=(251,))
        input_critic3 = Input(shape=(self.action_size,))
        x1 = Dense(20, activation='relu',kernel_initializer='glorot_uniform')(input_critic1)
        x2 = Dense(20, activation='relu', kernel_initializer='glorot_uniform')(input_critic2)
        x3 = Dense(20, activation='relu',kernel_initializer='glorot_uniform')(input_critic3)
        x = concatenate([x1,x2,x3],axis=1)
        x = Dense(20, activation='relu', kernel_initializer='glorot_uniform')(x)
        x = Dropout(0.1)(x)
        x = Dense(20, activation='relu', kernel_initializer='glorot_uniform')(x)
        x = Dropout(0.1)(x)
        x = Dense(10, activation='relu', kernel_initializer='glorot_uniform')(x)
        x = Dense(8, activation='relu',kernel_initializer='glorot_uniform')(x)
        output_critic = Dense(1, activation='linear', kernel_initializer='glorot_uniform')(x)
        model_critic = Model(inputs=[input_critic1, input_critic2, input_critic3], outputs=output_critic)
        model_critic.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.critic_lr))
        return model_critic
    def act(self, state, episode, f):
        scale = np.power(0.8, episode) 
        action = self.actor([state,f]) + truncnorm(-0.5, 0.5, loc=0,scale=scale).rvs() 
        return action
    def remember(self, state, action, reward, next_state, f_four, f_re, done): 
        if len(self.memory) < self.max_memory_size:
            self.memory.append((state, action, reward, next_state, f_four, f_re, done))
        else:
            print('--------------''\n',"remember over size,",'\n','--------------')
            self.memory.pop(0)
            self.memory.append((state, action, reward, next_state, f_four, f_re, done))
    def sample(self, batch_size): 
        fit = [getter[2] for getter in self.memory]
        fit = np.array(fit)
        priorities = np.exp(fit - np.max(fit))
        prob= priorities/np.sum(priorities)
        indices = np.random.choice(len(self.memory), batch_size, p=prob, replace=False)
        samples = itemgetter(*indices)(self.memory)
        return samples
    def replay(self, batch_size, step): 
        if len(self.memory) < batch_size:
            return
        #minibatch = self.sample(batch_size)
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, f, f1, done in minibatch:
            next_action = self.target_actor([next_state,f]) + truncnorm(-0.5, 0.5, loc=0).rvs()
            target_q1 = self.target1_critic([next_state, f, next_action])
            target_q2 = self.target2_critic([next_state, f, next_action])
            if np.sum(target_q1) > np.sum(target_q2):
                target_q = target_q2
            else:
                target_q = target_q1
            y = np.mean(reward + self.gamma * (1 - done) * target_q)
            with tf.GradientTape() as tape:
                value1 = 0.5*(y - self.critic1([state, f, action]))**2
            critic1_value = tape.gradient(value1, self.critic1.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(critic1_value, self.critic1.trainable_variables))
            with tf.GradientTape() as tape:
                value2 = 0.5*(y - self.critic2([state, f, action]))**2
            critic2_value = tape.gradient(value2, self.critic2.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(critic2_value, self.critic2.trainable_variables))
            if step % 3 == 0:
                with tf.GradientTape() as tape:
                    actor_action = self.actor([state,f])
                    critic_value = self.critic1([state, f, actor_action])
                    actor_loss = -tf.reduce_mean(critic_value)
                actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
                self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
                self.update_target_networks()
    def update_target_networks(self):
        actor_weights = self.actor.get_weights()
        target_actor_weights = self.target_actor.get_weights()
        critic1_weights = self.critic1.get_weights()
        target1_critic_weights = self.target1_critic.get_weights()
        critic2_weights = self.critic2.get_weights()
        target2_critic_weights = self.target2_critic.get_weights()
        for i in range(len(actor_weights)):
            target_actor_weights[i] = self.tau * actor_weights[i] + (1 - self.tau) * target_actor_weights[i]
        for i in range(len(critic1_weights)):
            target1_critic_weights[i] = self.tau * critic1_weights[i] + (1 - self.tau) * target1_critic_weights[i]
        for i in range(len(critic2_weights)):
            target2_critic_weights[i] = self.tau * critic2_weights[i] + (1 - self.tau) * target2_critic_weights[i]
        self.target_actor.set_weights(target_actor_weights)
        self.target1_critic.set_weights(target1_critic_weights)
        self.target2_critic.set_weights(target2_critic_weights)
