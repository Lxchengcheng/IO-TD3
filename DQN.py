from operator import itemgetter
import numpy as np
from scipy.stats import truncnorm
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
import random
Action = np.arange(0.0, 3.0, 0.01)
class Agent:
    def __init__(self, state_size, action_size, f_size, max_memory_size=10000, gamma=0.8, model_lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 0.95
        self.f_size = f_size
        self.max_memory_size = max_memory_size
        self.gamma = gamma
        self.max_current = 3.0
        self.index = 0
        self.model_lr = model_lr
        self.memory = []
        self.weight = []
        self.action = []
        self.model = self.build_model()
        self.target = self.build_model()
        self.model_optimizer = Adam(learning_rate=self.model_lr)

    def build_model(self):
        input_critic1 = Input(shape=(self.state_size,))
        input_critic2 = Input(shape=(251,))
        x1 = Dense(20, activation='relu',kernel_initializer='glorot_uniform')(input_critic1)
        x2 = Dense(20, activation='relu', kernel_initializer='glorot_uniform')(input_critic2)
        x = concatenate([x1,x2],axis=1)
        x = Dense(20, activation='relu', kernel_initializer='glorot_uniform')(x)
        x = Dropout(0.1)(x)
        x = Dense(20, activation='relu', kernel_initializer='glorot_uniform')(x)
        x = Dropout(0.1)(x)
        x = Dense(10, activation='relu', kernel_initializer='glorot_uniform')(x)
        x = Dense(8, activation='relu',kernel_initializer='glorot_uniform')(x)
        output_critic = Dense(300, activation='linear', kernel_initializer='glorot_uniform')(x)
        model_critic = Model(inputs=[input_critic1, input_critic2], outputs=output_critic)
        model_critic.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.model_lr))
        return model_critic

    def act(self, state, episode ,f_four):
        if np.random.rand() <= self.epsilon:
            Q = self.model([state, f_four])
            act_values_max = np.argmax(Q)
            action = Action[act_values_max]
        else:
            action = np.random.choice(Action)
        return action

    def remember(self, state, action, reward, next_state, f, done):
        if len(self.memory) < self.max_memory_size:
            self.memory.append((state, action, reward, next_state, f, done))
        else:
            #print('--------------''\n','remember=',episode,'\n''--------------')
            self.memory.pop(0)
            self.memory.append((state, action, reward, next_state, f, done))

    def replay(self, batch_size, step):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, f,  done in minibatch:
            target_q = self.target([next_state, f])
            qt_max = np.max(target_q)
            y = np.mean(reward + self.gamma * (1 - done) * qt_max)
            with tf.GradientTape() as tape:
                Qt = self.model([state, f])
                index = np.where(Action == action)
                Q_value = tf.gather(Qt, index,axis=1)
                value1 = 0.5*(y - Q_value)**2
                value1_mean = tf.reduce_mean(value1)
            modle_value = tape.gradient(value1_mean, self.model.trainable_variables)
            self.model_optimizer.apply_gradients(zip(modle_value, self.model.trainable_variables))
            if step % 3 == 0:
                self.target.set_weights(self.model.get_weights())