import random
import numpy as np
from collections import deque
from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizers import adam_v2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.python.client import device_lib
import pandas as pd # trust
import matplotlib.pyplot as plt
print(device_lib.list_local_devices())

class DQLAgent:
    def __init__(self, state_space, action_space, gamma=0.99, max_steps=1000):
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_decay = 0.998
        self.epsilon_min = 0.01
        self.beta = 30 # Constant in c^x, which defines discount reward for larger steps
        self.tot_reward = []
        self.batch_size = 64
        self.max_steps = max_steps
        self.memory = deque(maxlen=500000)
        self.osn = state_space
        self.an = action_space
        self.opt = adam_v2.Adam(learning_rate=0.001)
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        # Model maps from state space (continuous inputs, based on environment) to action space
        model.add(Dense(64, input_dim=self.osn, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.an, activation='softmax')) # actions: forward, left, right, back, rot clock, rot counter clock
        model.compile(loss='mse', optimizer=self.opt)
        return model

    def get_action(self, state):
        if random.random() <= self.epsilon:
            return random.randint(0, self.an - 1)
        # predict based on current environment inputs
        action = self.model.predict(state, verbose=0)
        return np.argmax(action[0])

    def memorize(self, state, action, reward, next_state, done, step):
        self.memory.append((state, action, reward, next_state, done, step))


    def replay_batch(self):
        # Choose a random sample from the memory to fit for this batch
        batch = random.sample(self.memory, self.batch_size)

        state = np.squeeze(np.array([i[0] for i in batch]))
        action = np.array([i[1] for i in batch])
        reward = np.array([i[2] for i in batch])
        next_state = np.squeeze(np.array([i[3] for i in batch]))
        done = np.array([i[4] for i in batch])
        step_discount = np.array([self.exponent_discount(i[5]) for i in batch])
        
        q_val = reward + self.gamma * np.amax(self.model.predict_on_batch(next_state), \
                                    axis=1) * (1 - done)
        target = self.model.predict_on_batch(state)
        idx = np.arange(self.batch_size)
        target[[idx], [action]] = q_val

        self.model.fit(state, target, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def step_discount(self, step):
        if step > 3 * self.max_steps / 4:
            return -.05 * step
        else:
            return 0

    def exponent_discount(self, step):
        return 1 + (-1/(.987 ** step))

    def save_model(self, path):
        self.model.save(path)
        print(self.tot_reward)

    def add_to_total_rewards(self, run_reward):
        self.tot_reward.append(run_reward)

class ImitateAgent:
    def __init__(self):
        pass

    def load_data(self, filename, labels):
        data_path = os.path.join("data", filename + ".csv")
        train = pd.read_csv(data_path)
        self.features = train.copy()
        self.an = len(labels)
        self.labels = pd.concat([self.features.pop(label) for label in labels], axis=1, join='inner')
        self.features = np.array(self.features)
        self.labels = np.array(self.labels)
    
    def build_model(self):
        # TODO: figure out how to normalize
        # normalize_layer = Normalization()
        # normalize_layer.adapt(self.features)
        model = Sequential([
                Dense(16, activation='relu'),
                Dense(self.an)
                ])

        model.compile(loss ='mse',
                      optimizer =adam_v2.Adam())
        
        self.model = model

    def train(self, epochs):
        print(self.features.shape)
        print("------------------")
        print(self.labels.shape)
        return self.model.fit(self.features, self.labels, epochs=epochs)

    def graph_loss(self, losses):

        plt.plot(range(1, len(losses) + 1), losses)
        fig_path = os.path.join("models", "imitation", "loss-11-9-22")
        plt.savefig(fig_path)
        #plt.show()

class A2CAgent:
    def __init__(self, state_space, action_space):

        self.tot_reward = []
        self.memory = deque(maxlen=500000)
        self.osn = state_space
        self.an = action_space
        self.opt = adam_v2.Adam(learning_rate=0.001)
        self.actor_model = self._build_actor_model()
        self.critic_model = self._build_critic_model()

    def _build_critic_model(self):
        model = Sequential()
        # Model maps from state space (continuous inputs, based on environment) to action space
        model.add(Dense(24, input_dim=self.osn, activation='relu'))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(self.an, activation='linear')) # actions: forward, left, right, back, rot clock, rot counter clock
        model.compile(loss='mse', optimizer=self.opt)
        return model

    def _build_actor_model(self):
        model = Sequential()
        # Model maps from state space (continuous inputs, based on environment) to action space
        model.add(Dense(24, input_dim=self.osn, activation='relu'))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(self.an, activation='softmax')) # actions: forward, left, right, back, rot clock, rot counter clock
        model.compile(loss='mse', optimizer=self.opt)
        return model

    def get_action(self, state):
        # Predict based on current environment inputs
        action_ps = self.actor_model.predict(state, verbose=0)
         # Sample from probability distribution, instead of taking max probability. Just for fun
        return np.random.choice(self.an, 1, p=action_ps)[0]

    def memorize(self, state, action, reward, next_state, done, step):
        self.memory.append((state, action, reward, next_state, done, step))
        
    def one_hot_encode_action(self, action, n_actions):
        encoded = np.zeros(n_actions, np.float32)
        encoded[action] = 1
        return encoded

    def train(self, state, action_ps, action, reward, done, next_state):
        """
        action: 
        done: int. 1 if complete, 0 if not
        """
        discount_factor = .7

        observation_reshaped = np.reshape(state, [1, self.osn])
        next_observation_reshaped = np.reshape(next_state, [1, self.osn])

        value_curr = np.asscalar(np.array(self.critic_model.predict(observation_reshaped)))
        value_next = np.asscalar(np.array(self.critic_model.predict(next_observation_reshaped)))

        TD_target = reward + (1 - done) * discount_factor * value_next
        advantage = TD_target - value_curr
        advantage_reshaped = np.vstack([advantage])

        TD_target_reshaped = np.vstack([TD_target])
        self.critic_model.train_on_batch(observation_reshaped, TD_target_reshaped)

        encoded_action = self.one_hot_encode_action(action, self.an)
        gradient = encoded_action - action_ps
        gradient_with_advantage = .0001 * gradient * advantage_reshaped + action_ps
        self.actor_model.train_on_batch(observation_reshaped, gradient_with_advantage)
        
        self.add_to_total_rewards(reward)


    def save_model(self, actor_path, critic_path):
        self.actor_model.save(actor_path)
        self.critic_model.save(critic_path)
        print(self.tot_reward)

    def add_to_total_rewards(self, run_reward):
        self.tot_reward.append(run_reward)


class BasicAgent:
    def __init__(self):
        self.model = self.build_model()

    def build_model(self):
        pass

    def get_action(self, state):
        return np.array([1.0, 0.0, 0.0])

