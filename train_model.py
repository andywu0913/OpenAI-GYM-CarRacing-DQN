import cv2
import random
import gym
import numpy as np
import tensorflow as tf
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K


RENDER = True
START_EPISODE = 1
END_EPISODES = 1000
TRAINING_BATCH_SIZE = 64
TARGET_MODEL_UPDATE_FREQUENCY = 5


class CarRacingDQNAgent:
    def __init__(self):
        self.action_space = [
            (-1, 1, 0.2), (0, 1, 0.2), (1, 1, 0.2), #           Action Space Structure
            (-1, 1,   0), (0, 1,   0), (1, 1,   0), #        (Steering Wheel, Gas, Break)
            (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2), # Range        -1~1       0~1   0~1
            (-1, 0,   0), (0, 0,   0), (1, 0,   0)
        ]
        self.frame_stack_num = 3
        self.memory = deque(maxlen=5000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9999
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    """Huber loss for Q Learning

    References: https://en.wikipedia.org/wiki/Huber_loss
                https://www.tensorflow.org/api_docs/python/tf/losses/huber_loss
    """
    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond  = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def q_loss(self, y_true, y_pred):
        pass

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Conv2D(filters=8, kernel_size=(7, 7), strides=3, activation='relu', input_shape=(96, 96, self.frame_stack_num)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(len(self.action_space), activation=None))
        model.compile(loss=self._huber_loss, optimizer=Adam(lr=self.learning_rate, epsilon=1e-7))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, self.action_space.index(action), reward, next_state, done))

    def act(self, state):
        if np.random.rand() > self.epsilon:
            act_values = self.model.predict(np.expand_dims(state, axis=0))
            action_index = np.argmax(act_values[0])
        else:
            action_index = random.randrange(len(self.action_space))
        return self.action_space[action_index]

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        train_state = []
        train_target = []
        for state, action_index, reward, next_state, done in minibatch:
            target = self.model.predict(np.expand_dims(state, axis=0))[0]
            if done:
                target[action_index] = reward
            else:
                # a = self.model.predict(next_state)[0]
                t = self.target_model.predict(np.expand_dims(next_state, axis=0))[0]
                target[action_index] = reward + self.gamma * np.amax(t)
                # target[0][action_index] = reward + self.gamma * t[np.argmax(a)]
            train_state.append(state)
            train_target.append(target)
        self.model.fit(np.array(train_state), np.array(train_target), epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)
        self.update_target_model()

    def save(self, name):
        self.target_model.save_weights(name)


def process_state_image(state):
    state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    state = state.astype(float)
    state /= 255.0
    return state

def generate_state_frame_stack_from_queue(deque):
    frame_stack = np.array(deque)
    # Move stack dimension to the channel dimension (stack, x, y) -> (x, y, stack)
    return np.transpose(frame_stack, (1, 2, 0))


if __name__ == '__main__':
    env = gym.make('CarRacing-v0')
    agent = CarRacingDQNAgent()
    # agent.load('./save/trial_100.h5')
    done = False

    for e in range(START_EPISODE, END_EPISODES+1):
        init_state = env.reset()
        init_state = process_state_image(init_state)
        total_reward = 0
        punishment_counter = 0
        state_frame_stack_queue = deque([init_state]*agent.frame_stack_num, maxlen=agent.frame_stack_num)
        time_frame_counter = 1
        while True:
            if RENDER:
                env.render()

            current_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)
            action = agent.act(current_state_frame_stack)
            next_state, reward, done, info = env.step(action)

            # If continually getting negative reward 10 times after the race map is fully loaded, punishment will be triggered
            punishment_counter = punishment_counter + 1 if time_frame_counter > 100 and reward < 0 else 0

            # Reward the model more if it hits full gas
            reward = reward*1.5 if action[1] == 1 and action[2] == 0 else reward

            total_reward += reward

            next_state = process_state_image(next_state)
            state_frame_stack_queue.append(next_state)
            next_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)

            agent.memorize(current_state_frame_stack, action, reward, next_state_frame_stack, done)

            if done or punishment_counter >= 10 or total_reward < 0:
                if e % TARGET_MODEL_UPDATE_FREQUENCY == 0:
                    agent.update_target_model()
                print('Episode: {}/{}, Score(Time Frames): {}, Total Reward(adjusted): {:.2}, Epsilon: {:.2}'.format(e, END_EPISODES, time_frame_counter, total_reward, agent.epsilon))
                break
            if len(agent.memory) > TRAINING_BATCH_SIZE:
                agent.replay(TRAINING_BATCH_SIZE)
            time_frame_counter += 1

        if (e) % 25 == 0:
            agent.save('./save/trial_{0}.h5'.format(e))

    env.close()
