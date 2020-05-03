import random
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

class CarRacingDQNAgent:
    def __init__(
        self,
        action_space    = [
            (-1, 1, 0.2), (0, 1, 0.2), (1, 1, 0.2), #           Action Space Structure
            (-1, 1,   0), (0, 1,   0), (1, 1,   0), #        (Steering Wheel, Gas, Break)
            (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2), # Range        -1~1       0~1   0~1
            (-1, 0,   0), (0, 0,   0), (1, 0,   0)
        ],
        frame_stack_num = 3,
        memory_size     = 5000,
        gamma           = 0.95,  # discount rate
        epsilon         = 1.0,   # exploration rate
        epsilon_min     = 0.1,
        epsilon_decay   = 0.9999,
        learning_rate   = 0.001
    ):
        self.action_space    = action_space
        self.frame_stack_num = frame_stack_num
        self.memory          = deque(maxlen=memory_size)
        self.gamma           = gamma
        self.epsilon         = epsilon
        self.epsilon_min     = epsilon_min
        self.epsilon_decay   = epsilon_decay
        self.learning_rate   = learning_rate
        self.model           = self.build_model()
        self.target_model    = self.build_model()
        self.update_target_model()

    def build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Conv2D(filters=6, kernel_size=(7, 7), strides=3, activation='relu', input_shape=(96, 96, self.frame_stack_num)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=12, kernel_size=(4, 4), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(216, activation='relu'))
        model.add(Dense(len(self.action_space), activation=None))
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=self.learning_rate, epsilon=1e-7))
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
                t = self.target_model.predict(np.expand_dims(next_state, axis=0))[0]
                target[action_index] = reward + self.gamma * np.amax(t)
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
