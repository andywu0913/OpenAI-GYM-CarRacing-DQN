import cv2
import gym
import numpy as np
from collections import deque
from train_model import CarRacingDQNAgent


EPISODES = 1

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
    agent.load('./save/trial_250.h5')
    done = False

    for e in range(EPISODES):
        init_state = env.reset()
        init_state = process_state_image(init_state)
        total_reward = 0
        punishment_counter = 0
        state_frame_stack_queue = deque([init_state]*agent.frame_stack_num, maxlen=agent.frame_stack_num)
        time_frame_counter = 1
        while True:
            env.render()

            current_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)
            action = agent.act(current_state_frame_stack)
            next_state, reward, done, info = env.step(action)

            total_reward += reward

            next_state = process_state_image(next_state)
            state_frame_stack_queue.append(next_state)
            next_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)

            if done:
                agent.update_target_model()
                print('Episode: {}/{}, Score(Time Frames): {}, Total Reward: {:.2}, Epsilon: {:.2}'.format(e+1, EPISODES, time_frame_counter, total_reward, agent.epsilon))
                break
            time_frame_counter += 1
