import gym
from collections import deque
from CarRacingDQNAgent import CarRacingDQNAgent
from common_functions import process_state_image
from common_functions import generate_state_frame_stack_from_queue

RENDER = True
START_EPISODE = 1
END_EPISODES = 1000
TRAINING_BATCH_SIZE = 64
SAVE_TRAINING_FREQUENCY = 25
UPDATE_TARGET_MODEL_FREQUENCY = 20

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

            reward = 0
            for _ in range(3):
                next_state, r, done, info = env.step(action)
                reward += r
                if done:
                    break

            # If continually getting negative reward 10 times after the race map is fully loaded, punishment will be triggered
            punishment_counter = punishment_counter + 1 if time_frame_counter > 100 and reward < 0 else 0

            # Reward the model more if it hits full gas
            reward = reward*1.2 if action[1] == 1 and action[2] == 0 else reward

            total_reward += reward

            if not done:
                next_state = process_state_image(next_state)
                state_frame_stack_queue.append(next_state)
                next_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)

                agent.memorize(current_state_frame_stack, action, reward, next_state_frame_stack, done)

            if done or punishment_counter >= 10 or total_reward < 0:
                print('Episode: {}/{}, Score(Time Frames): {}, Total Reward(adjusted): {:.2}, Epsilon: {:.2}'.format(e, END_EPISODES, time_frame_counter, total_reward, agent.epsilon))
                break
            if len(agent.memory) > TRAINING_BATCH_SIZE:
                agent.replay(TRAINING_BATCH_SIZE)
            time_frame_counter += 1

        if e % UPDATE_TARGET_MODEL_FREQUENCY == 0:
            agent.update_target_model()

        if e % SAVE_TRAINING_FREQUENCY == 0:
            agent.save('./save/trial_{}.h5'.format(e))

    env.close()
