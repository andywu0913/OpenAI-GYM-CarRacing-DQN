import gym

is_pressed_left  = False # control left
is_pressed_right = False # control right
is_pressed_space = False # control gas
is_pressed_shift = False # control break
is_pressed_esc   = False # exit the game
steering_wheel = 0 # init to 0
gas            = 0 # init to 0
break_system   = 0 # init to 0

def key_press(key, mod):
    global is_pressed_left
    global is_pressed_right
    global is_pressed_space
    global is_pressed_shift
    global is_pressed_esc

    if key == 65361:
        is_pressed_left = True
    if key == 65363:
        is_pressed_right = True
    if key == 32:
        is_pressed_space = True
    if key == 65505:
        is_pressed_shift = True
    if key == 65307:
        is_pressed_esc = True

def key_release(key, mod):
    global is_pressed_left
    global is_pressed_right
    global is_pressed_space
    global is_pressed_shift

    if key == 65361:
        is_pressed_left = False
    if key == 65363:
        is_pressed_right = False
    if key == 32:
        is_pressed_space = False
    if key == 65505:
        is_pressed_shift = False

def update_action():
    global steering_wheel
    global gas
    global break_system

    if is_pressed_left ^ is_pressed_right:
        if is_pressed_left:
            if steering_wheel > -1:
                steering_wheel -= 0.1
            else:
                steering_wheel = -1
        if is_pressed_right:
            if steering_wheel < 1:
                steering_wheel += 0.1
            else:
                steering_wheel = 1
    else:
        if abs(steering_wheel - 0) < 0.1:
            steering_wheel = 0
        elif steering_wheel > 0:
            steering_wheel -= 0.1
        elif steering_wheel < 0:
            steering_wheel += 0.1
    if is_pressed_space:
        if gas < 1:
            gas += 0.1
        else:
            gas = 1
    else:
        if gas > 0:
            gas -= 0.1
        else:
            gas = 0
    if is_pressed_shift:
        if break_system < 1:
            break_system += 0.1
        else:
            break_system = 1
    else:
        if break_system > 0:
            break_system -= 0.1
        else:
            break_system = 0

if __name__ == '__main__':
    env = gym.make('CarRacing-v0')
    state = env.reset()
    env.unwrapped.viewer.window.on_key_press = key_press
    env.unwrapped.viewer.window.on_key_release = key_release

    counter = 0
    total_reward = 0
    while not is_pressed_esc:
        env.render()
        update_action()
        action = [steering_wheel, gas, break_system]
        state, reward, done, info = env.step(action)
        counter += 1
        total_reward += reward
        print('Action:[{:+.1f}, {:+.1f}, {:+.1f}] Reward: {:.3f}'.format(action[0], action[1], action[2], reward))
        if done:
            print("Restart game after {} timesteps. Total Reward: {}".format(counter, total_reward))
            counter = 0
            total_reward = 0
            state = env.reset()
            continue

    env.close()
