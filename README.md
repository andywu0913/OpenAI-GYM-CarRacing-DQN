# OpenAI GYM CarRacing DQN

Training machines to play CarRacing from OpenAI GYM by implementing Deep Q Network(DQN)/Deep Q Learning with TensorFlow and Keras as the backend.

## Useage

### Keyboard Agent
To play the game with your keyboard, execute the following command.
```
python play_car_racing_with_keyboard.py
```
- Control the steering wheel by using the `left` and `right` key.
- Control the gas by using the `space` key.
- Control the break by using the `shift` key.

### Training the DQN
Training from scratch.
```
python train_model.py
```

### DQN Agent
After having the DQN model trained, let's see how well did the model learned about playing CarRacing.
```
python play_car_racing_by_the_model.py -m save/trial_XXX.h5 [-e 1]
```
- `-m` The path to the trained model.
- `-e` The number of episodes should the model play.
