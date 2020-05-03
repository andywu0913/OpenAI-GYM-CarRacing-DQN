# OpenAI GYM CarRacing DQN

Training machines to play CarRacing 2d from OpenAI GYM by implementing Deep Q Learning/Deep Q Network(DQN) with TensorFlow and Keras as the backend.

### Training Results
We can see that the scores(time frames elapsed) stop rising after around 500 episodes as well as the rewards. Thus let's terminate the training and evaluate the model using the last three saved weight files `trial_400.h5`, `trial_500.h5`, and `trial_600.h5`.
<img src="resources/training_results.png" width="600px">

#### Model Trained After 400 Episodes
The model knows it should follow the track to acquire rewards after training 400 episodes, and it also knows how to take short cuts. However, making a sharp right turn still seems difficult to it, which results in getting stuck out of the track.
<img src="resources/trial_400.gif" width="300px">

#### Model Trained After 500 Episodes
<img src="resources/trial_500.gif" width="300px">

#### Model Trained After 600 Episodes
<img src="resources/trial_600.gif" width="300px">

## Useage

### Keyboard Agent
To play the game with your keyboard, execute the following command.
```
python play_car_racing_with_keyboard.py
```
- Control the steering wheel by using the `left` and `right` key.
- Control the gas by using the `space` key.
- Control the break by using the `shift` key.

### Train the DQN
```
python train_model.py [-m save/trial_XXX.h5] [-s 1] [-e 1000] [-p 1.0]
```
- `-m` The path to the trained model if you wish to continue training after it.
- `-s` The starting training episode, default to 1.
- `-e` The ending training episode, default to 1000.
- `-p` The starting epsilon of the agent, default to 1.0.

### DQN Agent
After having the DQN model trained, let's see how well did the model learned about playing CarRacing.
```
python play_car_racing_by_the_model.py -m save/trial_XXX.h5 [-e 1]
```
- `-m` The path to the trained model.
- `-e` The number of episodes should the model play.

## File Structure

- `train_model.py` The training program.
- `common_functions.py` Some functions that will be used in multiple programs will be put in here.
- `CarRacingDQNAgent.py` The core DQN class. Anything related to the model is placed in here.
- `play_car_racing_by_the_model.py` The program for playing CarRacing by the model.
- `play_car_racing_with_keyboard.py` The program for playing CarRacing with the keyboard.
- `save/` The default folder to save the trained model.
