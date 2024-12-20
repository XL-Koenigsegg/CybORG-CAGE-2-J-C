import sys
sys.path.append('/home/jasonhuo/cage-challenge-2/CybORG')
import inspect
from CybORG import CybORG
from CybORG.Agents import *
from CybORG.Shared.Actions import *
from CybORG.Agents.Wrappers import *
import pandas as pd
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers, models
from pprint import pprint
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/total_reward_tracking')

path = str(inspect.getfile(CybORG))
path = path[:-10] + '/Shared/Scenarios/Scenario2.yaml'
env = CybORG(path, 'sim', agents={'Red':B_lineAgent})
blueWrap = BlueTableWrapper(env, output_mode='vector')

# --- limit the action space for specific scenarios
# --- potential solutions for learning
#     --- state representation of 52 bit vector causing the issue
#     --- epsilon decay, instead of having a fixed epsilon value of 0.5
#          --- gonna use decay instead
#     --- learning rate too high maybe?
#          --- this needs to be played around with
#          --- gonna use 0.001 for now
#     --- replay buffer
#        --- buffer storing the correct content?
#        --- buffer size too big? (maybe pop() function is not being utilized enough)
#           --- changing the buffer size to 50 and batch size to 10 to reduce memory usage and recent experience update
#     --- frequency of updating the target network (maybe more frequent?)
#        --- change it to every 2 episodes
#     --- the reward after Impact Op_Server0?
#        --- FOUND! The current reward display I have in the loop doesn't add together the total reward
#        --- But!! The reward and the content within the experience is correct



#rate of new information overrides old information || Alpha
#before it was 0.5, trying out 0.001 to lower the learning rate
learning_rate = 0.001

#short-sighted / long-sighted decision making factor || Gamma
discount_factor = 0.9

#exploration / exploitation
#used to be 0.5, now epsilon decay with 0.01 min and 0.995 decay factor and using initial value of 1.0
epsilon = 1.0
decay = 0.995
minimum = 0.01

def step(action):
   if action in (0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132):
       blueObs=blueWrap.step(action=Sleep(), agent ='Blue')

   elif action in (1, 13, 25, 37, 49, 61, 73, 85, 97, 109, 121, 133):
       blueObs=blueWrap.step(action=Monitor(session=0, agent='Blue'), agent='Blue')

   elif action in (2, 14, 26, 38, 50, 62, 74, 86, 98, 110, 122, 134):
       analyseMapping = {
          2: "User0",
          14: "User1",
          26: "User2",
          38: "User3",
          50: "User4",
          62: "Enterprise0",
          74: "Enterprise1",
          86: "Enterprise2",
          98: "Op_Host0",
          110: "Op_Host1",
          122: "Op_Host2",
          134: "Op_Server0"
       }

       hostname = analyseMapping.get(action)
       blueObs=blueWrap.step(action=Analyse(session=0, agent='Blue', hostname=hostname), agent='Blue')

   elif action in (3, 15, 27, 39, 51, 63, 75, 87, 99, 111, 123, 135):
       apacheMapping = {
          3: "User0",
          15: "User1",
          27: "User2",
          39: "User3",
          51: "User4",
          63: "Enterprise0",
          75: "Enterprise1",
          87: "Enterprise2",
          99: "Op_Host0",
          111: "Op_Host1",
          123: "Op_Host2",
          135: "Op_Server0"
       }

       hostname = apacheMapping.get(action)
       blueObs=blueWrap.step(action=DecoyApache(session=0, agent='Blue', hostname=hostname), agent='Blue')

   elif action in (4, 16, 28, 40, 52, 64, 76, 88, 100, 112, 124, 136):
       femitterMapping = {
          4: "User0",
          16: "User1",
          28: "User2",
          40: "User3",
          52: "User4",
          64: "Enterprise0",
          76: "Enterprise1",
          88: "Enterprise2",
          100: "Op_Host0",
          112: "Op_Host1",
          124: "Op_Host2",
          136: "Op_Server0"
       }

       hostname = femitterMapping.get(action)
       blueObs=blueWrap.step(action=DecoyFemitter(session=0, agent='Blue', hostname=hostname), agent='Blue')

   elif action in (5, 17, 29, 41, 53, 65, 77, 89, 101, 113, 125, 137):
       harakaSMPTMapping = {
          5: "User0",
          17: "User1",
          29: "User2",
          41: "User3",
          53: "User4",
          65: "Enterprise0",
          77: "Enterprise1",
          89: "Enterprise2",
          101: "Op_Host0",
          113: "Op_Host1",
          125: "Op_Host2",
          137: "Op_Server0"
       }

       hostname = harakaSMPTMapping.get(action)
       blueObs=blueWrap.step(action=DecoyHarakaSMPT(session=0, agent='Blue', hostname=hostname), agent='Blue')

   elif action in (6, 18, 30, 42, 54, 66, 78, 90, 102, 114, 126, 138):
       smssMapping = {
          6: "User0",
          18: "User1",
          30: "User2",
          42: "User3",
          54: "User4",
          66: "Enterprise0",
          78: "Enterprise1",
          90: "Enterprise2",
          102: "Op_Host0",
          114: "Op_Host1",
          126: "Op_Host2",
          138: "Op_Server0"
       }

       hostname = smssMapping.get(action)
       blueObs=blueWrap.step(action=DecoySmss(session=0, agent='Blue', hostname=hostname), agent='Blue')

   elif action in (7, 19, 31, 43, 55, 67, 79, 91, 103, 115, 127, 139):
       sshdMapping = {
          7: "User0",
          19: "User1",
          31: "User2",
          43: "User3",
          55: "User4",
          67: "Enterprise0",
          79: "Enterprise1",
          91: "Enterprise2",
          103: "Op_Host0",
          115: "Op_Host1",
          127: "Op_Host2",
          139: "Op_Server0"
       }

       hostname = sshdMapping.get(action)
       blueObs=blueWrap.step(action=DecoySSHD(session=0, agent='Blue', hostname=hostname), agent='Blue')

   elif action in (8, 20, 32, 44, 56, 68, 80, 92, 104, 116, 128, 140):
       svchostMapping = {
          8: "User0",
          20: "User1",
          32: "User2",
          44: "User3",
          56: "User4",
          68: "Enterprise0",
          80: "Enterprise1",
          92: "Enterprise2",
          104: "Op_Host0",
          116: "Op_Host1",
          128: "Op_Host2",
          140: "Op_Server0"
       }

       hostname = svchostMapping.get(action)
       blueObs=blueWrap.step(action=DecoySvchost(session=0, agent='Blue', hostname=hostname), agent='Blue')

   elif action in (9, 21, 33, 45, 57, 69, 81, 93, 105, 117, 129, 141):
       tomcatMapping = {
          9: "User0",
          21: "User1",
          33: "User2",
          45: "User3",
          57: "User4",
          69: "Enterprise0",
          81: "Enterprise1",
          93: "Enterprise2",
          105: "Op_Host0",
          117: "Op_Host1",
          129: "Op_Host2",
          141: "Op_Server0"
       }

       hostname = tomcatMapping.get(action)
       blueObs=blueWrap.step(action=DecoyTomcat(session=0, agent='Blue', hostname=hostname), agent='Blue')

   elif action in (10, 22, 34, 46, 58, 70, 82, 94, 106, 118, 130, 142):
       removeMapping = {
          10: "User0",
          22: "User1",
          34: "User2",
          46: "User3",
          58: "User4",
          70: "Enterprise0",
          82: "Enterprise1",
          94: "Enterprise2",
          106: "Op_Host0",
          118: "Op_Host1",
          130: "Op_Host2",
          142: "Op_Server0"
       }

       hostname = removeMapping.get(action)
       blueObs=blueWrap.step(action=Remove(session=0, agent='Blue', hostname=hostname), agent='Blue')

   elif action in (11, 23, 35, 47, 59, 71, 83, 95, 107, 119, 131, 143):
       restoreMapping = {
          11: "User0",
          23: "User1",
          35: "User2",
          47: "User3",
          59: "User4",
          71: "Enterprise0",
          83: "Enterprise1",
          95: "Enterprise2",
          107: "Op_Host0",
          119: "Op_Host1",
          131: "Op_Host2",
          143: "Op_Server0"
       }

       hostname = restoreMapping.get(action)
       blueObs=blueWrap.step(action=Restore(session=0, agent='Blue', hostname=hostname), agent='Blue')

   print("Blue Action: ", blueWrap.get_last_action('Blue'))

   print("Red Action: ", blueWrap.get_last_action('Red'))

   done = False

   next_state = blueObs.observation
   print('HELLOO THIS IS NEXT STATE',next_state)
   return next_state, done, blueObs  

def create_DQN(input_shape, actions):
    model = models.Sequential()
    model.add(layers.Dense(140, input_shape=(input_shape,), activation='relu'))
    model.add(layers.Dense(140,activation='relu'))
    model.add(layers.Dense(actions, activation='linear'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model

class ReplayBuffer:
    def __init__(self, max_size):
       self.buffer=[]
       self.max_size = max_size

    def add(self, experience):
       if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)

       print("This is the experience: ", experience)
       self.buffer.append(experience)

    def sample(self, batch_size):
       indices = np.random.choice(len(self.buffer), batch_size, replace=False)
       return [self.buffer[i] for i in indices]

def pick_action(state, model, epsilon, episode):
    if np.random.rand() < epsilon:
       return np.random.randint(0, 144)

    else:
       state_list = np.array(state, dtype=np.float32)
       print(state_list)

       if len(state_list) < input_shape:
            state_list = np.pad(state_list, (0, input_shape - len(state_list)), mode='constant')

       q_values = model.predict(state_list.reshape(1,-1))

       return np.argmax(q_values)

def train_DQN(model, target_model, replay_buffer, batch_size):
    if len(replay_buffer.buffer) < batch_size:
       return

    minibatch = replay_buffer.sample(batch_size)

    for state, action, reward, next_state, done in minibatch:
       state_list = np.array(state)

       if len(state_list) < input_shape:
            state_list = np.pad(state_list, (0, input_shape - len(state_list)), mode='constant')

       target_q = model.predict(state_list.reshape(1, input_shape))[0]

       if done:
           target_q[action] = reward
       else:
           next_state_list = np.array(next_state)
           next_q_values = target_model.predict(next_state_list.reshape(1, input_shape))[0]
           target_q[action] = reward + discount_factor * np.max(next_q_values)

       model.fit(state_list.reshape(1, input_shape), target_q[np.newaxis], epochs=5, verbose=0)
       #backwards propagation


def training_Loop(epsilon, model, target_model, replay_buffer, num_episodes, steps_per_episode):
    for episode in range(num_episodes):
        results = blueWrap.reset('Blue')
        state = [0] * input_shape
        total_reward = 0.0
        done = False

        for steps in range(steps_per_episode):

            action = pick_action(state, model, epsilon, episode)
            next_state, done, results = step(action)

            reward = results.reward
            total_reward += reward
            print("This is the reward: ", total_reward)

            replay_buffer.add((state, action, reward, next_state, done))

            train_DQN(model, target_model, replay_buffer, batch_size=35)

            state = next_state
            
            if done:
                break

        done = True

        if episode % 10 == 0:
            target_model.set_weights(model.get_weights())

        epsilon = max(minimum, epsilon * decay)

        #average_reward = total_reward / steps_per_episode
        writer.add_scalar('Total Reward per Episode', total_reward, episode)

        print(f"Episode {episode+1}/{num_episodes} - Total Reward: {total_reward}, Epsilon: {epsilon}")

    writer.close()

input_shape = 52
num_actions = 144
total_reward = 0.0

model = create_DQN(input_shape, num_actions)
target_model = create_DQN(input_shape, num_actions)
target_model.set_weights(model.get_weights())

replay_buffer = ReplayBuffer(max_size=1000)

training_Loop(epsilon, model, target_model, replay_buffer, num_episodes=7000, steps_per_episode=50)

# source /home/jasonhuo/CybORG-CAGE-2-J-C/venv/bin/activate
# tensorboard --logdir=runs/total_reward_tracking

# nohup python3 GoodCopy.py &
# ps -ef | grep GoodCopy.py
# tensorboard --logdir=C:/Users/qqcom/Downloads/total_reward_tracking
# tensorboard --logdir=C:/Users/Jason/Downloads/Total_Reward
# python -m tensorboard.main --logdir=C:/Users/Jason/Downloads/Total_Reward

# scp jasonhuo@134.117.214.176:/home/jasonhuo/CybORG-CAGE-2-J-C/runs/total_reward_tracking/ C:/Users/qqcom/Downloads/total_reward_tracking

# scp jasonhuo@134.117.214.176:/home/jasonhuo/CybORG-CAGE-2-J-C/runs/total_reward_tracking/events.out.tfevents.1731984380.sci-jgoa.3370632.0 C:/Users/Jason/Downloads/Total_Reward
# try with personal computer since the Intel Xeon processor is too slow