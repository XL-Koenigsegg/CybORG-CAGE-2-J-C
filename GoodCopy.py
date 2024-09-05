import sys
sys.path.append('/home/ubuntu/Downloads/cage-challenge-2/CybORG')
import inspect
from CybORG import CybORG
from CybORG.Agents import *
from CybORG.Shared.Actions import *
import pandas as pd
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers, models
from pprint import pprint

path = str(inspect.getfile(CybORG))
path = path[:-10] + '/Shared/Scenarios/Scenario2.yaml'
env = CybORG(path, 'sim')

class HostStatus:
    NOT_SCANNED = 0b00
    SCANNED_NOT_EXPLOITED = 0b01
    EXPLOITED_USER = 0b10
    EXPLOITED_ADMIN = 0b11

class AccessLevel:
    NONE = 0b00
    USER = 0b01
    ADMIN = 0b10
    UNKNOWN = 0b11

deviceNames = {
    'User0': 0,
    'User1': 1,
    'User2': 2,
    'User3': 3,
    'User4': 4,
    'Enterprise0': 5,
    'Enterprise1': 6,
    'Enterprise2': 7,
    'Op_Host0': 8,
    'Op_Host1': 9,
    'Op_Host2': 10,
    'Op_Server0': 11
}

def set_host_status(bit_vector, obs):
    affectedDevices = [key for key in obs if key != 'success']

    print("HELLO THIS IS BEING ATTACKED: ", affectedDevices)

    for key in affectedDevices:
       if not affectedDevices:
          return bit_vector

       else:
          #create the 4-bit representation for the device
          host_bits = (HostStatus.EXPLOITED_USER << 2) | AccessLevel.USER

          device_index = deviceNames.get(key)

          if device_index is None:
             continue

          #calculate the bit position in the bit vector
          bit_position = device_index * 4

          #clear the current 4 bits for the device
          bit_vector &= ~(0b1111 << bit_position)

          #set the new 4 bits for the device
          bit_vector |= (host_bits << bit_position)

   # print("HELLO THIS IS THE AFTERMATH: ", bin(bit_vector))

    return bit_vector

def get_host_status(bit_vector, host_index):
    bit_position = host_index * 4

    #extract the 4 bits of host bits
    host_bits = (bit_vector >> bit_position) & 0b1111

    #extract status and access levlel bits
    status = (host_bits >> 2) & 0b11
    access = host_bits & 0b11

    return status, access

def get_state(bit_vector):
    all_hosts_state = []

    for host_index in range(13):
       status, access = get_host_status(bit_vector, host_index)
       all_hosts_state.append((host_index, status, access))
    return all_hosts_state

def get_reward(bit_vector, action_used, redAction):
    reward_map = {
        0:-0.1,  # Subnet 1 Host 0
        1:-0.1,  # Subnet 1 Host 1
        2:-0.1,  # Subnet 1 Host 2
        3:-0.1,  # Subnet 1 Host 3
        4:-0.1,  # Subnet 1 Host 4
        5:-1,    # Subnet 2 Enterprise Server 0
        6:-1,    # Subnet 2 Enterprise Server 1
        7:-1,    # Subnet 2 Enterprise Server 2
        8:-0.1,  # Subnet 3 Host 0
        9:-0.1,  # Subnet 3 Host 1
        10:-0.1, # Subnet 3 Host 2
        11:-1    # Subnet 3 Operational Server 0
    }

    total_reward = 0
    state = get_state(bit_vector)

    #traverse through the current state vector and seek for root access-obtained hosts/servers
    for host_index, status, access in state:
       if access == AccessLevel.ADMIN:
          total_reward += reward_map.get(host_index, 0)

    #specific action rewards
    if action_used == 12:
       total_reward -= 1

    if redAction == "Impact":
       total_reward -= 10

    return total_reward

learning_rate = 0.5      #rate of new information overrides old information || Alpha
discount_factor = 0.5    #short-sighted / long-sighted decision making factor || Gamma
epsilon = 0.5            #exploration / exploitation

def step(action, env, results, bit_vector, agent):
    if action in (0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132):
       env.step(action=Sleep(), agent ='Blue')

    elif action in (1, 13, 25, 37, 49, 61, 73, 85, 97, 109, 121, 133):
       env.step(action=Monitor(session=0, agent='Blue'), agent='Blue')

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
       env.step(action=Analyse(session=0, agent='Blue', hostname=hostname), agent='Blue')

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
       env.step(action=DecoyApache(session=0, agent='Blue', hostname=hostname), agent='Blue')

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
       env.step(action=DecoyFemitter(session=0, agent='Blue', hostname=hostname), agent='Blue')

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
       env.step(action=DecoyHarakaSMPT(session=0, agent='Blue', hostname=hostname), agent='Blue')

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
       env.step(action=DecoySmss(session=0, agent='Blue', hostname=hostname), agent='Blue')

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
       env.step(action=DecoySSHD(session=0, agent='Blue', hostname=hostname), agent='Blue')

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
       env.step(action=DecoySvchost(session=0, agent='Blue', hostname=hostname), agent='Blue')

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
       env.step(action=DecoyTomcat(session=0, agent='Blue', hostname=hostname), agent='Blue')

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
       env.step(action=Remove(session=0, agent='Blue', hostname=hostname), agent='Blue')

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
       env.step(action=Restore(session=0, agent='Blue', hostname=hostname), agent='Blue')

    print(env.get_last_action('Blue'))

    #action = Analyse(hostname='User1',session=0,agent='Blue')
    #results = env.step(action = action, agent = 'Red')  #test the loop

    redAction_Space = results.action_space
    redObs = results.observation
    redAction = agent.get_action(redObs, redAction_Space)

    results = env.step(action=redAction, agent='Red')
    print(redAction)

    reward = get_reward(bit_vector, action, redAction)
    done = True

    next_state = env.get_observation('Blue')
    return next_state, reward, done, results

def create_DQN(input_shape, actions):
    model = models.Sequential()
    model.add(layers.Dense(128, input_shape=(input_shape,), activation='relu'))
    model.add(layers.Dense(129,activation='relu'))
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
       self.buffer.append(experience)

    def sample(self, batch_size):
       indices = np.random.choice(len(self.buffer), batch_size, replace=False)
       return [self.buffer[i] for i in indices]

def pick_action(state, model, epsilon):
    if np.random.rand() < epsilon:
       return np.random.randint(0, 144)

    else:
       state_vector = np.array([int(bit) for bit in bin(state)[2:].zfill(52)], dtype=np.float32)
       #print(state_vector)
       q_values = model.predict(state_vector.reshape(1,-1))
       return np.argmax(q_values)

def train_DQN(model, target_model, replay_buffer, batch_size):
    if len(replay_buffer.buffer) < batch_size:
       return

    minibatch = replay_buffer.sample(batch_size)

    for state, action, reward, next_state, done in minibatch:
       binary = format(state, '052b')
       binary_vector = np.array([int(bit) for bit in binary])

       #print("State:", binary_vector)
       #print("State size:", np.size(binary_vector))
       target_q = model.predict(binary_vector.reshape(1, 52))[0]

       if done:
           target_q[action] = reward
       else:
           next_binary = format(next_state, '052b')
           next_binary_vector = np.array([int(bit) for bit in next_binary])

           #print("Next State: ", next_binary_vector)
           #print("Next State Size: ", np.size(next_binary_vetor))

           next_q_values = target_model.predict(next_binary_vector.reshape(1, 52))[0]
           target_q[action] = reward + discount_factor * np.max(next_q_values)

       model.fit(binary_vector.reshape(1, 52), target_q[np.newaxis], epochs=1, verbose=0)

def training_Loop(env, model, traget_model, replay_buffer, num_episodes, results, agent):
    for episode in range(num_episodes):
       bit_vector = 0
       blue_observation = env.get_observation('Blue')
       bit_vector = set_host_status(bit_vector, blue_observation)
       done = False

       #pprint(blue_observation)

       while not done:
           #print(bit_vector)
           state = bit_vector
           action = pick_action(state, model, epsilon)
           #print("THIS IS THE ACTION NUMBERRRRRRRRRRR:", action)

           next_state, reward, done, results = step(action, env, results, bit_vector, agent)
           replay_buffer.add((state, action, reward, next_state, done))
           train_DQN(model, target_model, replay_buffer, batch_size=32)
           bit_vector = 0
           bit_vector = set_host_status(bit_vector, next_state)

       if episode % 10 ==0:
           target_model.set_weights(model.get_weights())

       print("WHILE LOOP DONEEEEEEEEEEEEEEEEE", episode)

input_shape = 52
num_actions = 144
results = env.reset('Red')
agent = B_lineAgent()

model = create_DQN(input_shape, num_actions)
target_model = create_DQN(input_shape, num_actions)
target_model.set_weights(model.get_weights())

replay_buffer = ReplayBuffer(max_size=500000)

training_Loop(env, model, target_model, replay_buffer, num_episodes=100, results=results, agent=agent)



