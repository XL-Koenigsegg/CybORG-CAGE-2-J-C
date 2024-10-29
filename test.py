import sys
sys.path.append('/home/ubuntu/Downloads/cage-challenge-2/CybORG')
import inspect
from pprint import pprint
from CybORG import CybORG
from CybORG.Agents.Wrappers import *
from CybORG.Agents import *
from CybORG.Shared.Actions import *

path = str(inspect.getfile(CybORG))
path = path[:-10] + '/Shared/Scenarios/Scenario2.yaml'

env = CybORG(path,'sim', agents={'Red':B_lineAgent})
blueWrap = BlueTableWrapper(env, output_mode='vector')
results = blueWrap.reset('Blue') 

# env = CybORG(path,'sim', agents={'Red':B_lineAgent})
# results = env.reset('Blue') 
# action_space = results.action_space
# actions = action_space['action']
# pprint(action_space)
# print(len(action_space))
# pprint(actions)

episode = 18

# true_state = blueWrap.get_agent_state('True')
# pprint(true_state['Enterprise0'])

# true_state = blueWrap.get_agent_state('True')
# pprint(true_state['Enterprise1'])

# true_state = blueWrap.get_agent_state('True')
# pprint(true_state['Enterprise2'])

# true_state = blueWrap.get_agent_state('True')
# pprint(true_state['Op_Server0'])
   
# action_space = results.action_space
# actions = action_space['session']
# pprint(actions)

# test = [0,0,0,0,0,0,0]
# print(test)

# if not test:
#     print("yea it's the same")


# print('Blue Action: ', env.get_last_action("Blue"))
# print('Red Action: ', env.get_last_action("Red"))
# print(blueWrap.get_rewards())

# results = blueWrap.step(action=DecoyTomcat(session=1, agent='Blue', hostname='Enterprise1'), agent='Blue')
# print('Blue Action: ', env.get_last_action("Blue"))
# print('Red Action: ', env.get_last_action("Red"))
# print(blueWrap.get_rewards())

# results = blueWrap.step(action=DecoyTomcat(session=1, agent='Blue', hostname='Enterprise2'), agent='Blue')
# print('Blue Action: ', env.get_last_action("Blue"))
# print('Red Action: ', env.get_last_action("Red"))
# print(blueWrap.get_rewards())

# results = blueWrap.step(action=DecoySSHD(session=1, agent='Blue', hostname='Enterprise2'), agent='Blue')
# print('Blue Action: ', env.get_last_action("Blue"))
# print('Red Action: ', env.get_last_action("Red"))
# print(blueWrap.get_rewards())

# results = blueWrap.step(action=DecoySSHD(session=1, agent='Blue', hostname='Op_Server0'), agent='Blue')
# print('Blue Action: ', env.get_last_action("Blue"))
# print('Red Action: ', env.get_last_action("Red"))
# print(blueWrap.get_rewards())

for i in range(episode):
    executed = False
    print("Loop: ", i + 1)
    true_state = blueWrap.get_agent_state('True')
    true_state = true_state['Enterprise0']
    for Sessions in true_state['Sessions']:
          if Sessions['Agent'] == 'Red':
                agent_ID = Sessions['ID']
                executed = True
                results = blueWrap.step(action=DecoySSHD(session=agent_ID, agent='Blue', hostname='Enterprise0'), agent='Blue')
                #print("THIS IS THE AGENT ID:::::::", agent_ID)
                break        
    if executed is not True:
              
        results = blueWrap.step(agent='Blue')

        #display in batch of 4, 4 bits per host, 13 hosts
        # for j in range(0, len(results.observation), 4):
        #         batch = results.observation[j:j+4]
        #         #print(batch)

        #observation = env.get_observation('Blue') 
        #pprint(observation)
        
    #print(results.observation)
    print('Blue Action: ', env.get_last_action("Blue"))
    print('Red Action: ', env.get_last_action("Red"))
    print("\n")

    #pprint(true_state)
    #print(blueWrap.get_rewards())

    

          

    

    print("\n")

# results = blueWrap.step(action=DecoyTomcat(session=0, agent='Blue', hostname='Enterprise0'), agent='Blue')

# print('Blue Action: ', env.get_last_action("Blue"))
# print('Red Action: ', env.get_last_action("Red"))
# print("\n")


# for i in range(episode):
#     print("Loop: ", i + 1)
#     true_state = blueWrap.get_agent_state('True')
#     true_state = true_state['Enterprise0']
    
# #     print('Blue Action: ', env.get_last_action("Blue"))
# #     print('Red Action: ', env.get_last_action("Red"))
# #     print("\n")

# #     pprint(true_state)
#     results = blueWrap.step(agent='Blue')

# #     results = blueWrap.step(action=DecoySSHD(session=2, agent='Blue', hostname='Enterprise0'), agent='Blue')


#     print('Blue Action: ', env.get_last_action("Blue"))
#     print('Red Action: ', env.get_last_action("Red"))
#     #print("\n")

#     #pprint(true_state)

#     print("\n")

#true_state = blueWrap.get_agent_state('True')
#pprint(true_state['Enterprise0'])