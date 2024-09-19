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

episode = 20

# test = [0,0,0,0,0,0,0]
# print(test)

# if not test:
#     print("yea it's the same")

for i in range(episode):
    
    results = blueWrap.step(agent='Blue')

    #display in batch of 4, 4 bits per host, 13 hosts
    for j in range(0, len(results.observation), 4):
            batch = results.observation[j:j+4]
            print(batch)

    #print(results.observation)

    print('Blue Action: ', env.get_last_action("Blue"))
    print('Red Action: ', env.get_last_action("Red"))
    print(blueWrap.get_rewards())

    

    print("\n")