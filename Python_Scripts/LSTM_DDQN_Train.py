# Scripts details


# ==========================================================================
# This script is used for running training imports all other scripts
#
# Car_EnvRL: This scripts contains following functions
#         >> Details of RL environment
#         >> Observe (State vector)
#         >> Reward Structure
#         >> Low-level planner for each high level goal and action choice
#         >> Step function
#         >> Reset function
#
# PID_Controller: This scripts contains following functions
#         >> PID Longitudinal control
#         >> PID Lateral Control
#
# Network_LSTM: This scripts contains following functions
#         >> High level network (Meta Model)
#         >> Low level network (Model)
#===========================================================================


from __future__ import print_function
# NORMAL FUNCTION IMPORTS

import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import math
from collections import deque
from tqdm import tqdm
import pylab
import csv

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- add PythonAPI for release mode --------------------------------------------
# These lines of code are important if you want to import other scripts
# ==============================================================================
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

import carla
from PID_Controller import VehiclePIDController
from Network_LSTM import Network_Model
from Car_EnvRL import CarEnv

# TENSORFLOW Imports 
import keras
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Add, Lambda, TimeDistributed, LSTM, concatenate
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K
from tensorflow.keras.layers import GaussianNoise


# The command to set frequency is: "carlaUE4 -benchmark -fps=30

# GLOBAL VARIABLES
# =========================================
#SHOW_PREVIEW = False
#IM_WIDTH = 640
#IM_HEIGHT = 480
SECONDS_PER_EPISODE = 20
REPLAY_MEMORY_SIZE = 10_000
MIN_REWARD = -200
EPISODES = 100
AGGREGATE_STATS_EVERY = 10
FPS = 60
name = "HDQN"
# ========================================


# This import and code prevents "prediction function callback" error
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# ======================================================================================================================================
# DQN Agent
# ======================================================================================================================================


class DQNAgent:
    def __init__(self):
        self.state_size = 14
        self.goal_size = 2
        self.action_size = 3
        self.vector_size = 15
        self.EPISODES = 9999
        self.memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.writing_bool = False
        self.reading_bool = False
        self.network = Network_Model()

        
        self.gamma = 0.8    # discount rate
        # The exploration has to be decreased after each batch of training
        self.epsilon = 1.0 # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999
        self.batch_size = 64
        self.train_start = 68
        self.TAU = 0.1
        self.scores, self.scores_r, self.scores_g, self.scores_a, self.episodes, self.episodes_r, self.average,self.average_r, self.average_g, self.average_a = [], [], [], [], [], [], [], [], [], [] 
        self.resume_training = False 


        # Create model for meta goals
        self.meta_model =  self.network.meta_model(input_shape = (3, 14), goal_space = self.goal_size)

        # create model for lower level actions
        self.model = self.network.controller_model(goal_shape = (1,), input_shape = (3,14), action_space = self.action_size)


        if (self.resume_training == True):
            print("Training resumed...")
            self.load()

        if (self.reading_bool == True):
            print("Building replay memory from csv file...")
            print("Might take few seconds")
            self.reading_and_replay()


    def remember(self, episode, state, history, goal, action, goal_reward, action_reward, next_state, next_history, done):
        self.memory.append((state, history, goal, action, goal_reward, action_reward, next_state, next_history, done))
        if episode >= 100:
            if len(self.memory) > self.train_start:
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay

        if self.writing_bool:
            with open('state_file1.csv', 'a+', newline = '') as csv_log_file:
                self.log_writer = csv.writer(csv_log_file)
                self.log_writer.writerow([float(state[0][0]), float(state[0][1]), float(state[0][2]) , float(state[0][3]), float(state[0][4]),float(state[0][5]),float(state[0][6]),float(state[0][7]),float(state[0][8]),float(state[0][9]), float(state[0][10]), float(state[0][11]), float(state[0][12]),float(state[0][13])])

            with open('next_state_file1.csv', 'a+', newline = '') as csv_log_file:
                self.log_writer = csv.writer(csv_log_file)
                self.log_writer.writerow([float(next_state[0][0]), float(next_state[0][1]), float(next_state[0][2]), float(next_state[0][3]), float(next_state[0][4]), float(next_state[0][5]), float(next_state[0][6]), float(next_state[0][7]), float(next_state[0][8]), float(next_state[0][9]), float(next_state[0][10]), float(next_state[0][11]), float(next_state[0][12]), float(next_state[0][13])])


            with open('other_files1.csv', 'a+', newline = '') as csv_other_file:
                self.other_writer = csv.writer(csv_other_file)
                self.other_writer.writerow([int(goal), int(action), int(goal_reward), int(action_reward), bool(done)])        


    def reading_and_replay(self):
        if self.reading_bool:
            state_list = []
            single_state_list = []
            with open('state_file.csv') as log_file:
                log_reader = csv.reader(log_file)
                for row in log_reader:
                    single_state_list.append(float(row[0]))
                    single_state_list.append(float(row[1]))                     
                    single_state_list.append(float(row[2]))
                    single_state_list.append(float(row[3]))
                    single_state_list.append(float(row[4]))
                    single_state_list.append(float(row[5]))
                    single_state_list.append(float(row[6]))
                    single_state_list.append(float(row[7]))
                    single_state_list.append(float(row[8]))
                    single_state_list.append(float(row[9]))
                    single_state_list.append(float(row[10]))
                    single_state_list.append(float(row[11]))                                                                               
                    single_state_list.append(float(row[12]))
                    single_state_list.append(float(row[13]))
                    single_state_list = tuple(single_state_list)

                    single_state_list = np.reshape(single_state_list, [1, 14])
                    #print(len(single_state_list))
                    #print(single_state_list)
                    state_list.append(single_state_list)
                    #print(state_list)
                    #print(state_list[0])
                    single_state_list = [] 
                    
            next_state_list = []
            single_state_list = []
            with open('next_state_file.csv') as log_file:
                log_reader = csv.reader(log_file)
                for row in log_reader:
                    single_state_list.append(float(row[0]))
                    single_state_list.append(float(row[1]))                     
                    single_state_list.append(float(row[2]))
                    single_state_list.append(float(row[3]))
                    single_state_list.append(float(row[4]))
                    single_state_list.append(float(row[5]))
                    single_state_list.append(float(row[6]))                                                                                                    
                    single_state_list.append(float(row[7]))
                    single_state_list.append(float(row[8]))
                    single_state_list.append(float(row[9]))
                    single_state_list.append(float(row[10]))
                    single_state_list.append(float(row[11]))
                    single_state_list.append(float(row[12]))
                    single_state_list.append(float(row[13]))
                    single_state_list = tuple(single_state_list)

                    single_state_list = np.reshape(single_state_list, [1, 14])
                    #print(len(single_state_list))
                    #print(single_state_list)
                    next_state_list.append(single_state_list)
                    #print(state_list)
                    #print(state_list[0])
                    single_state_list = []  

            action_list = []
            goal_list = []
            action_reward_list = []
            goal_reward_list = []
            done_list = []

            with open('other_files.csv') as log_other:
                log_reader_2 = csv.reader(log_other)
                for row in log_reader_2:
                    goal_list.append(int(row[0])) 
                    action_list.append(int(row[1]))                   
                    goal_reward_list.append(int(row[2]))
                    action_reward_list.append(int(row[3]))
                    done_list.append(eval(row[4]))


            for i in range(len(state_list)):
                #print(state_list[i])
                experience = (state_list[i], goal_list[i], action_list[i], goal_reward_list[i], action_reward_list[i], next_state_list[i], done_list[i])
                #print(experience)
                self.memory.append(experience)
            print(len(self.memory))


    def select_goal(self, history):
        if np.random.random() <= self.epsilon:
            #print("random--goal")
            return random.randrange(self.goal_size)
            time.sleep(1/FPS)
        else:
            #print("policy--GOAL")
            goal = np.argmax(self.meta_model.predict(history))
            return goal

    
    def act(self, goal, history):
        _goal = []
        _goal.append(goal)
        _goal = tuple(_goal)
        _goal = np.reshape(_goal,[1,1])   
        if np.random.random() <= self.epsilon:
            time.sleep(0.035)
            return random.randrange(self.action_size)
        else:
            #print("In here")
            action = np.argmax(self.model.predict([_goal, history]))
            #print("FINE UNTIL HERE CHECK 0")
            return action


    def replay(self):
        if len(self.memory) < self.train_start:
            print(len(self.memory))
            print("Returning back")
            return

        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))

        state = np.zeros((self.batch_size, self.state_size))
        history = np.zeros((self.batch_size, 3, 14))
        next_state = np.zeros((self.batch_size, self.state_size))
        next_history = np.zeros((self.batch_size, 3, 14))
        n_goal = np.zeros((self.batch_size, 1))        
        goal, action, goal_reward, action_reward, done = [], [], [], [], []

        for i in range(self.batch_size):
            state[i] = minibatch[i][0]
            history[i] = minibatch[i][1]
            goal.append(minibatch[i][2])
            np_goal = []
            np_goal.append(minibatch[i][2])
            np_goal = tuple(np_goal)
            np_goal = np.reshape(np_goal,[1,1])
            n_goal[i] = np_goal
            action.append(minibatch[i][3])
            goal_reward.append(minibatch[i][4])
            action_reward.append(minibatch[i][5])
            next_state[i] = minibatch[i][6]
            next_history[i] = minibatch[i][7]
            done.append(minibatch[i][8])


        # do batch predeiction to save speed for the goal network
        target = self.meta_model.predict(history)
        target_next = self.meta_model.predict(next_history)

        for i in range(self.batch_size):
            # correction on the Q value for the action used
            if done[i]:
                target[i][goal[i]] = goal_reward[i]
            else:
                target[i][goal[i]] = goal_reward[i] + self.gamma * (np.amax(target_next[i]))

        # Train the Neural Network with batches
        self.meta_model.fit(history, target, batch_size=self.batch_size, verbose=0)


        target2 = self.model.predict([n_goal, history])
        #print("FINE UNTIL HERE CHECK 1")
        target_next2 = self.model.predict([n_goal, next_history])
        #print("FINE UNTIL HERE CHECK 2")
        for i in range(self.batch_size):
            if done[i]:
                target2[i][action[i]] = action_reward[i]
            else:
                target2[i][action[i]] = action_reward[i] + self.gamma * (np.amax(target_next2[i]))

        #print("FINE UNTIL HERE CHECK 3")

        self.model.fit([n_goal, history], target2, batch_size=self.batch_size, verbose = 0) #, callbacks = [self.tensorboard] if log_this_step else None)



    def load(self):
        # Saving meta model
        episode = 1000 # The episode number you want to load
        self.meta_model = load_model("options_model"+str(episode)+".h5")
        self.model = load_model("planner_model"+str(episode)+".h5")

    def save(self, episode):
        # Saving the meta model
        print("Saving model and values...")
        self.meta_model.save("options_model"+str(episode)+".h5")
        # SAaving the planner model
        self.model.save("planner_model"+str(episode)+".h5")

        # Saving the values of the epsilon 
        f = open("HDQN_values.txt", "a")
        episode_str = str(episode)
        epsilon_str = str(self.epsilon)
        to_write = episode_str+" episode_epsilon_value"+" : "+ epsilon_str
        f.write("\n")
        f.write(to_write)
        f.close()

    pylab.figure(figsize=(18,9))
    def PlotReward(self, score, episode):
        self.scores_r.append(score)
        self.episodes_r.append(episode)
        self.average_r.append(sum(self.scores_r[-50:]) / len(self.scores_r[-50:]))
        pylab.plot(self.episodes_r, self.scores_r, 'C1')
        pylab.plot(self.episodes_r, self.average_r, 'b')
        pylab.ylabel('Score', fontsize=18)
        pylab.xlabel('Episodes', fontsize=18)
        dqn = 'Reward_LSTM_HDQN_Double'        

        try:
            pylab.savefig(dqn+".png")
        except OSError:
            pass

        return str(self.average_r[-1])[:5]            

'''
# Test the main
if __name__ == '__main__':
    agent = DQNAgent()
    env = CarEnv()
    episode_number = 1

    for episode in tqdm(range(episode_number + 10), ascii = True, unit = 'episodes'):

        env.collision_hist = []
        env.invasion_hist = []
        episode_reward = 0
        egoal_reward = 0
        eaction_reward = 0        
        step = 1
        state = env.reset()
        time.sleep(0.5)
        state = np.reshape(state, [1, agent.state_size])

        done = False
        episode_start = time.time()
        num_rand = random.randrange(0, 2)

        while True:
            if num_rand == 0:
                env.moving_vehicle.apply_control(carla.VehicleControl(throttle = 0.4, brake = 0.0, steer = 0.0))
                env.moving_vehicle2.apply_control(carla.VehicleControl(throttle = 0.6, brake = 0.0, steer = 0.0))
            else:
                env.moving_vehicle.apply_control(carla.VehicleControl(throttle = 0.6, brake = 0.0, steer = 0.0))
                env.moving_vehicle2.apply_control(carla.VehicleControl(throttle = 0.6, brake = 0.0, steer = 0.0))                
            #env.moving_vehicle2.apply_control(carla.VehicleControl(throttle = (random.randrange(8))/10, brake = 0.0, steer = 0.0))                                
            goal = np.argmax(agent.meta_model.predict(state))

            # Action selection based on weights
            _goal = []
            _goal.append(goal)             
            vector = tuple(np.concatenate([_goal, state[0]]))
            vector = np.reshape(vector, [1, 9])
            action = np.argmax(agent.model.predict(vector))

            #print("Into step")
            next_state, goal_reward, action_reward, done, _ = env.step(goal,action)
            #print("Out of step")
            egoal_reward += goal_reward
            eaction_reward += action_reward
            episode_reward += (goal_reward + action_reward)

            next_state = np.reshape(next_state, [1, agent.state_size])

            # Every step we update replay memory
            state = next_state
            step += 1

            if done:
                average = agent.PlotModel(episode_reward, episode)
                #agent.tensorboard.update_stats(reward_avg=average, episode_reward=episode_reward, options_reward=egoal_reward) #Episode_Reward=episode_reward, Options_reward=egoal_reward, Planner_Reward=eaction_reward)                    
                print('averag reward: ', average)
                print("Episode %d finished with total reward = %f." % (episode, episode_reward))
                for actor in env.actor_list:
                    actor.destroy()
                break
'''

if __name__ == '__main__':

    # Create agent and environment
    agent = DQNAgent()
    env = CarEnv()
    episode_number = 1

    # Iterate over episodes
    for episode in tqdm(range(episode_number, episode_number + 1000), ascii=True, unit='episodes'):
        env.collision_hist = []
        env.invasion_hist = []
        episode_reward = 0
        egoal_reward = 0
        eaction_reward = 0
        state = env.reset()
        time.sleep(0.5)
        state = np.reshape(state, [1, agent.state_size])
        done = False
        episode_start = time.time()
        num_rand = random.randrange(0, 4)
        step = 0
        # History from states
        history = np.stack((state, state, state), axis=0)
        history = np.reshape([history], (1,3,14))

        while True:
            if num_rand == 0:
                env.moving_vehicle.apply_control(carla.VehicleControl(throttle = 0.3, brake = 0.0, steer = 0.0))
                env.moving_vehicle2.apply_control(carla.VehicleControl(throttle = 0.3, brake = 0.0, steer = 0.0))
            elif num_rand == 1:
                env.moving_vehicle.apply_control(carla.VehicleControl(throttle = 0.4, brake = 0.0, steer = 0.0))
                env.moving_vehicle2.apply_control(carla.VehicleControl(throttle = 0.6, brake = 0.0, steer = 0.0))
            elif num_rand == 2:
                env.moving_vehicle.apply_control(carla.VehicleControl(throttle = 0.5, brake = 0.0, steer = 0.0))
                env.moving_vehicle2.apply_control(carla.VehicleControl(throttle = 0.5, brake = 0.0, steer = 0.0))
            elif num_rand == 3:
                env.moving_vehicle.apply_control(carla.VehicleControl(throttle = 0.3, brake = 0.0, steer = 0.0))
                env.moving_vehicle2.apply_control(carla.VehicleControl(throttle = 0.5, brake = 0.0, steer = 0.0)) 

            # Rule Based Episodes                               
            if episode < 100:
                goal, action = env.rule_based()
            else:
                goal = agent.select_goal(history)                   
                action = agent.act(goal, history)                  


            next_state, goal_reward, action_reward, done, _ = env.step(goal,action)
            egoal_reward += goal_reward
            eaction_reward += action_reward
            episode_reward += (goal_reward + action_reward)
            next_state = np.reshape(next_state, [1, agent.state_size])
           

            # Appending new state to the history containing three time step experiences
            next_history = next_state.reshape(1,1,14)
            next_history = np.append(next_history, history[:,:2,:], axis = 1)
            next_history = np.reshape([next_history], (1,3,14))         

            # Every step we update replay memory
            agent.remember(episode, state, history, goal, action, goal_reward, action_reward, next_state, next_history, done)

            state = next_state
            history = next_history
            step += 1
            #print("state vector", state)
            env.epi_steps += 1

            if done:
                for actor in env.actor_list:
                    actor.destroy()
                #average = agent.PlotModel(episode_reward, egoal_reward, eaction_reward, episode)
                average = agent.PlotReward(episode_reward, episode)
                #agent.tensorboard.update_stats(reward_avg=average, episode_reward=episode_reward, options_reward=egoal_reward) #Episode_Reward=episode_reward, Options_reward=egoal_reward, Planner_Reward=eaction_reward)
                print('averag reward: ', average)            
                print("Episode %d finished in %d steps with total reward = %f." % (episode, env.epi_steps, episode_reward))
                if (episode % 50) == 0:
                    agent.save(episode)
                #agent.remember(episode_buffer)
                agent.replay()
                if agent.epsilon > agent.epsilon_min:
                    agent.epsilon *= agent.epsilon_decay
                break
            #print("going into replay")
            # End of episode - destroy agents


